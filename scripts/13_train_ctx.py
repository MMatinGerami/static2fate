"""Train the flow model with one observed frame of the same cell as context.

Each training example is a cell, a context frame t_c drawn from 0 to 6 (0 to 3 h after drug)
and a target frame t drawn from t_c + 1 to 12. The model sees the context image and t_c and
generates frame t under the given fate (dropped 10% of the time for guidance). Same cells,
held-out wells and augmentation as 09_train_flow.py; context and target get the same flip and
rotation.

Usage: uv run scripts/13_train_ctx.py --minutes 60 [--device mps|cuda] [--size 64|128]
       [--init path/to/model.pt]   continue from saved (EMA) weights
       [--perceptual 0.1]          add a VGG16 feature term with this weight (static2fate.perceptual)
       [--ch 128]                  UNet width (64 by default)
       [--all-doses]               train on every dose, not only the four two-fate doses; the
                                   held-out wells, and so the evaluation, stay the same
       [--snapshot-every 120]      also keep the EMA weights as model_<step>.pt every so many
                                   minutes, to follow held-out loss over training
Writes <out>/model.pt (EMA weights), <out>/train_log.csv and <out>/run.json. A checkpoint is
saved every --save-every minutes so a cut Kaggle or Colab session loses little. On CUDA the
forward pass runs in float16 with loss scaling; on MPS and CPU everything stays float32.
"""

import argparse
import contextlib
import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.flow import UNet, flow_matching_losses  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_CTX = 6


def pairs(n: int, batch: int):
    ci = torch.randint(n, (batch,))
    tc = torch.randint(MAX_CTX + 1, (batch,))
    t = tc + 1 + (torch.rand(batch) * (12 - tc)).long()
    return ci, tc, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=float, default=15)
    ap.add_argument("--out", default=str(ROOT / "results/flow_ctx"))
    ap.add_argument("--init", default=None, help="model.pt to continue from")
    ap.add_argument("--perceptual", type=float, default=0.0, help="weight of the VGG16 feature term")
    ap.add_argument("--all-doses", action="store_true", help="train on all ten doses")
    ap.add_argument("--snapshot-every", type=float, default=0, help="minutes between kept snapshots")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = importlib.import_module("09_train_flow")

    x, fate, _ = train.load("train", args.size, doses=None if args.all_doses else train.DOSES)
    n = len(x)
    print(f"train cells {n}, size {args.size}, dies {fate.float().mean():.2f}", flush=True)
    model = UNet(ch=args.ch, context=True).to(args.device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=args.device, weights_only=True))
        print(f"continuing from {args.init}", flush=True)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    ema = UNet(ch=args.ch, context=True).to(args.device)
    ema.load_state_dict(model.state_dict())
    perceptual = None
    if args.perceptual > 0:
        from static2fate.perceptual import Perceptual

        perceptual = Perceptual().to(args.device)

    def save(step, t0):
        torch.save(ema.state_dict(), out / "model.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "device": args.device, "seed": args.seed, "steps": step, "batch": args.batch,
            "size": args.size, "ch": args.ch, "max_ctx_frame": MAX_CTX, "float16": amp,
            "init": args.init, "perceptual": args.perceptual, "all_doses": args.all_doses,
            "minutes": round((time.time() - t0) / 60, 1), "train_cells": n,
            "heldout_wells": train.HELDOUT_WELLS, "torch": torch.__version__}, indent=1))

    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log, step, t0, last_save, last_snap = [], 0, time.time(), time.time(), time.time()
    while time.time() - t0 < args.minutes * 60:
        ci, tc, t = pairs(n, args.batch)
        both = train.augment(torch.stack([x[ci, tc], x[ci, t]], 1)).to(args.device)
        with torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext():
            terms = flow_matching_losses(model, both[:, 1], t.to(args.device), fate[ci].to(args.device),
                                         perceptual=perceptual, ctx=both[:, 0], ctx_frame=tc.to(args.device))
            loss = terms["mse"] + args.perceptual * terms["perceptual"]
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step % 200 == 0:
            log.append({"step": step, "minutes": round((time.time() - t0) / 60, 2), "loss": loss.item(),
                        "mse": terms["mse"].item(), "perceptual": terms["perceptual"].item()})
            print(log[-1], flush=True)
        if args.snapshot_every and time.time() - last_snap > args.snapshot_every * 60:
            torch.save(ema.state_dict(), out / f"model_{step}.pt")
            last_snap = time.time()
        if time.time() - last_save > args.save_every * 60:
            save(step, t0)
            last_save = time.time()
        step += 1

    save(step, t0)
    print(f"done, {step} steps", flush=True)


if __name__ == "__main__":
    main()
