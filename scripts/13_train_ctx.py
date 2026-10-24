"""Train the flow model with one observed frame of the same cell as context.

Each training example is a cell, a context frame t_c drawn from 0 to 6 (0 to 3 h after drug)
and a target frame t drawn from t_c + 1 to 12. The model sees the context image and t_c and
generates frame t under the given fate (dropped 10% of the time for guidance). Same cells,
held-out wells and augmentation as 09_train_flow.py; context and target get the same flip and
rotation.

Usage: uv run scripts/13_train_ctx.py --minutes 60 [--device mps|cuda] [--size 64|128]
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
from static2fate.flow import UNet, flow_matching_loss  # noqa: E402

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
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = importlib.import_module("09_train_flow")

    x, fate, _ = train.load("train", args.size)
    n = len(x)
    print(f"train cells {n}, size {args.size}, dies {fate.float().mean():.2f}", flush=True)
    model = UNet(ch=args.ch, context=True).to(args.device)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    ema = UNet(ch=args.ch, context=True).to(args.device)
    ema.load_state_dict(model.state_dict())

    def save(step, t0):
        torch.save(ema.state_dict(), out / "model.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "device": args.device, "seed": args.seed, "steps": step, "batch": args.batch,
            "size": args.size, "ch": args.ch, "max_ctx_frame": MAX_CTX, "float16": amp,
            "minutes": round((time.time() - t0) / 60, 1), "train_cells": n,
            "heldout_wells": train.HELDOUT_WELLS, "torch": torch.__version__}, indent=1))

    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log, step, t0, last_save = [], 0, time.time(), time.time()
    while time.time() - t0 < args.minutes * 60:
        ci, tc, t = pairs(n, args.batch)
        both = train.augment(torch.stack([x[ci, tc], x[ci, t]], 1)).to(args.device)
        with torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext():
            loss = flow_matching_loss(model, both[:, 1], t.to(args.device), fate[ci].to(args.device),
                                      ctx=both[:, 0], ctx_frame=tc.to(args.device))
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
            log.append({"step": step, "minutes": round((time.time() - t0) / 60, 2), "loss": loss.item()})
            print(log[-1], flush=True)
        if time.time() - last_save > args.save_every * 60:
            save(step, t0)
            last_save = time.time()
        step += 1

    save(step, t0)
    print(f"done, {step} steps", flush=True)


if __name__ == "__main__":
    main()
