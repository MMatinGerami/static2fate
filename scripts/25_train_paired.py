"""Paired flow: learn the change from a cell's earlier image to its later one.

Same data, split and network as 13_train_ctx.py, but the path does not start at Gaussian
noise. Source = the context image plus sigma x noise, target = the same cell at a later frame,
so the velocity field transports the cell to its future and identity is kept by construction;
the noise is what gives different futures from the same start. Sampling starts from the real
0 h image plus the same noise (19_generate_samples.py reads sigma from run.json).

Usage: uv run scripts/25_train_paired.py --minutes 45 --sigma 0.5 [--device mps|cuda]
Writes <out>/model.pt, train_log.csv and run.json (out defaults to results/flow_paired_s<sigma>).
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=float, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out) if args.out else ROOT / f"results/flow_paired_s{args.sigma:g}"
    out.mkdir(parents=True, exist_ok=True)
    train = importlib.import_module("09_train_flow")
    ctx_train = importlib.import_module("13_train_ctx")

    x, fate, _ = train.load("train", args.size)
    n = len(x)
    print(f"train cells {n}, size {args.size}, sigma {args.sigma}, dies {fate.float().mean():.2f}", flush=True)
    model = UNet(ch=args.ch, context=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    ema = UNet(ch=args.ch, context=True).to(args.device)
    ema.load_state_dict(model.state_dict())

    def save(step, t0):
        torch.save(ema.state_dict(), out / "model.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "device": args.device, "seed": args.seed, "steps": step, "batch": args.batch,
            "size": args.size, "ch": args.ch, "max_ctx_frame": ctx_train.MAX_CTX, "sigma": args.sigma,
            "float16": amp, "minutes": round((time.time() - t0) / 60, 1), "train_cells": n,
            "heldout_wells": train.HELDOUT_WELLS, "torch": torch.__version__}, indent=1))

    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log, step, t0, last_save = [], 0, time.time(), time.time()
    while time.time() - t0 < args.minutes * 60:
        ci, tc, t = ctx_train.pairs(n, args.batch)
        both = train.augment(torch.stack([x[ci, tc], x[ci, t]], 1)).to(args.device)
        ctx, target = both[:, 0], both[:, 1]
        source = ctx + args.sigma * torch.randn_like(ctx)
        with torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext():
            loss = flow_matching_loss(model, target, t.to(args.device), fate[ci].to(args.device),
                                      source=source, ctx=ctx, ctx_frame=tc.to(args.device))
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
