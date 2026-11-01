"""Train the video model: the 3 to 6 h course of a cell as one clip, given an earlier frame.

Each training example is a cell, a context frame t_c drawn from 0 to 5 (0 to 2.5 h) and the
clip of frames 6 to 12 (seven frames, 3 to 6 h, every 30 min). The model sees the context
image and t_c and generates the whole clip under the given fate (dropped 10% of the time for
guidance). Same cells, held-out wells and augmentation as 13_train_ctx.py; context and clip
get the same flip and rotation. A batch of 16 clips is 112 images, close to the 64 images per
batch of the frame model times the seven frames it would take to cover the same span.

Usage: uv run scripts/28_train_video.py --minutes 60 [--device mps|cuda] [--size 64]
       [--ch 64] [--all-doses] [--snapshot-every 120] [--init model.pt]
--init takes a video model or a frame model (its temporal layers then start at zero).
Writes <out>/model.pt (EMA weights), <out>/train_log.csv and <out>/run.json ("clip" lists the
frames generated). Float16 on CUDA with loss scaling, float32 elsewhere.
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
from static2fate.video import VideoUNet, video_flow_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLIP = list(range(6, 13))  # 3 to 6 h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16, help="clips per step")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=float, default=15)
    ap.add_argument("--out", default=str(ROOT / "results/flow_video"))
    ap.add_argument("--init", default=None, help="model.pt to continue from")
    ap.add_argument("--all-doses", action="store_true", help="train on all ten doses")
    ap.add_argument("--snapshot-every", type=float, default=0, help="minutes between kept snapshots")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = importlib.import_module("09_train_flow")

    x, fate, _ = train.load("train", args.size, doses=None if args.all_doses else train.DOSES)
    n = len(x)
    print(f"train cells {n}, size {args.size}, clip frames {CLIP}, dies {fate.float().mean():.2f}", flush=True)
    model = VideoUNet(ch=args.ch).to(args.device)
    if args.init:
        state = torch.load(args.init, map_location=args.device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("temporal.") for k in missing):
            raise ValueError(f"{args.init} does not match: unexpected {unexpected}, missing {missing}")
        print(f"continuing from {args.init}" + (" (frame model, temporal layers at zero)" if missing else ""), flush=True)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    ema = VideoUNet(ch=args.ch).to(args.device)
    ema.load_state_dict(model.state_dict())
    clip_idx = torch.tensor(CLIP)

    def save(step, t0):
        torch.save(ema.state_dict(), out / "model.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "device": args.device, "seed": args.seed, "steps": step, "batch": args.batch,
            "size": args.size, "ch": args.ch, "clip": CLIP, "max_ctx_frame": CLIP[0] - 1,
            "float16": amp, "init": args.init, "all_doses": args.all_doses,
            "minutes": round((time.time() - t0) / 60, 1), "train_cells": n,
            "heldout_wells": train.HELDOUT_WELLS, "torch": torch.__version__}, indent=1))

    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log, step, t0, last_save, last_snap = [], 0, time.time(), time.time(), time.time()
    while time.time() - t0 < args.minutes * 60:
        ci = torch.randint(n, (args.batch,))
        tc = torch.randint(CLIP[0], (args.batch,))
        both = train.augment(torch.cat([x[ci, tc][:, None], x[ci][:, clip_idx]], 1)).to(args.device)
        frame = clip_idx[None].expand(args.batch, -1).to(args.device)
        with torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext():
            loss = video_flow_loss(model, both[:, 1:], frame, fate[ci].to(args.device), both[:, 0], tc.to(args.device))
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
        if step % 100 == 0:
            log.append({"step": step, "minutes": round((time.time() - t0) / 60, 2), "loss": loss.item()})
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
