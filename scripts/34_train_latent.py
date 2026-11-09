"""Train a flow model in the autoencoder's latent: the frame model (13) or the video model (28).

Inputs are the latents of 33_encode_latents.py, (cells, 13, 4, 32, 32) at unit scale. The
model is the same UNet as in pixel space with 4 + 4 input channels (code and context code) and
4 output channels at 32 px, so a step costs about a quarter of the 64 px model's and a
sixteenth of the 128 px model's while the decoder restores 128 px detail. Everything else is
as in 13_train_ctx.py and 28_train_video.py: context frame 0 to 6 (frame model) or 0 to 5
(video model, clip 6 to 12), fate dropped 10% of the time, flips and rotations (of the code,
see 33), EMA weights saved. A copy of the autoencoder (model.pt and run.json) is written next
to the model so 35_generate_latent.py can decode without the training machine's paths.

Usage: uv run scripts/34_train_latent.py --ae results/kaggle/ae128 --model frame|video
       --minutes 60 [--device mps|cuda] [--batch 64 (frame) | 16 (video)] [--ch 64]
       [--init model.pt] [--snapshot-every 120] [--no-augment] [--fates 3] [--out results/latent_frame]
--fates 3 conditions on three classes (survives / dies AnnexinV only / dies condensed, from
fate_labels.parquet through the tracks of the latents) instead of survives / dies.
Writes <out>/model.pt (EMA), <out>/ae/, <out>/train_log.csv and <out>/run.json.
"""

import argparse
import contextlib
import importlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.flow import UNet, flow_matching_loss  # noqa: E402
from static2fate.video import VideoUNet, video_flow_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_CTX = 6  # frame model, as 13_train_ctx.py
CLIP = list(range(6, 13))  # video model, as 28_train_video.py


def load_latents(path: Path):
    """z (cells, 13, 4, h, w) float32, fate (cells,) long, tracks."""
    d = np.load(path, allow_pickle=False)
    return torch.from_numpy(d["z"].astype(np.float32)), torch.from_numpy(d["fate"]), d["tracks"]


def build(model: str, ch: int, z: int, fates: int = 2):
    if model == "video":
        return VideoUNet(ch=ch, in_ch=z, n_fates=fates)
    return UNet(ch=ch, context=True, in_ch=z, n_fates=fates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", required=True, help="autoencoder directory (model.pt, run.json)")
    ap.add_argument("--latents", default=None, help="directory of 33_encode_latents.py, default <ae>/latents")
    ap.add_argument("--model", choices=["frame", "video"], default="frame")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=None, help="images (frame) or clips (video) per step")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=float, default=15)
    ap.add_argument("--snapshot-every", type=float, default=0, help="minutes between kept snapshots")
    ap.add_argument("--init", default=None, help="model.pt to continue from (video: a frame model also works)")
    ap.add_argument("--no-augment", action="store_true", help="no flips or rotations of the code")
    ap.add_argument("--fates", type=int, default=2, choices=[2, 3], help="fate classes (3: split dying cells)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    batch = args.batch or (16 if args.model == "video" else 64)
    out = Path(args.out or ROOT / f"results/latent_{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    train = importlib.import_module("09_train_flow")
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)

    ldir = Path(args.latents or Path(args.ae) / "latents")
    info = json.loads((ldir / "info.json").read_text())
    z, fate, tracks = load_latents(ldir / "train.npz")
    if args.fates != 2:
        fate = train.fate_classes(tracks, args.fates)
    n, nf = z.shape[:2]
    print(f"train cells {n}, latent {tuple(z.shape[2:])}, dies {(fate > 0).float().mean():.2f}, "
          f"{args.fates} fate classes {torch.bincount(fate, minlength=args.fates).tolist()}", flush=True)
    model = build(args.model, args.ch, info["z"], args.fates).to(args.device)
    if args.init:
        state = torch.load(args.init, map_location=args.device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("temporal.") for k in missing):
            raise ValueError(f"{args.init} does not match: unexpected {unexpected}, missing {missing}")
        print(f"continuing from {args.init}" + (" (temporal layers at zero)" if missing else ""), flush=True)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    ema = build(args.model, args.ch, info["z"], args.fates).to(args.device)
    ema.load_state_dict(model.state_dict())
    (out / "ae").mkdir(exist_ok=True)
    for f in ["model.pt", "run.json"]:
        shutil.copy(Path(args.ae) / f, out / "ae" / f)
    augment = (lambda x: x) if args.no_augment else train.augment
    clip_idx = torch.tensor(CLIP)
    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def save(step, t0):
        torch.save(ema.state_dict(), out / "model.pt")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        (out / "run.json").write_text(json.dumps({
            "model": args.model, "device": args.device, "seed": args.seed, "steps": step, "batch": batch,
            "size": info["size"], "z": info["z"], "latent_size": info["latent_size"],
            "latent_std": info["latent_std"], "ae": str(args.ae), "ch": args.ch, "fates": args.fates,
            "clip": CLIP if args.model == "video" else None,
            "max_ctx_frame": CLIP[0] - 1 if args.model == "video" else MAX_CTX,
            "float16": amp, "init": args.init, "augment": not args.no_augment,
            "minutes": round((time.time() - t0) / 60, 1), "train_cells": n,
            "heldout_wells": train.HELDOUT_WELLS, "torch": torch.__version__}, indent=1))

    log, step, t0, last_save, last_snap = [], 0, time.time(), time.time(), time.time()
    while time.time() - t0 < args.minutes * 60:
        ci = torch.randint(n, (batch,))
        with torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext():
            if args.model == "video":
                tc = torch.randint(CLIP[0], (batch,))
                both = augment(torch.cat([z[ci, tc][:, None], z[ci][:, clip_idx]], 1)).to(args.device)
                frame = clip_idx[None].expand(batch, -1).to(args.device)
                loss = video_flow_loss(model, both[:, 1:], frame, fate[ci].to(args.device), both[:, 0],
                                       tc.to(args.device))
            else:
                tc = torch.randint(MAX_CTX + 1, (batch,))
                t = tc + 1 + (torch.rand(batch) * (nf - 1 - tc)).long()
                both = augment(torch.stack([z[ci, tc], z[ci, t]], 1)).to(args.device)
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
