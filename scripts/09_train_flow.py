"""Train the conditional flow-matching model on the two-fate doses.

Images: every frame of every labelled cell at 4.88 to 39.06 nM, downsampled 128 -> 64 px and
scaled to [-1, 1]. Fields are split by well into train and held-out (one well per dose held out)
so the evaluation script can test on unseen cells. Random flips and 90 degree rotations.

Usage: uv run scripts/09_train_flow.py --minutes 30 [--device mps]
Writes results/flow/model.pt, results/flow/train_log.csv and results/flow/run.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from static2fate.flow import UNet, flow_matching_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "results/flow"
DOSES = [4.88, 9.77, 19.53, 39.06]
HELDOUT_WELLS = ["E-06", "E-07", "E-08", "E-09"]  # one replicate well per dose


def load(split: str):
    lab = pd.read_parquet(P / "fate_labels.parquet").set_index("track")
    tracks = np.load(P / "crops_tracks.npy")
    keep = np.isin(tracks, lab.index[lab["dose_nM"].isin(DOSES)])
    well = np.array([t[:4] for t in tracks])
    held = np.isin(well, HELDOUT_WELLS)
    keep &= held if split == "test" else ~held
    idx = np.flatnonzero(keep)
    x = np.load(P / "crops.npy", mmap_mode="r")[idx]
    x = torch.from_numpy(np.ascontiguousarray(x)).float().div(127.5).sub(1)  # (n, 13, 3, 128, 128)
    x = F.avg_pool2d(x.flatten(0, 1), 2).view(len(idx), 13, 3, 64, 64)
    fate = torch.from_numpy(lab.loc[tracks[idx], "apoptotic"].to_numpy().astype(np.int64))
    return x, fate, tracks[idx]


def augment(x):
    if torch.rand(1) < 0.5:
        x = x.flip(-1)
    if torch.rand(1) < 0.5:
        x = x.flip(-2)
    return torch.rot90(x, int(torch.randint(4, (1,))), dims=(-2, -1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    x, fate, _ = load("train")
    n = len(x)
    print(f"train cells {n}, images {n * 13}, dies {fate.float().mean():.2f}", flush=True)
    model = UNet().to(args.device)
    print(f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    ema = UNet().to(args.device)
    ema.load_state_dict(model.state_dict())

    log, step, t0 = [], 0, time.time()
    while time.time() - t0 < args.minutes * 60:
        ci = torch.randint(n, (args.batch,))
        fi = torch.randint(13, (args.batch,))
        xb = augment(x[ci, fi]).to(args.device)
        loss = flow_matching_loss(model, xb, fi.to(args.device), fate[ci].to(args.device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step % 200 == 0:
            log.append({"step": step, "minutes": round((time.time() - t0) / 60, 2), "loss": loss.item()})
            print(log[-1], flush=True)
        step += 1

    torch.save(ema.state_dict(), OUT / "model.pt")
    pd.DataFrame(log).to_csv(OUT / "train_log.csv", index=False)
    (OUT / "run.json").write_text(json.dumps({
        "device": args.device, "seed": args.seed, "steps": step, "batch": args.batch,
        "minutes": round((time.time() - t0) / 60, 1), "train_cells": n, "heldout_wells": HELDOUT_WELLS,
        "torch": torch.__version__}, indent=1))
    print(f"done, {step} steps", flush=True)


if __name__ == "__main__":
    main()
