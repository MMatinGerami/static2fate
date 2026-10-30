"""Is the context model limited by its capacity or by the data? Train against held-out loss.

For each model, the flow matching MSE (fate given, no dropout, no augmentation) is computed on
training and held-out cells with the same draws: per cell, K random (context frame, target
frame, flow time, noise) tuples from a fixed seed. Equal losses on both sets mean the model has
not memorised the training cells, so a larger model or longer training could still help; a
held-out loss above the training loss means it is fitting the 1,919 training cells and more
cells, not more capacity, is what is missing. Loss is also split by flow time s, since realism
is decided near s = 1 (fine detail), where the two can differ even if the mean agrees.

Usage: uv run scripts/27_fit_check.py name=dir [name=dir ...]
Writes results/tables/fit_check.csv.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.flow import UNet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = 16  # draws per cell
S_BINS = [0, 0.25, 0.5, 0.75, 0.9, 1.0]


def draws(n, k, size, seed):
    g = torch.Generator().manual_seed(seed)
    tc = torch.randint(7, (n, k), generator=g)
    t = tc + 1 + (torch.rand(n, k, generator=g) * (12 - tc)).long()
    s = torch.rand(n, k, generator=g)
    z = torch.randn(n, k, 3, size, size, generator=g)
    return tc, t, s, z


@torch.no_grad()
def losses(model, x, fate, device, batch=256):
    n, size = len(x), x.shape[-1]
    tc, t, s, z = draws(n, K, size, seed=0)
    ci = torch.arange(n)[:, None].expand(n, K).reshape(-1)
    tc, t, s, z = tc.reshape(-1), t.reshape(-1), s.reshape(-1), z.reshape(-1, 3, size, size)
    out = torch.empty(len(ci))
    for b in range(0, len(ci), batch):
        sl = slice(b, b + batch)
        c = ci[sl]
        x1, ctx = x[c, t[sl]].to(device), x[c, tc[sl]].to(device)
        x0, s4 = z[sl].to(device), s[sl].to(device)[:, None, None, None]
        xs = (1 - s4) * x0 + s4 * x1
        v = model(xs, s[sl].to(device), t[sl].to(device), fate[c].to(device), ctx=ctx, ctx_frame=tc[sl].to(device))
        out[sl] = ((v - (x1 - x0)) ** 2).mean((1, 2, 3)).cpu()
    return pd.DataFrame({"cell": ci.numpy(), "s": s.numpy(), "mse": out.numpy()})


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = {split: train.load(split, 64)[:2] for split in ("train", "test")}
    rows = []
    for name, d in models.items():
        run = json.loads((d / "run.json").read_text())
        model = UNet(ch=run["ch"], context=True).to(device)
        model.load_state_dict(torch.load(d / "model.pt", map_location=device, weights_only=True))
        model.eval()
        for split, (x, fate) in data.items():
            df = losses(model, x, fate, device)
            df["bin"] = pd.cut(df.s, S_BINS, include_lowest=True).astype(str)
            per_cell = df.groupby("cell").mse.mean()
            rows.append({"model": name, "split": split, "s": "all", "mse": per_cell.mean(),
                         "sem": per_cell.std() / np.sqrt(len(per_cell)), "cells": len(per_cell)})
            for b, g in df.groupby("bin"):
                pc = g.groupby("cell").mse.mean()
                rows.append({"model": name, "split": split, "s": b, "mse": pc.mean(),
                             "sem": pc.std() / np.sqrt(len(pc)), "cells": len(pc)})
        print(f"done {name}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/fit_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.pivot_table(index=["model", "s"], columns="split", values="mse").round(4).to_string())


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()})
