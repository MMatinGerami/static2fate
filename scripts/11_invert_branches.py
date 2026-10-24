"""Start from a real cell: invert its first frame, then generate both fates at later frames.

For every cell in the held-out wells:
  1. invert frame 0 (time of drug addition) to noise with the fate left out,
  2. regenerate frame 0 from that noise (round-trip check: does inversion keep the cell?),
  3. generate frames 6, 8, 10, 12 (3 h to 6 h) from the same noise, once with fate "survives"
     and once with "dies".

Two questions, both on the real future frames of the same cell, which the model never saw:
  A. Fate: is the branch with the cell's true fate closer to its real future than the other
     branch? Reported as the fraction of cells where it is, per frame and fate (0.5 = chance).
  B. Identity: is the generated future closer to this cell's real future than to the real future
     of another held-out cell with the same fate and dose? (0.5 = no identity kept.)
Distances are Euclidean in standardised readouts (nucleus area, nucleus Hoechst, cytoplasm green
and red; static2fate.readouts), standardised on the real held-out frames.
Writes results/flow/invert_branches.csv, invert_branches_summary.csv and invert_examples.png.
"""

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.flow import UNet, invert, sample  # noqa: E402
from static2fate.readouts import FEATURES, cell_readouts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/flow"
FUTURE = [6, 8, 10, 12]
BATCH = 64


def main(limit: int | None = None) -> None:
    train = importlib.import_module("09_train_flow")
    device = "mps"
    torch.manual_seed(0)
    model = UNet().to(device)
    model.load_state_dict(torch.load(OUT / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x, fate, tracks = train.load("test")
    if limit:
        x, fate, tracks = x[:limit], fate[:limit], tracks[:limit]
    labels = pd.read_parquet(ROOT / "data/processed/fate_labels.parquet").set_index("track")
    dose = labels.loc[tracks, "dose_nM"].to_numpy()
    condensed = labels.loc[tracks, "condensed"].to_numpy()
    n = len(x)
    print(f"held-out cells {n}, dies {fate.float().mean():.2f}", flush=True)

    recon = torch.zeros(n, 3, 64, 64)
    gen = torch.zeros(2, len(FUTURE), n, 3, 64, 64)  # fate, frame, cell
    for b in range(0, n, BATCH):
        sl = slice(b, min(b + BATCH, n))
        x0 = x[sl, 0].to(device)
        m = len(x0)
        z = invert(model, x0, torch.zeros(m, dtype=torch.long, device=device))
        null = torch.full((m,), 2, dtype=torch.long, device=device)
        recon[sl] = sample(model, torch.zeros(m, dtype=torch.long, device=device), null, guidance=1.0, x0=z.clone()).cpu()
        for k in (0, 1):
            for j, f in enumerate(FUTURE):
                gen[k, j, sl] = sample(model, torch.full((m,), f, device=device),
                                       torch.full((m,), k, device=device), x0=z.clone()).cpu()
        print(f"cells {sl.stop}/{n}", flush=True)

    real0 = cell_readouts(x[:, 0])
    rec = cell_readouts(recon)
    real_f = np.stack([cell_readouts(x[:, f]) for f in FUTURE])  # frame, cell, feat
    g = np.stack([np.stack([cell_readouts(gen[k, j]) for j in range(len(FUTURE))]) for k in (0, 1)])
    mu = np.nanmean(real_f.reshape(-1, 4), 0)
    sd = np.nanstd(real_f.reshape(-1, 4), 0)

    def z_(a):
        return (a - mu) / sd

    fate_np = fate.numpy()
    rng = np.random.default_rng(0)
    rows = []
    for j, f in enumerate(FUTURE):
        rf = z_(real_f[j])
        d_true = np.linalg.norm(z_(g[fate_np, j, np.arange(n)]) - rf, axis=1)
        d_other = np.linalg.norm(z_(g[1 - fate_np, j, np.arange(n)]) - rf, axis=1)
        # identity: same generated image vs the real future of a random other cell, same fate and dose
        d_self = d_true
        d_swap = np.full(n, np.nan)
        for i in range(n):
            pool = np.flatnonzero((fate_np == fate_np[i]) & (dose == dose[i]) & (np.arange(n) != i))
            if len(pool):
                d_swap[i] = np.linalg.norm(z_(g[fate_np[i], j, i]) - rf[rng.choice(pool)])
        for i in range(n):
            rows.append({"track": tracks[i], "dose_nM": dose[i], "fate": int(fate_np[i]),
                         "condensed": bool(condensed[i]), "frame": f,
                         "minutes": f * 30, "d_true": d_true[i], "d_other": d_other[i],
                         "d_self": d_self[i], "d_swap": d_swap[i]})
    df = pd.DataFrame(rows).dropna(subset=["d_true", "d_other"])
    df.to_csv(OUT / "invert_branches.csv", index=False)

    def frac(s, a, b):
        ok = s[[a, b]].dropna()
        return (ok[a] < ok[b]).mean()

    summ = []
    for (f, k), s in df.groupby(["frame", "fate"]):
        summ.append({"frame": f, "minutes": f * 30, "fate": "dies" if k else "survives", "n": len(s),
                     "true_branch_closer": frac(s, "d_true", "d_other"),
                     "own_cell_closer": frac(s, "d_self", "d_swap")})
    for (f, c), s in df[df.fate == 1].groupby(["frame", "condensed"]):
        summ.append({"frame": f, "minutes": f * 30, "fate": "dies, condensed" if c else "dies, AnnexinV only",
                     "n": len(s), "true_branch_closer": frac(s, "d_true", "d_other"),
                     "own_cell_closer": frac(s, "d_self", "d_swap")})
    for f, s in df.groupby("frame"):
        bal = np.mean([frac(s[s.fate == k], "d_true", "d_other") for k in (0, 1)])
        summ.append({"frame": f, "minutes": f * 30, "fate": "balanced", "n": len(s),
                     "true_branch_closer": bal, "own_cell_closer": frac(s, "d_self", "d_swap")})
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "invert_branches_summary.csv", index=False)
    rec_err = np.nanmedian(np.abs(z_(rec) - z_(real0)), 0)
    print("round-trip error at frame 0 (median |z| per readout):", dict(zip(FEATURES, rec_err.round(2))))
    print("pixel MSE round trip:", float(((recon - x[:, 0]) ** 2).mean()))
    print(sm.round(3).to_string(index=False))

    # figure: 4 dying and 4 surviving cells, real frames vs the two branches
    pick = list(np.flatnonzero(fate_np == 1)[:4]) + list(np.flatnonzero(fate_np == 0)[:4])
    fig, ax = plt.subplots(len(pick) * 3, len(FUTURE) + 2, figsize=(1.25 * (len(FUTURE) + 2), 1.25 * 3 * len(pick)))

    def show(a, im):
        im = ((im.numpy() + 1) / 2).clip(0, 1)
        a.imshow(np.dstack([im[2], im[1], im[0]]))
        a.set_xticks([])
        a.set_yticks([])

    for r, i in enumerate(pick):
        show(ax[3 * r, 0], x[i, 0])
        show(ax[3 * r + 1, 0], recon[i])
        ax[3 * r + 2, 0].axis("off")
        ax[3 * r, 1].axis("off")
        ax[3 * r + 1, 1].axis("off")
        ax[3 * r + 2, 1].axis("off")
        for j, f in enumerate(FUTURE):
            show(ax[3 * r, j + 2], x[i, f])
            show(ax[3 * r + 1, j + 2], gen[0, j, i])
            show(ax[3 * r + 2, j + 2], gen[1, j, i])
        truth = "dies" if fate_np[i] else "survives"
        ax[3 * r, 0].set_ylabel(f"real ({truth})", fontsize=6)
        ax[3 * r + 1, 0].set_ylabel("round trip", fontsize=6)
        ax[3 * r + 1, 2].set_ylabel("gen survives", fontsize=6)
        ax[3 * r + 2, 2].set_ylabel("gen dies", fontsize=6)
    ax[0, 0].set_title("0 h", fontsize=7)
    for j, f in enumerate(FUTURE):
        ax[0, j + 2].set_title(f"{f * 30 // 60} h {f * 30 % 60:02d}", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "invert_examples.png", dpi=90)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
