"""Evaluate the context model on held-out wells, against copying the context frame.

For every held-out cell and each context frame (0 h and 2 h), generate frames 6, 8, 10 and 12
(3 to 6 h) under both fates from the same noise. For each generated frame we ask:
  - beats copy: is the branch with the cell's true fate closer to the cell's real frame than
    the context frame itself is? (The trivial forecast is "nothing changes".)
  - true branch closer: is the true-fate branch closer to the real frame than the other branch?
The inversion model (11_invert_branches.py) is scored the same way from its saved distances,
so the three methods share one table. 95% CIs resample the 12 held-out fields.

Usage: uv run scripts/14_eval_ctx.py [model dir, default results/flow_ctx] [n cells]
Writes <model dir>/eval_cells.csv, results/tables/model_vs_copy.csv,
results/tables/ctx_readouts_6h_ctx{0,4}.csv and results/figures/fig8_context_vs_inversion.png.
"""

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import field_bootstrap, group_of  # noqa: E402
from static2fate.flow import UNet, sample  # noqa: E402
from static2fate.readouts import FEATURES, cell_readouts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FUTURE = [6, 8, 10, 12]
CONTEXTS = [0, 4]
BATCH = 64


def readouts(x):
    """Readouts are defined on 64 px crops; larger images are averaged down first."""
    if x.shape[-1] != 64:
        x = F.avg_pool2d(x, x.shape[-1] // 64)
    return cell_readouts(x)


def main(mdir: Path, limit: int | None) -> None:
    train = importlib.import_module("09_train_flow")
    run = json.loads((mdir / "run.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(ch=run["ch"], context=True).to(device)
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x, fate, tracks = train.load("test", run["size"])
    if limit:
        x, fate, tracks = x[:limit], fate[:limit], tracks[:limit]
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    condensed = lab.loc[tracks, "condensed"].to_numpy()
    n, size = len(x), run["size"]
    print(f"held-out cells {n}, size {size}, device {device}", flush=True)

    gen = torch.zeros(len(CONTEXTS), 2, len(FUTURE), n, 3, size, size)  # context, fate, frame, cell
    g = torch.Generator().manual_seed(0)
    for b in range(0, n, BATCH):
        sl = slice(b, min(b + BATCH, n))
        m = sl.stop - sl.start
        for j, f in enumerate(FUTURE):
            z = torch.randn(m, 3, size, size, generator=g)  # shared by both fates and contexts
            for c, tc in enumerate(CONTEXTS):
                for k in (0, 1):
                    gen[c, k, j, sl] = sample(
                        model, torch.full((m,), f, device=device), torch.full((m,), k, device=device),
                        x0=z.to(device), ctx=x[sl, tc].to(device),
                        ctx_frame=torch.full((m,), tc, device=device)).cpu()
        print(f"cells {sl.stop}/{n}", flush=True)

    real_f = np.stack([readouts(x[:, f]) for f in FUTURE])
    mu, sd = np.nanmean(real_f.reshape(-1, 4), 0), np.nanstd(real_f.reshape(-1, 4), 0)

    def z_(a):
        return (a - mu) / sd

    fate_np = fate.numpy()
    ft, idx = fate, torch.arange(n)
    rows = []
    for c, tc in enumerate(CONTEXTS):
        r_ctx = z_(readouts(x[:, tc]))
        for j, f in enumerate(FUTURE):
            rf = z_(real_f[j])
            g_true = z_(readouts(gen[c, :, j][ft, idx]))
            g_other = z_(readouts(gen[c, :, j][1 - ft, idx]))
            for i in range(n):
                rows.append({"method": f"context {tc * 30 // 60} h", "track": tracks[i], "field": tracks[i][:10],
                             "group": group_of(fate_np[i], condensed[i]), "hours": f / 2,
                             "d_true": np.linalg.norm(g_true[i] - rf[i]), "d_other": np.linalg.norm(g_other[i] - rf[i]),
                             "d_copy": np.linalg.norm(r_ctx[i] - rf[i])})
            if f == 12:
                g6 = readouts(gen[c, :, j][ft, idx])
                six = pd.DataFrame(np.r_[g6, real_f[j]], columns=FEATURES)
                six["source"] = ["generated"] * n + ["real"] * n
                six["fate"] = np.r_[fate_np, fate_np]
                six.groupby(["fate", "source"]).median().round(3).to_csv(
                    ROOT / f"results/tables/ctx_readouts_6h_ctx{tc}.csv")
    d = pd.DataFrame(rows)

    # inversion model, same cells, same metric (copy = its 0 h frame)
    inv = pd.read_csv(ROOT / "results/flow/invert_branches.csv")
    inv = inv[inv.track.isin(tracks)]
    r0 = pd.DataFrame(z_(readouts(x[:, 0])), index=tracks)
    rf_all = {f: pd.DataFrame(z_(real_f[j]), index=tracks) for j, f in enumerate(FUTURE)}
    inv = inv.assign(method="inversion 0 h", field=inv.track.str[:10], hours=inv.frame / 2,
                     group=[group_of(a, b) for a, b in zip(inv.fate, inv.condensed)],
                     d_copy=[np.linalg.norm(r0.loc[t].to_numpy() - rf_all[f].loc[t].to_numpy())
                             for t, f in zip(inv.track, inv.frame)])
    d = pd.concat([d, inv[d.columns]], ignore_index=True).dropna(subset=["d_true", "d_other", "d_copy"])
    d["beats_copy"] = (d.d_true < d.d_copy).astype(float)
    d["true_closer"] = (d.d_true < d.d_other).astype(float)
    d.to_csv(mdir / "eval_cells.csv", index=False)

    tab = pd.concat([field_bootstrap(d, col, ["method", "group", "hours"]).assign(measure=col)
                     for col in ("beats_copy", "true_closer")])
    tab.to_csv(ROOT / "results/tables/model_vs_copy.csv", index=False)
    print(tab[tab.measure == "beats_copy"].pivot_table(index=["group", "method"], columns="hours",
                                                       values="estimate").round(2).to_string())
    print(tab[tab.measure == "true_closer"].pivot_table(index=["group", "method"], columns="hours",
                                                        values="estimate").round(2).to_string())

    groups = ["dies, condensed", "dies, AnnexinV only", "survives"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for a, grp in zip(ax, groups):
        for meth, col in [("inversion 0 h", "grey"), ("context 0 h", "C0"), ("context 2 h", "C3")]:
            s = tab[(tab.measure == "beats_copy") & (tab.group == grp) & (tab.method == meth)].sort_values("hours")
            a.plot(s.hours, s.estimate, "-o", color=col, label=meth, ms=4)
            a.fill_between(s.hours, s.ci_low, s.ci_high, color=col, alpha=0.12)
        a.axhline(0.5, color="grey", ls=":")
        ns = tab[(tab.measure == "beats_copy") & (tab.group == grp)].n
        a.set_title(f"{grp} (n={int(ns.max()) if len(ns) else 0})", fontsize=9)
        a.set_xlabel("hours after staurosporine")
        a.set_xticks([3, 4, 5, 6])
    ax[0].set_ylabel("fraction of cells where model\nbeats copying the observed frame")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(ROOT / "results/figures/fig8_context_vs_inversion.png", dpi=110)

    # examples: 3 condensed, 3 AnnexinV-only, 3 survivors, context 2 h
    pick = [np.flatnonzero((fate_np == 1) & condensed)[:3], np.flatnonzero((fate_np == 1) & ~condensed)[:3],
            np.flatnonzero(fate_np == 0)[:3]]
    pick = [i for p in pick for i in p]
    fig, ax = plt.subplots(len(pick) * 2, len(FUTURE) + 1, figsize=(1.2 * (len(FUTURE) + 1), 1.2 * 2 * len(pick)))

    def show(a, im):
        im = ((im.numpy() + 1) / 2).clip(0, 1)
        a.imshow(np.dstack([im[2], im[1], im[0]]))
        a.set_xticks([])
        a.set_yticks([])

    for r, i in enumerate(pick):
        show(ax[2 * r, 0], x[i, 4])
        ax[2 * r + 1, 0].axis("off")
        for j, f in enumerate(FUTURE):
            show(ax[2 * r, j + 1], x[i, f])
            show(ax[2 * r + 1, j + 1], gen[1, fate_np[i], j, i])
        ax[2 * r, 0].set_ylabel(group_of(fate_np[i], condensed[i]).replace("dies, ", ""), fontsize=6)
        ax[2 * r + 1, 1].set_ylabel("generated", fontsize=6)
    ax[0, 0].set_title("2 h (given)", fontsize=7)
    for j, f in enumerate(FUTURE):
        ax[0, j + 1].set_title(f"{f // 2} h real", fontsize=7)
    plt.tight_layout()
    plt.savefig(mdir / "examples.png", dpi=90)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/flow_ctx",
         int(sys.argv[2]) if len(sys.argv) > 2 else None)
