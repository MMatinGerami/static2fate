"""Score the context model with several samples per cell, so it is compared fairly with copying.

14_eval_ctx.py drew one sample per cell. A single draw from a stochastic model is penalised for
its spread even when the spread is honest, while copying the 0 h frame has none. Here every
held-out cell gets M samples per future frame (true fate, context = its 0 h frame), and each
forecast is scored in the standardised readout space of 11 and 14:
  - energy score of the M samples (static2fate.evaluate.energy_score), which for a single
    deterministic forecast is the plain distance, so it is directly comparable with
  - copying the 0 h frame, and
  - the inversion model (deterministic, distances from results/flow/invert_branches.csv);
  - two stochastic baselines that never look at the image beyond the readouts, M draws each,
    to check that the model's credit for spread is not all it has: the cell's 0 h readouts
    plus the 0 h to frame change of a random training cell with the same fate ("copy + typical
    change"), and the readouts of a random training cell with the same fate at that frame;
  - also the distance of the mean readout over the M samples (the model as a point forecast)
    and of the first sample alone (the earlier one-sample comparison).
Paired differences against copy and inversion, 95% CIs from resampling the 12 held-out fields.

On CUDA the network runs in float16. Without results/flow/invert_branches.csv the inversion
columns are left empty.

Usage: uv run scripts/17_energy_score.py [model dir, default results/flow_ctx] [M, default 8] [cells]
       cells is a count (first n) or a range start:stop of the held-out cells, so one model can
       be split over several GPUs; readouts are standardised on all held-out cells either way
       uv run scripts/17_energy_score.py plot   (figure only, from the saved table)
Writes <model dir>/energy_cells.csv, results/tables/energy_score[_<model dir name>].csv and
results/figures/fig10_energy_score.png (default model only).
"""

import contextlib
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
from static2fate.evaluate import energy_score, field_bootstrap, group_of  # noqa: E402
from static2fate.flow import UNet, sample  # noqa: E402
from static2fate.readouts import cell_readouts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FUTURE = [6, 8, 10, 12]
CELLS_PER_BATCH = {64: 32, 128: 8}  # 8 samples each, conditional and null passes together


def readouts(x):
    if x.shape[-1] != 64:
        x = F.avg_pool2d(x, x.shape[-1] // 64)
    return cell_readouts(x)


def main(mdir: Path, m: int, cells: str | None) -> None:
    train = importlib.import_module("09_train_flow")
    run = json.loads((mdir / "run.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(ch=run["ch"], context=True).to(device)
    model.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x, fate, tracks = train.load("test", run["size"])
    real_all = np.stack([readouts(x[:, f]) for f in FUTURE])
    mu, sd = np.nanmean(real_all.reshape(-1, 4), 0), np.nanstd(real_all.reshape(-1, 4), 0)
    sel = slice(0, len(x))
    if cells:
        a, _, b = cells.partition(":")
        sel = slice(int(a), int(b)) if b else slice(0, int(a))
    x, fate, tracks, real_f = x[sel], fate[sel], tracks[sel], real_all[:, sel]
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    condensed = lab.loc[tracks, "condensed"].to_numpy()
    n, size = len(x), run["size"]
    print(f"held-out cells {n}, {m} samples each, size {size}, device {device}", flush=True)

    gen = np.full((len(FUTURE), n, m, 4), np.nan)  # readouts of every sample
    # float16 network passes on CUDA (the ODE state stays float32); MPS and CPU stay float32
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    g = torch.Generator().manual_seed(0)
    per_batch = CELLS_PER_BATCH[size]
    for b in range(0, n, per_batch):
        sl = slice(b, min(b + per_batch, n))
        k = sl.stop - sl.start
        ctx = x[sl, 0].repeat_interleave(m, 0).to(device)  # cell-major: cell 0 x m, cell 1 x m ...
        fk = fate[sl].repeat_interleave(m).to(device)
        for j, f in enumerate(FUTURE):
            z = torch.randn(k * m, 3, size, size, generator=g)
            with amp:
                out = sample(model, torch.full((k * m,), f, device=device), fk, x0=z.to(device), ctx=ctx,
                             ctx_frame=torch.zeros(k * m, dtype=torch.long, device=device)).float().cpu()
            gen[j, sl] = readouts(out).reshape(k, m, 4)
        print(f"cells {sl.stop}/{n}", flush=True)

    zs = (gen - mu) / sd
    rf = (real_f - mu) / sd
    r0 = (readouts(x[:, 0]) - mu) / sd
    inv_path = ROOT / "results/flow/invert_branches.csv"  # local only; absent on Kaggle
    inv = pd.read_csv(inv_path).set_index(["track", "frame"]) if inv_path.exists() else pd.DataFrame()

    # baselines from the training wells
    xt, fate_t, _ = train.load("train", run["size"])
    t0 = (readouts(xt[:, 0]) - mu) / sd
    tf = np.stack([(readouts(xt[:, f]) - mu) / sd for f in FUTURE])
    del xt
    fate_t = fate_t.numpy()
    rng = np.random.default_rng(0)

    rows = []
    for j, f in enumerate(FUTURE):
        for i in range(n):
            mean_ro = np.nanmean(zs[j, i], 0) if np.isfinite(zs[j, i]).all(1).any() else np.full(4, np.nan)
            key = (tracks[i], f)
            pool = rng.choice(np.flatnonzero(fate_t == int(fate[i])), m, replace=False)
            drift = r0[i] + (tf[j, pool] - t0[pool])
            rows.append({"track": tracks[i], "field": tracks[i][:10], "hours": f / 2,
                         "group": group_of(int(fate[i]), bool(condensed[i])),
                         "es_model": energy_score(zs[j, i], rf[j, i]),
                         "d_mean": np.linalg.norm(mean_ro - rf[j, i]),
                         "d_one": np.linalg.norm(zs[j, i, 0] - rf[j, i]),
                         "d_copy": np.linalg.norm(r0[i] - rf[j, i]),
                         "d_inversion": inv.loc[key, "d_true"] if key in inv.index else np.nan,
                         "es_drift": energy_score(drift, rf[j, i]),
                         "es_random_cell": energy_score(tf[j, pool], rf[j, i]),
                         "spread": np.nanmean(np.linalg.norm(zs[j, i][:, None] - zs[j, i][None], axis=2))})
    d = pd.DataFrame(rows).dropna(subset=["es_model", "d_copy"] + (["d_inversion"] if len(inv) else []))
    for a in ("es_model", "d_mean", "d_one", "d_inversion", "es_drift", "es_random_cell"):
        d[f"{a}_minus_copy"] = d[a] - d.d_copy
    d["es_model_minus_inversion"] = d.es_model - d.d_inversion
    d["es_model_minus_drift"] = d.es_model - d.es_drift
    d["es_beats_copy"] = (d.es_model < d.d_copy).astype(float)
    d.to_csv(mdir / "energy_cells.csv", index=False)

    cols = ["es_model_minus_copy", "d_mean_minus_copy", "d_one_minus_copy", "d_inversion_minus_copy",
            "es_drift_minus_copy", "es_random_cell_minus_copy", "es_model_minus_inversion",
            "es_model_minus_drift", "es_beats_copy"]
    tab = pd.concat([field_bootstrap(d, c, ["group", "hours"]).assign(measure=c) for c in cols])
    means = d.groupby(["group", "hours"])[["es_model", "es_drift", "es_random_cell", "d_mean", "d_one", "d_copy",
                                             "d_inversion", "spread"]].mean()
    tab = tab.merge(means.reset_index(), on=["group", "hours"])
    suffix = "" if mdir.resolve() == (ROOT / "results/flow_ctx").resolve() else f"_{mdir.name}"
    tab.to_csv(ROOT / f"results/tables/energy_score{suffix}.csv", index=False)
    print(means.round(2).to_string())
    print(tab.pivot_table(index=["measure", "group"], columns="hours", values="estimate").round(2).to_string())

    if not suffix:
        plot(tab, m)

def plot(tab: pd.DataFrame, m: int) -> None:
    groups = ["dies, condensed", "dies, AnnexinV only", "survives"]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for a, grp in zip(ax, groups):
        for meas, col, lab_ in [("d_inversion_minus_copy", "grey", "inversion"),
                                ("d_one_minus_copy", "C1", "context, 1 sample"),
                                ("es_drift_minus_copy", "C4", f"copy + typical change ({m})"),
                                ("es_model_minus_copy", "C0", f"context, energy score ({m})")]:
            s = tab[(tab.measure == meas) & (tab.group == grp)].sort_values("hours")
            a.plot(s.hours, s.estimate, "-o", color=col, label=lab_, ms=4)
            a.fill_between(s.hours, s.ci_low, s.ci_high, color=col, alpha=0.12)
        a.axhline(0, color="grey", ls=":")
        ns = tab[(tab.group == grp)].n
        a.set_title(f"{grp} (n={int(ns.max()) if len(ns) else 0})", fontsize=9)
        a.set_xlabel("hours after staurosporine")
        a.set_xticks([3, 4, 5, 6])
    ax[0].set_ylabel("score minus copying the 0 h frame\n(below 0 = better than copy)")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(ROOT / "results/figures/fig10_energy_score.png", dpi=110)


if __name__ == "__main__":
    if sys.argv[1:2] == ["plot"]:  # redraw the figure from the saved table
        plot(pd.read_csv(ROOT / "results/tables/energy_score.csv"), 8)
        sys.exit()
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/flow_ctx",
         int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         sys.argv[3] if len(sys.argv) > 3 else None)
