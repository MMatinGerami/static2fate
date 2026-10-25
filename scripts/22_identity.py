"""Identity: is the cell's own real future the nearest real candidate to its generated future?

The energy scores of 20 mix three things: realism, time after drug and whose cell it is. Here
every candidate is a real image at the target frame from a held-out cell of the same fate, so
the first two are held fixed and only identity is left. For each query the candidates are
ranked by Euclidean distance (features standardised as in 20) and the rank of the query cell's
own real future is recorded as a percentile: 0 when it is the nearest, 0.5 at chance.

Queries: each of the 8 saved samples of each model (ranks averaged per cell); the cell's own
0 h image ("copy 0 h"); and, for targets from 4 h on, the cell's real 3 h image ("real 3 h"),
a reference for how identifiable a cell is from a real frame 1 to 3 h earlier. Spaces and
resolutions as in 20, features from its cache (run 20 first). 95% CIs resample the 12 fields.

Usage: uv run scripts/22_identity.py
Writes results/tables/identity_rank.csv and results/figures/fig13_identity.png.
"""

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import field_bootstrap, group_of, rank_of_own  # noqa: E402

fs = importlib.import_module("20_feature_score")
ROOT = fs.ROOT
QUERIES = list(fs.MODELS) + ["copy 0 h", "real 3 h"]


def main() -> None:
    train = importlib.import_module("09_train_flow")
    _, tracks = train.split_index("test")
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    fate = lab.loc[tracks, "apoptotic"].to_numpy().astype(int)
    group = np.array([group_of(int(a), bool(c)) for a, c in zip(fate, lab.loc[tracks, "condensed"])])
    field = np.array([t[:10] for t in tracks])
    n = len(tracks)
    rows = []
    for res in fs.RES:
        real = {k: v.reshape(n, 1 + len(fs.FUTURE), -1) for k, v in np.load(fs.FEAT / f"test_{res}.npz").items()}
        gen = {}
        for name, d in fs.MODELS.items():
            tg, arrays = fs.load_samples(d)
            assert (tg == tracks).all(), name
            m = arrays[0].shape[2]
            key = f"{d.name}_{name.replace(' ', '').replace(',', '_')}_{res}.npz"
            gen[name] = {k: v.reshape(n, len(fs.FUTURE), m, -1) for k, v in np.load(fs.FEAT / key).items()}
        for sp in real:
            fut = real[sp][:, 1:].reshape(-1, real[sp].shape[-1])
            mu, sd = np.nanmean(fut, 0), np.nanstd(fut, 0) + 1e-8
            z = lambda a: (a - mu) / sd  # noqa: E731
            queries = {name: lambda j, name=name: z(gen[name][sp][:, j]) for name in fs.MODELS}
            queries["copy 0 h"] = lambda j: z(real[sp][:, 0])[:, None]
            queries["real 3 h"] = lambda j: z(real[sp][:, 1])[:, None] if j > 0 else None
            for j, f in enumerate(fs.FUTURE):
                target = z(real[sp][:, j + 1])
                for k in (0, 1):
                    cells = np.flatnonzero((fate == k) & np.isfinite(target).all(1))
                    cand = target[cells]
                    own = np.arange(len(cells))
                    for qname, qfn in queries.items():
                        q = qfn(j)
                        if q is None:
                            continue
                        q = q[cells]
                        okq = np.isfinite(q).all(-1)  # samples without a nucleus are left out
                        q = np.where(okq[..., None], q, 0.0)
                        pct, top1 = rank_of_own(q, cand, own)
                        pct, top1 = np.where(okq, pct, np.nan), np.where(okq, top1, np.nan)
                        for ci, i in enumerate(cells):
                            if np.isnan(pct[ci]).all():
                                continue
                            rows.append({"query": qname, "space": sp, "res": res, "track": tracks[i],
                                         "field": field[i], "group": group[i], "hours": f / 2,
                                         "candidates": len(cells), "pct_rank": np.nanmean(pct[ci]),
                                         "top1": np.nanmean(top1[ci])})
        print(f"done {res} px", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(fs.FEAT / "identity_cells.csv", index=False)
    by = ["query", "space", "res", "group", "hours"]
    tab = pd.concat([field_bootstrap(d, c, by).assign(measure=c) for c in ("pct_rank", "top1")])
    tab = tab.merge(d.groupby(by).candidates.first().reset_index(), on=by)
    tab.to_csv(ROOT / "results/tables/identity_rank.csv", index=False)
    pd.set_option("display.width", 250)
    for meas in ("pct_rank", "top1"):
        print(meas)
        print(tab[tab.measure == meas].pivot_table(index=["space", "res", "group"], columns=["query", "hours"],
                                                   values="estimate").round(3).to_string())

    groups = ["dies, condensed", "dies, AnnexinV only", "survives"]
    spaces = [("readouts", 64), ("handcrafted", 64), ("dinov2", 64), ("dinov2", 128)]
    fig, ax = plt.subplots(len(spaces), 3, figsize=(10, 2.6 * len(spaces)), sharex=True, sharey=True)
    styles = {"copy 0 h": ("grey", "--"), "real 3 h": ("black", ":")}
    for r, (sp, res) in enumerate(spaces):
        for c, grp in enumerate(groups):
            a = ax[r, c]
            for k, qname in enumerate(QUERIES):
                col, ls = styles.get(qname, (f"C{k}", "-"))
                s = tab[(tab.measure == "pct_rank") & (tab.space == sp) & (tab.res == res)
                        & (tab.group == grp) & (tab["query"] == qname)].sort_values("hours")
                a.plot(s.hours, s.estimate, ls, marker="o", color=col, label=qname, ms=3)
                a.fill_between(s.hours, s.ci_low, s.ci_high, color=col, alpha=0.1)
            a.axhline(0.5, color="grey", ls=":", lw=0.8)
            if r == 0:
                a.set_title(grp, fontsize=9)
            if c == 0:
                a.set_ylabel(f"{sp}, {res} px\npercentile rank of own future", fontsize=8)
            if r == len(spaces) - 1:
                a.set_xlabel("hours after staurosporine")
                a.set_xticks([3, 4, 5, 6])
    ax[0, 0].legend(fontsize=7)
    ax[0, 0].set_ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(ROOT / "results/figures/fig13_identity.png", dpi=110)


if __name__ == "__main__":
    main()
