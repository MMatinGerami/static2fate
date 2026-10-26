"""Identity against dynamics for the samplers that start from the cell (23_keep_cell.py).

x: percentile rank of the cell's own real future among same-fate real cells (22_identity.py;
0 = nearest, 0.5 = chance), lower is better. y: energy score of the samples minus the
"copy + typical change" baseline in readout space (20_feature_score.py), lower is better.
One point per sampler and the 0 h copy, per group, at 6 h, with field-bootstrap 95% CIs.
Usage: uv run scripts/24_tradeoff.py <tag> [name=dir ...]
The tag selects the tables written by 20 and 22 under STATIC2FATE_TAG=<tag>; the name=dir
pairs (same as given to 20 and 22) are used for the example panel.
Writes results/tables/tradeoff<tag>.csv, results/figures/fig14_identity_vs_dynamics<tag>.png
and fig14b_examples<tag>.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "_keep"
ARGS = dict(a.split("=", 1) for a in sys.argv[2:])
SPACE_ID, RES_ID = "dinov2", 64  # identity measured in DINOv2 space at 64 px
HOURS = 6.0

ident = pd.read_csv(ROOT / f"results/tables/identity_rank{TAG}.csv")
ident = ident[(ident.measure == "pct_rank") & (ident.space == SPACE_ID) & (ident.res == RES_ID) & (ident.hours == HOURS)]
ident = ident.rename(columns={"query": "sampler", "estimate": "identity", "ci_low": "id_low", "ci_high": "id_high"})
feat = pd.read_csv(ROOT / f"results/tables/feature_score{TAG}.csv")
dyn = feat[(feat.measure == "model_minus_drift") & (feat.space == "readouts") & (feat.hours == HOURS)]
dyn = dyn.rename(columns={"model": "sampler", "estimate": "dynamics", "ci_low": "dyn_low", "ci_high": "dyn_high"})
copy = feat[(feat.measure == "drift_minus_copy") & (feat.space == "readouts") & (feat.hours == HOURS)]
copy = copy.groupby("group").first().reset_index()  # same for every model
copy = pd.DataFrame({"sampler": "copy 0 h", "group": copy.group, "dynamics": -copy.estimate,
                     "dyn_low": -copy.ci_high, "dyn_high": -copy.ci_low})
dyn = pd.concat([dyn, copy])
t = ident.merge(dyn, on=["sampler", "group"])[["sampler", "group", "identity", "id_low", "id_high",
                                               "dynamics", "dyn_low", "dyn_high"]]
t.to_csv(ROOT / f"results/tables/tradeoff{TAG}.csv", index=False)
print(t.round(3).to_string(index=False))

groups = ["dies, condensed", "dies, AnnexinV only", "survives"]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
for a, grp in zip(ax, groups):
    s = t[t.group == grp]
    for k, (_, r) in enumerate(s.iterrows()):
        a.errorbar(r.identity, r.dynamics, xerr=[[r.identity - r.id_low], [r.id_high - r.identity]],
                   yerr=[[r.dynamics - r.dyn_low], [r.dyn_high - r.dynamics]], fmt="o", color=f"C{k}",
                   label=r.sampler, ms=5, capsize=2)
    a.axhline(0, color="grey", ls=":", lw=0.8)
    a.axvline(0.5, color="grey", ls=":", lw=0.8)
    a.set_title(grp, fontsize=9)
    a.set_xlabel("identity: rank of own future (0 = nearest)")
ax[0].set_ylabel("dynamics: energy score minus\n\"copy + typical change\" (readouts)")
ax[0].legend(fontsize=7)
plt.tight_layout()
plt.savefig(ROOT / f"results/figures/fig14_identity_vs_dynamics{TAG}.png", dpi=110)

# examples: real 0 h and 6 h, then one 6 h sample per sampler, for 3 cells of each group
import importlib  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
fs = importlib.import_module("20_feature_score")
train = importlib.import_module("09_train_flow")
DIRS = {k: Path(v) for k, v in ARGS.items()} or {
    "64 px, 8 h": ROOT / "results/kaggle/ctx64_long", "s0 0.3": ROOT / "results/keep_cell/ctx64_long_s03",
    "s0 0.5": ROOT / "results/keep_cell/ctx64_long_s05", "s0 0.7": ROOT / "results/keep_cell/ctx64_long_s07"}
x, fate, tracks = train.load("test", 64, [0, 12])
lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
cond = lab.loc[tracks, "condensed"].to_numpy()
pick = [i for sel in ((fate.numpy() == 1) & cond, (fate.numpy() == 1) & ~cond, fate.numpy() == 0)
        for i in np.flatnonzero(sel)[:3]]
samples = {name: fs.load_samples(d)[1][0] for name, d in DIRS.items()}
cols = ["real 0 h", "real 6 h"] + list(DIRS)
fig, ax = plt.subplots(len(pick), len(cols), figsize=(1.3 * len(cols), 1.3 * len(pick)))


def show(a, im):
    im = np.asarray(im, float)
    im = (im + 1) / 2 if im.max() <= 1 else im / 255
    a.imshow(np.dstack([im[2], im[1], im[0]]).clip(0, 1))
    a.set_xticks([])
    a.set_yticks([])


for r, i in enumerate(pick):
    show(ax[r, 0], x[i, 0].numpy())
    show(ax[r, 1], x[i, 1].numpy())
    for c, name in enumerate(DIRS):
        show(ax[r, c + 2], samples[name][i, 3, 0])
    ax[r, 0].set_ylabel(["condensed", "AnnexinV only", "survives"][r // 3], fontsize=6)
for c, name in enumerate(cols):
    ax[0, c].set_title(name, fontsize=7)
plt.tight_layout()
plt.savefig(ROOT / f"results/figures/fig14b_examples{TAG}.png", dpi=90)
