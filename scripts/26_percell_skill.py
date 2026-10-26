"""Per-cell skill: does a sampler's prediction follow this cell's own future, beyond its group?

Within each fate group and hour, the Spearman correlation across cells between a readout of
the real future (nucleus area, nucleus Hoechst, cytoplasm green and red at 64 px) and the same
readout of the prediction: the mean over a sampler's samples, the cell's own 0 h image
("copy 0 h") or its real 3 h image ("real 3 h", from 4 h on). A sampler that only knows the
fate and the time predicts the group mean, so its correlation is near 0; the copy's
correlation says how much of the future is already in the first frame. Features come from
the cache of 20_feature_score.py (run it first with the same name=dir pairs and tag).
95% CIs resample the 12 held-out fields.

Usage: STATIC2FATE_TAG=<tag> uv run scripts/26_percell_skill.py [name=dir ...]
Writes results/tables/percell_skill<tag>.csv.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import group_of  # noqa: E402
from static2fate.readouts import FEATURES  # noqa: E402

fs = importlib.import_module("20_feature_score")
ROOT = fs.ROOT
RES = 64


def boot_r(a, b, fields, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    r = spearmanr(a, b).statistic
    uf = np.unique(fields)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(uf, len(uf))
        idx = np.concatenate([np.flatnonzero(fields == f) for f in pick])
        bs.append(spearmanr(a[idx], b[idx]).statistic if len(idx) > 3 else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r, lo, hi


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    _, tracks = train.split_index("test")
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    fate = lab.loc[tracks, "apoptotic"].to_numpy().astype(int)
    group = np.array([group_of(int(a), bool(c)) for a, c in zip(fate, lab.loc[tracks, "condensed"])])
    field = np.array([t[:10] for t in tracks])
    n = len(tracks)
    real = np.load(fs.FEAT / f"test_{RES}.npz")["readouts"].reshape(n, 1 + len(fs.FUTURE), -1)
    preds = {"copy 0 h": {f: real[:, 0] for f in fs.FUTURE},
             "real 3 h": {f: real[:, 1] for f in fs.FUTURE if f > 6}}
    for name, d in models.items():
        tg, arrays = fs.load_samples(d)
        assert (tg == tracks).all()
        m = arrays[0].shape[2]
        key = f"{d.name}_{name.replace(' ', '').replace(',', '_')}_{RES}.npz"
        g = np.load(fs.FEAT / key)["readouts"].reshape(n, len(fs.FUTURE), m, -1)
        preds[name] = {f: np.nanmean(g[:, j], 1) for j, f in enumerate(fs.FUTURE)}
    rows = []
    for grp in np.unique(group):
        for j, f in enumerate(fs.FUTURE):
            for k, feat in enumerate(FEATURES):
                y = real[:, j + 1, k]
                for name, p in preds.items():
                    if f not in p:
                        continue
                    x = p[f][:, k]
                    ok = (group == grp) & np.isfinite(x) & np.isfinite(y)
                    r, lo, hi = boot_r(x[ok], y[ok], field[ok])
                    rows.append({"predictor": name, "group": grp, "hours": f / 2, "readout": feat,
                                 "spearman": r, "ci_low": lo, "ci_high": hi, "n": int(ok.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"results/tables/percell_skill{fs.TAG}.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[out.hours == 6].pivot_table(index=["readout", "group"], columns="predictor", values="spearman")
          .round(2).to_string())


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()} if args else fs.MODELS)
