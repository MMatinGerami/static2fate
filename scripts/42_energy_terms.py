"""Split the energy score of 20_feature_score.py into its two terms, per model and space.

ES = mean distance of the samples to the real frame - 0.5 x mean distance between samples
(static2fate.evaluate.energy_terms). "Model minus random" in the feature-score tables is the
difference of two such scores, so a model can beat or lose to the random same-fate cells on
accuracy, on spread, or on both; this script says which, from the features 20 cached in
results/features/ (so 20 must have run on these models first). Same standardisation, same
random pool construction (seed 0, drawn per cell), held-out cells at 6 h.

Usage: STATIC2FATE_TAG=<tag> uv run scripts/42_energy_terms.py [name=dir ...]
Writes results/tables/energy_terms<tag>.csv: per model, space, resolution and group the
field-bootstrap estimate and CI of d_model, s_model, d_random, s_random, d_copy and the
differences d_model_minus_random and s_model_minus_random.
"""

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import energy_terms, field_bootstrap, group_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "results/features"
TAG = os.environ.get("STATIC2FATE_TAG", "")
FRAME = 12  # 6 h


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    fs_mod = importlib.import_module("20_feature_score")
    frames = [0] + fs_mod.FUTURE
    j = fs_mod.FUTURE.index(FRAME)
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    _, fate_t, tracks_t = train.load("train", 64, frames=[0])
    _, fate_s, tracks_s = train.load("test", 64, frames=[0])
    fate_t, fate_s = fate_t.numpy(), fate_s.numpy()
    nt, ns = len(tracks_t), len(tracks_s)
    pos = {t: i for i, t in enumerate(tracks_s)}
    rows = []
    for res in fs_mod.RES:
        ft = {k: v.reshape(nt, len(frames), -1) for k, v in np.load(FEAT / f"train_{res}.npz").items()}
        fs = {k: v.reshape(ns, len(frames), -1) for k, v in np.load(FEAT / f"test_{res}.npz").items()}
        for name, d in models.items():
            tracks_g = np.concatenate([np.load(p) for p in sorted(d.glob("samples*_tracks.npy"))])
            key = f"{d.name}_{name.replace(' ', '').replace(',', '_')}_{res}.npz"
            fg_all = np.load(FEAT / key)
            for sp in fg_all.files:
                m = fg_all[sp].shape[0] // (len(tracks_g) * len(fs_mod.FUTURE))
                fg = fg_all[sp].reshape(len(tracks_g), len(fs_mod.FUTURE), m, -1)
                real_future = fs[sp][:, 1:].reshape(-1, fs[sp].shape[-1])
                mu, sd = np.nanmean(real_future, 0), np.nanstd(real_future, 0) + 1e-8
                z = lambda a: (a - mu) / sd  # noqa: E731
                rng = np.random.default_rng(0)
                for gi, t in enumerate(tracks_g):
                    i = pos[t]
                    pool = rng.choice(np.flatnonzero(fate_t == fate_s[i]), m, replace=False)
                    y = z(fs[sp][i, j + 1])
                    dm, sm = energy_terms(z(fg[gi, j]), y)
                    dr, sr = energy_terms(z(ft[sp][pool, j + 1]), y)
                    rows.append({"model": name, "space": sp, "res": res, "track": t, "field": t[:10],
                                 "group": group_of(int(fate_s[i]), bool(lab.loc[t, "condensed"])),
                                 "d_model": dm, "s_model": sm, "d_random": dr, "s_random": sr,
                                 "d_copy": float(np.linalg.norm(z(fs[sp][i, 0]) - y))})
            print(f"{name} at {res} px", flush=True)
    d = pd.DataFrame(rows).dropna()
    d["d_model_minus_random"] = d.d_model - d.d_random
    d["s_model_minus_random"] = d.s_model - d.s_random
    cols = ["d_model", "s_model", "d_random", "s_random", "d_copy", "d_model_minus_random", "s_model_minus_random"]
    tab = pd.concat([field_bootstrap(d, c, ["model", "space", "res", "group"]).assign(measure=c) for c in cols])
    tab.to_csv(ROOT / f"results/tables/energy_terms{TAG}.csv", index=False)
    pd.set_option("display.width", 250)
    for sp, res in [("dinov2", 64), ("dinov2", 128), ("handcrafted", 128), ("readouts", 64)]:
        t = tab[(tab.space == sp) & (tab.res == res)]
        print(f"\n{sp} at {res} px, 6 h: distance term d and spread term s (ES = d - 0.5 s)")
        print(t.pivot_table(index=["model"], columns=["measure", "group"], values="estimate")
              .reindex(list(models))[["d_model", "s_model", "d_random", "s_random", "d_copy"]].round(1).to_string())


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()})
