"""Can a linear probe tell generated futures from real ones in each feature space?

Real: the held-out cells at frames 6 to 12. Generated: the first of the M saved samples of the
same cell at the same frame (so the classes are balanced and cover the same cells and times).
Leave-one-field-out logistic regression on standardised features (cached by
20_feature_score.py); pooled out-of-fold AUROC. 0.5 means the space cannot see the difference.

Usage: uv run scripts/21_real_vs_generated.py [name=dir ...]   (default: the three models)
Writes results/tables/real_vs_generated_auroc.csv.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
fs = importlib.import_module("20_feature_score")
ROOT = fs.ROOT


def lofo_auroc(X, y, fields):
    scores = np.zeros(len(y))
    for f in np.unique(fields):
        te = fields == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        clf.fit(X[~te], y[~te])
        scores[te] = clf.decision_function(X[te])
    return roc_auc_score(y, scores)


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    _, tracks_s = train.split_index("test")
    pos = {t: i for i, t in enumerate(tracks_s)}
    rows = []
    for res in fs.RES:
        real = {k: v.reshape(len(tracks_s), 1 + len(fs.FUTURE), -1)[:, 1:] for k, v in
                np.load(fs.FEAT / f"test_{res}.npz").items()}  # (cells, 4 frames, d)
        for name, d in models.items():
            tracks_g, arrays = fs.load_samples(d)
            m = arrays[0].shape[2]
            key = f"{d.name}_{name.replace(' ', '').replace(',', '_')}_{res}.npz"
            gen = {k: v.reshape(len(tracks_g), len(fs.FUTURE), m, -1)[:, :, 0] for k, v in
                   np.load(fs.FEAT / key).items()}
            idx = np.array([pos[t] for t in tracks_g])
            fields = np.repeat([t[:10] for t in tracks_g], len(fs.FUTURE))
            for sp in gen:
                X = np.concatenate([real[sp][idx].reshape(-1, real[sp].shape[-1]),
                                    gen[sp].reshape(-1, gen[sp].shape[-1])])
                y = np.r_[np.zeros(len(idx) * len(fs.FUTURE)), np.ones(len(idx) * len(fs.FUTURE))]
                ok = np.isfinite(X).all(1)
                auc = lofo_auroc(X[ok], y[ok], np.r_[fields, fields][ok])
                rows.append({"model": name, "space": sp, "res": res, "auroc": auc, "n": int(ok.sum())})
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/tables/real_vs_generated_auroc.csv", index=False)
    print(out.pivot_table(index=["space", "res"], columns="model", values="auroc").round(3).to_string())


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()} if args else fs.MODELS)
