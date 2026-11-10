"""Distance of each model's samples and of the 0 h copy to the real 6 h frame, per cell, with
paired differences.

39_energy_masked.py reports the distance term of the energy score per model but not the copy's,
and its CIs are per model, so a one unit difference between two models cannot be read off it.
This script recomputes the masked DINOv2 distances (same masks, same features, same
normalisation as 39) for the 0 h copy and for the latent video models, pairs them per cell,
and bootstraps the differences over fields.

Usage: uv run scripts/43_masked_vs_copy.py [name=dir ...]
Writes results/tables/energy_masked_vs_copy.csv: per mask and group the field-bootstrap
estimate and CI of each distance and of the paired differences.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import field_bootstrap, group_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "latent video": ROOT / "results/kaggle/lat_video",
    "latent video, 3 fates": ROOT / "results/kaggle/lat_video_fate3",
    "latent video, seed 4": ROOT / "results/kaggle/lat_video_s4",
}
REFERENCE = "latent video, 3 fates"  # the model the differences are taken from


def main(models: dict[str, Path]) -> None:
    em = importlib.import_module("39_energy_masked")
    train = importlib.import_module("09_train_flow")
    fs = importlib.import_module("20_feature_score")
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    x_s, fate_s, tracks_s = train.load("test", 128, frames=[0, em.FRAME])
    u0, u6 = em.to_u8(x_s[:, 0]), em.to_u8(x_s[:, 1])
    del x_s
    fate_s = fate_s.numpy()
    pos = {t: i for i, t in enumerate(tracks_s)}
    j = fs.FUTURE.index(em.FRAME)
    gen = {}
    for name, d in models.items():
        tracks_g, arrays = fs.load_samples(d)
        gen[name] = (tracks_g, np.concatenate([np.asarray(a[:, j]) for a in arrays]))
    rows = []
    for mask, radius in em.MASKS.items():
        print(f"mask {mask}", flush=True)
        f6, f0 = em.masked_features(u6, radius), em.masked_features(u0, radius)
        mu, sd = f6.mean(0), f6.std(0) + 1e-8
        z = lambda a: (a - mu) / sd  # noqa: E731
        per = {}
        for name, (tracks_g, arr) in gen.items():
            m = arr.shape[1]
            fg = em.masked_features(arr.reshape(-1, *arr.shape[-3:]), radius).reshape(len(tracks_g), m, -1)
            per[name] = {t: float(np.linalg.norm(z(fg[gi]) - z(f6[pos[t]]), axis=1).mean())
                         for gi, t in enumerate(tracks_g)}
            print(f"  scored {name}", flush=True)
        for t in gen[REFERENCE][0]:
            i = pos[t]
            row = {"mask": mask, "track": t, "field": t[:10],
                   "group": group_of(int(fate_s[i]), bool(lab.loc[t, "condensed"])),
                   "0 h copy": float(np.linalg.norm(z(f0[i]) - z(f6[i])))}
            row.update({name: per[name][t] for name in models})
            rows.append(row)
    d = pd.DataFrame(rows)
    cols = ["0 h copy", *models]
    for name in models:
        d[f"{name} minus copy"] = d[name] - d["0 h copy"]
        cols.append(f"{name} minus copy")
    for name in models:
        if name != REFERENCE:
            d[f"{REFERENCE} minus {name}"] = d[REFERENCE] - d[name]
            cols.append(f"{REFERENCE} minus {name}")
    tab = pd.concat([field_bootstrap(d, c, ["mask", "group"]).assign(measure=c) for c in cols])
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(ROOT / "results/tables/energy_masked_vs_copy.csv", index=False)
    pd.set_option("display.width", 250)
    tab["cell"] = tab.apply(lambda r: f"{r.estimate:.1f} ({r.ci_low:.1f} to {r.ci_high:.1f})", axis=1)
    print(tab.pivot_table(index=["mask", "measure"], columns="group", values="cell", aggfunc="first")
          .reindex(cols, level=1).to_string())


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()} if args else MODELS)
