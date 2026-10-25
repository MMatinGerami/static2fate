"""Score the saved futures (19_generate_samples.py) in three feature spaces, at two resolutions.

Feature spaces: the four readouts of 17 (64 px only), the nine crop-level nucleus measurements
(static2fate.features.crop_features) and frozen DINOv2-base embeddings (768 d). Each space is
standardised per dimension on the real held-out frames 6 to 12. Scores per cell and frame:
energy score of the M samples (true fate, context 0 h), the "copy + typical change" baseline
(M draws, same construction as in 17), a random same-fate training cell at that frame (M
draws; unlike the drift baseline this stays on the manifold of real images, which matters in
DINOv2 space) and the distance of the 0 h copy. Resolutions: all
images at 64 px (128 px samples averaged down), and at 128 px (64 px samples bilinearly
upsampled, real crops native), so the 128 px model is also judged where its extra pixels can
count. Differences are paired by cell and frame; 95% CIs resample the 12 held-out fields.
Features are cached in results/features/.

Usage: uv run scripts/20_feature_score.py [name=dir ...]   (default: the three context models)
Writes results/tables/feature_score.csv, feature_score_vs_1h.csv,
results/figures/fig12_feature_score.png (model minus copy + typical change) and
fig12b_feature_score_vs_random_cell.png.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import energy_score, field_bootstrap, group_of  # noqa: E402
from static2fate.features import crop_features, dinov2_features  # noqa: E402
from static2fate.readouts import cell_readouts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "results/features"
MODELS = {
    "64 px, 1 h": ROOT / "results/flow_ctx",
    "64 px, 8 h": ROOT / "results/kaggle/ctx64_long",
    "128 px, 8 h": ROOT / "results/kaggle/ctx128",
}
FUTURE = [6, 8, 10, 12]
SPACES = ["readouts", "handcrafted", "dinov2"]
RES = [64, 128]
CHUNK = 256


def resize(x: torch.Tensor, res: int) -> torch.Tensor:
    s = x.shape[-1]
    if s == res:
        return x
    return F.avg_pool2d(x, s // res) if s > res else F.interpolate(x, size=res, mode="bilinear", align_corners=False)


def extract(key: str, images, res: int) -> dict[str, np.ndarray]:
    """images: uint8 (N, 3, s, s) array or memmap -> per-space (N, d) features, cached."""
    path = FEAT / f"{key}_{res}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = {sp: [] for sp in SPACES if sp != "readouts" or res == 64}
    for i in range(0, len(images), CHUNK):
        x = resize(torch.from_numpy(np.asarray(images[i:i + CHUNK])).float().div_(127.5).sub_(1), res)
        if "readouts" in out:
            out["readouts"].append(cell_readouts(x))
        out["handcrafted"].append(crop_features(x))
        out["dinov2"].append(dinov2_features(x))
        print(f"  {key} {res} px: {min(i + CHUNK, len(images))}/{len(images)}", flush=True)
    out = {k: np.concatenate(v) for k, v in out.items()}
    FEAT.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


def load_samples(d: Path):
    """Join samples*.npy of one model, ordered by their track files."""
    parts = sorted(d.glob("samples*_tracks.npy"))
    tracks = np.concatenate([np.load(p) for p in parts])
    arrays = [np.load(Path(str(p).replace("_tracks", "")), mmap_mode="r") for p in parts]
    return tracks, arrays


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    frames = [0] + FUTURE
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    x_t, fate_t, tracks_t = train.load("train", 128, frames)
    x_s, fate_s, tracks_s = train.load("test", 128, frames)
    to_u8 = lambda x: ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()  # noqa: E731
    u_t, u_s = to_u8(x_t.flatten(0, 1)), to_u8(x_s.flatten(0, 1))
    del x_t, x_s
    fate_t, fate_s = fate_t.numpy(), fate_s.numpy()
    nt, ns = len(tracks_t), len(tracks_s)
    pos = {t: i for i, t in enumerate(tracks_s)}
    rng = np.random.default_rng(0)
    rows = []
    for res in RES:
        print(f"real cells at {res} px", flush=True)
        ft = {k: v.reshape(nt, len(frames), -1) for k, v in extract("train", u_t, res).items()}
        fs = {k: v.reshape(ns, len(frames), -1) for k, v in extract("test", u_s, res).items()}
        for name, d in models.items():
            tracks_g, arrays = load_samples(d)
            m = arrays[0].shape[2]
            flat = np.concatenate([a.reshape(-1, *a.shape[-3:]) for a in arrays])  # (cells*4*M, 3, s, s)
            fg = {k: v.reshape(len(tracks_g), len(FUTURE), m, -1)
                  for k, v in extract(f"{d.name}_{name.replace(' ', '').replace(',', '_')}", flat, res).items()}
            for sp in fg:
                real_future = fs[sp][:, 1:].reshape(-1, fs[sp].shape[-1])
                mu, sd = np.nanmean(real_future, 0), np.nanstd(real_future, 0) + 1e-8
                z = lambda a: (a - mu) / sd  # noqa: E731
                for gi, t in enumerate(tracks_g):
                    i = pos[t]
                    pool = rng.choice(np.flatnonzero(fate_t == fate_s[i]), m, replace=False)
                    for j, f in enumerate(FUTURE):
                        y = z(fs[sp][i, j + 1])
                        drift = z(fs[sp][i, 0]) + z(ft[sp][pool, j + 1]) - z(ft[sp][pool, 0])
                        rows.append({"model": name, "space": sp, "res": res, "track": t, "field": t[:10],
                                     "group": group_of(int(fate_s[i]), bool(lab.loc[t, "condensed"])),
                                     "hours": f / 2, "es_model": energy_score(z(fg[sp][gi, j]), y),
                                     "es_drift": energy_score(drift, y),
                                     "es_random": energy_score(z(ft[sp][pool, j + 1]), y),
                                     "d_copy": float(np.linalg.norm(z(fs[sp][i, 0]) - y))})
            print(f"scored {name} at {res} px", flush=True)
    d = pd.DataFrame(rows).dropna()
    d["model_minus_copy"] = d.es_model - d.d_copy
    d["model_minus_drift"] = d.es_model - d.es_drift
    d["model_minus_random"] = d.es_model - d.es_random
    d["drift_minus_copy"] = d.es_drift - d.d_copy
    d["random_minus_copy"] = d.es_random - d.d_copy
    d.to_csv(FEAT / "feature_score_cells.csv", index=False)

    by = ["model", "space", "res", "group", "hours"]
    measures = ["model_minus_copy", "model_minus_drift", "model_minus_random", "drift_minus_copy", "random_minus_copy"]
    tab = pd.concat([field_bootstrap(d, c, by).assign(measure=c) for c in measures])
    tab.to_csv(ROOT / "results/tables/feature_score.csv", index=False)
    base = "64 px, 1 h"
    key = ["space", "res", "track", "field", "group", "hours"]
    w = d[d.model == base][key + ["es_model"]].rename(columns={"es_model": "es_base"})
    diffs = []
    for name in models:
        if name == base:
            continue
        j = d[d.model == name].merge(w, on=key)
        j["minus_1h"] = j.es_model - j.es_base
        diffs.append(field_bootstrap(j, "minus_1h", ["space", "res", "group", "hours"]).assign(model=name))
    vs = pd.concat(diffs) if diffs else pd.DataFrame()
    vs.to_csv(ROOT / "results/tables/feature_score_vs_1h.csv", index=False)
    pd.set_option("display.width", 200)
    for meas in ("model_minus_drift", "model_minus_random"):
        print(meas)
        print(tab[tab.measure == meas].pivot_table(
            index=["space", "res", "group"], columns=["model", "hours"], values="estimate").round(2).to_string())
    if len(vs):
        print(vs.pivot_table(index=["space", "res", "group"], columns=["model", "hours"], values="estimate")
              .round(2).to_string())

    plot(tab, list(models), "model_minus_drift", "copy + typical change", ROOT / "results/figures/fig12_feature_score.png")
    plot(tab, list(models), "model_minus_random", "random same-fate cell",
         ROOT / "results/figures/fig12b_feature_score_vs_random_cell.png")


def plot(tab: pd.DataFrame, models: list[str], measure: str, baseline: str, out: Path) -> None:
    groups = ["dies, condensed", "dies, AnnexinV only", "survives"]
    spaces = [("readouts", 64), ("handcrafted", 64), ("dinov2", 64), ("handcrafted", 128), ("dinov2", 128)]
    fig, ax = plt.subplots(len(spaces), 3, figsize=(10, 2.6 * len(spaces)), sharex=True)
    for r, (sp, res) in enumerate(spaces):
        for c, grp in enumerate(groups):
            a = ax[r, c]
            for k, name in enumerate(models):
                s = tab[(tab.measure == measure) & (tab.space == sp) & (tab.res == res)
                        & (tab.group == grp) & (tab.model == name)].sort_values("hours")
                a.plot(s.hours, s.estimate, "-o", color=f"C{k}", label=name, ms=3)
                a.fill_between(s.hours, s.ci_low, s.ci_high, color=f"C{k}", alpha=0.12)
            a.axhline(0, color="grey", ls=":")
            if r == 0:
                a.set_title(grp, fontsize=9)
            if c == 0:
                a.set_ylabel(f"{sp}, {res} px\nmodel minus {baseline}", fontsize=8)
            if r == len(spaces) - 1:
                a.set_xlabel("hours after staurosporine")
                a.set_xticks([3, 4, 5, 6])
    ax[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)

if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()} if args else MODELS)
