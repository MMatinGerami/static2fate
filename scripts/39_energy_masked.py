"""Does the DINOv2 energy score count the cell or its neighbours? Score masked crops.

The latent models' futures are further from the cell's own future in DINOv2 space than the
pixel models' (docs/results.md, "Generating in the latent"), and in the examples they add or
move neighbouring cells more often. To separate the two, the 6 h frame of every held-out
cell, of its generated futures and of the random same-fate training cells the score is
compared with, is masked to a disc around the crop centre before the features are taken:
"cell" (radius 40 px at 128 px, the central cell and little else) and "nucleus" (radius
24 px), next to the unmasked crop. Outside the disc the image is black; the edge is softened
over 3 px. Same energy score, same baselines and field bootstrap as 20_feature_score.py,
DINOv2 at 64 px only; the table also carries the score's two terms (mean distance to the real
frame, mean distance between samples) for the model and the random cells.

Usage: [STATIC2FATE_TAG=<tag>] uv run scripts/39_energy_masked.py [name=dir ...]   (default:
pixel frame, latent frame, latent video, ae128 reconstruction)
Writes results/tables/energy_masked<tag>.csv and results/figures/fig18_energy_masked<tag>.png.
"""

import importlib
import os
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
from static2fate.evaluate import energy_score, energy_terms, field_bootstrap, group_of  # noqa: E402
from static2fate.features import dinov2_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = os.environ.get("STATIC2FATE_TAG", "")  # suffix for the output table and figure
FRAME = 12  # 6 h
MASKS = {"none": None, "cell": 40, "nucleus": 24}
MODELS = {
    "frame model, 64 px, 8 h": ROOT / "results/kaggle/ctx64_long",
    "latent frame": ROOT / "results/kaggle/lat_frame",
    "latent video": ROOT / "results/kaggle/lat_video",
    "ae128 reconstruction": ROOT / "results/kaggle/ae128/recon",
}
GROUPS = ["survives", "dies, AnnexinV only", "dies, condensed"]


def disc_mask(size: int, radius: float, edge: float = 3.0) -> torch.Tensor:
    """(size, size) weights: 1 inside the disc, 0 outside, a linear ramp of `edge` px at the rim."""
    c = (size - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    r = torch.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    return ((radius - r) / edge + 0.5).clamp(0, 1)


def masked_features(images: np.ndarray, radius: float | None, chunk: int = 256) -> np.ndarray:
    """uint8 (n, 3, s, s) -> DINOv2 features at 64 px, the crop masked to the disc first.

    The radius is given at 128 px and scaled to the crop (the pixel models' samples are 64 px).
    """
    out = []
    size = images.shape[-1]
    w = disc_mask(size, radius * size / 128) if radius else None
    for i in range(0, len(images), chunk):
        x = torch.from_numpy(np.asarray(images[i:i + chunk])).float().div_(127.5).sub_(1)
        if w is not None:
            x = (x + 1) * w - 1  # black outside the disc
        out.append(dinov2_features(F.avg_pool2d(x, size // 64) if size > 64 else x))
    return np.concatenate(out)


def to_u8(x: torch.Tensor) -> np.ndarray:
    return ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    fs = importlib.import_module("20_feature_score")
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    x_t, fate_t, tracks_t = train.load("train", 128, frames=[FRAME])
    x_s, fate_s, tracks_s = train.load("test", 128, frames=[FRAME])
    u_t, u_s = to_u8(x_t[:, 0]), to_u8(x_s[:, 0])
    del x_t, x_s
    fate_t, fate_s = fate_t.numpy(), fate_s.numpy()
    pos = {t: i for i, t in enumerate(tracks_s)}
    j = fs.FUTURE.index(FRAME)
    gen = {}
    for name, d in models.items():
        tracks_g, arrays = fs.load_samples(d)
        gen[name] = (tracks_g, np.concatenate([np.asarray(a[:, j]) for a in arrays]))  # (cells, M, 3, s, s)
    rows = []
    for mask, radius in MASKS.items():
        print(f"mask {mask}", flush=True)
        ft, fts = masked_features(u_t, radius), masked_features(u_s, radius)
        mu, sd = fts.mean(0), fts.std(0) + 1e-8
        z = lambda a: (a - mu) / sd  # noqa: E731
        for name, (tracks_g, arr) in gen.items():
            m = arr.shape[1]
            fg = masked_features(arr.reshape(-1, *arr.shape[-3:]), radius).reshape(len(tracks_g), m, -1)
            rng = np.random.default_rng(0)  # the same random cells for every mask and model
            for gi, t in enumerate(tracks_g):
                i = pos[t]
                pool = rng.choice(np.flatnonzero(fate_t == fate_s[i]), m, replace=False)
                y = z(fts[i])
                d_model, s_model = energy_terms(z(fg[gi]), y)
                d_random, s_random = energy_terms(z(ft[pool]), y)
                rows.append({"mask": mask, "model": name, "track": t, "field": t[:10],
                             "group": group_of(int(fate_s[i]), bool(lab.loc[t, "condensed"])),
                             "es_model": energy_score(z(fg[gi]), y), "es_random": energy_score(z(ft[pool]), y),
                             "d_model": d_model, "s_model": s_model, "d_random": d_random, "s_random": s_random})
            print(f"  scored {name}", flush=True)
    d = pd.DataFrame(rows)
    d["model_minus_random"] = d.es_model - d.es_random
    d["d_model_minus_random"] = d.d_model - d.d_random  # the distance term of that difference
    d["s_model_minus_random"] = d.s_model - d.s_random  # and the spread term (ES = d - 0.5 s)
    tab = pd.concat([field_bootstrap(d, c, ["mask", "model", "group"]).assign(measure=c)
                     for c in ["model_minus_random", "d_model_minus_random", "s_model_minus_random",
                               "es_model", "es_random", "d_model", "s_model", "d_random", "s_random"]])
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(ROOT / f"results/tables/energy_masked{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab[tab.measure == "model_minus_random"].pivot_table(
        index=["model", "group"], columns="mask", values="estimate").round(2).to_string())
    figure(tab, list(models))


def figure(tab: pd.DataFrame, models: list[str]) -> None:
    t = tab[tab.measure == "model_minus_random"]
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(4.2 * len(GROUPS), 3.8), sharey=True)
    for ax, g in zip(axes, GROUPS):
        for k, name in enumerate(models):
            s = t[(t.model == name) & (t.group == g)].set_index("mask").loc[list(MASKS)]
            x = np.arange(len(MASKS)) + (k - (len(models) - 1) / 2) * min(0.15, 0.8 / len(models))
            ax.errorbar(x, s.estimate, yerr=[s.estimate - s.ci_low, s.ci_high - s.estimate], fmt="o",
                        ms=4, capsize=2, label=name)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(MASKS)))
        ax.set_xticklabels([f"{k}" + (f"\n(r = {r} px)" if r else "") for k, r in MASKS.items()])
        ax.set_title(g, fontsize=10)
    axes[0].set_ylabel("DINOv2 energy score minus random same-fate cell")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Held-out cells at 6 h, crops masked to a disc around the cell before DINOv2 (64 px)", fontsize=10)
    fig.tight_layout()
    (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f"results/figures/fig18_energy_masked{TAG}.png", dpi=130)


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()} if args else MODELS)
