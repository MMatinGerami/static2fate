"""Temporal consistency: do consecutive generated frames belong to one trajectory?

The frame model draws 3, 4, 5 and 6 h from separate noise, so between two of its frames the
cell can jump; the video model draws them together; real cells change smoothly. For every
held-out cell and sample, the change from one frame to the next is measured as the mean
absolute pixel difference and as the absolute change of the four readouts (nucleus area and
Hoechst, cytoplasm green and red), at 60 min steps (frames 6, 8, 10, 12, the only ones the
frame model has) and, for real and video, at 30 min steps. Also counted: how often the nucleus
area reverses direction over the three 60 min steps (a condensing nucleus should keep
shrinking). Means per fate group with 95% CIs from resampling the 12 held-out fields.

Usage: STATIC2FATE_TAG=<tag> uv run scripts/30_temporal.py [name=dir ...]
A dir with clips.npy (29_generate_clips.py) is a video model; one with samples.npy only
(19_generate_samples.py) is a frame model. Default: video=results/kaggle/video64 and
"frame model"=results/kaggle/ctx64_long.
Writes results/tables/temporal<tag>.csv and results/figures/fig15_temporal<tag>.png.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import field_bootstrap, group_of  # noqa: E402
from static2fate.readouts import FEATURES, cell_readouts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = os.environ.get("STATIC2FATE_TAG", "")
MODELS = {"video": ROOT / "results/kaggle/video64", "frame model": ROOT / "results/kaggle/ctx64_long"}
CLIP = list(range(6, 13))
HOUR = [6, 8, 10, 12]
MEASURES = ["pixel", *FEATURES]


def to64(arr) -> np.ndarray:
    """Average 128 px generated images down to the 64 px the readouts are defined at, in chunks."""
    if arr.shape[-1] == 64:
        return np.asarray(arr)
    f = arr.shape[-1] // 64
    out = np.empty((*arr.shape[:-2], 64, 64), np.uint8)
    for i in range(0, len(arr), 32):
        a = np.asarray(arr[i:i + 32]).astype(np.float32)
        out[i:i + 32] = a.reshape(*a.shape[:-2], 64, f, 64, f).mean((-1, -3)).round().clip(0, 255)
    return out


def load_generated(d: Path):
    """(tracks, frames, array (cells, M, len(frames), 3, s, s)) for a video or a frame model.

    Parts written for cell ranges (clips_0_400.npy and so on) are joined in track-file order.
    """
    kind = "clips" if list(d.glob("clips*_tracks.npy")) else "samples"
    parts = sorted(d.glob(f"{kind}*_tracks.npy"))
    tracks = np.concatenate([np.load(p) for p in parts])
    arr = np.concatenate([to64(np.load(Path(str(p).replace("_tracks", "")), mmap_mode="r")) for p in parts])
    if kind == "clips":
        return tracks, CLIP, arr
    return tracks, HOUR, arr.transpose(0, 2, 1, 3, 4, 5)  # (cells, 4, M, ...) -> (cells, M, 4, ...)


def changes(x: np.ndarray, frames: list[int], step: int) -> dict[str, np.ndarray]:
    """x (n, M, T, 3, s, s) uint8 -> per-cell mean over samples and steps of |difference|.

    step is in frames (1 = 30 min, 2 = 60 min); only frames `step` apart that are both present
    are used. Also "area_reversals": mean number of sign changes of the nucleus area increment.
    """
    n, m, t = x.shape[:3]
    pairs = [(i, j) for i, fi in enumerate(frames) for j, fj in enumerate(frames) if fj - fi == step]
    xf = x.astype(np.float32) / 127.5 - 1
    pix = np.mean([np.abs(xf[:, :, j] - xf[:, :, i]).mean((2, 3, 4)) for i, j in pairs], 0)  # (n, M)
    ro = cell_readouts(xf.reshape(-1, *xf.shape[-3:])).reshape(n, m, t, len(FEATURES))
    out = {"pixel": pix.mean(1)}
    for k, f in enumerate(FEATURES):
        d = np.stack([np.abs(ro[:, :, j, k] - ro[:, :, i, k]) for i, j in pairs], -1)  # (n, M, steps)
        out[f] = np.nanmean(d, (1, 2))
    if step == 2 and all(h in frames for h in HOUR):
        idx = [frames.index(h) for h in HOUR]
        inc = np.diff(ro[:, :, idx, 0], axis=2)  # (n, M, 3) nucleus area increments
        rev = (np.sign(inc[..., 1:]) * np.sign(inc[..., :-1]) < 0).sum(-1).astype(float)
        rev[np.isnan(inc).any(-1)] = np.nan
        out["area_reversals"] = np.nanmean(rev, 1)
    return out


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    x, fate, tracks = train.load("test", 64)
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    group = np.array([group_of(int(a), bool(lab.loc[t, "condensed"])) for a, t in zip(fate, tracks)])
    field = np.array([t[:10] for t in tracks])
    pos = {t: i for i, t in enumerate(tracks)}
    real = ((x[:, CLIP].numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)[:, None]  # (n, 1, 7, 3, s, s)
    sources = {"real": (tracks, CLIP, real)}
    for name, d in models.items():
        sources[name] = load_generated(d)
    rows = []
    for name, (tg, frames, arr) in sources.items():
        for step in (1, 2):
            if step == 1 and len(frames) < len(CLIP):
                continue
            ch = changes(arr, frames, step)
            idx = np.array([pos[t] for t in tg])
            for measure, v in ch.items():
                rows.extend({"model": name, "step_min": 30 * step, "measure": measure, "track": t,
                             "field": field[i], "group": group[i], "value": val}
                            for t, i, val in zip(tg, idx, v) if np.isfinite(val))
            print(f"{name}: {len(frames)} frames, step {30 * step} min", flush=True)
    d = pd.DataFrame(rows)
    out = field_bootstrap(d, "value", ["model", "step_min", "measure", "group"])
    out.to_csv(ROOT / f"results/tables/temporal{TAG}.csv", index=False)
    pd.set_option("display.width", 200)
    show = out[(out.step_min == 60) & out.measure.isin(["pixel", "nuc_area", "area_reversals"])]
    print(show.pivot_table(index=["measure", "group"], columns="model", values="estimate").round(3).to_string())
    figure(out, sources, pos, group, fate.numpy(), lab, tracks)


def figure(out, sources, pos, group, fate, lab, tracks):
    names = list(sources)
    groups = ["survives", "dies, AnnexinV only", "dies, condensed"]
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4])
    for col, (measure, label) in enumerate([("pixel", "mean |pixel change| per 60 min"),
                                            ("nuc_area", "|nucleus area change| (px) per 60 min"),
                                            ("area_reversals", "area direction reversals (0 to 2)")]):
        ax = fig.add_subplot(gs[0, col])
        sub = out[(out.step_min == 60) & (out.measure == measure)]
        for k, name in enumerate(names):
            s = sub[sub.model == name].set_index("group").reindex(groups)
            xs = np.arange(len(groups)) + (k - (len(names) - 1) / 2) * 0.22
            ax.errorbar(xs, s.estimate, yerr=[s.estimate - s.ci_low, s.ci_high - s.estimate], fmt="o", capsize=3, label=name)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace(", ", ",\n") for g in groups])
        ax.set_title(label, fontsize=10)
        if col == 0:
            ax.legend(fontsize=8)
    # example clips at 3, 4, 5, 6 h: a surviving and a condensed dying held-out cell
    _, _, real = sources["real"]
    area = cell_readouts(real[:, 0, [0, -1]].reshape(-1, 3, 64, 64).astype(np.float32) / 127.5 - 1)[:, 0].reshape(-1, 2)
    drop = area[:, 0] - area[:, 1]
    ok = np.isfinite(drop)
    common = np.isin(tracks, list(set.intersection(*(set(tg) for tg, _, _ in sources.values()))))
    ok &= common
    ex = []
    if (ok & (group == "survives")).any():
        ex.append(int(np.nanargmin(np.where(ok & (group == "survives"), np.abs(drop), np.nan))))
    if (ok & (group == "dies, condensed")).any():
        ex.append(int(np.nanargmax(np.where(ok & (group == "dies, condensed"), drop, np.nan))))
    hour_idx = {name: [frames.index(h) for h in HOUR] for name, (_, frames, _) in sources.items()}
    # one block of rows (sources) by columns (hours) per example cell, blocks side by side
    widths = [1] * len(HOUR) * len(ex)
    for b in range(1, len(ex)):
        widths.insert(b * len(HOUR) + b - 1, 0.3)  # gap between blocks
    inner = gs[1, :].subgridspec(len(names), len(widths), width_ratios=widths, wspace=0.05, hspace=0.05)
    for e, i in enumerate(ex):
        for k, name in enumerate(names):
            tg, frames, arr = sources[name]
            gi = int(np.flatnonzero(tg == tracks[i])[0])
            clip = np.asarray(arr[gi, 0])[hour_idx[name]]
            for j, h in enumerate(HOUR):
                ax = fig.add_subplot(inner[k, e * (len(HOUR) + 1) + j])
                im = clip[j].astype(np.float32) / 255
                ax.imshow(np.dstack([im[2], im[1], im[0]]).clip(0, 1))
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0 and e == 0:
                    ax.set_ylabel(name.replace(", ", ",\n"), fontsize=8)
                if k == 0:
                    ax.set_title(f"{group[i]}, {h / 2:.0f} h" if j == 0 else f"{h / 2:.0f} h", fontsize=8)
    fig.suptitle("Temporal consistency of generated futures (held-out cells, context 0 h, true fate)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / f"results/figures/fig15_temporal{TAG}.png", dpi=130)


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()} if args else MODELS)
