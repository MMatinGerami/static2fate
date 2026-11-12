"""Image galleries: many real cells, then real against generated, all with one display convention.

Real cells are cut from the raw TIFFs and drawn with static2fate.display: Hoechst blue,
ChromaLIVE 488 green, ChromaLIVE 561 red, linear, one intensity range per channel for the whole
plate (pooled from the first field of every well of row D at 0 h and 6 h, written to
results/tables/display_ranges.json). Cells and fields are picked with a fixed seed, not by eye.

  fig01_fields_by_dose.png   one field per dose at 0 h and 6 h, and its terminal AnnexinV image
  fig01b_field_over_time.png an untreated field and a 39.06 nM field, every hour
  fig02b_masks_registration.png  nucleus and cell masks and the terminal registration, three fields
  fig03_cells_<group>.png    24 cells per fate group at 19.53 and 39.06 nM, every hour
  fig04_colour_shift.png     12 cells with their 488 / 561 ratio and nucleus area over time
  fig07_real_vs_generated.png        held-out cells: real 0 h and 6 h, four generated 6 h frames
  fig08_generated_course.png         real 3 to 6 h course next to two generated clips
  fig09_counterfactual.png           the same 0 h cell drawn under each of the three fates

Figures 7 to 9 show what the model sees and draws, so they use the per-field scaling of
05_crops.py (said in their titles). They need results/kaggle/lat_video_fate3 (35_generate_latent.py)
and, for figure 9, results/guidance/counterfactual/fate{0,1,2} (same script with a fate class).

Usage: uv run scripts/47_gallery.py [real] [generated]   (both by default)
"""

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.display import (  # noqa: E402
    ANNEXIN,
    CHANNELS,
    GROUP_COLOURS,
    GROUPS,
    PIXEL_UM,
    composite,
    model_rgb,
    mosaic,
    plate_ranges,
    read_frame,
    scale_bar,
    unit,
    window,
)
from static2fate.evaluate import group_of  # noqa: E402
from static2fate.fields import Field  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/raw/images"
P = ROOT / "data/processed"
FIG = ROOT / "results/figures"
MODEL = ROOT / "results/kaggle/lat_video_fate3"
COUNTERFACTUAL = ROOT / "results/guidance/counterfactual"
DOSES = [0.0, 4.88, 9.77, 19.53, 39.06, 78.13, 156.25]
SEED = 0
BG = "#111111"


def labels() -> pd.DataFrame:
    lab = pd.read_parquet(P / "fate_labels.parquet")
    lab["group"] = [group_of(int(a), bool(c)) for a, c in zip(lab["apoptotic"], lab["condensed"], strict=True)]
    return lab.set_index("track")


def tracks_of(fields: list[str]) -> pd.DataFrame:
    tr = pd.concat([pd.read_parquet(P / f"tracks/{f}.parquet") for f in fields], ignore_index=True)
    tr["track"] = tr["well_fov"] + "_" + tr["particle"].astype(str)
    return tr[tr["frame"] <= 13]


def ranges() -> dict[str, tuple[float, float]]:
    path = ROOT / "results/tables/display_ranges.json"
    if path.exists():
        return {k: tuple(v) for k, v in json.loads(path.read_text()).items()}
    fields = [f"D-{c:02d}_F0001" for c in range(2, 12)]
    r = plate_ranges(IMAGES, fields)
    term = np.concatenate([read_frame(IMAGES, f, 14, ANNEXIN)[::8, ::8].ravel() for f in fields
                           if (IMAGES / f"{f}_T0014_Z0001_C{ANNEXIN:02d}.tif").exists()])
    r["annexin"] = tuple(float(v) for v in np.percentile(term, [0.5, 99.8]))
    path.write_text(json.dumps(r, indent=1))
    return r


def rgb_frame(well_fov: str, t: int, rng: dict) -> np.ndarray:
    return composite(*(read_frame(IMAGES, well_fov, t, c) for c in CHANNELS.values()), rng)


def first_labelled_field(lab: pd.DataFrame, dose: float, row: str = "D") -> str:
    s = lab[np.isclose(lab["dose_nM"], dose)]
    fields = sorted(s["well_fov"].unique(), key=lambda f: (not f.startswith(row), f))
    return fields[0]


def draw_fates(ax, tr: pd.DataFrame, lab: pd.DataFrame, frame: int, y0: int, x0: int, side: int, dy: float = 0, dx: float = 0):
    """Circle at every tracked, labelled nucleus of a frame, in the colour of its fate group."""
    for _, row in tr[tr["frame"] == frame].iterrows():
        if row["track"] not in lab.index:
            continue
        y, x = row["y"] - y0, row["x"] - x0
        if 0 <= y < side and 0 <= x < side:
            r = np.sqrt(row["area"] / np.pi) + 4
            ax.add_patch(plt.Circle((x + dx, y + dy), r, fill=False, lw=0.9, color=GROUP_COLOURS[lab.loc[row["track"], "group"]]))


def legend_fates(fig, y: float = 0.012) -> None:
    for i, g in enumerate(GROUPS):
        fig.text(0.02 + i * 0.2, y, f"○ {g}", color=GROUP_COLOURS[g], fontsize=9, ha="left")


def fields_by_dose(lab: pd.DataFrame, rng: dict) -> None:
    side, y0, x0 = 640, 630, 630
    fig, ax = plt.subplots(3, len(DOSES), figsize=(2.75 * len(DOSES), 8.9), facecolor=BG)
    for c, dose in enumerate(DOSES):
        f = first_labelled_field(lab, dose)
        tr = tracks_of([f])
        field = Field.load(IMAGES, f)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        for r, t in enumerate([1, 13]):
            ax[r, c].imshow(composite(field.live["hoechst"][t - 1][sl], field.live["cl488a"][t - 1][sl],
                                      field.live["cl561"][t - 1][sl], rng))
        draw_fates(ax[1, c], tr, lab, 13, y0, x0, side)
        ax[2, c].imshow(unit(field.terminal["annexin"][sl], *rng["annexin"]), cmap="gray", vmin=0, vmax=1)
        draw_fates(ax[2, c], tr, lab, 13, y0, x0, side)
        ax[0, c].set_title(f"{dose:g} nM\n{f}", color="white", fontsize=9)
        print(f"fields by dose: {dose:g} nM {f}", flush=True)
    for a, name in zip(ax[:, 0], ["0 h", "6 h, tracked nuclei\ncircled by fate", "AnnexinV after 6 h\n(registered)"], strict=True):
        a.set_ylabel(name, color="white", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.set_facecolor(BG)
        a.set_xlim(0, side)
        a.set_ylim(side, 0)
        for sp in a.spines.values():
            sp.set_visible(False)
    scale_bar(ax[0, 0], 30, side - 30, 50)
    fig.suptitle("HeLa cells under staurosporine, one field per dose (208 µm wide). Blue Hoechst, green ChromaLIVE 488, "
                 "red ChromaLIVE 561; one intensity range per channel for the whole plate", color="white", fontsize=10)
    legend_fates(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(FIG / "fig01_fields_by_dose.png", dpi=110, facecolor=BG)
    plt.close(fig)


def field_over_time(lab: pd.DataFrame, rng: dict) -> None:
    side, y0, x0 = 512, 694, 694
    frames = [1, 3, 5, 7, 9, 11, 13]
    picks = [first_labelled_field(lab, 0.0), first_labelled_field(lab, 39.06)]
    fig, ax = plt.subplots(2, len(frames), figsize=(2.6 * len(frames), 5.9), facecolor=BG)
    for r, f in enumerate(picks):
        tr = tracks_of([f])
        dose = lab.loc[lab["well_fov"] == f, "dose_nM"].iloc[0]
        for c, t in enumerate(frames):
            ax[r, c].imshow(rgb_frame(f, t, rng)[y0:y0 + side, x0:x0 + side])
            draw_fates(ax[r, c], tr, lab, t, y0, x0, side)
            if r == 0:
                ax[r, c].set_title(f"{(t - 1) // 2} h", color="white", fontsize=10)
        ax[r, 0].set_ylabel(f"{dose:g} nM\n{f}", color="white", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim(0, side)
        a.set_ylim(side - 1, 0)
        for sp in a.spines.values():
            sp.set_visible(False)
    scale_bar(ax[0, 0], 24, side - 24, 50)
    fig.suptitle("The same field every hour (166 µm wide); tracked nuclei circled by their fate at 6 h", color="white", fontsize=10)
    legend_fates(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(FIG / "fig01b_field_over_time.png", dpi=110, facecolor=BG)
    plt.close(fig)


def masks_registration(lab: pd.DataFrame, rng: dict) -> None:
    """What the label is measured on: nucleus masks, cell masks, and the registered terminal image."""
    from skimage.segmentation import find_boundaries

    from static2fate.cells import cell_masks
    from static2fate.fields import segment_nuclei_classical

    side, y0, x0 = 512, 694, 694
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    picks = [first_labelled_field(lab, d) for d in (0.0, 19.53, 39.06)]
    titles = ["Hoechst at 6 h\nnucleus masks", "all channels at 6 h\ncell masks (white), nuclei (yellow)",
              "AnnexinV after 6 h, registered\nsame cell masks", "Hoechst 6 h (magenta) and terminal (green)\nas deposited",
              "after registration\n(white = overlap)"]
    fig, ax = plt.subplots(len(picks), 5, figsize=(15.5, 3.25 * len(picks)), facecolor=BG)
    for r, f in enumerate(picks):
        field = Field.load(IMAGES, f)
        nuclei = segment_nuclei_classical(field.live["hoechst"][-1][None])[0]
        cells = cell_masks([field.live[c][-1] for c in ("cl488a", "cl488b", "cl561")], nuclei)
        h13 = unit(field.live["hoechst"][-1][sl], *rng["hoechst"])
        nb, cb = find_boundaries(nuclei[sl]), find_boundaries(cells[sl])
        ax[r, 0].imshow(h13, cmap="gray", vmin=0, vmax=1)
        ax[r, 0].contour(nb, levels=[0.5], colors="#ffdd33", linewidths=0.6)
        ax[r, 1].imshow(composite(field.live["hoechst"][-1][sl], field.live["cl488a"][-1][sl], field.live["cl561"][-1][sl], rng))
        ax[r, 1].contour(cb, levels=[0.5], colors="white", linewidths=0.6)
        ax[r, 1].contour(nb, levels=[0.5], colors="#ffdd33", linewidths=0.5)
        ax[r, 2].imshow(unit(field.terminal["annexin"][sl], *rng["annexin"]), cmap="gray", vmin=0, vmax=1)
        ax[r, 2].contour(cb, levels=[0.5], colors="white", linewidths=0.5)
        raw = unit(read_frame(IMAGES, field.terminal_name, 14, CHANNELS["hoechst"])[sl], *rng["hoechst"])
        reg = unit(field.terminal["hoechst"][sl], *rng["hoechst"])
        for c, t in [(3, raw), (4, reg)]:
            ax[r, c].imshow(np.dstack([h13, t, h13]))
        dose = lab.loc[lab["well_fov"] == f, "dose_nM"].iloc[0]
        ax[r, 0].set_ylabel(f"{dose:g} nM\n{f}", color="white", fontsize=9)
        ax[r, 4].text(8, 24, f"shift {field.shift[1]:+.0f}, {field.shift[0]:+.0f} px, r = {field.match_r:.2f}", color="white", fontsize=7)
        print(f"masks and registration: {f}", flush=True)
    for c, t in enumerate(titles):
        ax[0, c].set_title(t, color="white", fontsize=8.5)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    scale_bar(ax[0, 0], 24, side - 24, 50)
    fig.suptitle("Segmentation and registration behind the fate label (166 µm wide). The AnnexinV image is taken after the movie, "
                 "on a shifted stage position, so it is registered to the last live frame", color="white", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig02b_masks_registration.png", dpi=110, facecolor=BG)
    plt.close(fig)


def clean_tracks(lab: pd.DataFrame, doses: list[float], heldout_only: bool = False) -> pd.DataFrame:
    """Labelled cells whose 128 px window stays inside the field and whose nucleus never jumps over 25 px."""
    s = lab[lab["dose_nM"].isin(doses)]
    if heldout_only:
        s = s[s["well_fov"].str[:4].isin(["E-06", "E-07", "E-08", "E-09"])]
    tr = tracks_of(sorted(s["well_fov"].unique()))
    tr = tr[tr["track"].isin(s.index)].sort_values(["track", "frame"])
    g = tr.groupby("track")
    inside = (g["y"].min() > 64) & (g["x"].min() > 64) & (g["y"].max() < 1900 - 64) & (g["x"].max() < 1900 - 64)
    step = g.apply(lambda d: np.hypot(d["y"].diff(), d["x"].diff()).max(), include_groups=False)
    keep = inside & (step < 25) & (g.size() == 13)
    return s.loc[keep[keep].index]


class Frames:
    """Raw frames read once per (field, frame, channel)."""

    def __init__(self):
        self.cache = {}

    def get(self, well_fov: str, t: int, c: int) -> np.ndarray:
        key = (well_fov, t, c)
        if key not in self.cache:
            self.cache[key] = read_frame(IMAGES, well_fov, t, c)
        return self.cache[key]

    def cell(self, well_fov: str, t: int, y: float, x: float, rng: dict, size: int = 128) -> np.ndarray:
        return composite(*(window(self.get(well_fov, t, c), y, x, size) for c in CHANNELS.values()), rng)


def cell_galleries(lab: pd.DataFrame, rng: dict) -> None:
    ok = clean_tracks(lab, [19.53, 39.06])
    pos = tracks_of(sorted(ok["well_fov"].unique())).set_index(["track", "frame"])[["y", "x"]]
    frames = [1, 3, 5, 7, 9, 11, 13]
    rs = np.random.default_rng(SEED)
    for g in GROUPS:
        pick = rs.choice(ok.index[ok["group"] == g], 24, replace=False)
        pick = sorted(pick, key=lambda t: (ok.loc[t, "dose_nM"], t))
        reader = Frames()
        tiles = []
        for i in range(0, 24, 2):
            row = []
            for t in pick[i:i + 2]:
                f = ok.loc[t, "well_fov"]
                row += [reader.cell(f, k, *pos.loc[(t, k)], rng) for k in frames] + [None]
            tiles.append(row[:-1])
        img, tops, lefts = mosaic(tiles, gap=3, fill=17)
        fig, ax = plt.subplots(figsize=(img.shape[1] / 110, img.shape[0] / 110 + 0.75), facecolor=BG)
        ax.imshow(img)
        for j in range(2):
            for c, k in enumerate(frames):
                ax.text(lefts[j * 8 + c] + 64, -8, f"{(k - 1) // 2} h", color="white", ha="center", fontsize=8)
        for r in range(12):
            for j in range(2):
                t = pick[r * 2 + j]
                ax.text(lefts[j * 8] + 3, tops[r] + 12, f"{ok.loc[t, 'dose_nM']:.1f} nM", color="white", fontsize=6)
        scale_bar(ax, 6, img.shape[0] - 8, 10, label=False)
        ax.text(6 + 10 / PIXEL_UM + 6, img.shape[0] - 5, "10 µm", color="white", fontsize=6)
        ax.axis("off")
        n = int((ok["group"] == g).sum())
        ax.set_title(f"{g}: 24 cells drawn at random from {n} clean tracks at 19.53 and 39.06 nM (41.6 µm windows, raw "
                     "images, plate-wide scaling)", color=GROUP_COLOURS[g], fontsize=9, pad=16)
        fig.tight_layout()
        name = {"survives": "survive", "dies, AnnexinV only": "die_annexinv", "dies, condensed": "die_condensed"}[g]
        fig.savefig(FIG / f"fig03_cells_{name}.png", dpi=110, facecolor=BG)
        plt.close(fig)
        print(f"cell gallery: {g}", flush=True)


def colour_shift(lab: pd.DataFrame, rng: dict) -> None:
    ok = clean_tracks(lab, [39.06])
    tr = tracks_of(sorted(ok["well_fov"].unique()))
    tr = tr[tr["track"].isin(ok.index)].set_index(["track", "frame"])
    rs = np.random.default_rng(SEED)
    pick = [t for g, n in [("dies, condensed", 6), ("dies, AnnexinV only", 3), ("survives", 3)]
            for t in rs.choice(ok.index[ok["group"] == g], n, replace=False)]
    frames = [1, 3, 5, 7, 9, 11, 13]
    reader = Frames()
    fig, ax = plt.subplots(len(pick), 2, figsize=(11.5, 1.32 * len(pick)), facecolor=BG,
                           gridspec_kw={"width_ratios": [7.2, 2.4]})
    hours = np.arange(13) / 2
    for r, t in enumerate(pick):
        f, g = ok.loc[t, "well_fov"], ok.loc[t, "group"]
        row = [reader.cell(f, k, *tr.loc[(t, k), ["y", "x"]], rng) for k in frames]
        ax[r, 0].imshow(mosaic([row], gap=3, fill=17)[0])
        ax[r, 0].axis("off")
        ax[r, 0].text(4, 14, g, color=GROUP_COLOURS[g], fontsize=7)
        d = tr.loc[t].reindex(range(1, 14)).ffill()
        ratio = d["cl488a_ring_mean"] / (d["cl561_ring_mean"] + 1)
        a = ax[r, 1]
        a.set_facecolor(BG)
        a.plot(hours, np.log2(ratio / ratio.iloc[0]), color="#66dd66", lw=1.6, label="cytoplasm 488 / 561")
        a.plot(hours, np.log2(d["area"] / d["area"].iloc[0]), color="#6fa8ff", lw=1.6, label="nucleus area")
        a.axhline(0.5, color="#66dd66", lw=0.6, ls=":")
        a.axhline(-1, color="#6fa8ff", lw=0.6, ls=":")
        a.set_ylim(-2.6, 3.6)
        a.set_xlim(0, 6)
        a.tick_params(colors="white", labelsize=6)
        for sp in a.spines.values():
            sp.set_color("#666666")
        if r == 0:
            a.legend(fontsize=6, facecolor=BG, labelcolor="white", edgecolor="#444444", loc="upper left")
            for c, k in enumerate(frames):
                ax[r, 0].text(c * 131 + 64, -10, f"{(k - 1) // 2} h", color="white", ha="center", fontsize=8)
        if r == len(pick) - 1:
            a.set_xlabel("hours after drug", color="white", fontsize=7)
    fig.text(0.985, 0.5, "log2 change from 0 h (raw counts); dotted: the two thresholds", color="white", rotation=90,
             va="center", fontsize=7)
    fig.suptitle("Twelve cells at 39.06 nM drawn at random (6 condensed, 3 AnnexinV only, 3 surviving), "
                 "with the cytoplasm's 488 / 561 ratio and the nucleus area of each", color="white", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.98, 0.965))
    fig.savefig(FIG / "fig04_colour_shift.png", dpi=120, facecolor=BG)
    plt.close(fig)
    print("colour shift gallery", flush=True)


def heldout_picks(lab: pd.DataFrame, per_group: int, among: np.ndarray | None = None) -> list[str]:
    """Random held-out cells per group at 19.53 and 39.06 nM whose crop never leaves the field."""
    ok = lab[lab["dose_nM"].isin([19.53, 39.06]) & lab["well_fov"].str[:4].isin(["E-08", "E-09"])]
    tr = tracks_of(sorted(ok["well_fov"].unique()))
    g = tr[tr["track"].isin(ok.index)].groupby("track")
    inside = (g["y"].min() > 64) & (g["x"].min() > 64) & (g["y"].max() < 1900 - 64) & (g["x"].max() < 1900 - 64)
    ok = ok.loc[inside[inside].index]
    if among is not None:
        ok = ok[ok.index.isin(among)]
    rs = np.random.default_rng(SEED)
    return [t for g in GROUPS for t in rs.choice(ok.index[ok["group"] == g], per_group, replace=False)]


def tile_figure(tiles, col_titles, row_labels, row_colours, title: str, path: Path, group_gaps: bool = True) -> None:
    img, tops, lefts = mosaic(tiles, gap=3, fill=17)
    margin = 190  # image pixels kept free on the left for the row labels
    w, h = img.shape[1] + margin, img.shape[0]
    fig = plt.figure(figsize=(w / 100, h / 100 + 1.0), facecolor=BG)
    ax = fig.add_axes((0, 0, 1, h / (h + 100)))
    ax.imshow(img)
    ax.set_xlim(-margin, img.shape[1])
    ax.set_ylim(h, -34)
    for c, name in enumerate(col_titles):
        ax.text(lefts[c] + 64, -6, name, color="white", ha="center", va="bottom", fontsize=7.5)
    for r, (name, colour) in enumerate(zip(row_labels, row_colours, strict=True)):
        ax.text(-10, tops[r] + 64, name, color=colour, ha="right", va="center", fontsize=7.5)
    ax.axis("off")
    fig.text(0.5, 1 - 0.3 / (h / 100 + 1.0), title, color="white", fontsize=9, ha="center", va="center")
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print(path.name, flush=True)


def real_vs_generated(lab: pd.DataFrame) -> None:
    crops = np.load(P / "crops.npy", mmap_mode="r")
    cpos = {t: i for i, t in enumerate(np.load(P / "crops_tracks.npy"))}
    samples = np.load(MODEL / "samples.npy", mmap_mode="r")  # (cells, 4 frames [6, 8, 10, 12], M, 3, 128, 128)
    clips = np.load(MODEL / "clips.npy", mmap_mode="r")  # (cells, M, 7 frames 6..12, 3, 128, 128)
    spos = {t: i for i, t in enumerate(np.load(MODEL / "samples_tracks.npy"))}
    pick = heldout_picks(lab, 4)
    tiles = [[model_rgb(crops[cpos[t], 0]), model_rgb(crops[cpos[t], 12]), None,
              *(model_rgb(samples[spos[t], 3, k]) for k in range(4))] for t in pick]
    tile_figure(tiles, ["real 0 h\n(given)", "real 6 h", "", *[f"generated 6 h\nsample {k + 1}" for k in range(4)]],
                [f"{lab.loc[t, 'group']}\n{lab.loc[t, 'dose_nM']:.1f} nM" for t in pick],
                [GROUP_COLOURS[lab.loc[t, "group"]] for t in pick],
                "Held-out cells drawn at random, 4 per fate group. The model is given the 0 h frame and the fate group.\n"
                "Latent video model, three fate classes, guidance 2; per-field scaling, as the model sees it",
                FIG / "fig07_real_vs_generated.png")
    pick = heldout_picks(lab, 2)
    tiles, names, colours = [], [], []
    for t in pick:
        g = lab.loc[t, "group"]
        tiles.append([model_rgb(crops[cpos[t], f]) for f in [0, 6, 8, 10, 12]])
        names.append(f"real\n{g}")
        for k in range(2):
            tiles.append([model_rgb(crops[cpos[t], 0]), *(model_rgb(clips[spos[t], k, f - 6]) for f in [6, 8, 10, 12])])
            names.append(f"generated clip {k + 1}")
        colours += [GROUP_COLOURS[g], "white", "white"]
    tile_figure(tiles, ["0 h (given)", "3 h", "4 h", "5 h", "6 h"], names, colours,
                "Real course of a held-out cell and two generated clips\n(3 to 6 h drawn jointly from the 0 h frame and the fate group)",
                FIG / "fig08_generated_course.png")


def counterfactual(lab: pd.DataFrame) -> None:
    crops = np.load(P / "crops.npy", mmap_mode="r")
    cpos = {t: i for i, t in enumerate(np.load(P / "crops_tracks.npy"))}
    runs = []
    for k in range(3):
        parts = sorted((COUNTERFACTUAL / f"fate{k}").glob("samples_*_tracks.npy"))
        tracks = np.concatenate([np.load(p) for p in parts])
        arr = np.concatenate([np.load(str(p).replace("_tracks", ""), mmap_mode="r") for p in parts])
        runs.append(({t: i for i, t in enumerate(tracks)}, arr))
    pick = heldout_picks(lab, 3, among=np.array(list(runs[0][0])))
    tiles = [[model_rgb(crops[cpos[t], 0]), model_rgb(crops[cpos[t], 12]), None,
              *(model_rgb(runs[k][1][runs[k][0][t], 3, j]) for k in range(3) for j in range(2))] for t in pick]
    cols = ["real 0 h\n(given)", "real 6 h", ""] + [f"as {name}\nnoise {j + 1}" for name in
                                                      ["survives", "AnnexinV only", "condensed"] for j in range(2)]
    tile_figure(tiles, cols, [f"really {lab.loc[t, 'group']}\n{lab.loc[t, 'dose_nM']:.1f} nM" for t in pick],
                [GROUP_COLOURS[lab.loc[t, "group"]] for t in pick],
                "One 0 h cell, three generated futures at 6 h: the same two noise draws under each fate class.\n"
                "Only the future under the cell's real fate can be checked against the real 6 h frame",
                FIG / "fig09_counterfactual.png")


def main(parts: list[str]) -> None:
    importlib.import_module("09_train_flow")  # the held-out wells are those of the training script
    FIG.mkdir(parents=True, exist_ok=True)
    lab = labels()
    if "real" in parts:
        rng = ranges()
        print("display ranges:", {k: tuple(round(v) for v in r) for k, r in rng.items()}, flush=True)
        fields_by_dose(lab, rng)
        field_over_time(lab, rng)
        masks_registration(lab, rng)
        cell_galleries(lab, rng)
        colour_shift(lab, rng)
    if "generated" in parts:
        real_vs_generated(lab)
        if (COUNTERFACTUAL / "fate2").exists():
            counterfactual(lab)
        else:
            print("no counterfactual samples, skipping figure 9")


if __name__ == "__main__":
    main(sys.argv[1:] or ["real", "generated"])
