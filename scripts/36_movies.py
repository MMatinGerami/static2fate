"""Movies: real cells over 6 h next to generated futures, and one whole field with fate outlines.

Cell level: for each fate group, held-out cells with clean tracks (never near the field
border, no jump over 25 px between frames), chosen with a fixed seed. Each row is one cell:
its real course at 128 px, the same at 64 px (the resolution the generator works at), then
generated clips of the video model started from the frame model (results/kaggle/video64_init,
29_generate_clips.py). The generator saw only the 0 h frame, so until 3 h the generated
columns show that frame, marked "given"; from 3 h on they play the clip. One GIF per group with
three cells and three clips, and a strip for the README with one cell per group and two clips.

Field level: the central quarter of one held-out field at 19.53 nM over the 13 frames, colour
composite (red ChromaLIVE 561, green ChromaLIVE 488, blue Hoechst), every tracked and labelled
nucleus outlined in the colour of its fate.

Usage: uv run scripts/36_movies.py [video model dir] [field, default E-08_F0001]
Writes results/movies/cells_{survive,die_annexinv,die_condensed}.gif, strip.gif and field_<field>.gif.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import group_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "results/movies"
GROUPS = ["survives", "dies, AnnexinV only", "dies, condensed"]
COLOURS = {"survives": (80, 220, 120), "dies, AnnexinV only": (255, 170, 40), "dies, condensed": (255, 70, 70)}
TILE, GAP, HEAD = 128, 6, 18
FRAME_MS, LAST_MS = 400, 1500


def font(size: int):
    for name in ["Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rgb(im: np.ndarray) -> Image.Image:
    """(3, s, s) uint8 Hoechst, 488, 561 -> PIL image red 561, green 488, blue Hoechst, at TILE px."""
    pil = Image.fromarray(np.dstack([im[2], im[1], im[0]]))
    return pil.resize((TILE, TILE), Image.NEAREST) if pil.size != (TILE, TILE) else pil


def hhmm(f: int) -> str:
    return f"{f * 30 // 60} h {f * 30 % 60:02d}"


def clean_heldout_cells(train) -> pd.DataFrame:
    """Labelled cells of the held-out two-fate wells whose track stays inside and does not jump."""
    lab = pd.read_parquet(P / "fate_labels.parquet").set_index("track")
    _, tracks = train.split_index("test")
    lab = lab.loc[tracks]
    tr = pd.concat([pd.read_parquet(P / f"tracks/{w}.parquet") for w in lab["well_fov"].unique()], ignore_index=True)
    tr["track"] = tr["well_fov"] + "_" + tr["particle"].astype(str)
    tr = tr[tr["track"].isin(lab.index) & (tr["frame"] <= 13)].sort_values(["track", "frame"])
    g = tr.groupby("track")
    inside = g.apply(lambda d: (d.y.min() > 64) & (d.x.min() > 64) & (d.y.max() < 1900 - 64) & (d.x.max() < 1900 - 64))
    step = g.apply(lambda d: np.hypot(d.y.diff(), d.x.diff()).max())
    ok = lab.loc[inside[inside & (step < 25)].index].copy()
    ok["group"] = [group_of(int(a), bool(c)) for a, c in zip(ok["apoptotic"], ok["condensed"])]
    return ok


def load_clips(mdir: Path):
    parts = sorted(mdir.glob("clips*_tracks.npy"))
    tracks = np.concatenate([np.load(p) for p in parts])
    arr = np.concatenate([np.load(Path(str(p).replace("_tracks", "")), mmap_mode="r") for p in parts])
    return {t: i for i, t in enumerate(tracks)}, arr  # arr (cells, M, 7, 3, s, s), frames 6..12


def cell_movie(rows: list[tuple[str, int, str]], crops, clip_pos, clips, n_clips: int, title: str, path: Path):
    """rows: (track, crop index, group). Columns: real 128, real 64, generated clips."""
    cols = ["real, 128 px", "real, 64 px"] + [f"generated {k + 1}" for k in range(n_clips)]
    w = GAP + len(cols) * (TILE + GAP)
    h = HEAD + 22 + len(rows) * (TILE + GAP + 14)
    f_small, f_big = font(11), font(13)
    frames = []
    for f in range(13):
        im = Image.new("RGB", (w, h), (18, 18, 18))
        d = ImageDraw.Draw(im)
        d.text((GAP, 3), f"{title}    t = {hhmm(f)} after staurosporine", fill=(235, 235, 235), font=f_big)
        for j, c in enumerate(cols):
            d.text((GAP + j * (TILE + GAP), HEAD + 6), c, fill=(200, 200, 200), font=f_small)
        for r, (track, ci, grp) in enumerate(rows):
            y0 = HEAD + 22 + r * (TILE + GAP + 14)
            x = np.asarray(crops[ci, f])
            small = x.reshape(3, 64, 2, 64, 2).mean((2, 4)).round().astype(np.uint8)
            tiles = [rgb(x), rgb(small)]
            for k in range(n_clips):
                if f < 6:
                    ctx = np.asarray(crops[ci, 0]).reshape(3, 64, 2, 64, 2).mean((2, 4)).round().astype(np.uint8)
                    t = rgb(ctx).point(lambda v: v * 0.45)
                else:
                    t = rgb(np.asarray(clips[clip_pos[track], k, f - 6]))
                tiles.append(t)
            for j, t in enumerate(tiles):
                x0 = GAP + j * (TILE + GAP)
                im.paste(t, (x0, y0))
                if j >= 2 and f < 6:
                    d.text((x0 + 4, y0 + 4), "given: 0 h frame", fill=(230, 230, 230), font=f_small)
            d.rectangle([GAP - 2, y0 - 2, GAP + TILE + 1, y0 + TILE + 1], outline=COLOURS[grp], width=2)
            d.text((GAP, y0 + TILE + 1), f"{grp} ({track})", fill=COLOURS[grp], font=f_small)
        frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    save_gif(frames, path)


def save_gif(frames, path: Path):
    durations = [FRAME_MS] * (len(frames) - 1) + [LAST_MS]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    print(f"{path.relative_to(ROOT)}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB", flush=True)


def field_movie(well_fov: str, path: Path):
    crops_mod = importlib.import_module("05_crops")
    stack = crops_mod.load_scaled(well_fov)  # (13, 3, 1900, 1900) uint8
    lab = pd.read_parquet(P / "fate_labels.parquet").set_index("track")
    tr = pd.read_parquet(P / f"tracks/{well_fov}.parquet")
    tr["track"] = tr["well_fov"] + "_" + tr["particle"].astype(str)
    tr = tr[tr["track"].isin(lab.index) & (tr["frame"] <= 13)]
    y0, x0, side = 475, 475, 950  # central quarter, shown at half resolution
    f_small, f_big = font(12), font(14)
    frames = []
    for f in range(13):
        im = stack[f, :, y0:y0 + side, x0:x0 + side]
        pil = Image.fromarray(np.dstack([im[2], im[1], im[0]])).resize((side // 2, side // 2), Image.BILINEAR)
        canvas = Image.new("RGB", (side // 2, side // 2 + 40), (18, 18, 18))
        canvas.paste(pil, (0, 0))
        d = ImageDraw.Draw(canvas)
        for _, row in tr[tr["frame"] == f + 1].iterrows():
            yy, xx = (row["y"] - y0) / 2, (row["x"] - x0) / 2
            if not (0 <= yy < side // 2 and 0 <= xx < side // 2):
                continue
            grp = group_of(int(lab.loc[row["track"], "apoptotic"]), bool(lab.loc[row["track"], "condensed"]))
            r = max(4.0, np.sqrt(row["area"] / np.pi) / 2 + 2)
            d.ellipse([xx - r, yy - r, xx + r, yy + r], outline=COLOURS[grp], width=1)
        dose = lab.loc[lab["well_fov"] == well_fov, "dose_nM"].iloc[0]
        d.text((6, side // 2 + 4), f"{well_fov}, {dose:g} nM staurosporine, t = {hhmm(f)}", fill=(235, 235, 235), font=f_big)
        x = 6
        for grp in GROUPS:
            d.ellipse([x, side // 2 + 26, x + 9, side // 2 + 35], outline=COLOURS[grp], width=2)
            d.text((x + 13, side // 2 + 23), grp, fill=COLOURS[grp], font=f_small)
            x += 13 + d.textlength(grp, font=f_small) + 16
        frames.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    save_gif(frames, path)


def main(mdir: Path, well_fov: str) -> None:
    train = importlib.import_module("09_train_flow")
    OUT.mkdir(parents=True, exist_ok=True)
    ok = clean_heldout_cells(train)
    crops = np.load(P / "crops.npy", mmap_mode="r")
    crop_pos = {t: i for i, t in enumerate(np.load(P / "crops_tracks.npy"))}
    clip_pos, clips = load_clips(mdir)
    ok = ok[ok.index.isin(crop_pos) & ok.index.isin(clip_pos)]
    rng = np.random.default_rng(0)
    picks = {g: list(rng.choice(ok.index[ok["group"] == g], 3, replace=False)) for g in GROUPS}
    print({g: len(ok[ok["group"] == g]) for g in GROUPS}, "clean held-out cells", flush=True)
    titles = {"survives": ("Held-out cells that survive", "cells_survive.gif"),
              "dies, AnnexinV only": ("Held-out cells that die, AnnexinV only", "cells_die_annexinv.gif"),
              "dies, condensed": ("Held-out cells that die, condensed nucleus", "cells_die_condensed.gif")}
    for g in GROUPS:
        rows = [(t, crop_pos[t], g) for t in picks[g]]
        cell_movie(rows, crops, clip_pos, clips, 3, titles[g][0], OUT / titles[g][1])
    rows = [(picks[g][0], crop_pos[picks[g][0]], g) for g in GROUPS]
    cell_movie(rows, crops, clip_pos, clips, 2, "Real course and generated futures", OUT / "strip.gif")
    field_movie(well_fov, OUT / f"field_{well_fov}.gif")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/kaggle/video64_init",
         sys.argv[2] if len(sys.argv) > 2 else "E-08_F0001")
