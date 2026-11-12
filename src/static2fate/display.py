"""One display convention for every image figure of real cells.

Colours: Hoechst blue, ChromaLIVE 488 (channel 488a) green, ChromaLIVE 561 red, AnnexinV grey.
Intensities: linear, one (low, high) pair per channel for the whole plate (plate_ranges), so a
cell is drawn on the same scale at every dose and time. This is for looking at cells; the
models use the per-field scaling of 05_crops.py, and figures of model inputs and outputs say so.
Pixel size: 0.325 um, from utils/pixel_dimension.toml of the Way lab's analysis code
(Zenodo 10.5281/zenodo.17419412).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

PIXEL_UM = 0.325
CHANNELS = {"hoechst": 1, "cl488a": 2, "cl561": 4}
ANNEXIN = 5
GROUPS = ["survives", "dies, AnnexinV only", "dies, condensed"]
GROUP_COLOURS = {"survives": "#4da6ff", "dies, AnnexinV only": "#ffa726", "dies, condensed": "#ff4d4d"}


def read_frame(images: Path, well_fov: str, t: int, c: int) -> np.ndarray:
    return tifffile.imread(images / f"{well_fov}_T{t:04d}_Z0001_C{c:02d}.tif")


def plate_ranges(images: Path, fields: list[str], frames: tuple[int, ...] = (1, 13),
                 q: tuple[float, float] = (0.5, 99.8), step: int = 8) -> dict[str, tuple[float, float]]:
    """(low, high) per live channel from every `step`-th pixel of the given fields and frames, pooled."""
    out = {}
    for name, c in CHANNELS.items():
        px = np.concatenate([read_frame(images, f, t, c)[::step, ::step].ravel() for f in fields for t in frames])
        lo, hi = np.percentile(px, q)
        out[name] = (float(lo), float(hi))
    return out


def unit(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear map of [lo, hi] to [0, 1], clipped."""
    return np.clip((np.asarray(img, np.float32) - lo) / (hi - lo), 0, 1)


def composite(hoechst: np.ndarray, green: np.ndarray, red: np.ndarray, ranges: dict[str, tuple[float, float]]) -> np.ndarray:
    """Raw channel images -> (H, W, 3) uint8: red 561, green 488, blue Hoechst."""
    rgb = np.dstack([unit(red, *ranges["cl561"]), unit(green, *ranges["cl488a"]), unit(hoechst, *ranges["hoechst"])])
    return (rgb * 255).round().astype(np.uint8)


def model_rgb(x: np.ndarray) -> np.ndarray:
    """(3, s, s) uint8 in the models' channel order (Hoechst, 488, 561) -> (s, s, 3) uint8 RGB."""
    return np.dstack([x[2], x[1], x[0]])


def window(img: np.ndarray, y: float, x: float, size: int) -> np.ndarray:
    """size x size window of a 2-D image centred on (y, x), zero-padded at the border."""
    h = size // 2
    yi, xi = round(float(y)), round(float(x))
    pad = np.pad(img, ((h, h), (h, h)))
    return pad[yi:yi + size, xi:xi + size]


def mosaic(tiles: list[list[np.ndarray | None]], gap: int = 4, fill: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid of equal-size (s, s, 3) uint8 tiles (None leaves a hole) -> image, row tops, column lefts."""
    s = next(t for row in tiles for t in row if t is not None).shape[0]
    rows, cols = len(tiles), max(len(r) for r in tiles)
    img = np.full((rows * s + (rows - 1) * gap, cols * s + (cols - 1) * gap, 3), fill, np.uint8)
    tops, lefts = np.arange(rows) * (s + gap), np.arange(cols) * (s + gap)
    for r, row in enumerate(tiles):
        for c, t in enumerate(row):
            if t is not None:
                img[tops[r]:tops[r] + s, lefts[c]:lefts[c] + s] = t
    return img, tops, lefts


def scale_bar(ax, x: float, y: float, um: float, pixel_um: float = PIXEL_UM, zoom: float = 1.0, label: bool = True) -> None:
    """White bar of `um` micrometres with its left end at (x, y) in image pixels; zoom = display px per raw px."""
    length = um / pixel_um * zoom
    ax.plot([x, x + length], [y, y], color="white", lw=2.5, solid_capstyle="butt")
    if label:
        ax.text(x + length / 2, y - 4, f"{um:g} µm", color="white", ha="center", va="bottom", fontsize=7)
