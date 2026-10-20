"""One field of view of S-BIAD2515: load, register the terminal frame, segment, measure, track.

Frames 1..13 are live (Hoechst C01, ChromaLIVE C02 C03 C04, 30 min apart). Frame 14 is the
fixed terminal image (Hoechst C01, AnnexinV C05). The terminal image is shifted by up to ~180 px
against frame 13. The terminal stain was acquired with six fields per well (documented in the
Way lab's 1.pre_process_data/scripts/2.sort_files_properly.py: timelapse FOV3 = terminal FOV4,
FOV4 = terminal FOV5, terminal FOV3 and FOV6 unused); the BioStudies copy carries the raw
numbering with only F0001 to F0004 deposited, so the file named F0003 is the unused field, the
file named F0004 belongs to live F0003, and live F0004 has no partner. The partner of each live
field is therefore found by cross-correlation against all terminal files of the well (which
reproduces their mapping in all 30 wells), and a field without a convincing partner gets no fate.

Nuclei are segmented on Hoechst and linked over time with trackpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology
from skimage.measure import regionprops_table
from skimage.registration import phase_cross_correlation
from skimage.segmentation import expand_labels, watershed
from skimage.transform import downscale_local_mean

LIVE_FRAMES = range(1, 14)
TERMINAL_FRAME = 14
LIVE_CHANNELS = {"hoechst": 1, "cl488a": 2, "cl488b": 3, "cl561": 4}
TERMINAL_CHANNELS = {"hoechst": 1, "annexin": 5}
RING_PX = 8  # cytoplasm proxy: ring of this width around the nucleus
MIN_MATCH_R = 0.3  # Pearson r between live frame 13 and the shifted terminal Hoechst image


def frame_path(images: Path, well_fov: str, t: int, c: int) -> Path:
    return images / f"{well_fov}_T{t:04d}_Z0001_C{c:02d}.tif"


def read_tiff(path: Path) -> np.ndarray | None:
    """None for a file that is missing or truncated (two of the 6,480 deposited files are)."""
    try:
        img = tifffile.imread(path)
    except Exception:
        return None
    return img if img.shape == (1900, 1900) else None


def match_terminal(live_last: np.ndarray, terminals: dict[str, np.ndarray]) -> tuple[str | None, np.ndarray, float]:
    """Which terminal Hoechst image is this live field, and by what (dy, dx) shift?

    Cross-correlation (no phase normalisation; the normalised version returns (0, 0) because the
    fixed image has a different intensity structure) at 4x downsampling, refined at full
    resolution for the winner. Returns (terminal well_fov or None, shift, r).
    """
    a_small = downscale_local_mean(live_last.astype(np.float32), (4, 4))
    best = (None, np.zeros(2), -1.0)
    for name, img in terminals.items():
        b_small = downscale_local_mean(img.astype(np.float32), (4, 4))
        sh, _, _ = phase_cross_correlation(a_small, b_small, normalization=None)
        r = _shifted_corr(a_small, b_small, sh)
        if r > best[2]:
            best = (name, sh * 4, r)
    name, sh, r = best
    if name is None or r < MIN_MATCH_R:
        return None, np.zeros(2), r
    a = live_last.astype(np.float32)
    b = terminals[name].astype(np.float32)
    fine, _, _ = phase_cross_correlation(a, b, normalization=None)
    if np.abs(fine - sh).max() <= 8:
        sh = fine
    return name, np.asarray(sh, dtype=float), _shifted_corr(a, b, sh)


def _shifted_corr(a: np.ndarray, b: np.ndarray, shift) -> float:
    bs = ndi.shift(b, shift, order=1, cval=0)
    m = (bs > 0) & (a > 0)
    return float(np.corrcoef(a[m], bs[m])[0, 1]) if m.sum() > 500 else -1.0


@dataclass
class Field:
    well_fov: str
    live: dict[str, np.ndarray]  # channel -> (13, H, W)
    terminal: dict[str, np.ndarray] | None  # channel -> (H, W), registered onto live frame 13
    terminal_name: str | None  # which deposited terminal field it was
    shift: tuple[float, float]
    match_r: float

    @classmethod
    def load(cls, images: Path, well_fov: str) -> "Field":
        live = {}
        for name, c in LIVE_CHANNELS.items():
            frames = [read_tiff(frame_path(images, well_fov, t, c)) for t in LIVE_FRAMES]
            if any(f is None for f in frames):
                raise FileNotFoundError(f"{well_fov}: unreadable {name} frame(s) {[t for t, f in zip(LIVE_FRAMES, frames) if f is None]}")
            live[name] = np.stack(frames)
        well = well_fov.split("_")[0]
        candidates = sorted(p.name.rsplit("_T", 1)[0] for p in images.glob(f"{well}_F*_T{TERMINAL_FRAME:04d}_Z0001_C01.tif"))
        hoechst = {c: read_tiff(frame_path(images, c, TERMINAL_FRAME, 1)) for c in candidates}
        hoechst = {c: img for c, img in hoechst.items() if img is not None}
        name, shift, r = match_terminal(live["hoechst"][-1], hoechst)
        terminal = None
        if name is not None:
            raw = {ch: tifffile.imread(frame_path(images, name, TERMINAL_FRAME, c)) for ch, c in TERMINAL_CHANNELS.items()}
            terminal = {
                ch: ndi.shift(img.astype(np.float32), shift, order=1, cval=0).astype(np.uint16) for ch, img in raw.items()
            }
        return cls(well_fov, live, terminal, name, (float(shift[0]), float(shift[1])), r)

    def hoechst_stack(self) -> np.ndarray:
        """Hoechst frames 1..13 plus the registered terminal frame when there is one."""
        frames = [self.live["hoechst"]]
        if self.terminal is not None:
            frames.append(self.terminal["hoechst"][None])
        return np.concatenate(frames)


def segment_nuclei_cellpose(stack: np.ndarray, model, diameter: float | None = None) -> np.ndarray:
    """Cellpose label masks for each frame of a (T, H, W) stack."""
    masks = [model.eval(frame, diameter=diameter, flow_threshold=0.4, cellprob_threshold=0.0)[0] for frame in stack]
    return np.stack(masks).astype(np.int32)


def segment_nuclei_classical(stack: np.ndarray, sigma: float = 2, min_area: int = 80, min_distance: int = 9) -> np.ndarray:
    """Otsu threshold on smoothed Hoechst, then distance-transform watershed to split touching nuclei.

    About 1 s per 1900x1900 frame on CPU; Cellpose-SAM takes minutes per frame on this machine.
    The threshold is taken per frame, so brightening of condensed apoptotic nuclei does not
    shift the foreground of the other frames.
    """
    out = []
    for frame in stack:
        sm = filters.gaussian(frame.astype(np.float32), sigma, preserve_range=True)
        fg = sm > filters.threshold_otsu(sm)
        fg = ndi.binary_fill_holes(morphology.remove_small_objects(fg, max_size=min_area))
        dist = ndi.distance_transform_edt(fg)
        peaks = feature.peak_local_max(dist, min_distance=min_distance, labels=measure.label(fg))
        markers = np.zeros(fg.shape, np.int32)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        out.append(watershed(-dist, markers, mask=fg))
    return np.stack(out).astype(np.int32)


def measure_frame(labels: np.ndarray, channels: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-nucleus centroid, area and per-channel mean inside the nucleus and in a ring around it."""
    props = regionprops_table(labels, properties=("label", "centroid", "area", "eccentricity", "solidity"))
    df = pd.DataFrame(props).rename(columns={"centroid-0": "y", "centroid-1": "x"})
    ring = expand_labels(labels, RING_PX)
    ring[labels > 0] = 0
    idx = df["label"].to_numpy()
    with np.errstate(invalid="ignore"):  # a ring can be empty at the image border
        for name, img in channels.items():
            img = img.astype(np.float64)
            df[f"{name}_nuc_mean"] = ndi.mean(img, labels, idx)
            df[f"{name}_ring_mean"] = ndi.mean(img, ring, idx)
            df[f"{name}_ring_sum"] = ndi.sum(img, ring, idx)
    return df


def measure_field(field: Field, masks: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(LIVE_FRAMES):
        chans = {name: field.live[name][i] for name in LIVE_CHANNELS}
        df = measure_frame(masks[i], chans)
        df["frame"] = t
        rows.append(df)
    if field.terminal is not None:
        df = measure_frame(masks[-1], field.terminal)
        df["frame"] = TERMINAL_FRAME
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "well_fov", field.well_fov)
    return out


def link_tracks(df: pd.DataFrame, search_range: float = 70, memory: int = 2) -> pd.DataFrame:
    """trackpy linking on nucleus centroids with a constant-velocity predictor; adds 'particle'.

    HeLa nuclei at 30 min intervals: nearest-neighbour displacement p50 13 px, p90 44 px, p99 89 px
    (C-02_F0001), so a 30 px search range fragments most tracks.
    """
    tp.quiet()
    pred = tp.predict.NearestVelocityPredict()
    return pred.link_df(df, search_range=search_range, memory=memory, t_column="frame")


def process_field(images: Path, well_fov: str, model=None, diameter: float | None = None) -> tuple[pd.DataFrame, dict]:
    """model=None uses the classical segmenter; otherwise a cellpose CellposeModel."""
    field = Field.load(images, well_fov)
    stack = field.hoechst_stack()
    masks = segment_nuclei_classical(stack) if model is None else segment_nuclei_cellpose(stack, model, diameter)
    df = link_tracks(measure_field(field, masks))
    info = {
        "well_fov": well_fov,
        "terminal_field": field.terminal_name,
        "match_r": round(field.match_r, 3),
        "shift_dy": field.shift[0],
        "shift_dx": field.shift[1],
        "nuclei_per_frame": [int(m.max()) for m in masks],
        "n_tracks": int(df["particle"].nunique()),
    }
    return df, info
