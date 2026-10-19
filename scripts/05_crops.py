"""Single-cell patches for every complete track at every live frame.

128x128 px around the nucleus centroid, three channels (Hoechst, ChromaLIVE 488a, ChromaLIVE
561), scaled to uint8 with one pair of percentiles per field and channel over all 13 frames, so
that a cell getting brighter over time stays brighter in the patch. Patches that would leave the
image are zero-padded. Output: data/processed/crops.npz with arrays
  x      uint8 (n, 13, 3, 128, 128)
  track  str   (n,)
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/raw/images"
P = ROOT / "data/processed"
SIZE = 128
CHANNELS = {"hoechst": 1, "cl488a": 2, "cl561": 4}


def load_scaled(well_fov: str) -> np.ndarray:
    """(13, 3, H, W) uint8 stack for one field."""
    out = []
    for c in CHANNELS.values():
        stack = np.stack([tifffile.imread(IMAGES / f"{well_fov}_T{t:04d}_Z0001_C{c:02d}.tif") for t in range(1, 14)]).astype(np.float32)
        lo, hi = np.percentile(stack, [0.5, 99.8])
        out.append(np.clip((stack - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8))
    return np.stack(out, axis=1)


def crop(stack: np.ndarray, y: float, x: float) -> np.ndarray:
    h = SIZE // 2
    yi, xi = int(round(y)), int(round(x))
    pad = np.pad(stack, ((0, 0), (0, 0), (h, h), (h, h)))
    return pad[:, :, yi : yi + SIZE, xi : xi + SIZE]


def main() -> None:
    labels = pd.read_parquet(P / "fate_labels.parquet")
    tracks = pd.concat([pd.read_parquet(p) for p in glob.glob(str(P / "tracks/*.parquet"))], ignore_index=True)
    tracks["track"] = tracks["well_fov"] + "_" + tracks["particle"].astype(str)
    tracks = tracks[tracks["track"].isin(labels["track"]) & (tracks["frame"] <= 13)]

    xs, names = [], []
    for well_fov, g in tracks.groupby("well_fov"):
        stack = load_scaled(well_fov)
        for track, tg in g.groupby("track"):
            tg = tg.set_index("frame").sort_index()
            # a track may skip up to two frames (trackpy memory); reuse the previous position
            frames = []
            last = None
            for t in range(1, 14):
                if t in tg.index:
                    last = (tg.loc[t, "y"], tg.loc[t, "x"])
                frames.append(crop(stack[t - 1 : t], *last)[0] if last is not None else np.zeros((3, SIZE, SIZE), np.uint8))
            xs.append(np.stack(frames))
            names.append(track)
        print(f"{well_fov}: {g['track'].nunique()} tracks")
    x = np.stack(xs)
    np.savez_compressed(P / "crops.npz", x=x, track=np.array(names))
    print(f"wrote {x.shape} to crops.npz")


if __name__ == "__main__":
    main()
