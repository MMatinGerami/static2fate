"""Controls for the anchored sampler (40): pixel-space composites of a 0 h rim and a 6 h disc.

Anchored sampling keeps the 0 h neighbourhood outside a disc and draws the cell inside it. Two
composites made in pixel space, with the same disc, say what the scores make of that layout
on their own: "real 6 h in the disc" pastes the cell's real future into its 0 h crop (the
ceiling for any sampler that holds the rim; one image per cell, so its energy score is a
plain distance, comparable with the 0 h copy's); "free samples in the disc" pastes the free
latent video model's M samples into the same rim (the free model's cells with a 0 h
neighbourhood instead of an invented one, comparable with the free model). Radii at 128 px,
edge softened over 4 px (one code pixel of the latent sampler). Output in the samples layout
of 19_generate_samples.py so the scoring chain runs with name=dir arguments.

Usage: uv run scripts/41_composite_control.py [free model dir, default results/kaggle/lat_video] [radii, default 40,56]
Writes results/anchor/composite_real_r<r>/ and results/anchor/composite_<model dir name>_r<r>/.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/anchor"
FUTURE = [6, 8, 10, 12]


def disc_weights(size: int, radius: float, edge: float = 4.0) -> np.ndarray:
    c = (size - 1) / 2
    yy, xx = np.mgrid[:size, :size]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
    return np.clip((radius - r) / edge + 0.5, 0, 1).astype(np.float32)


def composite(disc: np.ndarray, rim: np.ndarray, w: np.ndarray) -> np.ndarray:
    """uint8 (..., 3, s, s): disc inside the weights, rim outside; w (s, s) in [0, 1]."""
    return (w * disc.astype(np.float32) + (1 - w) * rim.astype(np.float32)).round().clip(0, 255).astype(np.uint8)


def write(d: Path, samples: np.ndarray, tracks: np.ndarray, meta: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "samples.npy", samples)
    np.save(d / "samples_tracks.npy", tracks)
    (d / "run.json").write_text(json.dumps(meta, indent=1))
    print(f"{d.name}: {samples.shape}", flush=True)


def main(free_dir: Path, radii: list[float]) -> None:
    train = importlib.import_module("09_train_flow")
    x, _, tracks = train.load("test", 128, frames=[0] + FUTURE)
    u8 = ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()  # (n, 5, 3, 128, 128)
    rim0, real = u8[:, 0], u8[:, 1:]  # 0 h and the four future frames
    free = np.load(free_dir / "samples.npy", mmap_mode="r")  # (n, 4, M, 3, 128, 128)
    assert (np.load(free_dir / "samples_tracks.npy") == tracks).all()
    for radius in radii:
        w = disc_weights(128, radius)
        meta = {"radius_px": radius, "edge_px": 4.0, "free_fraction": float(w.mean())}
        write(OUT / f"composite_real_r{radius:g}", composite(real, rim0[:, None], w)[:, :, None], tracks,
              {"disc": "real future frame", "rim": "real 0 h frame", "m": 1, **meta})
        out = np.empty(free.shape, dtype=np.uint8)
        for i in range(len(tracks)):
            out[i] = composite(np.asarray(free[i]), rim0[i][None, None], w)
        write(OUT / f"composite_{free_dir.name}_r{radius:g}", out, tracks,
              {"disc": f"samples of {free_dir.relative_to(ROOT)}", "rim": "real 0 h frame", "m": free.shape[2], **meta})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "results/kaggle/lat_video",
         [float(r) for r in sys.argv[2].split(",")] if len(sys.argv) > 2 else [40.0, 56.0])
