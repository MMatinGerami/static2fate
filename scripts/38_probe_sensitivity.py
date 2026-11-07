"""What does the real-vs-generated probe react to? Score near-copies of the real frames.

The DINOv2 probe tells the autoencoder's reconstruction of a held-out cell from the cell
itself at AUROC 0.995 although the two agree to 33 dB (docs/results.md). To see what it
reads, the real held-out frames 6, 8, 10 and 12 are written in the samples layout of
19_generate_samples.py under small, known changes, and scored with the same chain
(20_feature_score.py then 21_real_vs_generated.py, tag _probe):

- copy: the real frame itself (the probe must give 0.5);
- blur 0.5 px, blur 1 px: a Gaussian blur of that width at 128 px, which removes camera
  noise and a little detail but no biology;
- ae128 + noise: the reconstruction with Gaussian noise added, per channel at the standard
  deviation of the residual (real minus reconstruction) over the held-out images, so it
  has the real image's noise energy but not its noise pattern.

Usage: uv run scripts/38_probe_sensitivity.py [ae recon dir, default results/kaggle/ae128/recon]
Writes results/probe_check/<variant>/samples.npy (cells, 4, 1, 3, 128, 128) uint8 and
samples_tracks.npy, then prints the scoring command.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/probe_check"
FUTURE = [6, 8, 10, 12]


def blur(x8: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of uint8 images (..., 3, s, s), per channel, back to uint8."""
    flat = x8.reshape(-1, *x8.shape[-3:]).astype(np.float32)
    out = np.empty_like(flat)
    for i, im in enumerate(flat):
        out[i] = gaussian_filter(im, (0, sigma, sigma), mode="reflect")
    return out.round().clip(0, 255).astype(np.uint8).reshape(x8.shape)


def residual_std(real8: np.ndarray, recon8: np.ndarray) -> np.ndarray:
    """Per-channel standard deviation (in uint8 units) of real minus reconstruction."""
    d = real8.astype(np.float32) - recon8.astype(np.float32)
    axes = tuple(i for i in range(d.ndim) if i != d.ndim - 3)
    return d.std(axis=axes)


def add_noise(x8: np.ndarray, std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(x8.shape, dtype=np.float32) * std.reshape(-1, 1, 1)
    return (x8.astype(np.float32) + noise).round().clip(0, 255).astype(np.uint8)


def variants(real8: np.ndarray, recon8: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    std = residual_std(real8, recon8)
    print(f"residual std per channel (uint8 units): {np.round(std, 2)}", flush=True)
    return {"copy": real8.copy(), "blur05": blur(real8, 0.5), "blur1": blur(real8, 1.0),
            "ae128_noise": add_noise(recon8, std, rng)}


def main(recon_dir: Path) -> None:
    train = importlib.import_module("09_train_flow")
    x, _, tracks = train.load("test", 128, frames=FUTURE)
    real8 = ((x + 1) * 127.5).round().clamp(0, 255).to("cpu").numpy().astype(np.uint8)  # (n, 4, 3, 128, 128)
    recon = np.load(recon_dir / "samples.npy")  # (n, 4, 1, 3, 128, 128)
    assert (np.load(recon_dir / "samples_tracks.npy") == tracks).all()
    recon8 = recon[:, :, 0]
    for name, arr in variants(real8, recon8).items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "samples.npy", arr[:, :, None])
        np.save(d / "samples_tracks.npy", tracks)
        print(f"{name}: {arr.shape} -> {d}", flush=True)
    names = {"copy": "real copy", "blur05": "blur 0.5 px", "blur1": "blur 1 px", "ae128_noise": "ae128 + noise"}
    args = " ".join(f'"{v}={OUT / k}"' for k, v in names.items())
    print(f"score with:\nSTATIC2FATE_TAG=_probe uv run scripts/20_feature_score.py {args}\n"
          f"STATIC2FATE_TAG=_probe uv run scripts/21_real_vs_generated.py {args}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/kaggle/ae128/recon")
