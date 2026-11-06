"""Encode every frame of the two-fate cells with a trained autoencoder, for the latent flow models.

Both splits (train and held-out wells) at 128 px, posterior mean divided by the autoencoder's
latent_std, saved as float16: train.npz and test.npz with z (cells, 13, 4, 32, 32), fate and
tracks, plus info.json. Augmentation of the flow models happens in the latent (flips and 90
degree rotations of the 32 x 32 code), which is only right if the autoencoder is equivariant
to them; info.json reports, on 256 held-out images, the PSNR of decoding a rotated and a
flipped code against the rotated and flipped image, next to the plain reconstruction PSNR.

Usage: uv run scripts/33_encode_latents.py [ae dir, default results/autoencoder] [out dir,
       default <ae dir>/latents]
"""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.latent import decode, encode, load_autoencoder, uint8_to_float  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (10 * torch.log10(4 / ((a - b) ** 2).mean((1, 2, 3)))).mean().item()


def main(adir: Path, out: Path) -> None:
    train = importlib.import_module("09_train_flow")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)
    ae, std, run = load_autoencoder(adir, device)
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    out.mkdir(parents=True, exist_ok=True)
    info = {"ae": str(adir), "latent_std": std, "z": run["z"], "factor": run["factor"], "size": run["size"],
            "latent_size": run["size"] // run["factor"], "doses": train.DOSES, "dtype": "float16"}
    for split in ["train", "test"]:
        t0 = time.time()
        x8, tracks = train.load_uint8(split)
        n, nf = x8.shape[:2]
        z = torch.empty(n, nf, run["z"], info["latent_size"], info["latent_size"], dtype=torch.float16)
        for b in range(0, n, 64):
            zb = encode(ae, uint8_to_float(x8[b:b + 64].flatten(0, 1)), std, batch=128)
            z[b:b + 64] = zb.view(-1, nf, *zb.shape[1:]).half()
            if device == "mps":
                torch.mps.empty_cache()
            print(f"{split}: {min(b + 64, n)}/{n} cells", flush=True)
        fate = lab.loc[tracks, "apoptotic"].to_numpy().astype(np.int64)
        np.savez(out / f"{split}.npz", z=z.numpy(), fate=fate, tracks=tracks)
        info[split] = {"cells": n, "frames": nf, "minutes": round((time.time() - t0) / 60, 1),
                       "z_std": z.float().std().item()}
        if split == "test":  # equivariance of the autoencoder to the latent-space augmentation
            g = torch.Generator().manual_seed(0)
            pick = torch.randperm(n, generator=g)[:256]
            x = uint8_to_float(x8[pick, 8])
            zz = z[pick, 8].float()
            info["equivariance_psnr"] = {
                "plain": psnr(decode(ae, zz, std), x),
                "rot90": psnr(decode(ae, torch.rot90(zz, 1, (-2, -1)), std), torch.rot90(x, 1, (-2, -1))),
                "flip": psnr(decode(ae, zz.flip(-1), std), x.flip(-1))}
        del x8
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


if __name__ == "__main__":
    adir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/autoencoder"
    main(adir, Path(sys.argv[2]) if len(sys.argv) > 2 else adir / "latents")
