"""Moving between pixels and the autoencoder's latent, for the flow models that generate there.

A trained autoencoder directory holds model.pt and run.json (31_train_autoencoder.py), whose
"latent_std" scales the posterior mean to unit scale: the flow models see mean / latent_std
and their output is multiplied back before decoding. load_autoencoder reads the directory;
encode and decode work in batches on any device and return CPU tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .autoencoder import AutoEncoder


def load_autoencoder(mdir: Path, device: str) -> tuple[AutoEncoder, float, dict]:
    """The autoencoder of a directory in eval mode, its latent_std and its run.json."""
    run = json.loads((Path(mdir) / "run.json").read_text())
    ae = AutoEncoder(ch=run["ch"], z=run["z"]).to(device)
    ae.load_state_dict(torch.load(Path(mdir) / "model.pt", map_location=device, weights_only=True))
    ae.eval()
    return ae, float(run["latent_std"]), run


@torch.no_grad()
def encode(ae: AutoEncoder, x: torch.Tensor, std: float, batch: int = 64) -> torch.Tensor:
    """Images (n, 3, s, s) in [-1, 1] -> scaled posterior means (n, z, s / f, s / f), on the CPU."""
    device = next(ae.parameters()).device
    out = []
    for b in range(0, len(x), batch):
        mean, _ = ae.encode(x[b:b + batch].to(device))
        out.append((mean.float() / std).cpu())
    return torch.cat(out)


@torch.no_grad()
def decode(ae: AutoEncoder, z: torch.Tensor, std: float, batch: int = 64) -> torch.Tensor:
    """Scaled latents (n, z, h, w) -> images (n, 3, s, s) clamped to [-1, 1], on the CPU."""
    device = next(ae.parameters()).device
    out = []
    for b in range(0, len(z), batch):
        out.append(ae.decode(z[b:b + batch].to(device) * std).float().clamp(-1, 1).cpu())
    return torch.cat(out)


def uint8_to_float(x8: torch.Tensor) -> torch.Tensor:
    return x8.float().div_(127.5).sub_(1)


def float_to_uint8(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
