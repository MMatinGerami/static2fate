import importlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from static2fate.autoencoder import AutoEncoder, Discriminator, hinge_d_loss, kl_loss, latent_std  # noqa: E402


def test_autoencoder_shapes_at_two_sizes():
    torch.manual_seed(0)
    ae = AutoEncoder(ch=16, z=4)
    assert ae.factor == 4
    for size in (64, 128):
        x = torch.rand(2, 3, size, size) * 2 - 1
        mean, logvar = ae.encode(x)
        assert mean.shape == logvar.shape == (2, 4, size // 4, size // 4)
        recon, _, _ = ae(x)
        assert recon.shape == x.shape and torch.isfinite(recon).all()
        assert ae.decode(mean).shape == x.shape


def test_losses_are_finite_and_train_both_halves():
    torch.manual_seed(0)
    ae = AutoEncoder(ch=16, z=4)
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    recon, mean, logvar = ae(x)
    loss = (recon - x).abs().mean() + 1e-6 * kl_loss(mean, logvar)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ae.parameters())
    assert kl_loss(torch.zeros(2, 4, 8, 8), torch.zeros(2, 4, 8, 8)) == 0  # unit Gaussian posterior


def test_discriminator_patch_logits_and_hinge_loss():
    torch.manual_seed(0)
    d = Discriminator(ch=16)
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    logits = d(x)
    assert logits.shape[:2] == (2, 1) and logits.shape[2] > 1  # one logit per patch
    real, fake = torch.full((2, 1, 14, 14), 2.0), torch.full((2, 1, 14, 14), -2.0)
    assert hinge_d_loss(real, fake) == 0  # already separated by the margin
    assert hinge_d_loss(fake, real) == 3


def test_latent_std_matches_direct_computation():
    torch.manual_seed(0)
    ae = AutoEncoder(ch=16, z=4).eval()
    x = torch.rand(6, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        mean, _ = ae.encode(x)
    assert np.isclose(latent_std(ae, x, batch=4), mean.pow(2).mean().sqrt().item(), rtol=1e-4)


def test_reconstruct_keeps_layout_and_range():
    rec = importlib.import_module("32_reconstruct")
    torch.manual_seed(0)
    ae = AutoEncoder(ch=16, z=4).eval()
    x = torch.rand(5, 3, 64, 64) * 2 - 1
    out = rec.reconstruct(ae, x, "cpu", batch=2)
    assert out.shape == x.shape and out.abs().max() <= 1
    assert rec.FUTURE == [6, 8, 10, 12]
