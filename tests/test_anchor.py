"""Anchored sampling (40_anchor_neighbourhood.py): outside the disc the state follows the context."""

import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from static2fate.flow import UNet, sample  # noqa: E402

anchor = importlib.import_module("40_anchor_neighbourhood")


def test_disc_weights_free_centre_anchored_rim():
    w = anchor.disc_weights(32, 10.0)
    assert w.shape == (32, 32) and w[15, 15] == 1 and w[16, 16] == 1
    assert w[0, 0] == 0 and w[15, 0] == 0 and w[15, 31] == 0
    assert 0.25 < w.mean() < 0.35  # pi * 10^2 / 32^2 = 0.31


def test_anchored_sampler_holds_the_rim_and_frees_the_centre():
    torch.manual_seed(0)
    m = UNet(ch=16, cdim=32, context=True, in_ch=4).eval()
    noise = torch.randn(2, 4, 32, 32)
    ctx = torch.randn(2, 4, 32, 32)
    free = anchor.disc_weights(32, 10.0)
    frame, fate = torch.tensor([6, 12]), torch.tensor([0, 1])
    cf = torch.zeros(2, dtype=torch.long)
    out = sample(m, frame, fate, steps=3, x0=noise.clone(), clamp=False, size=32,
                 project=anchor.anchored(noise, ctx, free), ctx=ctx, ctx_frame=cf)
    plain = sample(m, frame, fate, steps=3, x0=noise.clone(), clamp=False, size=32, ctx=ctx, ctx_frame=cf)
    rim = free == 0
    assert torch.allclose(out[:, :, rim], ctx[:, :, rim])  # the rim is the context's code at s = 1
    assert not torch.allclose(out[:, :, 15, 15], ctx[:, :, 15, 15])  # the centre is generated
    assert not torch.allclose(out[:, :, rim], plain[:, :, rim])
