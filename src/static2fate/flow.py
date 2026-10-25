"""Small conditional flow-matching model for 64 px single-cell images.

The model learns a velocity field v(x_s, s | frame, fate) that moves Gaussian noise (s = 0) to a
cell image (s = 1) along straight paths (rectified flow / conditional flow matching). The
condition is the time after drug (frame 0..12) and the fate observed at 6 h (0 survives,
1 dies). Asking for both fates at the same frame gives two counterfactual images of the same
moment, which is the simplest version of the two-future question.

With context=True the model also sees one observed image of the same cell (concatenated as three
extra input channels) and the frame it was taken at, and predicts the cell at a later frame.
This lets the prediction depend on how far along that particular cell already is.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cdim: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(cdim, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.emb(c)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class UNet(nn.Module):
    """Three resolutions (64, 32, 16), two blocks each, conditioning added through FiLM."""

    def __init__(self, ch: int = 64, cdim: int = 256, n_frames: int = 13, n_fates: int = 2,
                 context: bool = False):
        super().__init__()
        self.context = context
        if context:
            self.ctx_frame = nn.Embedding(n_frames, cdim)
        self.s_mlp = nn.Sequential(nn.Linear(ch, cdim), nn.SiLU(), nn.Linear(cdim, cdim))
        self.frame = nn.Embedding(n_frames, cdim)
        self.fate = nn.Embedding(n_fates + 1, cdim)  # last index = no fate (classifier-free guidance)
        self.ch = ch
        self.inp = nn.Conv2d(6 if context else 3, ch, 3, padding=1)
        self.d1 = nn.ModuleList([ResBlock(ch, ch, cdim), ResBlock(ch, ch, cdim)])
        self.d2 = nn.ModuleList([ResBlock(ch, 2 * ch, cdim), ResBlock(2 * ch, 2 * ch, cdim)])
        self.d3 = nn.ModuleList([ResBlock(2 * ch, 4 * ch, cdim), ResBlock(4 * ch, 4 * ch, cdim)])
        self.mid = ResBlock(4 * ch, 4 * ch, cdim)
        self.u3 = nn.ModuleList([ResBlock(8 * ch, 4 * ch, cdim), ResBlock(4 * ch, 4 * ch, cdim)])
        self.u2 = nn.ModuleList([ResBlock(6 * ch, 2 * ch, cdim), ResBlock(2 * ch, 2 * ch, cdim)])
        self.u1 = nn.ModuleList([ResBlock(3 * ch, ch, cdim), ResBlock(ch, ch, cdim)])
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, x, s, frame, fate, ctx=None, ctx_frame=None):
        c = self.s_mlp(sinusoidal(s * 1000, self.ch)) + self.frame(frame) + self.fate(fate)
        if self.context:
            x = torch.cat([x, ctx], 1)
            c = c + self.ctx_frame(ctx_frame)
        h1 = self.inp(x)
        for b in self.d1:
            h1 = b(h1, c)
        h2 = F.avg_pool2d(h1, 2)
        for b in self.d2:
            h2 = b(h2, c)
        h3 = F.avg_pool2d(h2, 2)
        for b in self.d3:
            h3 = b(h3, c)
        m = self.mid(F.avg_pool2d(h3, 2), c)
        u = F.interpolate(m, scale_factor=2)
        u = self.u3[1](self.u3[0](torch.cat([u, h3], 1), c), c)
        u = F.interpolate(u, scale_factor=2)
        u = self.u2[1](self.u2[0](torch.cat([u, h2], 1), c), c)
        u = F.interpolate(u, scale_factor=2)
        u = self.u1[1](self.u1[0](torch.cat([u, h1], 1), c), c)
        return self.out(u)


def flow_matching_loss(model, x1, frame, fate, p_drop: float = 0.1, **cond):
    """Straight-path conditional flow matching; fate is dropped 10% of the time for guidance.

    Extra conditions (ctx, ctx_frame) are passed through to the model unchanged.
    """
    x0 = torch.randn_like(x1)
    s = torch.rand(x1.shape[0], device=x1.device)
    xs = (1 - s[:, None, None, None]) * x0 + s[:, None, None, None] * x1
    drop = torch.rand(x1.shape[0], device=x1.device) < p_drop
    fate = torch.where(drop, torch.full_like(fate, 2), fate)
    return F.mse_loss(model(xs, s, frame, fate, **cond), x1 - x0)


@torch.no_grad()
def invert(model, x1, frame, steps: int = 50):
    """Integrate the ODE backwards from a real image to its noise, without a fate (null class).

    Deterministic, like DDIM inversion: the noise found here regenerates (approximately) the
    same cell, and can then be pushed forward under either fate.
    """
    x = x1.clone()
    null = torch.full((len(x1),), 2, device=x1.device, dtype=torch.long)
    for i in reversed(range(steps)):
        s = torch.full((len(x1),), (i + 1) / steps, device=x1.device)
        x = x - model(x, s, frame, null) / steps
    return x


@torch.no_grad()
def sample(model, frame, fate, steps: int = 50, guidance: float = 2.0, generator=None, x0=None,
           size: int = 64, x_start=None, s0: float = 0.0, **cond):
    """Euler integration from noise; the same x0 can be reused to compare two fates.

    With x_start and s0 > 0 the integration starts part way along the path, from
    (1 - s0) * noise + s0 * x_start (as in SDEdit), so the result keeps the large-scale layout
    of x_start, more so for larger s0, while the model fills in what the condition asks for.
    """
    device = frame.device
    x = x0 if x0 is not None else torch.randn(len(frame), 3, size, size, device=device, generator=generator)
    first = 0
    if x_start is not None:
        first = round(s0 * steps)
        x = (1 - first / steps) * x + (first / steps) * x_start
    n = len(frame)
    both_fates = torch.cat([fate, torch.full_like(fate, 2)])
    frame2 = torch.cat([frame, frame])
    cond2 = {k: torch.cat([v, v]) for k, v in cond.items()}
    for i in range(first, steps):
        s = torch.full((2 * n,), i / steps, device=device)
        v = model(torch.cat([x, x]), s, frame2, both_fates, **cond2)  # conditional and null in one pass
        v_c, v_u = v[:n], v[n:]
        x = x + (v_u + guidance * (v_c - v_u)) / steps
    return x.clamp(-1, 1)
