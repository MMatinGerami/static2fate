"""Joint generation of a cell's future as a short clip, so the frames belong to one trajectory.

The context model (flow.UNet with context=True) draws each future frame on its own: asked for
3 h and 6 h it starts from two noise draws, so the two images need not be the same future. Here
the velocity field acts on a whole clip x of shape (B, T, 3, H, W), the frames 6 to 12 (3 to
6 h) at once, from one noise draw and one flow time s. The spatial layers are the ResBlocks of
flow.py applied to every frame (frames folded into the batch); after each resolution a
temporal attention layer mixes the T positions at every pixel (Ho et al., 2022,
arXiv:2204.03458), so what happens in one frame is seen when the next is drawn. The attention
output projections start at zero, so at initialisation the model is the per-frame model.
Conditions as in the context model: fate, the context image (concatenated to every frame) and
its frame t_c; each frame also gets its own time-after-drug embedding.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .flow import UNet, sinusoidal


class TemporalAttention(nn.Module):
    """Self-attention over the frame axis, separately at every pixel. Residual, zero-initialised."""

    def __init__(self, ch: int, n_frames: int = 13, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.pos = nn.Embedding(n_frames, ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, h: torch.Tensor, n_frames: int, frame: torch.Tensor) -> torch.Tensor:
        """h (B*T, C, H, W), frame (B*T,) the frame index of each image."""
        bt, c, hh, ww = h.shape
        b, t = bt // n_frames, n_frames
        z = h.view(b, t, c, hh * ww).permute(0, 3, 1, 2).reshape(b * hh * ww, t, c)
        pos = self.pos(frame).view(b, 1, t, c).expand(b, hh * ww, t, c).reshape(b * hh * ww, t, c)
        q = self.norm(z) + pos
        a, _ = self.attn(q, q, q, need_weights=False)
        z = z + a
        return z.view(b, hh * ww, t, c).permute(0, 2, 3, 1).reshape(bt, c, hh, ww)


class VideoUNet(UNet):
    """The context UNet with a temporal attention layer after every stage."""

    def __init__(self, ch: int = 64, cdim: int = 256, n_frames: int = 13, n_fates: int = 2, heads: int = 4,
                 in_ch: int = 3):
        super().__init__(ch, cdim, n_frames, n_fates, context=True, in_ch=in_ch)
        widths = [ch, 2 * ch, 4 * ch, 4 * ch, 4 * ch, 2 * ch, ch]  # d1, d2, d3, mid, u3, u2, u1
        self.temporal = nn.ModuleList([TemporalAttention(w, n_frames, heads) for w in widths])

    def forward(self, x, s, frame, fate, ctx, ctx_frame):
        """x (B, T, C, H, W) clip at flow time s (B,); frame (B, T); fate, ctx_frame (B,); ctx (B, C, H, W)."""
        b, t = x.shape[:2]
        c = self.s_mlp(sinusoidal(s * 1000, self.ch)) + self.fate(fate) + self.ctx_frame(ctx_frame)
        fr = frame.reshape(-1)
        c = c[:, None].expand(b, t, -1).reshape(b * t, -1) + self.frame(fr)
        x = torch.cat([x, ctx[:, None].expand(b, t, *ctx.shape[1:])], 2).flatten(0, 1)
        tm = self.temporal
        h1 = self.inp(x)
        for blk in self.d1:
            h1 = blk(h1, c)
        h1 = tm[0](h1, t, fr)
        h2 = F.avg_pool2d(h1, 2)
        for blk in self.d2:
            h2 = blk(h2, c)
        h2 = tm[1](h2, t, fr)
        h3 = F.avg_pool2d(h2, 2)
        for blk in self.d3:
            h3 = blk(h3, c)
        h3 = tm[2](h3, t, fr)
        m = tm[3](self.mid(F.avg_pool2d(h3, 2), c), t, fr)
        u = F.interpolate(m, scale_factor=2)
        u = tm[4](self.u3[1](self.u3[0](torch.cat([u, h3], 1), c), c), t, fr)
        u = F.interpolate(u, scale_factor=2)
        u = tm[5](self.u2[1](self.u2[0](torch.cat([u, h2], 1), c), c), t, fr)
        u = F.interpolate(u, scale_factor=2)
        u = tm[6](self.u1[1](self.u1[0](torch.cat([u, h1], 1), c), c), t, fr)
        return self.out(u).view(b, t, self.in_ch, *x.shape[-2:])


def video_flow_loss(model, clip, frame, fate, ctx, ctx_frame, p_drop: float = 0.1):
    """Straight-path flow matching on whole clips; one s and one noise draw per clip."""
    x0 = torch.randn_like(clip)
    s = torch.rand(clip.shape[0], device=clip.device)
    s5 = s[:, None, None, None, None]
    xs = (1 - s5) * x0 + s5 * clip
    drop = torch.rand(clip.shape[0], device=clip.device) < p_drop
    fate = torch.where(drop, torch.full_like(fate, 2), fate)
    return F.mse_loss(model(xs, s, frame, fate, ctx, ctx_frame), clip - x0)


@torch.no_grad()
def sample_video(model, frame, fate, ctx, ctx_frame, steps: int = 50, guidance: float = 2.0,
                 generator=None, x0=None, clamp: bool = True, project=None):
    """Euler integration from noise to a clip (B, T, C, H, W); classifier-free guidance on fate.

    `project(x, s)`, if given, is applied to the state after every step (s is the time reached).
    Clamped to [-1, 1] for images; clamp=False for latents.
    """
    b, t = frame.shape
    device = frame.device
    x = x0 if x0 is not None else torch.randn(b, t, *ctx.shape[1:], device=device, generator=generator)
    both_fates = torch.cat([fate, torch.full_like(fate, 2)])
    frame2, ctx2, cf2 = torch.cat([frame, frame]), torch.cat([ctx, ctx]), torch.cat([ctx_frame, ctx_frame])
    for i in range(steps):
        s = torch.full((2 * b,), i / steps, device=device)
        v = model(torch.cat([x, x]), s, frame2, both_fates, ctx2, cf2)
        v_c, v_u = v[:b], v[b:]
        x = x + (v_u + guidance * (v_c - v_u)) / steps
        if project is not None:
            x = project(x, (i + 1) / steps)
    return x.clamp(-1, 1) if clamp else x
