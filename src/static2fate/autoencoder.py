"""Autoencoder for 128 px cell crops: the latent in which the next flow models will generate.

Every pixel-space model so far hits the same floor: a frozen encoder tells generated from
real without error, and the floor does not move with training time, width or more cells
(docs/results.md). The pixel loss asks the velocity field to regress texture, and the mean
of many plausible textures is a blur. Generating in the latent of an autoencoder trained to
reconstruct these cells moves texture out of the generator: the decoder owns it, the flow
model only has to place the cell's state in a 4 x 32 x 32 code (Rombach et al., 2022,
arXiv:2112.10752).

The autoencoder is a KL-regularised convolutional encoder and decoder (downsampling factor
4, four latent channels, GroupNorm and SiLU, no attention). Trained with an L1 pixel loss, the
VGG16 perceptual distance of static2fate.perceptual, a very small KL term that keeps the
latent close to unit scale, and optionally a PatchGAN discriminator with a hinge loss. The
latent used downstream is the posterior mean divided by its standard deviation over training
images (latent_std), so the flow model sees codes of unit scale. VGG16 is the training signal
and DINOv2 the judge, as before, so the probe stays independent of what was optimised.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, ch: int, z: int, mult: tuple[int, ...], blocks: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, ch, 3, padding=1)]
        cin = ch
        for i, m in enumerate(mult):
            for _ in range(blocks):
                layers.append(ResBlock(cin, ch * m))
                cin = ch * m
            if i < len(mult) - 1:
                layers.append(nn.Conv2d(cin, cin, 3, stride=2, padding=1))
        layers += [nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, 2 * z, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, ch: int, z: int, mult: tuple[int, ...], blocks: int):
        super().__init__()
        cin = ch * mult[-1]
        layers: list[nn.Module] = [nn.Conv2d(z, cin, 3, padding=1)]
        for i, m in enumerate(reversed(mult)):
            for _ in range(blocks):
                layers.append(ResBlock(cin, ch * m))
                cin = ch * m
            if i < len(mult) - 1:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cin, 3, padding=1)]
        layers += [nn.GroupNorm(8, cin), nn.SiLU()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(cin, 3, 3, padding=1)  # the last layer, for the adaptive GAN weight

    def forward(self, z):
        return self.out(self.net(z))


class AutoEncoder(nn.Module):
    """x (n, 3, s, s) in [-1, 1] <-> latent (n, z, s / 4, s / 4) for mult of length 3."""

    def __init__(self, ch: int = 64, z: int = 4, mult: tuple[int, ...] = (1, 2, 4), blocks: int = 2):
        super().__init__()
        self.z = z
        self.factor = 2 ** (len(mult) - 1)
        self.encoder = Encoder(ch, z, mult, blocks)
        self.decoder = Decoder(ch, z, mult, blocks)

    def encode(self, x):
        """Posterior mean and log variance, each (n, z, s / f, s / f)."""
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, logvar.clamp(-30, 20)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample: bool = True):
        mean, logvar = self.encode(x)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp() if sample else mean
        return self.decode(z), mean, logvar


def kl_loss(mean, logvar):
    """KL to a unit Gaussian, summed over the latent and averaged over the batch."""
    return 0.5 * (mean ** 2 + logvar.exp() - 1 - logvar).sum((1, 2, 3)).mean()


class Discriminator(nn.Module):
    """PatchGAN: one logit per 16 x 16 patch (receptive field) of a 128 px image."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, 2 * ch, 4, stride=2, padding=1), nn.GroupNorm(8, 2 * ch), nn.LeakyReLU(0.2),
            nn.Conv2d(2 * ch, 4 * ch, 4, stride=2, padding=1), nn.GroupNorm(8, 4 * ch), nn.LeakyReLU(0.2),
            nn.Conv2d(4 * ch, 4 * ch, 4, stride=1, padding=1), nn.GroupNorm(8, 4 * ch), nn.LeakyReLU(0.2),
            nn.Conv2d(4 * ch, 1, 4, stride=1, padding=1))

    def forward(self, x):
        return self.net(x)


def hinge_d_loss(real_logits, fake_logits):
    return 0.5 * (F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean())


@torch.no_grad()
def latent_std(ae: AutoEncoder, x: torch.Tensor, batch: int = 64) -> float:
    """Standard deviation of the posterior means over images x (n, 3, s, s); divides the latent."""
    device = next(ae.parameters()).device
    sq, n = 0.0, 0
    for b in range(0, len(x), batch):
        mean, _ = ae.encode(x[b:b + batch].to(device))
        sq += (mean.float() ** 2).sum().item()
        n += mean.numel()
    return (sq / n) ** 0.5
