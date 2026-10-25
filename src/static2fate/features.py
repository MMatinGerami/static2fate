"""Richer per-image features than static2fate.readouts, computed the same way for real and
generated crops.

crop_features: the nucleus (Otsu object nearest the crop centre, as in readouts) described by
area, eccentricity and solidity, plus the mean of each channel inside it and in a ring around
it. This is the crop-level version of the handcrafted set used by the fate baseline, which
came from whole-field segmentation and so cannot be computed on a generated crop.

dinov2_features: frozen DINOv2-base CLS embedding of the crop resized to 224 px, as in the
baseline. Both accept (n, 3, s, s) in [-1, 1] at any size s; the ring width scales with s.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import expand_labels

FEATURES_HAND = ["nuc_area", "nuc_ecc", "nuc_solidity", "hoechst_nuc", "hoechst_ring",
                 "green_nuc", "green_ring", "red_nuc", "red_ring"]
RING_PX_128 = 8  # as fields.RING_PX at the native 128 px
DINO = "facebook/dinov2-base"
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_dino = {}


def _np(x) -> np.ndarray:
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, float)


def crop_features(x) -> np.ndarray:
    """(n, 3, s, s) in [-1, 1], channels Hoechst, 488, 561 -> (n, 9); NaN row if no nucleus."""
    a = _np(x)
    s = a.shape[-1]
    ring_px = max(1, round(RING_PX_128 * s / 128))
    out = np.full((len(a), len(FEATURES_HAND)), np.nan)
    for i, im in enumerate(a):
        h = im[0]
        lab = label(h > threshold_otsu(h))
        props = regionprops(lab)
        if not props:
            continue
        p = min(props, key=lambda q: (q.centroid[0] - s / 2) ** 2 + (q.centroid[1] - s / 2) ** 2)
        nuc = lab == p.label
        ring = (expand_labels(nuc.astype(int), ring_px) > 0) & ~nuc
        with np.errstate(invalid="ignore"):
            out[i] = [p.area, p.eccentricity, p.solidity,
                      *(v for c in range(3) for v in (im[c][nuc].mean(), im[c][ring].mean()))]
    return out


def dinov2_features(x, device: str | None = None, batch: int = 64) -> np.ndarray:
    """(n, 3, s, s) in [-1, 1] -> (n, 768) CLS embeddings of frozen DINOv2-base."""
    from transformers import AutoModel

    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device not in _dino:
        _dino[device] = AutoModel.from_pretrained(DINO).eval().to(device)
    model = _dino[device]
    xt = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), batch):
            b = (xt[i:i + batch].float() + 1) / 2
            b = torch.nn.functional.interpolate(b, size=224, mode="bilinear", align_corners=False)
            b = ((b - _MEAN) / _STD).to(device)
            out.append(model(pixel_values=b).pooler_output.float().cpu().numpy())
    return np.concatenate(out)
