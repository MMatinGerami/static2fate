"""Per-image cell readouts used to compare real and generated 64 px crops.

The nucleus is the Otsu-thresholded Hoechst object closest to the crop centre. A per-image
threshold is used because generated images are slightly dimmer than real ones overall; a fixed
threshold made generated nuclei look far smaller than they are (notebook, 24 Oct).
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

FEATURES = ["nuc_area", "nuc_hoechst", "cyto_green", "cyto_red"]


def cell_readouts(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """x (n, 3, 64, 64) in [-1, 1], channels Hoechst, ChromaLIVE 488, ChromaLIVE 561.

    Returns (n, 4): nucleus area (px), mean Hoechst in the nucleus, and mean green and red in a
    12 px ring around the nucleus (the cytoplasm signal that shifts to green in apoptosis).
    """
    a = x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, float)
    out = np.full((len(a), 4), np.nan)
    yy, xx = np.mgrid[:64, :64]
    for i, im in enumerate(a):
        h = im[0]
        lab = label(h > threshold_otsu(h))
        props = regionprops(lab)
        if not props:
            continue
        p = min(props, key=lambda q: (q.centroid[0] - 32) ** 2 + (q.centroid[1] - 32) ** 2)
        nuc = lab == p.label
        cy, cx = p.centroid
        r = np.hypot(yy - cy, xx - cx)
        ring = (~nuc) & (r < np.sqrt(p.area / np.pi) + 12)
        out[i] = [p.area, h[nuc].mean(), im[1][ring].mean(), im[2][ring].mean()]
    return out
