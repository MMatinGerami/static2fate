"""Cell masks from the ChromaLIVE channels, seeded by the nuclei, and per-cell AnnexinV.

The terminal image has no ChromaLIVE channels (Hoechst and AnnexinV only), so cell masks are
drawn on live frame 13 and applied to the registered terminal AnnexinV image, which is what the
Way lab did too. Cells barely move between frame 13 and fixation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.segmentation import watershed


def cell_masks(chroma: list[np.ndarray], nuclei: np.ndarray, sigma: float = 2, max_radius: int = 60) -> np.ndarray:
    """Seeded watershed on the summed ChromaLIVE signal; every nucleus gets one cell.

    Foreground is an Otsu threshold on the log of the smoothed sum (the log tames the bright
    perinuclear vesicles). Cells are also limited to max_radius px from their nucleus so a
    nucleus at the edge of a dim region cannot claim a whole patch of background.
    """
    total = np.sum([c.astype(np.float32) for c in chroma], axis=0)
    sm = filters.gaussian(total, sigma, preserve_range=True)
    lg = np.log1p(sm)
    fg = lg > filters.threshold_otsu(lg)
    fg = morphology.closing(fg, morphology.disk(3))
    fg = ndi.binary_fill_holes(fg) | (nuclei > 0)
    dist_to_nucleus = ndi.distance_transform_edt(nuclei == 0)
    fg &= dist_to_nucleus <= max_radius
    edges = filters.sobel(lg)
    return watershed(edges, markers=nuclei, mask=fg).astype(np.int32)


def annexin_per_cell(cells: np.ndarray, nuclei: np.ndarray, annexin: np.ndarray, hoechst_terminal: np.ndarray) -> pd.DataFrame:
    """AnnexinV in the whole cell and in the cytoplasm (cell minus nucleus), per nucleus label."""
    idx = np.unique(cells)
    idx = idx[idx > 0]
    cyto = cells.copy()
    cyto[nuclei > 0] = 0
    img = annexin.astype(np.float64)
    with np.errstate(invalid="ignore"):
        df = pd.DataFrame(
            {
                "label": idx,
                "cell_area": ndi.sum(np.ones_like(cells), cells, idx),
                "annexin_cell_mean": ndi.mean(img, cells, idx),
                "annexin_cell_sum": ndi.sum(img, cells, idx),
                "annexin_cyto_mean": ndi.mean(img, cyto, idx),
                "annexin_cyto_sum": ndi.sum(img, cyto, idx),
                "annexin_cyto_p90": [np.percentile(img[cyto == i], 90) if (cyto == i).any() else np.nan for i in idx],
                "hoechst_terminal_in_cell": ndi.mean(hoechst_terminal.astype(np.float64), cells, idx),
            }
        )
    return df
