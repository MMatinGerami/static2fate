"""Per-cell AnnexinV at the terminal frame, using cell masks drawn on live frame 13.

For every field with a terminal partner: re-segment nuclei on frame 13 exactly as in
01_segment_track.py (so labels match the tracks parquet), draw cell masks from the ChromaLIVE
channels, apply them to the registered AnnexinV image, and join the result onto the frame-13
rows of the tracks parquet by (well_fov, label). Writes data/processed/cell_annexin.parquet and
a QC panel for the first field.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from static2fate.cells import annexin_per_cell, cell_masks  # noqa: E402
from static2fate.fields import LIVE_CHANNELS, Field, segment_nuclei_classical  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/raw/images"
TRACKS = ROOT / "data/processed/tracks"


def qc_panel(field: Field, nuclei: np.ndarray, cells: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    sl = (slice(600, 1100), slice(600, 1100))

    def norm(a):
        a = a.astype(float)
        lo, hi = np.percentile(a, [1, 99.5])
        return np.clip((a - lo) / (hi - lo), 0, 1)

    total = sum(field.live[c][-1].astype(float) for c in ("cl488a", "cl488b", "cl561"))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(norm(total[sl]), cmap="gray")
    ax[0].contour(find_boundaries(cells[sl]), levels=[0.5], colors="cyan", linewidths=0.5)
    ax[0].contour(find_boundaries(nuclei[sl]), levels=[0.5], colors="r", linewidths=0.4)
    ax[0].set_title("frame 13: ChromaLIVE sum, cell (cyan) and nucleus (red) masks")
    ax[1].imshow(norm(field.terminal["annexin"][sl]), cmap="gray")
    ax[1].contour(find_boundaries(cells[sl]), levels=[0.5], colors="cyan", linewidths=0.5)
    ax[1].set_title("terminal AnnexinV (registered) with frame-13 cell masks")
    rgb = np.dstack([norm(field.live["hoechst"][-1][sl]), norm(field.terminal["hoechst"][sl]), np.zeros((500, 500))])
    ax[2].imshow(rgb)
    ax[2].set_title("Hoechst frame 13 (red) vs terminal (green) after registration")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=80)


def main() -> None:
    rows = []
    first = True
    for p in sorted(TRACKS.glob("*.parquet")):
        tracks = pd.read_parquet(p)
        if tracks["frame"].max() < 14:
            continue
        well_fov = p.stem
        field = Field.load(IMAGES, well_fov)
        nuclei = segment_nuclei_classical(field.live["hoechst"][-1][None])[0]
        chroma = [field.live[c][-1] for c in ("cl488a", "cl488b", "cl561")]
        cells = cell_masks(chroma, nuclei)
        df = annexin_per_cell(cells, nuclei, field.terminal["annexin"], field.terminal["hoechst"])
        df.insert(0, "well_fov", well_fov)
        rows.append(df)
        print(f"{well_fov}: {len(df)} cells, median cell area {df.cell_area.median():.0f} px, "
              f"median cyto AnnexinV {df.annexin_cyto_mean.median():.0f}")
        if first:
            qc_panel(field, nuclei, cells, ROOT / "results/qc" / f"{well_fov}_cell_masks.png")
            first = False
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(ROOT / "data/processed/cell_annexin.parquet", index=False)
    print(f"wrote {len(out)} cells")


if __name__ == "__main__":
    main()
