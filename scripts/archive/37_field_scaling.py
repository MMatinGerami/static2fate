"""Does the per-field channel scaling of the crops carry the field's death rate?

05_crops.py scales each channel of each field with one pair of percentiles (0.5 and 99.8)
over all 13 frames. If the 99.8th percentile of a ChromaLIVE channel is set by the brightest
dying cells, a field with many dying cells is compressed and a survivor there looks dimmer
than a survivor in a quiet field: a field-level cue to fate, and a distortion of the colour
of apoptosis across doses. Here the constants are recomputed for every field (percentiles on
every fourth pixel) and related to the field's fraction of labelled cells that die.

Usage: uv run scripts/archive/37_field_scaling.py
Writes results/tables/field_scaling.csv (one row per field: lo and hi per channel, dose, cells,
death fraction), results/tables/field_scaling_summary.csv (per channel: spread of hi across
fields, Spearman with the death fraction over all fields and within dose) and
results/figures/supporting/fig17_field_scaling.png.
"""

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))  # scripts/, for the current pipeline

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"
IMAGES = ROOT / "data/raw/images"
PCT = [0.5, 99.8]
STRIDE = 4


def field_constants(well_fov: str, channels: dict[str, int]) -> dict[str, float]:
    out = {}
    for name, c in channels.items():
        frames = []
        for t in range(1, 14):
            try:
                frames.append(tifffile.imread(IMAGES / f"{well_fov}_T{t:04d}_Z0001_C{c:02d}.tif")[::STRIDE, ::STRIDE])
            except Exception:
                return {}
        lo, hi = np.percentile(np.stack(frames).astype(np.float32), PCT)
        out[f"{name}_lo"], out[f"{name}_hi"] = float(lo), float(hi)
    return out


def main() -> None:
    crops = importlib.import_module("05_crops")
    lab = pd.read_parquet(P / "fate_labels.parquet")
    per_field = lab.groupby("well_fov").agg(dose_nM=("dose_nM", "first"), cells=("track", "size"),
                                            dies=("apoptotic", "mean"), condensed=("condensed", "mean"))
    rows = []
    for i, (wf, r) in enumerate(per_field.iterrows()):
        k = field_constants(wf, crops.CHANNELS)
        if k:
            rows.append({"well_fov": wf, **r.to_dict(), **k})
        print(f"{i + 1}/{len(per_field)} {wf} {'ok' if k else 'unreadable'}", flush=True)
    tab = pd.DataFrame(rows)
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(ROOT / "results/tables/field_scaling.csv", index=False)

    summary = []
    for ch in crops.CHANNELS:
        hi = tab[f"{ch}_hi"]
        within = []
        for _, d in tab.groupby("dose_nM"):
            if d["dies"].nunique() > 1 and len(d) >= 4:
                within.append(spearmanr(d[f"{ch}_hi"], d["dies"]).correlation)
        summary.append({"channel": ch, "fields": len(tab), "hi_min": hi.min(), "hi_median": hi.median(),
                        "hi_max": hi.max(), "hi_max_over_min": hi.max() / hi.min(), "hi_cv": hi.std() / hi.mean(),
                        "spearman_hi_vs_dies_all": spearmanr(hi, tab["dies"]).correlation,
                        "spearman_hi_vs_dose": spearmanr(hi, tab["dose_nM"]).correlation,
                        "spearman_hi_vs_dies_within_dose_mean": float(np.mean(within)),
                        "within_dose_n": len(within),
                        "lo_median": tab[f"{ch}_lo"].median(), "lo_cv": tab[f"{ch}_lo"].std() / tab[f"{ch}_lo"].mean()})
    summary = pd.DataFrame(summary)
    summary.to_csv(ROOT / "results/tables/field_scaling_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    doses = sorted(tab["dose_nM"].unique())
    cmap = plt.get_cmap("viridis")
    for ax, ch in zip(axes, crops.CHANNELS):
        for j, d in enumerate(doses):
            m = tab["dose_nM"] == d
            ax.scatter(tab.loc[m, "dies"], tab.loc[m, f"{ch}_hi"], s=18, color=cmap(j / (len(doses) - 1)), label=f"{d:g}")
        ax.set_xlabel("fraction of labelled cells that die")
        ax.set_ylabel(f"{ch}: 99.8th percentile (white point)")
        ax.set_title(ch)
    axes[-1].legend(title="nM", fontsize=7, title_fontsize=7, ncol=2)
    fig.suptitle("Per-field scaling constants of the crops against the field's death rate", fontsize=10)
    fig.tight_layout()
    (ROOT / "results/figures/supporting").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results/figures/supporting/fig17_field_scaling.png", dpi=130)


if __name__ == "__main__":
    main()
