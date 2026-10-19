"""Final per-track fate table: one row per complete track with the cell-level AnnexinV readout.

Label: apoptotic = AnnexinV cytoplasm 90th percentile above the control wells' 90th percentile
of that same quantity. By construction about 10% of control cells are called positive; some of
those are real spontaneous apoptosis, some are not. Writes data/processed/fate_labels.parquet.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"


def main() -> None:
    tracks = pd.concat([pd.read_parquet(p) for p in glob.glob(str(P / "tracks/*.parquet"))], ignore_index=True)
    f13 = tracks[tracks["frame"] == 13][["well_fov", "label", "particle"]]
    cells = pd.read_parquet(P / "cell_annexin.parquet")
    j = f13.merge(cells, on=["well_fov", "label"], how="inner")
    j["track"] = j["well_fov"] + "_" + j["particle"].astype(str)
    j = j.drop(columns=["well_fov", "label", "particle"])

    fate = pd.read_parquet(P / "fate_table.parquet")
    fate = fate[fate["complete"]].merge(j, on="track", how="inner")
    fate["area_ratio"] = fate["terminal_area"] / fate["initial_area"]
    fate["hoechst_ratio"] = fate["terminal_hoechst_mean"] / fate["initial_hoechst_mean"]

    ctrl = fate[fate["dose_nM"] == 0]
    thr = float(np.percentile(ctrl["annexin_cyto_p90"].dropna(), 90))
    fate = fate.dropna(subset=["annexin_cyto_p90"])
    fate["annexin_threshold"] = thr
    fate["apoptotic"] = fate["annexin_cyto_p90"] > thr
    fate.to_parquet(P / "fate_labels.parquet", index=False)

    print(f"{len(fate)} complete tracks with a cell-level AnnexinV value; threshold {thr:.0f}")
    print(fate.groupby("dose_nM").agg(n=("apoptotic", "size"), apoptotic=("apoptotic", "mean")).round(2).to_string())
    print("\nmedian nucleus area ratio and Hoechst ratio by dose and label:")
    print(fate.groupby(["dose_nM", "apoptotic"])[["area_ratio", "hoechst_ratio"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
