"""One row per track: dose, frames covered, terminal AnnexinV, and the live trajectory summary.

Reads data/processed/tracks/*.parquet, writes data/processed/fate_table.parquet and prints
per-dose counts. 'complete' tracks start at frame 1 and reach the terminal frame 14.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "data/processed/tracks"
PLATEMAP = ROOT / "data/raw/zenodo/17419422/WayScience-live_cell_timelapse_apoptosis-1ad7ea4/data/platemap_6hr_4ch.csv"


def main() -> None:
    plate = pd.read_csv(PLATEMAP).drop_duplicates("well").set_index("well")["dose"]
    frames = [pd.read_parquet(p) for p in sorted(TRACKS.glob("*.parquet"))]
    if not frames:
        sys.exit("no track files")
    df = pd.concat(frames, ignore_index=True)
    df["well"] = df["well_fov"].str.split("_").str[0]
    df["dose_nM"] = df["well"].map(plate)
    df["track"] = df["well_fov"] + "_" + df["particle"].astype(str)

    g = df.groupby("track")
    out = g.agg(
        well_fov=("well_fov", "first"),
        dose_nM=("dose_nM", "first"),
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        n_frames=("frame", "size"),
    )
    has_terminal = df.groupby("well_fov")["frame"].max().eq(14)
    out["has_terminal"] = out["well_fov"].map(has_terminal)
    out["complete"] = (out["first_frame"] == 1) & (out["last_frame"] == 14)

    term = df[df["frame"] == 14].set_index("track")
    out["annexin_ring_sum"] = term["annexin_ring_sum"]
    out["annexin_ring_mean"] = term["annexin_ring_mean"]
    out["terminal_area"] = term["area"]
    out["terminal_hoechst_mean"] = term["hoechst_nuc_mean"]

    first = df[df["frame"] == 1].set_index("track")
    out["initial_area"] = first["area"]
    out["initial_hoechst_mean"] = first["hoechst_nuc_mean"]

    out.to_parquet(ROOT / "data/processed/fate_table.parquet")

    print(f"{len(out)} tracks, {int(out.complete.sum())} complete (frame 1 to 14); "
          f"fields with a terminal frame: {int(has_terminal.sum())}/{len(has_terminal)}")
    print(out.groupby("dose_nM").agg(tracks=("complete", "size"), with_terminal=("has_terminal", "sum"),
                                     complete=("complete", "sum")).to_string())
    comp = out[out.complete]
    print("\nterminal AnnexinV ring mean, complete tracks, per dose (p10 / p50 / p90):")
    q = comp.groupby("dose_nM")["annexin_ring_mean"].quantile([0.1, 0.5, 0.9]).unstack()
    print(q.round(0).to_string())
    print("\nterminal nucleus area / initial nucleus area, complete tracks, per dose (p10 / p50 / p90):")
    ratio = (comp["terminal_area"] / comp["initial_area"]).groupby(comp["dose_nM"]).quantile([0.1, 0.5, 0.9]).unstack()
    print(ratio.round(2).to_string())


if __name__ == "__main__":
    main()
