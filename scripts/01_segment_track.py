"""Segment, register and track every downloaded field; write one parquet per field.

Usage: uv run scripts/01_segment_track.py [--device mps|cpu] [--only C-02_F0001 ...]
Output: data/processed/tracks/<well_fov>.parquet (one row per nucleus per frame, with 'particle'
track ids) and data/processed/tracks/<well_fov>.json (shift, nuclei per frame, track count).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from cellpose import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from static2fate.fields import process_field  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data/raw/images"
OUT = ROOT / "data/processed/tracks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", choices=["classical", "cellpose"], default="classical")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fields = sorted({p.name.rsplit("_T", 1)[0] for p in IMAGES.glob("*_T0001_Z0001_C01.tif")})
    if args.only:
        fields = [f for f in fields if f in args.only]
    model = None
    if args.segmenter == "cellpose":
        model = models.CellposeModel(gpu=args.device != "cpu", device=torch.device(args.device), use_bfloat16=False)

    for well_fov in fields:
        if (OUT / f"{well_fov}.parquet").exists():
            print(f"{well_fov}: done already")
            continue
        t0 = time.time()
        df, info = process_field(IMAGES, well_fov, model, args.diameter)
        df.to_parquet(OUT / f"{well_fov}.parquet", index=False)
        info["seconds"] = round(time.time() - t0)
        info["segmenter"] = args.segmenter
        (OUT / f"{well_fov}.json").write_text(json.dumps(info, indent=1))
        print(f"{well_fov}: terminal={info['terminal_field']} r={info['match_r']} shift=({info['shift_dy']:.0f},{info['shift_dx']:.0f}) "
              f"nuclei/frame={info['nuclei_per_frame']} tracks={info['n_tracks']} {info['seconds']}s")


if __name__ == "__main__":
    main()
