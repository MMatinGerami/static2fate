"""Write the cells used by the flow models as a dataset folder for Kaggle or Colab.

Every labelled cell of all ten doses (4.3 GB). The training scripts select the two-fate doses
unless told otherwise, so earlier jobs read the same cells as before. The files keep their
names, so pointing STATIC2FATE_DATA at the folder is enough for 09_train_flow.load and the
training scripts.

Usage: uv run scripts/export_kaggle.py [kaggle-username]
Writes data/kaggle/static2fate-crops/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "data/kaggle/static2fate-crops"


def main(user: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(P / "fate_labels.parquet")
    tracks = np.load(P / "crops_tracks.npy")
    keep = np.flatnonzero(np.isin(tracks, lab["track"]))
    crops = np.load(P / "crops.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(OUT / "crops.npy", mode="w+", dtype=crops.dtype,
                                    shape=(len(keep),) + crops.shape[1:])
    for b in range(0, len(keep), 256):
        out[b:b + 256] = crops[keep[b:b + 256]]
    out.flush()
    np.save(OUT / "crops_tracks.npy", tracks[keep])
    lab[lab["track"].isin(tracks[keep])].to_parquet(OUT / "fate_labels.parquet")
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "static2fate HeLa staurosporine single-cell crops",
        "id": f"{user}/static2fate-crops",
        "licenses": [{"name": "CC0-1.0"}],
        "description": "Single-cell crops (13 frames x 3 channels x 128 px, uint8) of HeLa cells "
                       "at ten staurosporine doses (0 to 156 nM) with fate labels, derived from "
                       "BioImage Archive S-BIAD2515 (Lippincott et al., CC0). Code: "
                       "github.com/MMatinGerami/static2fate"}, indent=1))
    print(f"{len(keep)} cells, {out.nbytes / 1e9:.2f} GB -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "USERNAME")
