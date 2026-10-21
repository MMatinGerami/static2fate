"""Figure for the README: one cell that dies and one that survives at 19.53 nM, every 30 min.

Candidates are cells at 19.53 nM whose crop never touches the image border and whose nucleus
never moves more than 25 px between frames (no track jumps). The dying cell is the one with the
strongest nuclear condensation (area halved, Hoechst brightest); the surviving cell keeps its
nucleus size and has the lowest terminal AnnexinV. Crops are shown as a colour composite
(red ChromaLIVE 561, green ChromaLIVE 488, blue Hoechst).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
DOSE = 19.53
H = 64  # half crop size


def main() -> None:
    lab = pd.read_parquet(P / "fate_labels.parquet")
    lab = lab[lab["dose_nM"] == DOSE].set_index("track")
    tr = pd.concat([pd.read_parquet(f) for f in P.glob("tracks/*.parquet")], ignore_index=True)
    tr["track"] = tr["well_fov"] + "_" + tr["particle"].astype(str)
    tr = tr[tr["track"].isin(lab.index) & (tr["frame"] <= 13)].sort_values(["track", "frame"])
    g = tr.groupby("track")
    inside = g.apply(lambda d: (d.y.min() > H) & (d.x.min() > H) & (d.y.max() < 1900 - H) & (d.x.max() < 1900 - H))
    step = g.apply(lambda d: np.hypot(d.y.diff(), d.x.diff()).max())
    ok = lab.loc[inside[inside & (step < 25)].index]
    x = np.load(P / "crops.npy", mmap_mode="r")
    tracks = np.load(P / "crops_tracks.npy")
    idx = {t: i for i, t in enumerate(tracks)}
    ok = ok[ok.index.isin(idx)]
    dies = ok[ok["apoptotic"] & ok["condensed"]].sort_values("hoechst_ratio", ascending=False).index[0]
    calm = ok[~ok["apoptotic"] & ok["area_ratio"].between(0.8, 1.3)]
    lives = calm.sort_values("annexin_cyto_p90").index[0]
    print(f"{len(ok)} clean candidates")

    frames = range(0, 13, 2)
    fig, ax = plt.subplots(2, len(frames), figsize=(1.6 * len(frames), 3.6))
    for r, (track, name) in enumerate([(lives, "survives"), (dies, "dies")]):
        for c, f in enumerate(frames):
            im = np.asarray(x[idx[track], f])
            ax[r, c].imshow(np.dstack([im[2], im[1], im[0]]))
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"{f * 30 // 60} h {f * 30 % 60:02d}", fontsize=9)
        ax[r, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("Two HeLa cells at 19.5 nM staurosporine, one survives and one dies", fontsize=10)
    plt.tight_layout()
    out = ROOT / "results/figures/fig3_two_cells.png"
    plt.savefig(out, dpi=110)
    print(f"{out.name}: survives {lives}, dies {dies}")


if __name__ == "__main__":
    main()
