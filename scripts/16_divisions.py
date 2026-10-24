"""Flag divisions in every track and see how they overlap with the fate labels.

Uses the per-frame nuclei in data/processed/tracks/*.parquet (live frames 1 to 13) and
static2fate.divisions. Two ways a division can contaminate the "dies" label:
  1. a cell divides during the movie and is still called apoptotic at the end;
  2. a cell is in mitosis at the last frames, so its nucleus is small and bright, which the
     label's "condensed" clause reads as apoptosis. Mitotic cells do not expose
     phosphatidylserine, so condensed but AnnexinV-negative cells with a late onset are the
     suspects.
To see whether dividing cells reach the labelled set at all, the nucleus count of every field
at frame 13 over frame 1 is also tabulated: in a growing population it estimates the fraction
of cells that divided (minus deaths).
Writes data/processed/divisions.parquet, results/tables/divisions_by_fate.csv,
results/tables/nucleus_count_growth.csv,
results/tables/divisions_by_dose.csv and image strips for checking by eye in results/qc/.
"""

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from static2fate.divisions import find_divisions  # noqa: E402
from static2fate.evaluate import group_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
LIVE = list(range(1, 14))


def strips(tracks: list[str], title: str, out: Path) -> None:
    """All 13 live frames of each cell (Hoechst blue, 488 green, 561 red), one row per cell."""
    names = np.load(P / "crops_tracks.npy")
    crops = np.load(P / "crops.npy", mmap_mode="r")
    where = {t: i for i, t in enumerate(names)}
    tracks = [t for t in tracks if t in where]
    fig, ax = plt.subplots(len(tracks), 13, figsize=(13 * 0.9, len(tracks) * 0.95), squeeze=False)
    for r, t in enumerate(tracks):
        c = crops[where[t]].astype(float) / 255
        for f in range(13):
            ax[r, f].imshow((np.dstack([c[f, 2], c[f, 1], c[f, 0]]) * 1.3).clip(0, 1))
            ax[r, f].axis("off")
        ax[r, 0].set_title(t, fontsize=5, loc="left")
    fig.suptitle(title, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=70)
    plt.close(fig)


def main() -> None:
    out = []
    for path in sorted(glob.glob(str(P / "tracks/*.parquet"))):
        f = pd.read_parquet(path, columns=["well_fov", "frame", "particle", "x", "y", "area", "hoechst_nuc_mean"])
        d = find_divisions(f, LIVE)
        d["track"] = f.well_fov.iloc[0] + "_" + d.particle.astype(str)
        out.append(d)
    div = pd.concat(out, ignore_index=True)
    counts = []
    for path in sorted(glob.glob(str(P / "tracks/*.parquet"))):
        n = pd.read_parquet(path, columns=["well_fov", "frame"]).groupby(["well_fov", "frame"]).size()
        wf = n.index[0][0]
        counts.append({"well_fov": wf, "n_frame1": n[(wf, 1)], "n_frame13": n[(wf, 13)]})
    lab = pd.read_parquet(P / "fate_labels.parquet")
    j = lab[["track", "dose_nM", "apoptotic", "annexin_positive", "condensed"]].merge(div, on="track", how="left")
    j["group"] = [group_of(int(a), bool(c)) for a, c in zip(j.apoptotic, j.condensed)]
    # late condensation: still condensed at frame 13, started in the last hour (frames 12 to 13)
    j["late_condensed"] = j.condensed_at_end.fillna(False).astype(bool) & (j.end_onset_frame >= 12)
    j["late_condensed_annexin_neg"] = j.late_condensed & ~j.annexin_positive
    j.to_parquet(P / "divisions.parquet")
    growth = pd.DataFrame(counts).merge(lab.groupby("well_fov").dose_nM.first().reset_index(), on="well_fov")
    growth["growth"] = growth.n_frame13 / growth.n_frame1
    g = growth.groupby("dose_nM").agg(fields=("growth", "size"), median_growth=("growth", "median"),
                                      q25=("growth", lambda v: v.quantile(0.25)), q75=("growth", lambda v: v.quantile(0.75)))
    g.round(3).to_csv(ROOT / "results/tables/nucleus_count_growth.csv")
    print(g.round(3).to_string())
    print(f"labelled tracks {len(j)}, with all 13 live frames {int(j.complete_live.sum())}")

    cols = ["mitosis_like", "division", "late_condensed", "late_condensed_annexin_neg"]
    jj = j[j.complete_live == True].copy()  # noqa: E712
    jj[cols] = jj[cols].fillna(False).astype(bool)

    def summarise(by):
        g = jj.groupby(by)
        t = g[cols].sum().astype(int)
        t.insert(0, "n", g.size())
        for c in cols:
            t[f"{c}_frac"] = (t[c] / t.n).round(3)
        return t

    by_fate = summarise(["group"])
    by_dose = summarise(["dose_nM", "group"])
    by_fate.to_csv(ROOT / "results/tables/divisions_by_fate.csv")
    by_dose.to_csv(ROOT / "results/tables/divisions_by_dose.csv")
    print(by_fate.to_string())
    print(by_dose[["n", "division", "division_frac", "late_condensed_annexin_neg"]].to_string())

    qc = ROOT / "results/qc"
    qc.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    ctrl = jj[(jj.dose_nM == 0) & (jj.group == "survives")].track
    strips(list(rng.choice(ctrl, 16, replace=False)), "random control survivors", qc / "divisions_random_controls.png")
    strips(list(jj[jj.division].track[:16]), "flagged divisions", qc / "divisions_flagged.png")
    late = jj[jj.late_condensed_annexin_neg & (jj.group == "dies, condensed")].track
    strips(list(late[:16]), "condensed in the last hour, AnnexinV negative, labelled dies",
           qc / "divisions_late_condensed.png")


if __name__ == "__main__":
    main()
