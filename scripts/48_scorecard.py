"""One figure for the final model: the latent video model with three fate classes, guidance 2.

Reads the tables the scoring chain wrote for the 816 held-out cells (tags _guide, _probe and
_recon) and draws six panels, each with the reference a number has to be read against:

A. Realism: AUROC of a linear probe telling generated from real 6 h frames on nine handcrafted
   nucleus measurements at 128 px, next to the same probe on the autoencoder's reconstruction of
   the real frame and on a 1 px blur of it.
B. Whose future: mean DINOv2 distance at 64 px from the generated 6 h frames to the cell's real
   6 h frame, next to the cell's own 0 h frame (copy) and a random real cell of the same fate.
C. Identity: percentile rank of the cell's own real future among all held-out cells of its fate
   (0 nearest, 0.5 chance), for the generated frames, the 0 h copy and the real 3 h frame.
D, E. The two looks of dying: fraction of 6 h frames with a condensed nucleus and with a green
   cytoplasm, real against generated.
F. Motion: reversals of the nucleus-area change over three hourly steps (1 = random).

Usage: uv run scripts/48_scorecard.py
Writes results/figures/fig06_scorecard.png and results/tables/scorecard.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results/tables"
MODEL = "latent video, 3 fates"
GROUPS = ["survives", "dies, AnnexinV only", "dies, condensed"]
SHORT = ["survives", "dies,\nAnnexinV only", "dies,\ncondensed"]
COLOUR = {"model": "tab:purple", "real": "0.35", "copy": "tab:olive", "random": "0.75", "real 3 h": "tab:cyan"}


def pick(d: pd.DataFrame, **sel) -> pd.DataFrame:
    for k, v in sel.items():
        d = d[d[k] == v]
    return d.set_index("group").reindex(GROUPS)


def bars(ax, series: list[tuple[str, pd.DataFrame, str]], ylabel: str, title: str, ylim=None) -> list[dict]:
    w = 0.8 / len(series)
    rows = []
    for i, (name, s, colour) in enumerate(series):
        x = np.arange(len(GROUPS)) + (i - (len(series) - 1) / 2) * w
        ax.bar(x, s["estimate"], w * 0.92, color=colour, label=name,
               yerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]], capsize=2, error_kw={"lw": 0.8})
        rows += [{"panel": title, "series": name, "group": g, "estimate": s.loc[g, "estimate"],
                  "ci_low": s.loc[g, "ci_low"], "ci_high": s.loc[g, "ci_high"]} for g in GROUPS]
    ax.set_xticks(range(len(GROUPS)), SHORT, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7.5)
    return rows


def main() -> None:
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.6))
    rows = []

    probe = {"generated 6 h frames": pd.read_csv(TAB / "real_vs_generated_auroc_guide.csv").query("model == @MODEL"),
             "autoencoder reconstruction\nof the real frame": pd.read_csv(TAB / "real_vs_generated_auroc_recon.csv").query("model == 'ae128'"),
             "real frame blurred by 1 px": pd.read_csv(TAB / "real_vs_generated_auroc_probe.csv").query("model == 'blur 1 px'")}
    vals = {k: float(d[(d["space"] == "handcrafted") & (d["res"] == 128)]["auroc"].iloc[0]) for k, d in probe.items()}
    a = ax[0, 0]
    a.barh(list(vals)[::-1], list(vals.values())[::-1], color=[COLOUR["random"], COLOUR["random"], COLOUR["model"]])
    for i, v in enumerate(list(vals.values())[::-1]):
        a.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    a.axvline(0.5, color="k", lw=0.6)
    a.set_xlim(0.45, 0.8)
    a.set_xlabel("AUROC of a probe against real frames (0.5 = cannot tell)", fontsize=9)
    a.tick_params(axis="y", labelsize=8)
    a.set_title("A. Realism, handcrafted probe at 128 px", fontsize=10, loc="left")
    rows += [{"panel": "A. Realism", "series": k.replace("\n", " "), "group": "all", "estimate": v} for k, v in vals.items()]

    e = pd.read_csv(TAB / "energy_terms_guide.csv")
    e = e[(e["model"] == MODEL) & (e["space"] == "dinov2") & (e["res"] == 64)]
    rows += bars(ax[0, 1], [("generated", pick(e, measure="d_model"), COLOUR["model"]),
                            ("the cell's own 0 h frame", pick(e, measure="d_copy"), COLOUR["copy"]),
                            ("a random real cell, same fate", pick(e, measure="d_random"), COLOUR["random"])],
                 "DINOv2 distance to the cell's real 6 h frame", "B. Whose future is it", ylim=(33, 41))

    i = pd.read_csv(TAB / "identity_rank_guide.csv")
    i = i[(i["space"] == "dinov2") & (i["res"] == 64) & (i["hours"] == 6) & (i["measure"] == "pct_rank")]
    rows += bars(ax[0, 2], [("generated", pick(i, query=MODEL), COLOUR["model"]),
                            ("the cell's own 0 h frame", pick(i, query="copy 0 h"), COLOUR["copy"]),
                            ("its real 3 h frame", pick(i, query="real 3 h"), COLOUR["real 3 h"])],
                 "rank of the cell's own real 6 h frame", "C. Identity (0 = nearest, 0.5 = chance)", ylim=(0, 0.75))
    ax[0, 2].axhline(0.5, color="k", lw=0.6, ls=":")

    m = pd.read_csv(TAB / "guidance_mixture_guide.csv")
    for a, measure, title in [(ax[1, 0], "p_condensed", "D. Nucleus condensed at 6 h"),
                              (ax[1, 1], "p_green", "E. Cytoplasm turned green at 6 h")]:
        rows += bars(a, [("real", pick(m, model="real 6 h", measure=measure), COLOUR["real"]),
                         ("generated", pick(m, model=MODEL, measure=measure), COLOUR["model"])],
                     "fraction of frames", title, ylim=(0, 1))

    t = pd.read_csv(TAB / "temporal_guide.csv")
    t = t[(t["measure"] == "area_reversals") & (t["step_min"] == 60)]
    rows += bars(ax[1, 2], [("real", pick(t, model="real"), COLOUR["real"]),
                            ("generated", pick(t, model=MODEL), COLOUR["model"])],
                 "direction reversals of nucleus area, 3 to 6 h", "F. Motion (1 = random)", ylim=(0, 1.4))
    ax[1, 2].axhline(1.0, color="k", lw=0.6, ls=":")

    fig.suptitle("Latent video model, three fate classes, guidance 2: 816 held-out cells (571 / 150 / 95), 8 futures each, "
                 "95% CIs resample the 12 fields", fontsize=10.5)
    fig.tight_layout()
    (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results/figures/fig06_scorecard.png", dpi=140)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "scorecard.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
