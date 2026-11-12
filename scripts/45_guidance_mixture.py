"""Is "dies" drawn as a mixture? Condensation and colour of each sample, by fate group and sampler.

The two-class label "dies" covers two looks at 6 h, a condensed nucleus with a strongly green
cytoplasm (55% of dying training cells) and a normal-looking nucleus with a milder shift (45%).
A sampler told "dies" should draw both in about those proportions; with classifier-free
guidance it may collapse onto the typical one. For every held-out cell and every 6 h sample of
each sampler, from the handcrafted features cached at 64 px by 20_feature_score.py (run it
first with the same name=dir pairs):

- condensed: the label's own rule applied in crop space, nucleus area below half the cell's real
  0 h area and nucleus Hoechst above twice it (grey levels);
- green: the cytoplasm ring's green / red ratio more than 1.41x the real 0 h one (44_green_red.py).

Per cell: the fraction of its samples that are condensed and that are green, and whether the
samples disagree (mixed, some condensed and some not). Per group: the mean of each over cells,
the fraction of cells whose samples all agree, and the real 6 h frame under the same rules, with
95% CIs over the 12 held-out fields. The real frame's condensed call is also compared with the
label (which used raw whole-field intensities) as a check on the crop-space rule.

Usage: STATIC2FATE_TAG=<tag> uv run scripts/45_guidance_mixture.py [name=dir ...]
Writes results/tables/guidance_mixture<tag>.csv and results/figures/supporting/fig20_guidance_mixture<tag>.png.
"""

import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import field_bootstrap  # noqa: E402
from static2fate.features import FEATURES_HAND  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "results/features"
TAG = os.environ.get("STATIC2FATE_TAG", "")
RES = 64
FUTURE = [6, 8, 10, 12]
GROUPS = ["survives", "dies, AnnexinV only", "dies, condensed"]
MODELS = {
    "latent video": ROOT / "results/kaggle/lat_video",
    "latent video, guidance 1": ROOT / "results/guidance/lat_video_g1",
    "latent video, 3 fates": ROOT / "results/kaggle/lat_video_fate3",
}
AREA, HOE, G, R = (FEATURES_HAND.index(k) for k in ["nuc_area", "hoechst_nuc", "green_ring", "red_ring"])


def to8(v: np.ndarray) -> np.ndarray:
    """Mean of [-1, 1] pixels -> grey levels."""
    return (np.asarray(v, float) + 1) * 127.5


def calls(feat: np.ndarray, real0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """feat (..., 9) at 6 h, real0 (..., 9) broadcastable at 0 h -> (condensed, green) 0/1 floats, NaN if no nucleus.

    The label's rule, area below half and Hoechst above twice the 0 h value, in crop space; green
    is a ring ratio up by more than 2^0.5 from the 0 h frame.
    """
    gr = importlib.import_module("44_green_red")
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = (feat[..., AREA] / real0[..., AREA] < 0.5) & (to8(feat[..., HOE]) / to8(real0[..., HOE]) > 2)
        dlog = gr.log2_change(gr.ratio(to8(feat[..., G]), to8(feat[..., R])), gr.ratio(to8(real0[..., G]), to8(real0[..., R])))
    ok = np.isfinite(feat[..., AREA]) & np.isfinite(real0[..., AREA])
    return np.where(ok, cond.astype(float), np.nan), np.where(np.isfinite(dlog), (dlog > gr.RISE).astype(float), np.nan)


def main(models: dict[str, Path]) -> None:
    train = importlib.import_module("09_train_flow")
    fs = importlib.import_module("20_feature_score")
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    _, tracks_s = train.split_index("test")
    fate_s = train.fate_classes(tracks_s, 3).numpy()  # 0 survives, 1 AnnexinV only, 2 condensed
    group = np.array(GROUPS)[fate_s]
    real = np.load(FEAT / f"test_{RES}.npz")["handcrafted"].reshape(len(tracks_s), 1 + len(FUTURE), -1)
    r0, r6 = real[:, 0], real[:, 1 + FUTURE.index(12)]
    base = pd.DataFrame({"track": tracks_s, "field": [t[:10] for t in tracks_s], "group": group})
    cond, green = calls(r6, r0)
    rows = [base.assign(model="real 6 h", p_condensed=cond, p_green=green, mixed=0.0, all_condensed=cond, none_condensed=1 - cond)]
    agree = np.nanmean(cond == lab.loc[tracks_s, "condensed"].to_numpy().astype(float))
    print(f"real 6 h: crop-space condensed call agrees with the label for {agree:.0%} of cells "
          f"({np.nanmean(cond):.0%} called condensed, label {lab.loc[tracks_s, 'condensed'].mean():.0%})", flush=True)
    pos = pd.Series(np.arange(len(tracks_s)), index=tracks_s)
    for name, d in models.items():
        tg, _ = fs.load_samples(d)
        key = f"{d.name}_{name.replace(' ', '').replace(',', '_')}_{RES}.npz"
        feat = np.load(FEAT / key)["handcrafted"]
        m = len(feat) // (len(tg) * len(FUTURE))
        f6 = feat.reshape(len(tg), len(FUTURE), m, -1)[:, FUTURE.index(12)]  # (cells, m, 9)
        p = pos.loc[tg].to_numpy()
        c, g = calls(f6, r0[p][:, None])
        pc, pg = np.nanmean(c, 1), np.nanmean(g, 1)
        rows.append(base.iloc[p].assign(model=name, p_condensed=pc, p_green=pg, mixed=((pc > 0) & (pc < 1)).astype(float),
                                        all_condensed=(pc == 1).astype(float), none_condensed=(pc == 0).astype(float)))
    cells = pd.concat(rows, ignore_index=True)
    measures = ["p_condensed", "p_green", "mixed", "all_condensed", "none_condensed"]
    tab = pd.concat([field_bootstrap(cells.dropna(subset=[c]), c, ["model", "group"]).assign(measure=c) for c in measures])
    (ROOT / "results/tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(ROOT / f"results/tables/guidance_mixture{TAG}.csv", index=False)
    pd.set_option("display.width", 220)
    order = ["real 6 h", *models]
    for c in measures:
        print(f"\n{c} (mean over held-out cells; for a sampler, over its samples first)")
        print(tab[tab.measure == c].pivot_table(index="model", columns="group", values="estimate").reindex(order)[GROUPS].round(2).to_string())
    figure(tab, cells, order)


def figure(tab: pd.DataFrame, cells: pd.DataFrame, order: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    colours = {"survives": "tab:blue", "dies, AnnexinV only": "tab:orange", "dies, condensed": "tab:red"}
    for ax, c, title in zip(axes[:2], ["p_condensed", "p_green"],
                            ["fraction of 6 h samples with a condensed nucleus", "fraction of 6 h samples turned green"], strict=True):
        t = tab[tab.measure == c]
        for k, g in enumerate(GROUPS):
            s = t[t.group == g].set_index("model").reindex(order)
            x = np.arange(len(order)) + (k - 1) * 0.25
            ax.errorbar(x, s.estimate, yerr=[s.estimate - s.ci_low, s.ci_high - s.estimate], fmt="o", ms=4, capsize=2,
                        color=colours[g], label=g)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    s = cells[(cells.group == "dies, AnnexinV only") & (cells.model != "real 6 h")]
    for name in order[1:]:
        v = s.loc[s.model == name, "p_condensed"].dropna()
        ax.hist(v, bins=np.linspace(0, 1, 9), histtype="step", lw=1.5, label=name, density=True)
    ax.set_xlabel("fraction of a cell's 8 samples with a condensed nucleus")
    ax.set_title("AnnexinV-only dying cells: how the 8 samples split", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Held-out cells at 6 h, crop-space rules at 64 px, 95% CI over 12 fields", fontsize=10)
    fig.tight_layout()
    (ROOT / "results/figures/supporting").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / f"results/figures/supporting/fig20_guidance_mixture{TAG}.png", dpi=130)


if __name__ == "__main__":
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    main({k: Path(v) for k, v in args.items()} if args else MODELS)
