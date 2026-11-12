"""Does the cytoplasm turn from red to green when a cell dies? Real tracks, then generated futures.

In the colour images (red ChromaLIVE 561, green ChromaLIVE 488, blue Hoechst) dying cells look as
if their cytoplasm goes from red to green. Two questions, measured rather than eyeballed:

1. Real cells, raw intensities. For every complete labelled track the mean of the 488a and 561
   channels in the cytoplasm ring at each of the 13 frames (data/processed/tracks, camera counts,
   no display scaling) and their ratio R = 488 / 561. Per cell, the log2 change of R from 0 h.
   By fate group: the time course, the fraction of cells whose ratio rises by more than 2^0.5
   (1.41x) by 6 h, and the AUROC at each frame of that change for dying against surviving cells,
   on the held-out wells and on all wells of the two-fate doses. The fate labels come from the
   AnnexinV channel and from Hoechst / nucleus area, so the ratio is not part of its own label.
2. Generated cells, crop space. For the 816 held-out cells, the ring means of green and red in
   the handcrafted features cached by 20_feature_score.py at 64 px (real 0 h and 6 h frames,
   and every 6 h sample of each model). The per-field display scaling makes the two channels'
   absolute ratio arbitrary, so the measure is the log2 change from the cell's own real 0 h
   frame, averaged over a model's samples; the fraction of cells whose change passes 2^0.5; and,
   as the eye sees it, the fraction whose green ring is brighter than its red ring at 6 h.

Usage: uv run scripts/44_green_red.py
Writes results/tables/green_red_tracks.csv (group x frame time course), green_red_auroc.csv,
green_red_change.csv (per-group fractions and means, real and generated) and
results/figures/supporting/fig19_green_red.png.
"""

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.evaluate import field_bootstrap, group_of  # noqa: E402
from static2fate.features import FEATURES_HAND  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FEAT = ROOT / "results/features"
GROUPS = ["survives", "dies, AnnexinV only", "dies, condensed"]
RISE = 0.5  # log2 units: a ratio up by more than 1.41x counts as "turned green"
EPS = 1.0  # one camera count (part 1) or one grey level (part 2) in the denominator
RES = 64
FUTURE = [6, 8, 10, 12]
# name -> (samples directory under results/kaggle, stem of the 20_feature_score cache)
MODELS = {
    "frame model, 64 px, 8 h": ("ctx64_long", "ctx64_long_64px_8h"),
    "latent frame": ("lat_frame", "lat_frame_latentframe"),
    "latent video": ("lat_video", "lat_video_latentvideo"),
    "latent video, 3 fates": ("lat_video_fate3", "lat_video_fate3_latentvideo_3fates"),
    "ae128 reconstruction": ("ae128/recon", "recon_ae128"),
}
G, R = FEATURES_HAND.index("green_ring"), FEATURES_HAND.index("red_ring")


def ratio(green: np.ndarray, red: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Green over red with `eps` in the denominator; NaN where either is missing or negative."""
    g, r = np.asarray(green, float), np.asarray(red, float)
    out = g / (r + eps)
    bad = ~np.isfinite(out) | (g < 0) | (r < 0)
    return np.where(bad, np.nan, out)


def log2_change(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """log2(after / before) of two ratio arrays, NaN where either side is not positive."""
    a, b = np.asarray(after, float), np.asarray(before, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log2(a / b)
    return np.where((a > 0) & (b > 0), out, np.nan)


def labelled_tracks(train) -> pd.DataFrame:
    """One row per complete track of the two-fate doses: group, field, held-out flag."""
    lab = pd.read_parquet(P / "fate_labels.parquet")
    lab = lab[lab["complete"] & lab["dose_nM"].isin(train.DOSES)].copy()
    lab["group"] = [group_of(int(a), bool(c)) for a, c in zip(lab["apoptotic"], lab["condensed"], strict=True)]
    lab["field"] = lab["well_fov"]
    lab["heldout"] = lab["well_fov"].str[:4].isin(train.HELDOUT_WELLS)
    return lab[["track", "group", "field", "heldout", "dose_nM"]]


def ring_ratios(lab: pd.DataFrame) -> pd.DataFrame:
    """Per track and frame: raw ring means of 488a and 561, their ratio, the log2 change from frame 1."""
    tr = pd.concat([pd.read_parquet(p) for p in sorted((P / "tracks").glob("*.parquet"))], ignore_index=True)
    tr["track"] = tr["well_fov"] + "_" + tr["particle"].astype(str)
    tr = tr[tr["track"].isin(lab["track"])][["track", "frame", "cl488a_ring_mean", "cl561_ring_mean"]]
    tr["ratio"] = ratio(tr["cl488a_ring_mean"], tr["cl561_ring_mean"])
    first = tr[tr["frame"] == 1].set_index("track")
    tr = tr.join(first[["ratio", "cl488a_ring_mean", "cl561_ring_mean"]].add_suffix("_0"), on="track")
    tr["dlog_ratio"] = log2_change(tr["ratio"], tr["ratio_0"])
    tr["dlog_green"] = log2_change(tr["cl488a_ring_mean"], tr["cl488a_ring_mean_0"])
    tr["dlog_red"] = log2_change(tr["cl561_ring_mean"], tr["cl561_ring_mean_0"])
    return tr.merge(lab, on="track")


def auroc_by_frame(tr: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """AUROC of the change from 0 h for dying vs surviving cells at each frame, field-bootstrap CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for subset, d in [("held-out wells", tr[tr["heldout"]]), ("all wells", tr)]:
        fields = d["field"].unique()
        for frame in range(2, 14):
            s = d[d["frame"] == frame]
            for col, sign in [("dlog_ratio", 1), ("dlog_green", 1), ("dlog_red", -1)]:
                v = s[["field", "group", col]].dropna()
                y = (v["group"] != "survives").astype(int).to_numpy()
                x = sign * v[col].to_numpy()
                by_field = {f: np.flatnonzero(v["field"].to_numpy() == f) for f in fields}
                bs = np.empty(n_boot)
                for b in range(n_boot):
                    pick = np.concatenate([by_field[f] for f in fields[rng.integers(len(fields), size=len(fields))]])
                    bs[b] = roc_auc_score(y[pick], x[pick]) if 0 < y[pick].sum() < len(pick) else np.nan
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                rows.append({"subset": subset, "frame": frame, "hours": (frame - 1) / 2, "measure": col,
                             "auroc": roc_auc_score(y, x), "ci_low": lo, "ci_high": hi, "n": len(v)})
    return pd.DataFrame(rows)


def generated_changes(train, lab: pd.DataFrame) -> pd.DataFrame:
    """Per held-out cell: log2 change of the ring ratio from the real 0 h frame, real 6 h and each model."""
    fs = importlib.import_module("20_feature_score")
    _, tracks_s = train.split_index("test")
    real = np.load(FEAT / f"test_{RES}.npz")["handcrafted"].reshape(len(tracks_s), 1 + len(FUTURE), -1)
    to8 = lambda v: (v + 1) * 127.5  # noqa: E731  features are means of [-1, 1] pixels
    r0 = ratio(to8(real[:, 0, G]), to8(real[:, 0, R]))
    j = 1 + FUTURE.index(12)
    r6 = ratio(to8(real[:, j, G]), to8(real[:, j, R]))
    info = lab.set_index("track").loc[tracks_s]
    base = pd.DataFrame({"track": tracks_s, "group": info["group"].to_numpy(), "field": info["field"].to_numpy()})
    out = [base.assign(model="real 6 h", dlog=log2_change(r6, r0), green_over_red=(to8(real[:, j, G]) > to8(real[:, j, R])).astype(float))]
    for name, (d, stem) in MODELS.items():
        tg, _ = fs.load_samples(ROOT / "results/kaggle" / d)
        feats = np.load(FEAT / f"{stem}_{RES}.npz")["handcrafted"]
        m = len(feats) // (len(tg) * len(FUTURE))
        g6 = feats.reshape(len(tg), len(FUTURE), m, -1)[:, FUTURE.index(12)]  # (cells, m, 9)
        pos = pd.Series(np.arange(len(tracks_s)), index=tracks_s).loc[tg].to_numpy()
        rg = ratio(to8(g6[..., G]), to8(g6[..., R]))  # (cells, m)
        with np.errstate(invalid="ignore"):
            dlog = np.nanmean(log2_change(rg, r0[pos][:, None]), axis=1)
            over = np.nanmean((to8(g6[..., G]) > to8(g6[..., R])).astype(float), axis=1)
        out.append(base.iloc[pos].assign(model=name, dlog=dlog, green_over_red=over))
    return pd.concat(out, ignore_index=True)


def summarise(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean log2 change, fraction rising past RISE and fraction green over red, with field CIs."""
    d = d.dropna(subset=["dlog"]).copy()
    d["rises"] = (d["dlog"] > RISE).astype(float)
    parts = []
    for col in ["dlog", "rises"] + (["green_over_red"] if "green_over_red" in d else []):
        parts.append(field_bootstrap(d, col, by).assign(measure=col))
    med = d.groupby(by)["dlog"].median().rename("median_dlog").reset_index()
    return pd.concat(parts, ignore_index=True).merge(med, on=by)


def figure(course: pd.DataFrame, auc: pd.DataFrame, real: pd.DataFrame, gen: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colours = {"survives": "tab:blue", "dies, AnnexinV only": "tab:orange", "dies, condensed": "tab:red"}
    ax = axes[0, 0]
    for g in GROUPS:
        s = course[course["group"] == g]
        ax.plot(s["hours"], s["ratio_median"], "o-", ms=3, color=colours[g], label=f"{g} (n = {int(s['n'].iloc[0])})")
        ax.fill_between(s["hours"], s["ratio_q25"], s["ratio_q75"], color=colours[g], alpha=0.15)
    ax.set_xlabel("hours after drug")
    ax.set_ylabel("cytoplasm ring 488 / 561 (raw counts), median and IQR")
    ax.set_title("Real tracks, two-fate doses, all wells", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    names = {"dlog_ratio": "log2 change of 488 / 561", "dlog_green": "log2 change of 488", "dlog_red": "minus log2 change of 561"}
    for col, name in names.items():
        s = auc[(auc["subset"] == "held-out wells") & (auc["measure"] == col)]
        ax.plot(s["hours"], s["auroc"], "o-", ms=3, label=name)
        ax.fill_between(s["hours"], s["ci_low"], s["ci_high"], alpha=0.15)
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_xlabel("hours after drug")
    ax.set_ylabel("AUROC, dying vs surviving, change from 0 h")
    ax.set_title("Held-out wells, 95% CI over 12 fields", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    data = [real.loc[(real["group"] == g) & (real["frame"] == 13), "dlog_ratio"].dropna() for g in GROUPS]
    ax.violinplot(data, positions=range(len(GROUPS)), showmedians=True)
    ax.axhline(RISE, color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUPS, fontsize=9)
    ax.set_ylabel("log2 change of 488 / 561, 0 h to 6 h")
    ax.set_title("Real cells at 6 h; dashed line is the 1.41x threshold", fontsize=10)
    ax = axes[1, 1]
    rows = ["real 6 h", *MODELS]
    t = gen[gen["measure"] == "dlog"]
    for k, g in enumerate(GROUPS):
        s = t[t["group"] == g].set_index("model").loc[rows]
        x = np.arange(len(rows)) + (k - 1) * 0.25
        ax.errorbar(x, s["estimate"], yerr=[s["estimate"] - s["ci_low"], s["ci_high"] - s["estimate"]], fmt="o",
                    ms=4, capsize=2, color=colours[g], label=g)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(rows, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("mean log2 change of green / red ring, 6 h vs real 0 h")
    ax.set_title("Held-out cells, crop space at 64 px, 95% CI over 12 fields", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (ROOT / "results/figures/supporting").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results/figures/supporting/fig19_green_red.png", dpi=130)


def main() -> None:
    train = importlib.import_module("09_train_flow")
    lab = labelled_tracks(train)
    tr = ring_ratios(lab)
    print(f"{len(lab)} complete tracks at the two-fate doses, {lab['heldout'].sum()} in the held-out wells", flush=True)
    q = lambda p: (lambda s: s.quantile(p))  # noqa: E731
    course = tr.groupby(["group", "frame"]).agg(
        n=("track", "size"), green_median=("cl488a_ring_mean", "median"), red_median=("cl561_ring_mean", "median"),
        ratio_median=("ratio", "median"), ratio_q25=("ratio", q(0.25)), ratio_q75=("ratio", q(0.75)),
        dlog_ratio_median=("dlog_ratio", "median")).reset_index()
    course["hours"] = (course["frame"] - 1) / 2
    auc = auroc_by_frame(tr)
    end = tr[tr["frame"] == 13].rename(columns={"dlog_ratio": "dlog"})
    change = pd.concat([
        summarise(end, ["group"]).assign(space="raw tracks", subset="all wells", model="real 6 h"),
        summarise(end[end["heldout"]], ["group"]).assign(space="raw tracks", subset="held-out wells", model="real 6 h"),
        summarise(end.assign(dose=end["dose_nM"]), ["group", "dose"]).assign(space="raw tracks", subset="all wells", model="real 6 h"),
    ], ignore_index=True)
    gen = generated_changes(train, lab)
    gen_sum = summarise(gen, ["model", "group"]).assign(space="crops 64 px", subset="held-out wells")
    change = pd.concat([change, gen_sum], ignore_index=True)
    tables = ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    course.to_csv(tables / "green_red_tracks.csv", index=False)
    auc.to_csv(tables / "green_red_auroc.csv", index=False)
    change.to_csv(tables / "green_red_change.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nmedian ring 488 / 561 by group and hour (raw counts, all two-fate wells)")
    print(course.pivot_table(index="group", columns="hours", values="ratio_median").round(2).to_string())
    print("\nfraction of cells whose ratio rises by more than 1.41x by 6 h (estimate, 95% CI)")
    for _, r in change[(change["measure"] == "rises") & (change["space"] == "raw tracks") & change["dose"].isna()].iterrows():
        print(f"  {r['subset']:15s} {r['group']:20s} {r['estimate']:.2f} ({r['ci_low']:.2f} to {r['ci_high']:.2f}), n = {r['n']}")
    print("\nAUROC of the change from 0 h, dying vs surviving, held-out wells")
    print(auc[auc["subset"] == "held-out wells"].pivot_table(index="measure", columns="hours", values="auroc").round(2).to_string())
    print("\ngenerated vs real at 6 h, crop space: mean log2 change / fraction rising / fraction green over red")
    for meas in ["dlog", "rises", "green_over_red"]:
        print(meas)
        print(gen_sum[gen_sum["measure"] == meas].pivot_table(index="model", columns="group", values="estimate").round(2).to_string())
    figure(course, auc, tr, gen_sum)


if __name__ == "__main__":
    main()
