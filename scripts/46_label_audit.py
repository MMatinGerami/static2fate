"""How clean is the fate label, and how early is the nuclear event readable?

The fate label of 04_fate_labels.py is "AnnexinV above the 90th percentile of untreated cells,
or a condensed nucleus". This script checks it against the two markers that do not depend on
the AnnexinV image, both read from the live movie: nuclear condensation (area below half and
Hoechst above twice the cell's own 0 h values) and the colour shift of the cytoplasm (ChromaLIVE
488 / 561 ring ratio up by more than 1.41x, 44_green_red.py).

A. Funnel: nuclei segmented, complete tracks, labelled cells, cells at the doses the models use.
B. Dose response of each marker, and the share of the "dies, AnnexinV only" label that the
   threshold alone explains: by construction 10% of untreated cells are AnnexinV-positive, so
   at dose d the expected background is (control AnnexinV-only rate) x (non-condensed cells at
   d). 95% CIs resample fields.
C. Agreement of the three markers by dose.
D. Timing: for each cell the first frame from which a marker holds until the last live frame
   (onset), and the cumulative fraction of cells past onset by hour and dose.
E. Readability from one frame, logistic regression on the handcrafted nucleus measurements or
   on frozen DINOv2 embeddings of that frame.
   (1) The protocol of 06_baseline.py (trained on all doses, fields held out, AUROC within a
   dose) for the fate label in untreated wells, where it marks only the top 10% of the
   AnnexinV readout, next to 19.53 nM.
   (2) At 39.06 nM, the one dose where every marker splits the cells about in half, trained
   within the dose, leave one field out, for each marker as the target. Condensation and the
   colour shift are defined relative to the cell's own 0 h frame, so a 0 h measurement is
   part of their definition; the AnnexinV call is not.
F. Order of events: per cell, hours from the colour shift to nuclear condensation.

Usage: uv run scripts/46_label_audit.py
Writes results/tables/label_audit_{funnel,dose,markers,onset,readability,definition,lag}.csv and
results/figures/fig02_labels.png, fig05_readability.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from static2fate.audit import background_share, cumulative_incidence, sustained_onset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
TAB = ROOT / "results/tables"
FIG = ROOT / "results/figures"
HAND = ["area", "eccentricity", "solidity", "hoechst_nuc_mean", "hoechst_ring_mean", "cl488a_nuc_mean",
        "cl488a_ring_mean", "cl488b_nuc_mean", "cl488b_ring_mean", "cl561_nuc_mean", "cl561_ring_mean"]
MODEL_DOSES = [4.88, 9.77, 19.53, 39.06]
HELDOUT_WELLS = ["E-06", "E-07", "E-08", "E-09"]
RISE = 0.5  # log2 units, as in 44_green_red.py
MIXED_DOSE = 39.06  # the dose where each marker splits the cells about in half
N_BOOT = 1000
COLOURS = {"AnnexinV positive": "tab:purple", "nucleus condensed": "tab:red", "cytoplasm turned green": "tab:green"}


def load() -> tuple[pd.DataFrame, dict[str, np.ndarray], int]:
    """Labelled cells (one row each), their per-frame measurements as (cells, 13) arrays, and the
    number of nuclei segmented at 0 h in all fields."""
    lab = pd.read_parquet(P / "fate_labels.parquet")
    lab["field"] = lab["well_fov"]
    lab["well"] = lab["well_fov"].str[:4]
    tr = pd.concat([pd.read_parquet(p) for p in sorted((P / "tracks").glob("*.parquet"))], ignore_index=True)
    tr["track"] = tr["well_fov"] + "_" + tr["particle"].astype(str)
    nuclei = int((tr["frame"] == 1).sum())
    tr = tr[tr["track"].isin(lab["track"]) & (tr["frame"] <= 13)]
    wide = {}
    for col in HAND:
        w = tr.pivot_table(index="track", columns="frame", values=col, aggfunc="first").reindex(
            index=lab["track"], columns=range(1, 14))
        wide[col] = w.ffill(axis=1).bfill(axis=1).to_numpy()  # a track may skip up to two frames
    return lab.reset_index(drop=True), wide, nuclei


def markers(lab: pd.DataFrame, wide: dict[str, np.ndarray]) -> pd.DataFrame:
    """Adds the live-movie markers per cell: onset frames (1 to 13, NaN if never) and calls at 6 h."""
    area = wide["area"] / wide["area"][:, :1]
    hoechst = wide["hoechst_nuc_mean"] / wide["hoechst_nuc_mean"][:, :1]
    ratio = wide["cl488a_ring_mean"] / (wide["cl561_ring_mean"] + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog = np.log2(ratio / ratio[:, :1])
    lab = lab.copy()
    lab["onset_condensed"] = sustained_onset((area < 0.5) & (hoechst > 2)) + 1
    lab["onset_green"] = sustained_onset(dlog > RISE) + 1
    lab["live_condensed"] = lab["onset_condensed"].notna()
    lab["green"] = lab["onset_green"].notna()
    lab["group"] = np.where(~lab["apoptotic"], "survives",
                            np.where(lab["condensed"], "dies, condensed", "dies, AnnexinV only"))
    return lab


def boot_fields(lab: pd.DataFrame, stat, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% CI of stat(dataframe) from resampling fields with replacement."""
    rng = np.random.default_rng(seed)
    parts = [g for _, g in lab.groupby("field")]
    bs = [stat(pd.concat([parts[i] for i in rng.integers(len(parts), size=len(parts))])) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def funnel(lab: pd.DataFrame, nuclei: int) -> pd.DataFrame:
    ft = pd.read_parquet(P / "fate_table.parquet")
    model = lab[lab["dose_nM"].isin(MODEL_DOSES)]
    held = model["well"].isin(HELDOUT_WELLS)
    rows = [("nuclei segmented at 0 h, 119 fields", nuclei),
            ("tracked from 0 h to the terminal image", int(ft["complete"].sum())),
            ("with a cell mask and an AnnexinV value (labelled)", len(lab)),
            ("at the four doses the models use (4.88 to 39.06 nM)", len(model)),
            ("of these, training wells (rows C and D)", int((~held).sum())),
            ("of these, held-out wells (row E, 12 fields)", int(held.sum()))]
    return pd.DataFrame(rows, columns=["step", "cells"])


def dose_table(lab: pd.DataFrame) -> pd.DataFrame:
    ctrl = lab[lab["dose_nM"] == 0]
    rows = []
    for d, s in lab.groupby("dose_nM"):
        both = pd.concat([ctrl, s]) if d != 0 else s

        def share(x: pd.DataFrame, d=d) -> float:
            c, t = x[x["dose_nM"] == 0], x[x["dose_nM"] == d]
            if not len(c) or not len(t):
                return np.nan
            base = (c["group"] == "dies, AnnexinV only").mean()
            return background_share((t["group"] == "dies, AnnexinV only").sum(), (~t["condensed"]).sum(), base)

        lo, hi = boot_fields(both, share)
        rows.append({"dose_nM": d, "n": len(s), "wells": s["well"].nunique(), "fields": s["field"].nunique(),
                     "annexin_positive": s["annexin_positive"].mean(), "condensed": s["condensed"].mean(),
                     "green": s["green"].mean(), "apoptotic": s["apoptotic"].mean(),
                     "annexin_only": (s["group"] == "dies, AnnexinV only").mean(),
                     "background_share_of_annexin_only": share(both), "share_ci_low": lo, "share_ci_high": hi})
    out = pd.DataFrame(rows)
    for m in ["annexin_positive", "condensed", "green"]:
        ci = [boot_fields(s, lambda x, m=m: x[m].mean()) for _, s in lab.groupby("dose_nM")]
        out[f"{m}_ci_low"], out[f"{m}_ci_high"] = zip(*ci, strict=True)
    return out


def marker_table(lab: pd.DataFrame) -> pd.DataFrame:
    """Conditional rates between the three markers, by dose."""
    rows = []
    for d, s in lab.groupby("dose_nM"):
        neither = s[~s["condensed"] & ~s["green"]]
        rows.append({
            "dose_nM": d, "n": len(s),
            "annexin_given_condensed": s.loc[s["condensed"], "annexin_positive"].mean(),
            "annexin_given_green_only": s.loc[~s["condensed"] & s["green"], "annexin_positive"].mean(),
            "annexin_given_neither": neither["annexin_positive"].mean(),
            "green_given_condensed": s.loc[s["condensed"], "green"].mean(),
            "green_given_annexin_only": s.loc[s["group"] == "dies, AnnexinV only", "green"].mean(),
            "green_given_survives": s.loc[s["group"] == "survives", "green"].mean(),
            "all_three": (s["annexin_positive"] & s["condensed"] & s["green"]).mean(),
            "none": (~s["annexin_positive"] & ~s["condensed"] & ~s["green"]).mean(),
            "n_condensed": int(s["condensed"].sum()), "n_annexin_only": int((s["group"] == "dies, AnnexinV only").sum()),
            "n_survives": int((s["group"] == "survives").sum())})
    return pd.DataFrame(rows)


def onset_table(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, s in lab.groupby("dose_nM"):
        for marker, col in [("nucleus condensed", "onset_condensed"), ("cytoplasm turned green", "onset_green")]:
            ci = cumulative_incidence(s[col].to_numpy(), 13)
            rows += [{"dose_nM": d, "marker": marker, "frame": f + 1, "hours": f / 2, "fraction": ci[f], "n": len(s)}
                     for f in range(13)]
    return pd.DataFrame(rows)


def oof_scores(X: np.ndarray, y: np.ndarray, fields: np.ndarray, folds: int | None = 10) -> np.ndarray:
    """Out-of-fold scores, (cells, 13): one classifier per frame; fields kept together in `folds`
    groups, or each field left out in turn when folds is None."""
    scores = np.full(X.shape[:2], np.nan)
    cv = LeaveOneGroupOut() if folds is None else GroupKFold(n_splits=folds)
    for tr, te in cv.split(X[:, 0], y, fields):
        for f in range(X.shape[1]):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            clf.fit(X[tr, f], y[tr])
            scores[te, f] = clf.predict_proba(X[te, f])[:, 1]
    return scores


def auroc_ci(y: np.ndarray, s: np.ndarray, fields: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = {f: np.flatnonzero(fields == f) for f in np.unique(fields)}
    keys = list(idx)
    bs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[keys[i]] for i in rng.integers(len(keys), size=len(keys))])
        bs.append(roc_auc_score(y[pick], s[pick]) if 0 < y[pick].sum() < len(pick) else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def auroc_rows(y: np.ndarray, s: np.ndarray, fields: np.ndarray, **tags) -> list[dict]:
    rows = []
    for f in range(s.shape[1]):
        a, lo, hi = auroc_ci(y, s[:, f], fields)
        rows.append({**tags, "frame": f + 1, "hours": f / 2, "auroc": a, "ci_low": lo, "ci_high": hi,
                     "n": len(y), "positives": int(y.sum())})
    return rows


def readability(lab: pd.DataFrame, feats: dict[str, np.ndarray]) -> pd.DataFrame:
    fields = lab["field"].to_numpy()
    rows = []
    y = lab["apoptotic"].to_numpy().astype(int)
    s = oof_scores(feats["handcrafted"], y, fields, folds=10)
    for d in [0.0, 19.53]:
        m = (lab["dose_nM"] == d).to_numpy()
        rows += auroc_rows(y[m], s[m], fields[m], protocol="trained on all doses", features="handcrafted",
                           target="apoptotic", dose_nM=d)
    m = np.isclose(lab["dose_nM"], MIXED_DOSE)
    for target in ["annexin_positive", "condensed", "green"]:
        y = lab[target].to_numpy().astype(int)[m]
        for feat, X in feats.items():
            s = oof_scores(X[m], y, fields[m], folds=None)
            rows += auroc_rows(y, s, fields[m], protocol="trained within the dose", features=feat, target=target,
                               dose_nM=MIXED_DOSE)
            print(f"scored {feat} / {target}", flush=True)
    return pd.DataFrame(rows)


def definition_check(lab: pd.DataFrame, wide: dict[str, np.ndarray]) -> pd.DataFrame:
    """Is a 0 h signal for condensation biology, or built into how condensation is defined?

    The label's rule is relative to the cell's own 0 h nucleus (area halved, Hoechst doubled),
    so a large, dim nucleus at 0 h meets it more easily. An absolute rule on the last live
    frame (area below the 5th and Hoechst above the 95th percentile of untreated cells) favours
    nuclei that are small and bright to begin with. AUROC of two 0 h measurements for each
    rule, within the mixed dose: if the sign flips with the rule, the 0 h signal is the rule's.
    """
    ctrl = (lab["dose_nM"] == 0).to_numpy()
    a13, h13 = wide["area"][:, 12], wide["hoechst_nuc_mean"][:, 12]
    absolute = (a13 < np.percentile(a13[ctrl], 5)) & (h13 > np.percentile(h13[ctrl], 95))
    m = np.isclose(lab["dose_nM"], MIXED_DOSE)
    rows = []
    for rule, y in [("ratio to the cell's own 0 h (the label)", lab["condensed"].to_numpy()),
                    ("absolute, from untreated cells", absolute)]:
        rows.append({"rule": rule, "rate_at_mixed_dose": y[m].mean(), "rate_untreated": y[ctrl].mean(),
                     "auroc_area_0h": roc_auc_score(y[m], wide["area"][m, 0]),
                     "auroc_hoechst_0h": roc_auc_score(y[m], wide["hoechst_nuc_mean"][m, 0])})
    return pd.DataFrame(rows)


def lag_table(lab: pd.DataFrame) -> pd.DataFrame:
    """Per dose: among cells whose nucleus condenses in the movie, when did the cytoplasm turn green?"""
    rows = []
    for d, s in lab[lab["live_condensed"]].groupby("dose_nM"):
        if len(s) < 30:
            continue
        lag = (s["onset_condensed"] - s["onset_green"]) / 2  # hours, positive = green first
        rows.append({"dose_nM": d, "condensing_cells": len(s), "turned_green": s["green"].mean(),
                     "green_first": (lag > 0).mean(), "same_frame": (lag == 0).mean(), "condensed_first": (lag < 0).mean(),
                     "median_lag_h": lag.median(), "lag_q25_h": lag.quantile(0.25), "lag_q75_h": lag.quantile(0.75)})
    return pd.DataFrame(rows)


def fig_labels(dose: pd.DataFrame, mk: pd.DataFrame, onset: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2))
    x = np.arange(len(dose))
    ticks = [f"{d:g}" for d in dose["dose_nM"].round(1)]
    a = ax[0, 0]
    for m, name in [("annexin_positive", "AnnexinV positive"), ("condensed", "nucleus condensed"),
                    ("green", "cytoplasm turned green")]:
        a.errorbar(x, dose[m], yerr=[dose[m] - dose[f"{m}_ci_low"], dose[f"{m}_ci_high"] - dose[m]], marker="o", ms=4,
                   capsize=2, color=COLOURS[name], label=name)
    a.axhline(0.10, color="tab:purple", ls=":", lw=1)
    a.text(0.02, 0.115, "AnnexinV threshold: 10% of untreated cells by construction", fontsize=7.5, color="tab:purple")
    a.set_xticks(x, ticks)
    a.set_xlabel("staurosporine (nM)")
    a.set_ylabel("fraction of cells at 6 h")
    a.set_title("A. Three markers of apoptosis by dose", fontsize=10, loc="left")
    a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.92))
    a = ax[0, 1]
    a.bar(x, dose["background_share_of_annexin_only"], color="tab:purple", alpha=0.6,
          yerr=[dose["background_share_of_annexin_only"] - dose["share_ci_low"],
                dose["share_ci_high"] - dose["background_share_of_annexin_only"]], capsize=2)
    a.axvspan(3.5, 7.5, color="grey", alpha=0.12)
    a.text(5.5, 1.02, "doses the models use", ha="center", fontsize=8)
    a.set_xticks(x, ticks)
    a.set_ylim(0, 1.1)
    a.set_xlabel("staurosporine (nM)")
    a.set_ylabel('share of "dies, AnnexinV only" cells\nexpected from the threshold alone')
    a.set_title("B. How much of the AnnexinV-only label is background", fontsize=10, loc="left")
    a = ax[1, 0]
    w = 0.26
    for i, (col, name, c) in enumerate([("green_given_condensed", "dies, condensed", "tab:red"),
                                        ("green_given_annexin_only", "dies, AnnexinV only", "tab:orange"),
                                        ("green_given_survives", "survives", "tab:blue")]):
        a.bar(x + (i - 1) * w, mk[col], w, color=c, label=name)
    a.set_xticks(x, ticks)
    a.set_xlabel("staurosporine (nM)")
    a.set_ylabel("fraction whose cytoplasm turned green by 6 h")
    a.set_title("C. The colour shift within each label group", fontsize=10, loc="left")
    a.legend(fontsize=8)
    a = ax[1, 1]
    shades = {0.0: 0.25, 19.53: 0.5, 39.06: 0.75, 78.13: 1.0}
    for marker, ls in [("nucleus condensed", "-"), ("cytoplasm turned green", "--")]:
        for d, alpha in shades.items():
            s = onset[(onset["marker"] == marker) & np.isclose(onset["dose_nM"], d)]
            a.plot(s["hours"], s["fraction"], ls, color=COLOURS[marker], alpha=alpha, lw=1.8,
                   label=f"{marker}, {d:g} nM")
    a.set_xlabel("hours after drug")
    a.set_ylabel("cumulative fraction of cells past onset")
    a.set_title("D. When each marker appears (live movie only)", fontsize=10, loc="left")
    a.legend(fontsize=7, ncol=2)
    fig.suptitle("Fate labels of 6,676 tracked HeLa cells: three markers, one plate (95% CIs resample fields)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_labels.png", dpi=140)
    plt.close(fig)


def fig_readability(read: pd.DataFrame, lab: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

    def curve(a, colour, ls, label, **sel):
        s = read
        for k, v in sel.items():
            s = s[np.isclose(s[k], v)] if k == "dose_nM" else s[s[k] == v]
        a.plot(s["hours"], s["auroc"], ls, color=colour, marker="o", ms=3, label=label)
        a.fill_between(s["hours"], s["ci_low"], s["ci_high"], color=colour, alpha=0.12)

    a = ax[0]
    curve(a, "tab:purple", "-", "19.53 nM (31% labelled dying)", protocol="trained on all doses", dose_nM=19.53)
    curve(a, "grey", "-", "untreated wells (11% labelled dying)", protocol="trained on all doses", dose_nM=0.0)
    a.set_title("A. The fate label, with and without drug", fontsize=10, loc="left")
    a = ax[1]
    names = {"annexin_positive": "AnnexinV positive", "condensed": "nucleus condensed", "green": "cytoplasm turned green"}
    for target, name in names.items():
        curve(a, COLOURS[name], "-", f"{name}, handcrafted", protocol="trained within the dose", target=target,
              features="handcrafted")
        curve(a, COLOURS[name], ":", f"{name}, DINOv2", protocol="trained within the dose", target=target,
              features="DINOv2")
    a.set_title(f"B. Each marker as the target, {MIXED_DOSE:g} nM only", fontsize=10, loc="left")
    for a in ax[:2]:
        a.axhline(0.5, color="k", lw=0.6)
        a.set_ylim(0.38, 1.0)
        a.set_xlabel("hours after drug (the one frame the classifier sees)")
        a.set_ylabel("AUROC, held-out fields")
        a.legend(fontsize=7.5, loc="upper left")
    a = ax[2]
    s = lab[lab["live_condensed"] & lab["green"] & lab["dose_nM"].isin([39.06, 78.13, 156.25])]
    lag = (s["onset_condensed"] - s["onset_green"]) / 2
    a.hist(lag, bins=np.arange(-3.25, 5.5, 0.5), color="tab:green", alpha=0.7)
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel("hours from the colour shift to nuclear condensation\n(positive: the cytoplasm turned green first)")
    a.set_ylabel("cells")
    a.set_title(f"C. Order of the two live markers, 39 to 156 nM (n = {len(s)})", fontsize=10, loc="left")
    fig.suptitle("How early is fate readable from one frame? 95% CIs resample fields", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_readability.png", dpi=140)
    plt.close(fig)


def main() -> None:
    lab, wide, nuclei = load()
    lab = markers(lab, wide)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    fn = funnel(lab, nuclei)
    fn.to_csv(TAB / "label_audit_funnel.csv", index=False)
    print(fn.to_string(index=False))
    dose = dose_table(lab)
    dose.to_csv(TAB / "label_audit_dose.csv", index=False)
    print(dose[["dose_nM", "n", "annexin_positive", "condensed", "green", "annexin_only",
                "background_share_of_annexin_only", "share_ci_low", "share_ci_high"]].round(2).to_string(index=False))
    model = lab[lab["dose_nM"].isin(MODEL_DOSES)]
    ctrl = lab[lab["dose_nM"] == 0]
    base = (ctrl["group"] == "dies, AnnexinV only").mean()
    pooled = background_share((model["group"] == "dies, AnnexinV only").sum(), (~model["condensed"]).sum(), base)
    print(f"model doses pooled: background share of the AnnexinV-only label {pooled:.2f}")
    agree = (lab["condensed"] == lab["live_condensed"]).mean()
    print(f"live-movie condensation call agrees with the label's (terminal image) for {agree:.1%} of cells")
    mk = marker_table(lab)
    mk.to_csv(TAB / "label_audit_markers.csv", index=False)
    print(mk.round(2).to_string(index=False))
    onset = onset_table(lab)
    onset.to_csv(TAB / "label_audit_onset.csv", index=False)
    print(onset[onset["frame"].isin([5, 9, 13])].pivot_table(index=["marker", "dose_nM"], columns="hours",
                                                             values="fraction").round(2).to_string())
    fig_labels(dose, mk, onset)

    hand = np.log1p(np.stack([wide[c] for c in HAND], axis=-1).clip(0))
    hand = np.where(np.isfinite(hand), hand, np.nanmedian(hand, axis=0, keepdims=True))
    tracks = np.load(P / "crops_tracks.npy")
    emb = np.load(P / "dinov2_base_embeddings.npy")
    emb = emb[pd.Series(np.arange(len(tracks)), index=tracks).loc[lab["track"]].to_numpy()]
    read = readability(lab, {"handcrafted": hand, "DINOv2": emb})
    read.to_csv(TAB / "label_audit_readability.csv", index=False)
    print(read[read["frame"].isin([1, 5, 7, 9, 13])].pivot_table(index=["protocol", "target", "features", "dose_nM"],
                                                                 columns="hours", values="auroc").round(2).to_string())
    check = definition_check(lab, wide)
    check.to_csv(TAB / "label_audit_definition.csv", index=False)
    print(check.round(2).to_string(index=False))
    lag = lag_table(lab)
    lag.to_csv(TAB / "label_audit_lag.csv", index=False)
    print(lag.round(2).to_string(index=False))
    fig_readability(read, lab)

if __name__ == "__main__":
    main()
