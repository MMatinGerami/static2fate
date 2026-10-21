"""Does a video encoder see fate earlier than a single frame?

For the cells at the four two-fate doses (4.88, 9.77, 19.53, 39.06 nM), embed the clip of frames
1..k with frozen V-JEPA 2 (ViT-L, mean-pooled tokens, native 128 px) and compare, at the same
k, with three controls on the same cells and the same leave-one-field-out logistic regression:
  dinov2_frame   DINOv2-base CLS of frame k alone (the single-frame baseline)
  dinov2_mean    DINOv2-base CLS averaged over frames 1..k (as many frames as V-JEPA sees,
                 but no temporal model), so a V-JEPA gain cannot just be "more frames"
  hand_frame     the 11 handcrafted nucleus measurements at frame k
AUROC is computed within each dose and averaged over the four doses. The interval is a
bootstrap over fields within dose (1,000 resamples, seed 0) of the out-of-fold scores.

Usage: uv run scripts/07_vjepa.py --frames 2 4 6 8 [--device mps]
Writes results/tables/vjepa_vs_frame.csv, data/processed/vjepa_k{k}.npy, and the run metadata
next to the table.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import VJEPA2Model

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
DOSES = [4.88, 9.77, 19.53, 39.06]
MODEL = "facebook/vjepa2-vitl-fpc64-256"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
HAND = ["area", "eccentricity", "solidity", "hoechst_nuc_mean", "hoechst_ring_mean", "cl488a_nuc_mean",
        "cl488a_ring_mean", "cl488b_nuc_mean", "cl488b_ring_mean", "cl561_nuc_mean", "cl561_ring_mean"]


def embed_clips(model, x: np.ndarray, k: int, device: str, batch: int = 32) -> np.ndarray:
    """x uint8 (n, 13, 3, 128, 128); returns (n, 1024) mean-pooled V-JEPA 2 tokens of frames 1..k."""
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            v = torch.from_numpy(np.asarray(x[i : i + batch, :k])).float() / 255
            v = ((v - MEAN) / STD).to(device)
            h = model(pixel_values_videos=v, skip_predictor=True).last_hidden_state
            out.append(h.mean(1).float().cpu().numpy())
    return np.concatenate(out)


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    s = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(X[tr], y[tr])
        s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def within_dose_auroc(y, s, dose, field, rng=None) -> float:
    """Mean over doses of the within-dose AUROC; with rng, resample fields within each dose."""
    vals = []
    for d in DOSES:
        m = np.flatnonzero(dose == d)
        if rng is not None:
            fields = np.unique(field[m])
            pick = rng.choice(fields, size=len(fields), replace=True)
            m = np.concatenate([m[field[m] == f] for f in pick])
        if 0 < y[m].sum() < len(m):
            vals.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    labels = pd.read_parquet(P / "fate_labels.parquet").set_index("track")
    all_tracks = np.load(P / "crops_tracks.npy")
    crops = np.load(P / "crops.npy", mmap_mode="r")
    keep_crop = np.isin(all_tracks, labels.index[labels["dose_nM"].isin(DOSES)])
    idx = np.flatnonzero(keep_crop)
    tracks = all_tracks[idx]
    lab = labels.loc[tracks]
    y = lab["apoptotic"].to_numpy().astype(int)
    dose = lab["dose_nM"].to_numpy()
    field = lab["well_fov"].to_numpy()
    print(f"{len(y)} cells at doses {DOSES}, {y.mean():.2f} apoptotic, {len(np.unique(field))} fields")

    # the DINOv2 cache from 06_baseline.py covers the labelled tracks in crop order
    lab_tracks = all_tracks[np.isin(all_tracks, labels.index)]
    dino = np.load(P / "dinov2_base_embeddings.npy")
    pos = pd.Index(lab_tracks).get_indexer(tracks)
    assert (pos >= 0).all() and len(lab_tracks) == len(dino), "DINOv2 cache does not match the crops"
    dino = dino[pos]

    meas = pd.concat([pd.read_parquet(p) for p in glob.glob(str(P / "tracks/*.parquet"))], ignore_index=True)
    meas["track"] = meas["well_fov"] + "_" + meas["particle"].astype(str)
    meas = meas[meas["track"].isin(tracks)]

    model = VJEPA2Model.from_pretrained(MODEL).eval().to(args.device)
    rng = np.random.default_rng(0)
    rows = []
    t0 = time.time()
    for k in args.frames:
        cache = P / f"vjepa_k{k}.npy"
        if cache.exists():
            vj = np.load(cache)
        else:
            vj = embed_clips(model, crops[idx], k, args.device)
            np.save(cache, vj)
        hand = meas[meas["frame"] == k].set_index("track").reindex(tracks)[HAND]
        hand = np.log1p(hand.fillna(hand.median()).to_numpy().clip(0))
        feats = {
            "vjepa_clip": vj,
            "dinov2_mean": dino[:, :k].mean(1),
            "dinov2_frame": dino[:, k - 1],
            "hand_frame": hand,
        }
        scores = {name: oof_scores(X, y, field) for name, X in feats.items()}
        for name, s in scores.items():
            boot = [within_dose_auroc(y, s, dose, field, np.random.default_rng(b)) for b in range(1000)]
            diff = [within_dose_auroc(y, scores["vjepa_clip"], dose, field, np.random.default_rng(b))
                    - within_dose_auroc(y, s, dose, field, np.random.default_rng(b)) for b in range(1000)]
            rows.append({
                "frames": k, "minutes_last_frame": (k - 1) * 30, "features": name,
                "auroc": within_dose_auroc(y, s, dose, field),
                "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                "vjepa_minus_this": float(np.mean(diff)) if name != "vjepa_clip" else 0.0,
                "diff_ci_low": np.percentile(diff, 2.5) if name != "vjepa_clip" else 0.0,
                "diff_ci_high": np.percentile(diff, 97.5) if name != "vjepa_clip" else 0.0,
                **{f"dose_{d}": roc_auc_score(y[dose == d], s[dose == d]) for d in DOSES},
            })
        sub = pd.DataFrame(rows)
        sub = sub[sub["frames"] == k]
        print(f"k={k} ({(k - 1) * 30} min): " + " | ".join(
            f"{r.features} {r.auroc:.3f} [{r.ci_low:.3f},{r.ci_high:.3f}]" for r in sub.itertuples()), flush=True)

    out = ROOT / "results/tables/vjepa_vs_frame.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    meta = {"model": MODEL, "input_px": 128, "pooling": "mean over tokens", "device": args.device,
            "doses": DOSES, "n_cells": int(len(y)), "n_fields": int(len(np.unique(field))),
            "classifier": "StandardScaler + LogisticRegression(C=0.1), leave-one-field-out",
            "bootstrap": "1000 resamples of fields within dose, seeds 0..999",
            "minutes": round((time.time() - t0) / 60, 1), "torch": torch.__version__}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(df[["frames", "features", "auroc", "ci_low", "ci_high", "vjepa_minus_this", "diff_ci_low", "diff_ci_high"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
