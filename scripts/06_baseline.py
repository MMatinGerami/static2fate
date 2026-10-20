"""Single-frame fate baseline: can a frozen encoder plus logistic regression tell, from one live
frame at time t, whether the cell will be AnnexinV-positive at 6 h?

Two feature sets per frame: (a) frozen DINOv2-base CLS embedding of the 128 px patch, (b) the
handcrafted nucleus measurements from the tracks parquet (area, eccentricity, solidity, channel
means in nucleus and ring). Evaluation is leave-one-field-out over the 12 fields; AUROC on the
pooled out-of-fold scores, overall and within each dose. 'dose only' is the AUROC of the dose
value itself, the floor any pooled number has to clear. Writes results/tables/baseline_auroc.csv
and results/figures/fig2_single_frame_baseline.png.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
MODEL = "facebook/dinov2-base"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HAND = ["area", "eccentricity", "solidity", "hoechst_nuc_mean", "hoechst_ring_mean", "cl488a_nuc_mean",
        "cl488a_ring_mean", "cl488b_nuc_mean", "cl488b_ring_mean", "cl561_nuc_mean", "cl561_ring_mean"]


def embed(x: np.ndarray, device: str = "mps", batch: int = 64) -> np.ndarray:
    """x uint8 (n, 3, 128, 128) -> (n, 768) CLS embeddings."""
    model = AutoModel.from_pretrained(MODEL).eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            b = torch.from_numpy(x[i : i + batch]).float() / 255
            b = torch.nn.functional.interpolate(b, size=224, mode="bilinear", align_corners=False)
            b = ((b - MEAN) / STD).to(device)
            out.append(model(pixel_values=b).pooler_output.float().cpu().numpy())
    return np.concatenate(out)


def lofo_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, dose: np.ndarray) -> dict:
    """Leave-one-field-out logistic regression; AUROC overall and within each dose."""
    scores = np.zeros(len(y))
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr])
        scores[te] = clf.predict_proba(X[te])[:, 1]
    res = {"all": roc_auc_score(y, scores)}
    for d in np.unique(dose):
        m = dose == d
        res[f"dose_{d}"] = roc_auc_score(y[m], scores[m]) if 0 < y[m].sum() < m.sum() else np.nan
    return res


def main() -> None:
    labels = pd.read_parquet(P / "fate_labels.parquet").set_index("track")
    all_tracks = np.load(P / "crops_tracks.npy")
    keep = np.isin(all_tracks, labels.index)
    tracks = all_tracks[keep]
    x = np.load(P / "crops.npy", mmap_mode="r")
    labels = labels.loc[tracks]
    y = labels["apoptotic"].to_numpy().astype(int)
    groups = labels["well_fov"].to_numpy()
    dose = labels["dose_nM"].to_numpy()
    print(f"{len(y)} tracks, {y.mean():.2f} apoptotic; dose-only AUROC {roc_auc_score(y, dose):.3f}")

    meas = pd.concat([pd.read_parquet(p) for p in glob.glob(str(P / "tracks/*.parquet"))], ignore_index=True)
    meas["track"] = meas["well_fov"] + "_" + meas["particle"].astype(str)
    meas = meas[meas["track"].isin(tracks)]

    rows = []
    cache = P / "dinov2_base_embeddings.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        idx = np.flatnonzero(keep)
        emb = np.stack([embed(np.asarray(x[idx, t])) for t in range(13)], axis=1)  # (n, 13, 768)
        np.save(cache, emb)
    for t in range(1, 14):
        r = lofo_auroc(emb[:, t - 1], y, groups, dose)
        rows.append({"frame": t, "minutes": (t - 1) * 30, "features": "dinov2_base", **r})
        m = meas[meas["frame"] == t].set_index("track").reindex(tracks)[HAND]
        m = m.fillna(m.median())
        r = lofo_auroc(np.log1p(m.to_numpy().clip(0)), y, groups, dose)
        rows.append({"frame": t, "minutes": (t - 1) * 30, "features": "nucleus_measurements", **r})
        print(f"frame {t:2d} ({(t - 1) * 30:3d} min): dinov2 {rows[-2]['all']:.3f} | hand {rows[-1]['all']:.3f} | "
              f"within 19.53 nM: dinov2 {rows[-2].get('dose_19.53', np.nan):.3f} hand {rows[-1].get('dose_19.53', np.nan):.3f}")
    df = pd.DataFrame(rows)
    (ROOT / "results/tables").mkdir(exist_ok=True, parents=True)
    df.to_csv(ROOT / "results/tables/baseline_auroc.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for feat, ls in [("dinov2_base", "-"), ("nucleus_measurements", "--")]:
        s = df[df["features"] == feat]
        ax[0].plot(s["minutes"], s["all"], ls, marker="o", label=feat)
        for d, col in [(9.77, "C1"), (19.53, "C2"), (78.13, "C3")]:
            ax[1].plot(s["minutes"], s[f"dose_{d}"], ls, marker="o", color=col, label=f"{feat}, {d} nM")
    ax[0].axhline(roc_auc_score(y, dose), color="grey", ls=":", label="dose only")
    ax[0].set_title("all doses pooled (leave-one-field-out)")
    ax[1].set_title("within one dose")
    for a in ax:
        a.set_xlabel("minutes after staurosporine (frame used)")
        a.set_ylabel("AUROC for AnnexinV-positive at 6 h")
        a.set_ylim(0.4, 1)
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(ROOT / "results/figures/fig2_single_frame_baseline.png", dpi=90)


if __name__ == "__main__":
    main()
