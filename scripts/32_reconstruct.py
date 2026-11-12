"""How much of a held-out cell survives a trip through the autoencoder?

The autoencoder's reconstruction is the ceiling of any model that generates in its latent: if
the probe tells reconstructions from real cells, or the readouts of a reconstruction do not
track the cell's own, latent generation inherits that. For the 816 held-out cells of the
two-fate doses, frames 6, 8, 10 and 12 (the frames the scoring chain reads) are encoded with
the posterior mean and decoded. Written in the layout of 19_generate_samples.py with one
"sample" per frame, so 20_feature_score.py and 21_real_vs_generated.py score the
reconstruction like a model (tag it, say, _recon). Also reported directly, per fate group:
PSNR, L1, VGG16 perceptual distance, and the Spearman correlation across cells between the
readouts (nucleus area, Hoechst, green, red at 64 px) of the real frame and of its
reconstruction.

Usage: uv run scripts/32_reconstruct.py [ae dir, default results/autoencoder] [name]
Writes <ae dir>/recon/samples.npy (cells, 4, 1, 3, 128, 128) uint8 and samples_tracks.npy,
results/tables/autoencoder_<name>.csv and results/figures/supporting/fig16_recon_<name>.png (copies of
both in <ae dir>/recon/).
"""

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.autoencoder import AutoEncoder  # noqa: E402
from static2fate.evaluate import group_of  # noqa: E402
from static2fate.perceptual import Perceptual  # noqa: E402
from static2fate.readouts import FEATURES, cell_readouts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FUTURE = [6, 8, 10, 12]


@torch.no_grad()
def reconstruct(ae: AutoEncoder, x: torch.Tensor, device: str, batch: int = 32) -> torch.Tensor:
    """x (n, 3, s, s) in [-1, 1] -> reconstruction from the posterior mean, clamped to [-1, 1]."""
    out = torch.empty_like(x)
    for b in range(0, len(x), batch):
        mean, _ = ae.encode(x[b:b + batch].to(device))
        out[b:b + batch] = ae.decode(mean).float().clamp(-1, 1).cpu()
    return out


def main(mdir: Path, name: str) -> None:
    train = importlib.import_module("09_train_flow")
    run = json.loads((mdir / "run.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)
    ae = AutoEncoder(ch=run["ch"], z=run["z"]).to(device)
    ae.load_state_dict(torch.load(mdir / "model.pt", map_location=device, weights_only=True))
    ae.eval()
    size = run["size"]
    x, fate, tracks = train.load("test", size, frames=FUTURE)
    lab = pd.read_parquet(train.P / "fate_labels.parquet").set_index("track")
    group = np.array([group_of(int(a), bool(lab.loc[t, "condensed"])) for a, t in zip(fate, tracks)])
    n = len(x)
    flat = x.flatten(0, 1)
    recon = reconstruct(ae, flat, device)
    print(f"reconstructed {n} cells x {len(FUTURE)} frames at {size} px on {device}", flush=True)

    rdir = mdir / "recon"
    rdir.mkdir(exist_ok=True)
    u8 = ((recon + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).view(n, len(FUTURE), 1, 3, size, size)
    np.save(rdir / "samples.npy", u8.numpy())
    np.save(rdir / "samples_tracks.npy", tracks)

    perceptual = Perceptual().to(device)
    with torch.no_grad():
        per = torch.cat([perceptual(recon[b:b + 64].to(device), flat[b:b + 64].to(device)).cpu()
                         for b in range(0, len(flat), 64)])
    mse = ((recon - flat) ** 2).mean((1, 2, 3))
    psnr = 10 * torch.log10(4 / mse)  # range of [-1, 1] is 2
    l1 = (recon - flat).abs().mean((1, 2, 3))
    small = (F.avg_pool2d(flat, size // 64), F.avg_pool2d(recon, size // 64)) if size != 64 else (flat, recon)
    ro_real, ro_rec = cell_readouts(small[0]), cell_readouts(small[1])
    g4 = np.repeat(group, len(FUTURE))
    rows = []
    for grp in ["survives", "dies, AnnexinV only", "dies, condensed", "all"]:
        m = np.ones(len(g4), bool) if grp == "all" else g4 == grp
        row = {"name": name, "group": grp, "images": int(m.sum()), "psnr": psnr[m].mean().item(),
               "l1": l1[m].mean().item(), "perceptual": per[m].mean().item()}
        for k, f in enumerate(FEATURES):
            ok = m & np.isfinite(ro_real[:, k]) & np.isfinite(ro_rec[:, k])
            row[f"spearman_{f}"] = spearmanr(ro_real[ok, k], ro_rec[ok, k]).correlation
        ok = m & np.isfinite(ro_real[:, 0]) & np.isfinite(ro_rec[:, 0])
        row["nuc_area_ratio"] = float(np.median(ro_rec[ok, 0] / ro_real[ok, 0]))  # reconstructed / real
        rows.append(row)
    tab = pd.DataFrame(rows)
    for d in (ROOT / "results/tables", rdir):  # the copy next to the model travels with Kaggle outputs
        d.mkdir(parents=True, exist_ok=True)
        tab.to_csv(d / f"autoencoder_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))
    (ROOT / "results/figures/supporting").mkdir(parents=True, exist_ok=True)
    figure(x, recon.view_as(x), group, name, [ROOT / "results/figures/supporting", rdir])


def figure(x, recon, group, name, dirs):
    """Real against reconstructed 6 h frame for two cells per fate group."""
    groups = ["survives", "dies, AnnexinV only", "dies, condensed"]
    rng = np.random.default_rng(0)
    picks = [i for g in groups for i in rng.choice(np.flatnonzero(group == g), 2, replace=False)]
    fig, axes = plt.subplots(2, len(picks), figsize=(2 * len(picks), 4.4))
    for j, i in enumerate(picks):
        for k, (im, lab) in enumerate([(x[i, -1], "real"), (recon[i, -1], "reconstruction")]):
            a = (im.numpy() + 1) / 2
            axes[k, j].imshow(np.dstack([a[2], a[1], a[0]]).clip(0, 1))
            axes[k, j].set_xticks([])
            axes[k, j].set_yticks([])
            if j == 0:
                axes[k, j].set_ylabel(lab)
        axes[0, j].set_title(group[i].replace(", ", ",\n"), fontsize=8)
    fig.suptitle(f"Held-out cells at 6 h through the autoencoder ({name})", fontsize=10)
    fig.tight_layout()
    for d in dirs:
        fig.savefig(d / f"fig16_recon_{name}.png", dpi=130)


if __name__ == "__main__":
    mdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/autoencoder"
    main(mdir, sys.argv[2] if len(sys.argv) > 2 else mdir.name)
