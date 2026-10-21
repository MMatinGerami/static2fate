"""Evaluate the flow-matching model against held-out wells.

1. Counterfactual grid: the same noise is integrated twice, once conditioned on "survives" and
   once on "dies", at frames 0, 2, ..., 12. If the model learned the biology, the two branches
   should look alike early and diverge later.
2. A simple quantitative check on two nuclear readouts that mark apoptosis, computed the same
   way on real held-out crops and on generated images: nucleus area (pixels above a fixed
   Hoechst threshold) and mean Hoechst inside the nucleus. For each frame and fate, compare the
   generated mean with the real mean, and compare the dies-minus-survives gap over time.

Writes results/flow/counterfactual_grid.png, results/flow/nuclear_readouts.csv and
results/flow/nuclear_readouts.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static2fate.flow import UNet, sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/flow"
FRAMES = list(range(0, 13, 2))
N_GEN = 128  # generated images per frame and fate
THR = -0.2  # Hoechst threshold in [-1, 1] units for the nucleus mask


def nuclear(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """x (n, 3, 64, 64) in [-1, 1]; channel 0 is Hoechst. Central nucleus only (16 px radius)."""
    h = x[:, 0]
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    centre = ((yy - 32) ** 2 + (xx - 32) ** 2 < 16**2).to(h.device)
    mask = (h > THR) & centre
    area = mask.flatten(1).sum(1).float()
    mean = (h * mask).flatten(1).sum(1) / area.clamp(min=1)
    return area.cpu().numpy(), mean.cpu().numpy()


def main() -> None:
    import importlib

    train = importlib.import_module("09_train_flow")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(OUT / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x, fate, _ = train.load("test")
    print(f"held-out cells {len(x)}, dies {fate.float().mean():.2f}")

    g = torch.Generator(device="cpu").manual_seed(0)
    x0 = torch.randn(6, 3, 64, 64, generator=g).to(device)
    fig, ax = plt.subplots(12, len(FRAMES), figsize=(1.3 * len(FRAMES), 12 * 1.3))
    for c, f in enumerate(FRAMES):
        for k in range(2):
            fr = torch.full((6,), f, device=device)
            imgs = sample(model, fr, torch.full((6,), k, device=device), x0=x0.clone())
            for i in range(6):
                im = ((imgs[i].cpu().numpy() + 1) / 2).clip(0, 1)
                a = ax[2 * i + k, c]
                a.imshow(np.dstack([im[2], im[1], im[0]]))
                a.set_xticks([])
                a.set_yticks([])
                if c == 0:
                    a.set_ylabel(f"#{i} {'dies' if k else 'survives'}", fontsize=7)
        ax[0, c].set_title(f"{f * 30 // 60} h {f * 30 % 60:02d}", fontsize=8)
    fig.suptitle("Same noise, two fates (rows in pairs)", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "counterfactual_grid.png", dpi=90)
    plt.close()

    rows = []
    for f in FRAMES:
        for k in (0, 1):
            real = x[fate == k][:, f]
            ra, rm = nuclear(real)
            fr = torch.full((N_GEN,), f, device=device)
            gen = sample(model, fr, torch.full((N_GEN,), k, device=device),
                         generator=torch.Generator(device="cpu").manual_seed(f * 10 + k) if device == "cpu" else None)
            ga, gm = nuclear(gen)
            rows.append({"frame": f, "minutes": f * 30, "fate": "dies" if k else "survives",
                         "real_area": ra.mean(), "gen_area": ga.mean(),
                         "real_hoechst": rm.mean(), "gen_hoechst": gm.mean(), "n_real": len(real)})
        print(rows[-2:], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "nuclear_readouts.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for j, (col, name) in enumerate([("area", "nucleus area (px)"), ("hoechst", "mean Hoechst in nucleus")]):
        for fate_name, colr in [("survives", "C0"), ("dies", "C3")]:
            s = df[df.fate == fate_name]
            ax[j].plot(s.minutes, s[f"real_{col}"], "-o", color=colr, label=f"real, {fate_name}")
            ax[j].plot(s.minutes, s[f"gen_{col}"], "--s", color=colr, label=f"generated, {fate_name}")
        ax[j].set_xlabel("minutes after staurosporine")
        ax[j].set_ylabel(name)
        ax[j].legend(fontsize=7)
    ax[0].set_title("held-out wells vs generated, by fate", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "nuclear_readouts.png", dpi=90)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
