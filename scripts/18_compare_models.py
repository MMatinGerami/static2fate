"""Compare the context models trained so far on the same held-out cells.

  64 px, 1 h    results/flow_ctx            Apple M5 Pro (MPS), 7,162 steps
  64 px, 8 h    results/kaggle/ctx64_long   Kaggle T4, 98,298 steps, seed 1
  128 px, 8 h   results/kaggle/ctx128       Kaggle T4, 50,999 steps, batch 32; evaluated in two
                                            halves (ctx128_a, ctx128_b), joined here

All were scored by 17_energy_score.py (8 samples per cell, energy score on the four readouts,
real and generated 128 px images averaged down to 64 px before the readout). Differences are
paired by cell and frame; 95% CIs resample the 12 held-out fields. Models whose files are
missing are skipped. Writes results/tables/model_comparison.csv and
results/figures/fig11_model_comparison.png.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from static2fate.evaluate import field_bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "64 px, 1 h": [ROOT / "results/flow_ctx"],
    "64 px, 8 h": [ROOT / "results/kaggle/ctx64_long"],
    "128 px, 8 h": [ROOT / "results/kaggle/ctx128_a", ROOT / "results/kaggle/ctx128_b"],
}
KEY = ["track", "field", "group", "hours"]


def main() -> None:
    cells, logs = {}, {}
    for name, dirs in MODELS.items():
        files = [d / "energy_cells.csv" for d in dirs if (d / "energy_cells.csv").exists()]
        if len(files) < len(dirs):
            print(f"skipping {name}: {len(files)} of {len(dirs)} files")
            continue
        cells[name] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        logs[name] = pd.read_csv(dirs[0] / "train_log.csv")
    base = "64 px, 1 h"
    wide = cells[base][KEY + ["es_model", "es_drift", "d_copy"]].rename(columns={"es_model": f"es {base}"})
    for name, d in cells.items():
        if name != base:
            wide = wide.merge(d[KEY + ["es_model"]].rename(columns={"es_model": f"es {name}"}), on=KEY)
    print(f"cells in common: {wide.track.nunique()}")
    cols = []
    for name in cells:
        wide[f"{name} minus copy"] = wide[f"es {name}"] - wide.d_copy
        wide[f"{name} minus drift"] = wide[f"es {name}"] - wide.es_drift
        cols += [f"{name} minus copy", f"{name} minus drift"]
        if name != base:
            wide[f"{name} minus {base}"] = wide[f"es {name}"] - wide[f"es {base}"]
            cols.append(f"{name} minus {base}")
    tab = pd.concat([field_bootstrap(wide, c, ["group", "hours"]).assign(measure=c) for c in cols])
    tab.to_csv(ROOT / "results/tables/model_comparison.csv", index=False)
    show = tab.pivot_table(index=["measure", "group"], columns="hours", values="estimate").round(2)
    print(show.to_string())

    groups = ["dies, condensed", "dies, AnnexinV only", "survives"]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    for a, grp in zip(ax[:3], groups):
        for k, name in enumerate(cells):
            s = tab[(tab.measure == f"{name} minus drift") & (tab.group == grp)].sort_values("hours")
            a.plot(s.hours, s.estimate, "-o", color=f"C{k}", label=name, ms=4)
            a.fill_between(s.hours, s.ci_low, s.ci_high, color=f"C{k}", alpha=0.12)
        a.axhline(0, color="grey", ls=":")
        a.set_title(f"{grp} (n={int(tab[tab.group == grp].n.max())})", fontsize=9)
        a.set_xlabel("hours after staurosporine")
        a.set_xticks([3, 4, 5, 6])
    ax[0].set_ylabel("energy score minus\n\"copy + typical change\" baseline")
    ax[0].legend(fontsize=7)
    for k, (name, log) in enumerate(logs.items()):
        sm = log.loss.rolling(10, min_periods=1).mean()
        ax[3].plot(log.step, sm, color=f"C{k}", label=name, lw=1)
    ax[3].set_xscale("log")
    ax[3].set_xlabel("training step")
    ax[3].set_ylabel("flow matching loss (smoothed)")
    ax[3].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(ROOT / "results/figures/fig11_model_comparison.png", dpi=110)


if __name__ == "__main__":
    main()
