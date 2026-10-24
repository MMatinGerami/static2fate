"""Figure: does the true-fate branch, started from a real cell, beat copying its first frame?

Reads results/tables/invert_vs_copy_baseline.csv (fraction of held-out cells where the
generated future under the cell's true fate is closer to its real future than its own 0 h frame
is, with field-bootstrap 95% CIs) and writes results/figures/fig6_inversion_vs_copy.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

d = pd.read_csv(ROOT / "results/tables/invert_vs_copy_baseline.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for g, c in [("dies, condensed", "C3"), ("survives", "C0"), ("dies, AnnexinV only", "C1")]:
    s = d[d.group == g].sort_values("hours")
    ax.plot(s.hours, s.model_beats_copy, "-o", color=c, label=f"{g} (n={s.n.iloc[0]})")
    ax.fill_between(s.hours, s.ci_low, s.ci_high, color=c, alpha=0.15)
ax.axhline(0.5, color="grey", ls=":")
ax.set_xlabel("hours after staurosporine")
ax.set_ylabel("fraction of cells where model\nbeats copying the 0 h frame")
ax.set_ylim(0.2, 0.85)
ax.set_xticks([3, 4, 5, 6])
ax.legend(fontsize=7, loc="lower left")
ax.set_title("Started from each held-out cell's 0 h image", fontsize=9)
plt.tight_layout()
plt.savefig(ROOT / "results/figures/fig6_inversion_vs_copy.png", dpi=110)
print("ok")
