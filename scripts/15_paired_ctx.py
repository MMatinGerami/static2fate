"""Paired comparison on the same cells: context model (0 h frame) minus inversion model.

Both start from the cell's 0 h image and are scored against the same copy-0h baseline, so the
per-cell difference in "beats copy" is paired. 95% CIs resample the 12 held-out fields.
Reads results/flow_ctx/eval_cells.csv (14_eval_ctx.py), writes
results/tables/ctx_vs_inversion_paired.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

d = pd.read_csv(ROOT / "results/flow_ctx/eval_cells.csv")
w = d.pivot_table(index=["track", "field", "group", "hours"], columns="method", values="beats_copy")
w = w.dropna(subset=["context 0 h", "inversion 0 h"]).reset_index()
w["diff"] = w["context 0 h"] - w["inversion 0 h"]
rng = np.random.default_rng(0)
fields = w.field.unique()
rows = []
for (g, h), s in w.groupby(["group", "hours"]):
    per = s.groupby("field")["diff"].agg(["sum", "size"]).reindex(fields, fill_value=0)
    bs = []
    for _ in range(2000):
        p = per.iloc[rng.integers(len(fields), size=len(fields))]
        bs.append(p["sum"].sum() / p["size"].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append({"group": g, "hours": h, "context_minus_inversion": s["diff"].mean(),
                 "ci_low": lo, "ci_high": hi, "n": len(s)})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/tables/ctx_vs_inversion_paired.csv", index=False)
print(out.round(2).to_string(index=False))
