"""Comparisons of generated futures with the real future of the same held-out cell.

Distances are Euclidean in standardised readouts (static2fate.readouts). The unit of
resampling for confidence intervals is the imaging field, since cells in one field share
focus, density and illumination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_of(fate: int, condensed: bool) -> str:
    if fate == 0:
        return "survives"
    return "dies, condensed" if condensed else "dies, AnnexinV only"


def field_bootstrap(d: pd.DataFrame, col: str, by: list[str], n_boot: int = 2000, seed: int = 0):
    """Mean of a 0/1 column per group with 95% CIs from resampling fields with replacement."""
    rng = np.random.default_rng(seed)
    fields = d["field"].unique()
    out = []
    for key, s in d.groupby(by):
        per = s.groupby("field")[col].agg(["sum", "size"]).reindex(fields, fill_value=0)
        bs = np.empty(n_boot)
        for b in range(n_boot):
            pick = per.iloc[rng.integers(len(fields), size=len(fields))]
            bs[b] = pick["sum"].sum() / pick["size"].sum() if pick["size"].sum() else np.nan
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        out.append(dict(zip(by, key if isinstance(key, tuple) else (key,)),
                        estimate=s[col].mean(), ci_low=lo, ci_high=hi, n=len(s)))
    return pd.DataFrame(out)
