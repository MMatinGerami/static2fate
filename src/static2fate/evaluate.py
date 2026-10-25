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


def energy_score(samples: np.ndarray, y: np.ndarray) -> float:
    """Energy score of an ensemble forecast (m, d) for one observation y (d,); lower is better.

    ES = mean_i |x_i - y| - 0.5 mean_{i != j} |x_i - x_j| (Gneiting and Raftery 2007,
    doi:10.1198/016214506000001437). Averaging the second term over distinct pairs only makes
    it an unbiased estimate of the score of the distribution the samples come from, whatever
    their number. With one sample it is the plain distance, so a deterministic forecast such as
    copying a frame is scored on the same scale. Samples with missing readouts are dropped;
    NaN if none is left.
    """
    x = samples[~np.isnan(samples).any(axis=1)]
    if len(x) == 0 or np.isnan(y).any():
        return np.nan
    first = np.linalg.norm(x - y, axis=1).mean()
    if len(x) == 1:
        return float(first)
    second = np.linalg.norm(x[:, None] - x[None], axis=2).sum() / (len(x) * (len(x) - 1))
    return float(first - 0.5 * second)
