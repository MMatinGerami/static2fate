"""Comparisons of generated futures with the real future of the same held-out cell.

Distances are Euclidean in standardised readouts (static2fate.readouts). The unit of
resampling for confidence intervals is the imaging field, since cells in one field share
focus, density and illumination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


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


def rank_of_own(queries: np.ndarray, candidates: np.ndarray, own: np.ndarray):
    """Where does each query's own cell rank among real candidates, by Euclidean distance?

    queries (n, m, d): m forecasts per cell; candidates (K, d): one real image per candidate
    cell at the target frame, including the n query cells' own; own (n,): row of each query
    cell in candidates. Returns percentile ranks (n, m) in [0, 1], 0 when the own cell is the
    nearest and 0.5 at chance, and top1 (n, m) booleans. Ties count against the query.
    """
    n, m, d = queries.shape
    q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).reshape(n * m, d)
    c = torch.from_numpy(np.ascontiguousarray(candidates, dtype=np.float32))
    dist = torch.cdist(q, c).reshape(n, m, -1)
    d_own = dist[torch.arange(n)[:, None], torch.arange(m)[None, :], torch.as_tensor(own)[:, None]]
    closer = (dist <= d_own[:, :, None]).sum(-1) - 1  # candidates at least as close, minus own
    pct = closer.numpy() / max(len(candidates) - 1, 1)
    return pct, closer.numpy() == 0
