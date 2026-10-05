"""Helpers for the inspection-reliability notebooks (06, 08-11)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from holdout import SEED


def onehot(codes) -> tuple[sp.csr_matrix, pd.Index]:
    c, levels = pd.factorize(codes)
    return sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)), shape=(len(c), len(levels))), levels


def two_way_effects(rows: pd.DataFrame, y: str = "score", unit: str = "address",
                    rater: str = "inspector", period: str = "fiscal_year") -> dict[str, pd.Series]:
    """Least-squares fit of y = unit + rater + period + error on sparse dummies.

    Returns centred rater and period effects and the unit effects. Rater effects are
    identified from units seen by more than one rater.
    """
    v = rows[y].to_numpy(float)
    v = v - v.mean()
    U, u_lv = onehot(rows[unit])
    R, r_lv = onehot(rows[rater])
    P, p_lv = onehot(rows[period])
    beta = lsqr(sp.hstack([U, R, P]).tocsr(), v, atol=1e-10, btol=1e-10, iter_lim=50_000)[0]
    nU, nR = len(u_lv), len(r_lv)
    rater_eff = pd.Series(beta[nU:nU + nR], index=r_lv)
    period_eff = pd.Series(beta[nU + nR:], index=p_lv)
    unit_eff = pd.Series(beta[:nU], index=u_lv)
    return {
        "rater": rater_eff - rater_eff.mean(),
        "period": period_eff - period_eff.mean(),
        "unit": unit_eff + rater_eff.mean() + period_eff.mean(),
    }


def consecutive_pairs(insp: pd.DataFrame, different_fy: bool = True) -> pd.DataFrame:
    """Each inspection paired with the same property's previous one."""
    s = insp.sort_values(["address", "date_opened", "sr_number"])
    g = s.groupby("address")
    p = s.assign(**{f"prev_{c}": g[c].shift() for c in ["score", "bottom10", "inspector", "fiscal_year", "date_opened"]})
    p = p.dropna(subset=["prev_score"])
    if different_fy:
        p = p[p.fiscal_year != p.prev_fiscal_year]
    p = p[(p.inspector >= 0) & (p.prev_inspector >= 0)].copy()
    p["same_inspector"] = p.inspector == p.prev_inspector
    p["gap_years"] = (p.date_opened - p.prev_date_opened).dt.days / 365.25
    return p


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], cluster: str = "address",
                      n: int = 1000, seed: int = SEED) -> tuple[float, float, float]:
    """Point estimate and 95% interval, resampling clusters (properties) with replacement."""
    rng = np.random.default_rng(seed)
    groups = df.groupby(cluster).indices
    keys = np.array(list(groups))
    idx = [groups[k] for k in keys]
    est = stat(df)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([idx[i] for i in pick])
        draws.append(stat(df.iloc[rows]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return est, lo, hi


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary ratings."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


TIER_DTA = "tier_observations.dta"
TIER_CACHE = "tier_observations.parquet"


def tier_observations(data=None) -> pd.DataFrame:
    """Observation rows with the pipeline's rubric-aligned tier (Step 9A), from
    `tier_observations.dta` (shared drive `_data/_processed/`), cached as parquet."""
    from holdout import DATA

    data = data or DATA
    cache = data / TIER_CACHE
    if not cache.exists():
        cols = ["sr_number", "subject", "comments", "life_hazard", "tier", "score", "fiscal_year"]
        parts = []
        with pd.read_stata(data / TIER_DTA, iterator=True, chunksize=20_000, columns=cols) as r:
            parts.extend(r)
        pd.concat(parts, ignore_index=True).to_parquet(cache, index=False)
    obs = pd.read_parquet(cache)
    obs["sr_number"] = obs.sr_number.astype(str).str.strip()
    return obs
