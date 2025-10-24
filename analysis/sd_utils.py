from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Literal, Tuple, Optional, Dict

# Optional dependency for consistent SD tests (kept from your version)
import pysdtest

# ---------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------

Mode = Literal["strict", "pysdtest", "epsilon", "left_tail", "right_tail", "bootstrap_prob"]

def sd_compare(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    order: int = 1,
    mode: Mode = "strict",
    # Common knobs
    ngrid: int = 200,
    alpha: float = 0.10,           # significance for decision (when relevant)
    # For pysdtest (consistent tests under dependence)
    resampling: Literal["bootstrap", "subsampling", "paired bootstrap", "stationary_bootstrap"] = "bootstrap",
    nboot: int = 500,
    b1: Optional[int] = None,
    b2: Optional[int] = None,
    quiet: bool = True,
    # For epsilon near-dominance
    epsilon: float = 0.0,
    # For tail-restricted dominance
    alpha_tail: float = 0.20,      # compare only up to Q_alpha_tail (left tail)
    # For bootstrap probability dominance
    B: int = 1000,
    block_len: int = 10,
    random_state: Optional[int] = None,
) -> dict:
    """
    Compare two samples of lambda (λ) under various stochastic-dominance flavors.

    Returns a dict with:
      - 'decision': bool or None
      - 'stat': float (test statistic or gap measure)
      - 'pvalue': float or None
      - 'w_ij': float or None (Pr[i dominates j] when mode='bootstrap_prob')
      - 'mode': str (echo)
      - 'order': int (echo)
      - 'details': dict (aux info)
    """
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    rng = np.random.default_rng(random_state)

    if mode == "pysdtest":
        stat, pval = sd_stat_pvalue(
            x, y, s=order, ngrid=ngrid,
            resampling=resampling if resampling != "stationary_bootstrap" else "bootstrap",
            nboot=nboot, b1=b1, b2=b2, quiet=quiet
        )
        decision = bool(pval < alpha)
        return {
            "decision": decision, "stat": stat, "pvalue": pval,
            "w_ij": None, "mode": mode, "order": order, "details": {}
        }

    if mode == "strict":
        stat, violated = _sd_gap(x, y, order=order, ngrid=ngrid)
        decision = not violated  # no positive violation => x SD-dominates y
        return {
            "decision": decision, "stat": stat, "pvalue": None,
            "w_ij": None, "mode": mode, "order": order,
            "details": {"gap_definition": "sup positive envelope of F_x - F_y (or its integrals)"}
        }

    if mode == "epsilon":
        stat, violated = _sd_gap(x, y, order=order, ngrid=ngrid)
        decision = (stat <= epsilon + 1e-12)  # allow ε slack
        return {
            "decision": decision, "stat": stat, "pvalue": None,
            "w_ij": None, "mode": mode, "order": order,
            "details": {"epsilon": epsilon}
        }

    if mode == "left_tail":
        stat, violated = _sd_gap_left_tail(x, y, order=order, alpha_tail=alpha_tail, ngrid=ngrid)
        decision = not violated
        return {
            "decision": decision, "stat": stat, "pvalue": None,
            "w_ij": None, "mode": mode, "order": order,
            "details": {"alpha_tail": alpha_tail}
        }

    if mode == "right_tail":
        stat, violated = _sd_gap_right_tail(x, y, order=order, alpha_tail=alpha_tail, ngrid=ngrid)
        decision = not violated
        return {
            "decision": decision, "stat": stat, "pvalue": None,
            "w_ij": None, "mode": mode, "order": order,
            "details": {"alpha_tail": alpha_tail}
        }

    if mode == "bootstrap_prob":
        # Probability that x SD-dominates y (strict) under time-series dependence
        w = _dominance_probability(
            x, y, order=order, B=B, block_len=block_len, ngrid=ngrid,
            rng=rng
        )
        return {
            "decision": None, "stat": float("nan"), "pvalue": None,
            "w_ij": w, "mode": mode, "order": order,
            "details": {"B": B, "block_len": block_len}
        }

    raise ValueError(f"Unknown mode '{mode}'")

def sd_pairwise_network(
    data: pd.DataFrame,
    *,
    order: int = 1,
    mode: Mode = "strict",
    keep_threshold: float = 0.60,     # used for 'bootstrap_prob'
    **kwargs,
) -> dict:
    """
    Build an SD network from a wide DataFrame of λ (columns = assets).

    Returns:
      - 'A': adjacency (DataFrame, 1 if i dominates j, else 0 or NaN if not applicable)
      - 'W': weight matrix (DataFrame or None). For 'bootstrap_prob', W[i,j]=Pr(i dominates j).
      - 'meta': dict with parameters used.
    """
    cols = list(data.columns)
    n = len(cols)
    A = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    W = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float) if mode == "bootstrap_prob" else None

    for i, ci in enumerate(cols):
        xi = _to_1d_array(data[ci].dropna().values)
        if xi.size == 0:
            continue
        for j, cj in enumerate(cols):
            if i == j:
                continue
            yj = _to_1d_array(data[cj].dropna().values)
            if yj.size == 0:
                continue

            res = sd_compare(xi, yj, order=order, mode=mode, **kwargs)

            if mode == "bootstrap_prob":
                w = res["w_ij"]
                W.loc[ci, cj] = w
                A.loc[ci, cj] = int(w is not None and w >= keep_threshold)
            else:
                A.loc[ci, cj] = int(bool(res["decision"]))

    meta = dict(order=order, mode=mode, keep_threshold=keep_threshold, **kwargs)
    return {"A": A, "W": W, "meta": meta}

# ---------------------------------------------------------------------
# Your original PySDTest wrapper (retained and slightly hardened)
# ---------------------------------------------------------------------

def sd_stat_pvalue(
    x,
    y,
    s: int = 1,
    ngrid: int = 100,
    resampling: str = "bootstrap",
    nboot: int = 200,
    b1: int | None = None,
    b2: int | None = None,
    quiet: bool = True,
    debug: bool = False,
    method: str = "perm",     # kept for API compatibility
    B: int | None = None,     # alias for number of bootstraps
) -> Tuple[float, float]:
    """
    Wrapper for PySDTest's test_sd function (consistent SD testing).

    Parameters
    ----------
    x, y         : 1-D array-likes (NumPy arrays, pandas Series)
    s            : SD order (1=FSD, 2=SSD, 3=SD3)
    nboot        : number of bootstrap/subsampling draws
    ngrid        : number of grid points for ECDF evaluation
    resampling   : 'bootstrap' | 'subsampling' | 'paired bootstrap'
                   ('stationary_bootstrap' is routed to 'bootstrap' here;
                    the time dependence is handled by our own resampler when needed)
    method       : kept for API compatibility with upstream code
    B            : alias for nboot; overrides if provided
    b1, b2       : subsample sizes for sample1 and sample2 (only for subsampling)
    quiet        : if False, prints detailed output from PySDTest
    debug        : if True, prints debug info
    """
    if B is not None:
        nboot = int(B)

    x_arr = _to_1d_array(x)
    y_arr = _to_1d_array(y)

    if x_arr.size == 0 or y_arr.size == 0:
        if debug:
            print("[WARN] sd_stat_pvalue called with empty sample(s); returning NaN result")
        return float("nan"), float("nan")

    test = pysdtest.test_sd(
        sample1=x_arr,
        sample2=y_arr,
        ngrid=ngrid,
        s=s,
        resampling=resampling,
        b1=b1,
        b2=b2,
        nboot=nboot,
        quiet=quiet,
    )
    test.testing()

    stat = getattr(test, "statistic", None)
    pval = getattr(test, "pvalue", None)

    if stat is None or pval is None:
        result = getattr(test, "result", {})
        stat = result.get("statistic") or result.get("test_stat") or 0.0
        pval = result.get("pvalue") or result.get("p_value") or result.get("p_val") or 1.0

    if stat is None:
        stat = 0.0
    if pval is None:
        pval = 1.0

    if debug and pval >= 0.999:
        print("---- SD DEBUG ----")
        print(f"Order s={s}, nboot={nboot}, resampling={resampling}")
        print("x head:", x_arr[:5], "y head:", y_arr[:5])
        print("Result dict:", getattr(test, "result", {}))
        print("------------------")

    return float(stat), float(pval)

# ---------------------------------------------------------------------
# Core nonparametric building blocks (strict, ε, tail-restricted)
# ---------------------------------------------------------------------

def _to_1d_array(a) -> np.ndarray:
    if isinstance(a, pd.Series):
        a = a.values
    a = np.asarray(a)
    return a.ravel().astype(float)

def _grid_from_samples(x: np.ndarray, y: np.ndarray, ngrid: int) -> np.ndarray:
    lo = np.nanmin([np.nanmin(x), np.nanmin(y)])
    hi = np.nanmax([np.nanmax(x), np.nanmax(y)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        # fall back: jitter a tiny window around the point
        lo, hi = float(lo), float(hi)
        span = 1e-6 if hi == lo else (hi - lo)
        lo -= 0.5 * span
        hi += 0.5 * span
    return np.linspace(lo, hi, num=max(5, int(ngrid)))

def _ecdf_at(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x_sorted = np.sort(x)
    return np.searchsorted(x_sorted, grid, side="right") / max(1, x_sorted.size)

def _integrate_envelope(diff: np.ndarray, dx: float, order: int) -> np.ndarray:
    """
    For order=1: envelope = F_x - F_y
    For order=2: envelope = integral of (F_x - F_y)
    For order=3: integral of integral ...
    Returns envelope at grid points (cumulative integrals).
    """
    env = diff.copy()
    for _ in range(1, order):
        env = np.cumsum(env) * dx
    return env

def _sd_gap(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int = 1,
    ngrid: int = 200,
) -> Tuple[float, bool]:
    """
    Compute the *positive* envelope gap:
      gap = sup_z max( 0, E_s(z) ), where E_s is the s-th order envelope of F_x - F_y.
    If gap>0, SD is violated (x does NOT dominate y).
    """
    grid = _grid_from_samples(x, y, ngrid)
    Fx = _ecdf_at(x, grid)
    Fy = _ecdf_at(y, grid)
    diff = Fx - Fy
    dx = (grid[-1] - grid[0]) / (len(grid) - 1)
    env = _integrate_envelope(diff, dx, order)
    gap = float(np.max(np.maximum(env, 0.0)))
    violated = gap > 0.0 + 1e-12
    return gap, violated

def _sd_gap_left_tail(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int = 1,
    alpha_tail: float = 0.20,
    ngrid: int = 200,
) -> Tuple[float, bool]:
    """
    Same as _sd_gap, but restrict the comparison to z <= pooled Q_alpha_tail.
    """
    pooled = np.concatenate([x, y])
    q = np.nanquantile(pooled, max(0.0, min(1.0, alpha_tail)))
    grid_full = _grid_from_samples(x, y, ngrid * 2)
    grid = grid_full[grid_full <= q]
    if grid.size < 5:
        grid = np.linspace(np.nanmin(pooled), q, num=max(5, ngrid // 2))
    Fx = _ecdf_at(x, grid)
    Fy = _ecdf_at(y, grid)
    diff = Fx - Fy
    dx = (grid[-1] - grid[0]) / (len(grid) - 1)
    env = _integrate_envelope(diff, dx, order)
    gap = float(np.max(np.maximum(env, 0.0)))
    violated = gap > 0.0 + 1e-12
    return gap, violated

def _sd_gap_right_tail(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int = 1,
    alpha_tail: float = 0.20,
    ngrid: int = 200,
) -> Tuple[float, bool]:
    """
    Same as _sd_gap, but restrict the comparison to z >= pooled Q_(1-alpha_tail).
    This focuses on the RIGHT tail (high values = high systemic risk).
    """
    pooled = np.concatenate([x, y])
    q = np.nanquantile(pooled, max(0.0, min(1.0, 1.0 - alpha_tail)))  # RIGHT tail cutoff
    grid_full = _grid_from_samples(x, y, ngrid * 2)
    grid = grid_full[grid_full >= q]  # Focus on values >= quantile (high risk)
    if grid.size < 5:
        grid = np.linspace(q, np.nanmax(pooled), num=max(5, ngrid // 2))
    Fx = _ecdf_at(x, grid)
    Fy = _ecdf_at(y, grid)
    diff = Fx - Fy
    dx = (grid[-1] - grid[0]) / (len(grid) - 1) if len(grid) > 1 else 1e-6
    env = _integrate_envelope(diff, dx, order)
    gap = float(np.max(np.maximum(env, 0.0)))
    violated = gap > 0.0 + 1e-12
    return gap, violated

# ---------------------------------------------------------------------
# Bootstrap dominance probability with stationary bootstrap
# ---------------------------------------------------------------------

def _stationary_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Politis–Romano stationary bootstrap index generator.
    Draws a bootstrap sample of length n from a single time series of length n.
    """
    if block_len <= 1:
        # i.i.d. fallback
        return rng.integers(0, n, size=n)

    p = 1.0 / float(block_len)
    idx = np.empty(n, dtype=int)
    # start point
    idx0 = rng.integers(0, n)
    idx[0] = idx0
    for t in range(1, n):
        if rng.random() < p:
            # start a new block
            idx[t] = rng.integers(0, n)
        else:
            # continue block (wrap-around)
            idx[t] = (idx[t - 1] + 1) % n
    return idx

def _dominance_probability(
    x: np.ndarray,
    y: np.ndarray,
    *,
    order: int,
    B: int,
    block_len: int,
    ngrid: int,
    rng: np.random.Generator,
) -> float:
    """
    Estimate Pr( x SD-dominates y ) using stationary bootstrap within each series.
    Dominance check is *strict* SD (no ε slack, full support).
    """
    n_x = x.size
    n_y = y.size
    wins = 0
    for b in range(B):
        ix = _stationary_bootstrap_indices(n_x, block_len, rng)
        iy = _stationary_bootstrap_indices(n_y, block_len, rng)
        xb = x[ix]
        yb = y[iy]
        _, violated = _sd_gap(xb, yb, order=order, ngrid=ngrid)
        if not violated:
            wins += 1
    return wins / float(B)

# ---------------------------------------------------------------------
# Utilities for reporting / rankings
# ---------------------------------------------------------------------

def copeland_scores(W: pd.DataFrame) -> pd.Series:
    """
    Copeland score = out-prob minus in-prob for each node, using W[i,j]=Pr(i dominates j).
    """
    out_prob = W.sum(axis=1, skipna=True)
    in_prob = W.sum(axis=0, skipna=True)
    score = out_prob - in_prob
    return score.sort_values(ascending=False)

# ---------------------------------------------------------------------
# Notes:
#  - 'strict' and 'epsilon' use our envelope gaps (deterministic decision).
#  - 'left_tail' uses the same envelope but truncates support to bad tail.
#  - 'pysdtest' calls consistent tests (Barrett–Donald / LMW family via PySDTest).
#  - 'bootstrap_prob' returns an edge weight w_ij = Pr(i dominates j), which you
#     can threshold into a graph and/or use for Copeland rankings.
# ---------------------------------------------------------------------
