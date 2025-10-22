"""
sd_network.py
-------------
Construct a directed dominance graph from a single FRM λ-vector or λ-samples.

• If you pass only scalars (one λ per asset), we use a fast scalar rule:
  edge i -> j if λ_i > λ_j.

• If you pass arrays (sample of λ per asset), we run statistical SD tests
  and add edge i -> j if asset i stochastically dominates j at order s.

Usage:
    from analysis.sd_network import dominance_graph_single
"""

import numpy as np
import networkx as nx
from analysis.sd_utils import sd_stat_pvalue


def _is_arraylike(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))


def dominance_graph_single(
    lambda_vec,
    s: int = 2,
    alpha: float = 0.05,
    ngrid: int = 100,
    nboot: int = 200,
    method: str = "perm",     # NEW: "perm" (permutation) or "ks" (SD1 only, if supported)
    B: int | None = None,     # NEW: alias for number of permutations; overrides nboot if given
    debug: bool = False,
) -> nx.DiGraph:
    """
    Build a directed graph G where nodes are tickers (index/keys of lambda_vec)
    and there is an edge i -> j if:
      • (scalar mode)           λ_i > λ_j
      • (distributional mode)   sample_i SD-dominates sample_j at order s (p <= alpha)

    Parameters
    ----------
    lambda_vec : pd.Series | dict | list | np.ndarray
        Mapping from asset -> scalar or array of λ values. If ANY entry is array-like
        (len>1), distributional SD mode is used; otherwise scalar mode is used.
    s : int
        Order of stochastic dominance (1 = FSD, 2 = SSD).
    alpha : float
        Significance threshold for the (one-sided) SD test.
    ngrid : int
        Number of ECDF grid points used by the SD test (if applicable).
    nboot : int
        Number of bootstrap/permutation replications (if applicable).
    method : str
        "perm" for permutation p-values (recommended), or "ks" for one-sided KS (SD1 only; if supported by sd_utils).
    B : int | None
        Alias for number of permutations/bootstraps. If provided, overrides nboot.
    debug : bool
        If True, forwards debug flag to sd_stat_pvalue.

    Returns
    -------
    G : networkx.DiGraph
        Directed graph where edge i->j indicates dominance.
    """
    # Harmonize nboot/B
    if B is not None:
        nboot = int(B)

    # Extract tickers/keys and values in order
    try:
        tickers = list(lambda_vec.index)
        samples = list(lambda_vec.values)
    except AttributeError:
        # lambda_vec may be dict-like or list/tuple/ndarray
        if isinstance(lambda_vec, dict):
            tickers = list(lambda_vec.keys())
            samples = [lambda_vec[k] for k in tickers]
        else:
            tickers = list(range(len(lambda_vec)))
            samples = list(lambda_vec)

    N = len(tickers)
    G = nx.DiGraph()
    G.add_nodes_from(tickers)

    # Decide mode: scalar vs distributional
    sizes = []
    for v in samples:
        arr = np.atleast_1d(v)
        sizes.append(arr.size)
    use_distributional = any(sz > 1 for sz in sizes)

    if not use_distributional:
        # --- Scalar fallback: edge i->j if lambda_i > lambda_j ---
        lam = [float(np.atleast_1d(v).item()) for v in samples]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                li = lam[i]
                lj = lam[j]
                if np.isfinite(li) and np.isfinite(lj) and li > lj:
                    G.add_edge(tickers[i], tickers[j], weight=float(li - lj), pvalue=0.0)
        return G

    # --- Distributional SD: arrays per asset, test SD(i > j) one-sided ---
    # Minimal sample size per asset to attempt a test
    min_len = 5

    for i in range(N):
        xi = np.asarray(samples[i])
        xi = xi[np.isfinite(xi)]
        if xi.size < min_len:
            continue

        for j in range(N):
            if i == j:
                continue
            xj = np.asarray(samples[j])
            xj = xj[np.isfinite(xj)]
            if xj.size < min_len:
                continue

            if debug:
                print(f"[SD] {tickers[i]} (n={xi.size}) vs {tickers[j]} (n={xj.size}) | s={s}, method={method}, nboot={nboot}")

            # One-sided dominance test: does i dominate j?
            # sd_stat_pvalue should accept (x, y, s, method=?, nboot=?, ngrid=?, debug=?)
            stat_ij, p_ij = sd_stat_pvalue(
                x=xi, y=xj, s=s, method=method, nboot=nboot, ngrid=ngrid, debug=debug
            )
            stat_ji, p_ji = sd_stat_pvalue(
                x=xj, y=xi, s=s, method=method, nboot=nboot, ngrid=ngrid, debug=debug
            )

            # Add i->j if i dominates j significantly and NOT vice-versa
            cond_ij = (stat_ij is not None) and (p_ij is not None) and (stat_ij > 0) and (p_ij <= alpha)
            cond_ji = (stat_ji is not None) and (p_ji is not None) and (stat_ji > 0) and (p_ji <= alpha)

            if cond_ij and not cond_ji:
                G.add_edge(tickers[i], tickers[j], weight=float(stat_ij), pvalue=float(p_ij))

            # If both cond_ij and cond_ji, treat as a tie and add no edge (conservative)
            # If neither, no edge.

    return G
