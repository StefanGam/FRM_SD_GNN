#!/usr/bin/env python3
"""
run_crypto.py
-------------
Full pipeline runner for FRM/SD/Factor analysis on cryptos (monthly/weekly).
Avoids recomputation: saves/loads lambda matrices and SD networks for fast reruns!
"""


import os
import argparse, yaml, pickle
from pathlib import Path
import pandas as pd
import numpy as np

# progress bar
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kw):  # fallback no-op if tqdm missing
        return x

def main():
    # --- CLI and config ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_network', action='store_true',
                        help="Recompute FRM/SD network (default: False, just reload pickles if present)")
    parser.add_argument('--config', type=str, default="config.yml", help="Path to config file")
    args = parser.parse_args()
    with_network = args.with_network

    # --- Load config ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)

    # --- Data path setup ---
    networks_path = outputs / "networks.pkl"
    lambda_path = outputs / "lambda_mat_full.pkl"
    frm_idx_path = outputs / "frm_index_full.pkl"

    # --- 1. Data Prep (skip if already prepped) ---
    returns = pd.read_csv(cfg['returns_path'], index_col=0, parse_dates=True)

    # Optional time-slice for fast tests (from config.yml:test_window)
    tw = cfg.get("test_window", {}) or {}
    start = tw.get("start")
    end = tw.get("end")
    last_days = tw.get("last_days")
    last_weeks = tw.get("last_weeks")

    if start or end:
        start_dt = pd.to_datetime(start) if start else returns.index.min()
        end_dt = pd.to_datetime(end) if end else returns.index.max()
        returns = returns.loc[(returns.index >= start_dt) & (returns.index <= end_dt)]
        print(
            f"[INFO] Sliced by date range to {returns.index.min():%Y-%m-%d} .. {returns.index.max():%Y-%m-%d} ({len(returns)} rows)")
    elif last_weeks:
        returns = returns.tail(int(last_weeks))
        print(f"[INFO] Sliced to last {last_weeks} rows (weekly config) → {len(returns)} rows")
    elif last_days:
        returns = returns.tail(int(last_days))
        print(f"[INFO] Sliced to last {last_days} rows (daily config) → {len(returns)} rows")

    # Normalize timezones to tz-naive everywhere
    returns.index = pd.to_datetime(returns.index)
    try:
        returns.index = returns.index.tz_localize(None)
    except AttributeError:
        # already tz-naive
        pass

    ret_log = returns.copy()
    print("Data periods:", len(ret_log), "| Rolling window:", cfg["window"])
    print("ret_log shape:", ret_log.shape, ", window:", cfg["window"])

    # --- 2. Compute or load FRM/λ-matrix ---
    if with_network or not lambda_path.exists() or not frm_idx_path.exists():
        print("[INFO] Computing FRM/λ-matrix ...")
        from analysis.frm_asgl import compute_frm

        # Auto-shrink window for short samples: need at least (window+1) rows to get 1 window
        n_obs = len(ret_log)
        cfg_window = int(cfg["window"])
        eff_window = min(cfg_window, max(2, n_obs - 2))
        if eff_window < cfg_window:
            print(f"[INFO] Adjusting FRM window from {cfg_window} to {eff_window} for short sample of {n_obs} rows.")

        frm_out = compute_frm(
            ret_log,
            window=eff_window,
            tau=cfg["tau"],
            n_jobs=cfg.get("n_jobs", 4),
            progress=True,
            step=cfg["step"],
            lambda_grid=cfg["lambda_grid"],
            n_folds=cfg.get("n_folds", 2),
            bootstrap=cfg.get("bootstrap", 0)
        )
        frm_idx = frm_out["frm_index"]
        full_lambda_mat = frm_out["lambda_mat"]

        # If FRM produced no windows, short-circuit gracefully
        if full_lambda_mat is None or full_lambda_mat.empty or frm_idx is None or frm_idx.empty:
            print("[WARN] FRM output is empty (likely due to too few rows after time slicing). "
                  "Increase test_window.last_days or reduce window. Skipping SD and factor.")
            # Save empty placeholders so reload doesn't loop forever
            full_lambda_mat = pd.DataFrame()
            frm_idx = pd.DataFrame()
            # Make λ and FRM indices tz-naive for consistency with returns
            full_lambda_mat.index = pd.to_datetime(full_lambda_mat.index)
            frm_idx.index = pd.to_datetime(frm_idx.index)
            try:
                full_lambda_mat.index = full_lambda_mat.index.tz_localize(None)
                frm_idx.index = frm_idx.index.tz_localize(None)
            except AttributeError:
                pass

            full_lambda_mat.to_pickle(lambda_path)
            frm_idx.to_pickle(frm_idx_path)

            return

        # Save outputs as pickle for fast reload
        full_lambda_mat.to_pickle(lambda_path)
        frm_idx.to_pickle(frm_idx_path)
        # Also save as CSV for reference
        full_lambda_mat.to_csv(outputs / "lambda_mat_full.csv")
        frm_idx.to_csv(outputs / "frm_index_full.csv")
    else:
        print("[INFO] Loading FRM/λ-matrix from file ...")
        full_lambda_mat = pd.read_pickle(lambda_path)
        frm_idx = pd.read_pickle(frm_idx_path)
        print(f"[INFO] Loaded lambda_mat ({full_lambda_mat.shape}), frm_idx ({frm_idx.shape}) from pickles.")
        if full_lambda_mat is None or full_lambda_mat.empty or frm_idx is None or frm_idx.empty:
            print("[WARN] Stored FRM output is empty. Nothing to do. Exiting.")
            return

    # If λ is empty for any reason, stop early (safety)
    if full_lambda_mat.empty:
        print("[WARN] λ-matrix empty; skipping network, factor, and econ tests.")
        return

    # --- 3. Compute or load networks ---
    if with_network or not networks_path.exists():
        print("[INFO] Computing and saving SD networks...")
        from analysis.sd_network import dominance_graph_single

        sd_cfg = cfg.get("sd", {}) or {}
        sd_mode = sd_cfg.get("method", "scalar").lower()  # "scalar" | "distributional"
        s_order = int(sd_cfg.get("order", 1))  # SD order (1/2) if distributional
        alpha = float(sd_cfg.get("alpha", 0.05))
        lookback = int(sd_cfg.get("lookback", 126))
        pmethod = sd_cfg.get("pval", "perm")  # "perm" or "ks"
        perm_B = int(sd_cfg.get("perm_B", 200))

        network_list = []

        if sd_mode == "distributional":
            # Per-date λ-distributions from trailing lookback window
            dates = full_lambda_mat.index
            for t in tqdm(range(len(dates)), total=len(dates), desc="SD networks"):
                date = dates[t]
                start = max(0, t - lookback + 1)
                hist = full_lambda_mat.iloc[start:t + 1]  # lookback × assets
                # Each asset gets an array of λ's (drop NaNs)
                sample_series = hist.apply(lambda col: col.dropna().values, axis=0)
                # dominance_graph_single detects arrays and runs SD tests via sd_utils
                G = dominance_graph_single(
                    sample_series, s=s_order, alpha=alpha, method=pmethod, B=perm_B
                )
                network_list.append((date, G))
        else:
            # Fast scalar fallback (edge i->j if lambda_i > lambda_j)
            for date, row in tqdm(full_lambda_mat.iterrows(), total=len(full_lambda_mat), desc="SD networks"):
                G = dominance_graph_single(row)
                network_list.append((date, G))

        with open(networks_path, "wb") as f:
            pickle.dump(network_list, f)
        print(f"[INFO] Saved networkx graphs for all periods to {networks_path}")
    else:
        print("[INFO] Loading precomputed networks from file...")
        with open(networks_path, "rb") as f:
            network_list = pickle.load(f)
        print(f"[INFO] Loaded {len(network_list)} networks from {networks_path}")

    # --- 4. Centralities ---
    from analysis.features import compute_graph_centralities
    centralities = []
    for date, G in network_list:
        cent = compute_graph_centralities(G)
        flat = {f"{asset}_{metric}": val
                for metric, asset_dict in cent.items()
                for asset, val in asset_dict.items()}
        # force tz-naive date for alignment with returns
        d = pd.Timestamp(date)
        if d.tzinfo is not None:
            d = d.tz_convert("UTC").tz_localize(None)
        flat['Date'] = d
        centralities.append(flat)

    if not centralities:
        print("[WARN] No networks / centralities computed. Skipping factor and econ tests.")
        return

    centrality_df = pd.DataFrame(centralities).set_index("Date").sort_index()
    centrality_df.index = pd.to_datetime(centrality_df.index)
    try:
        centrality_df.index = centrality_df.index.tz_localize(None)
    except AttributeError:
        pass
    centrality_df.to_csv(outputs / "centralities.csv")

    # --- 5. Factor construction ---
    from analysis.factor import build_HL_factor
    factor_series = build_HL_factor(
        centrality_df=centrality_df,
        returns_df=returns,
        metric=cfg["centrality_metric"],
        top_n=cfg["factor_high"],
        bottom_n=cfg["factor_low"],
        out_path=cfg["factor_output"]
    )

    # --- 6. Performance/Econ test (use the SAME sliced returns) ---
    from analysis.econ_test import run_econ_test
    sliced_returns_path = outputs / "returns_sliced.csv"
    returns.to_csv(sliced_returns_path)
    run_econ_test(
        factor_path=cfg["factor_output"],
        returns_path=str(sliced_returns_path),
        out_dir="outputs"
    )

    print("[PIPELINE DONE] All outputs saved in 'outputs/'.")

if __name__ == "__main__":
    main()
