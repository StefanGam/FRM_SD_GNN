#!/usr/bin/env python3
"""
main.py
--------
End-to-end driver for crypto/FRM/SD pipeline with monthly or weekly data, config-driven!
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from analysis import prep
from analysis.frm_asgl      import compute_frm
from analysis.sd_network    import dominance_graph_single
from analysis.features      import compute_graph_centralities
from analysis.factor        import build_HL_factor

def parse_args():
    p = argparse.ArgumentParser(description="FRM/SD pipeline")
    p.add_argument("--sample", type=int, default=None,
                   help="randomly keep only N assets (asset-level only)")
    p.add_argument("--n_jobs", type=int, default=None,
                   help="override number of parallel jobs for FRM")
    p.add_argument("--level", choices=["asset", "sector"], default="asset",
                   help="run at 'asset' or 'sector' aggregation level")
    p.add_argument("--config", type=Path, default=Path("config.yml"),
                   help="path to YAML config")
    p.add_argument("--last-days", type=int, default=None,
                   help="run only on the last N calendar days (daily freq)")
    p.add_argument("--last-weeks", type=int, default=None,
                   help="run only on the last N calendar weeks (weekly freq)")
    p.add_argument("--start-date", type=str, default=None,
                   help="ISO start date (e.g., 2024-01-01) to slice returns")
    p.add_argument("--end-date", type=str, default=None,
                   help="ISO end date (inclusive, e.g., 2024-12-31)")
    return p.parse_args()

def load_cfg(path: Path) -> dict:
    if path.exists():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            print(f"[WARN] could not parse {path}; using defaults")
            cfg = {}
    else:
        cfg = {}
    return cfg

def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    root     = Path(__file__).resolve().parent
    data_dir = root / "data"
    outputs  = root / "outputs"
    outputs.mkdir(exist_ok=True)

    # 1. Load or build returns (robust to empty/missing/dir paths)
    returns_path_str = cfg.get("returns_path", None)
    returns_path = Path(returns_path_str).expanduser() if returns_path_str else None

    freq = str(cfg.get("frequency", "daily")).lower()
    universe = str(cfg.get("universe", "crypto")).lower()

    def _is_valid_file(p: Path | None) -> bool:
        try:
            return p is not None and p.is_file()
        except Exception:
            return False

    if _is_valid_file(returns_path):
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        print(f"[INFO] Loaded {returns.shape[0]}×{returns.shape[1]} returns from {returns_path}")
    else:
        # Warn if user set something that isn't a file (e.g., ".", folder, or bad path)
        if returns_path_str not in (None, "", "null"):
            print(f"[WARN] returns_path={returns_path_str!r} is not a readable file; falling back to auto-build.")

        if universe == "crypto":
            folder = cfg.get("crypto", {}).get("folder", "data/cryptos")

            if freq.startswith("w"):
                weekly = prep.prepare_crypto_weekly(folder=folder, week_ending=cfg.get("week_ending", "SUN"))
                returns = weekly["ret_log_W"]
                print(f"[INFO] Built {returns.shape[0]}×{returns.shape[1]} weekly log-returns from {folder}")

            elif freq.startswith("m"):
                # Monthly auto-build from daily crypto closes (month-end)
                daily = prep.prepare_crypto_daily(folder=folder)
                prices_D = daily["prices_D"]
                prices_M = prices_D.resample("M").last()
                returns = np.log(prices_M).diff().dropna(how="all")
                print(f"[INFO] Built {returns.shape[0]}×{returns.shape[1]} monthly log-returns (M-end) from {folder}")

            else:
                # default: daily
                daily = prep.prepare_crypto_daily(folder=folder)
                returns = daily["ret_log_D"]
                print(f"[INFO] Built {returns.shape[0]}×{returns.shape[1]} daily log-returns from {folder}")
        else:
            raise ValueError("No valid returns_path and non-crypto universe not implemented for auto-build.")


    # 2. (Optional) Asset subsample
    if args.sample:
        returns = returns.sample(n=args.sample, axis=1, random_state=42)
        print(f"[INFO] Sub-sampled to {returns.shape[1]} assets")

    # 2b. Optional time-window slicing for fast testing
    # Config-driven overrides CLI when both provided
    test_cfg = cfg.get("test_window", {}) or {}
    start_date = args.start_date or test_cfg.get("start")
    end_date   = args.end_date   or test_cfg.get("end")
    last_days  = args.last_days  or test_cfg.get("last_days")
    last_weeks = args.last_weeks or test_cfg.get("last_weeks")

    # Date-range slice first (if provided)
    if start_date or end_date:
        returns = returns.loc[
            (returns.index >= (pd.to_datetime(start_date) if start_date else returns.index.min())) &
            (returns.index <= (pd.to_datetime(end_date)   if end_date   else returns.index.max()))
        ]
        print(f"[INFO] Sliced by date range to {returns.index.min():%Y-%m-%d} .. {returns.index.max():%Y-%m-%d} ({len(returns)} rows)")

    # Last N periods (exclusive with date range; apply only if no date range given)
    elif str(cfg.get("frequency", "daily")).lower().startswith("w") and last_weeks:
        returns = returns.tail(int(last_weeks))
        print(f"[INFO] Sliced to last {last_weeks} weeks → {len(returns)} rows")
    elif last_days:
        # For daily data, just keep the last N rows (data are daily already)
        returns = returns.tail(int(last_days))
        print(f"[INFO] Sliced to last {last_days} days → {len(returns)} rows")

    # 3. Prepare returns for pipeline
    # (returns are already returns; keep as ret_log)
    ret_log = returns

    # 4. Compute FRM/λ-matrix
    nj = args.n_jobs or cfg.get("n_jobs", 4)

    # Compute an effective window if the test slice is short
    n_obs = len(returns)
    cfg_window = int(cfg.get("window", 63))
    eff_window = min(cfg_window, max(2, n_obs - 2))
    if eff_window < cfg_window:
        print(f"[INFO] Adjusting FRM window from {cfg_window} to {eff_window} for short test sample of {n_obs} rows.")

    print(f"[INFO] Computing FRM (window={eff_window}, step={cfg['step']}, freq={freq}) …")
    frm_out = compute_frm(
        returns,
        window=eff_window,
        step=cfg['step'],
        tau=cfg['tau'],
        n_jobs=nj,
        n_folds=cfg['n_folds'],
    )

    frm_idx = frm_out["frm_index"]
    full_lambda_mat = frm_out["lambda_mat"]

    # short-circuit if empty (too little data after slicing)
    if frm_idx is None or len(frm_idx) == 0 or full_lambda_mat is None or full_lambda_mat.empty:
        print("[WARN] FRM index is empty (likely due to too few rows after time slicing). "
              "Increase test_window (e.g., last_days) or reduce window. Skipping SD network and factor.")
        return

    # 5. Save λ-matrix & FRM index
    full_lambda_mat.to_csv(outputs / "lambda_mat_full.csv")
    frm_idx.to_csv(outputs / "frm_index_full.csv")
    print(f"[INFO] Saved lambda matrix {full_lambda_mat.shape} and FRM index ({len(frm_idx)} rows)")

    # 6. Build scalar SD networks and extract centralities for each period
    centralities = []
    for date, row in full_lambda_mat.iterrows():
        G = dominance_graph_single(row)  # Scalar network
        cent = compute_graph_centralities(G)
        # Flatten: {'metric': {asset: value}} to {'metric_asset': value}
        flat = {f"{metric}_{asset}": val for metric, asset_dict in cent.items() for asset, val in asset_dict.items()}
        flat['Date'] = date
        centralities.append(flat)

    centrality_df = pd.DataFrame(centralities).set_index('Date')
    centrality_df.to_csv(outputs / "centralities.csv")
    print(f"[INFO] Saved SD network centralities ({centrality_df.shape}) to outputs/centralities.csv")

    # 7. Build Network Risk factor (High-Low) and save
    print(f"[INFO] Building Network Risk factor (High minus Low portfolios) …")
    from analysis.factor import build_HL_factor
    factor_series = build_HL_factor(
        centrality_df=centrality_df,
        returns_df=returns,
        metric=cfg["centrality_metric"],
        top_n=cfg["factor_high"],
        bottom_n=cfg["factor_low"],
        out_path=cfg["factor_output"]
    )
    print(f"[INFO] Done. Network risk factor shape: {factor_series.shape}")

    print(f"[DONE] {datetime.now():%Y-%m-%d %H:%M}")

if __name__ == "__main__":
    main()
