import pandas as pd
import numpy as np

def build_HL_factor(
    centrality_df,
    returns_df,
    metric='eigenvector',
    top_n=3,
    bottom_n=3,
    out_path="outputs/NetworkRisk.csv"
):
    """
    For each period:
        1) rank assets by centrality; 2) long top_n, short bottom_n; 3) factor = H - L.
    """

    # --- normalize indices (tz-naive) and align on common dates ---
    returns_df.index = pd.to_datetime(returns_df.index)
    try:
        returns_df.index = returns_df.index.tz_localize(None)
    except AttributeError:
        pass

    centrality_df.index = pd.to_datetime(centrality_df.index)
    try:
        centrality_df.index = centrality_df.index.tz_localize(None)
    except AttributeError:
        pass

    common = returns_df.index.intersection(centrality_df.index)
    if len(common) == 0:
        print("[HL-Factor] No overlapping dates between centrality and returns; nothing to do.")
        pd.Series([], name="NetworkRisk").to_csv(out_path)
        return

    centrality_df = centrality_df.loc[common].sort_index()
    returns_df    = returns_df.loc[common].sort_index()

    metric_cols = [c for c in centrality_df.columns if c.endswith("_" + metric) or c.startswith(metric + "_")]
    if not metric_cols:
        print("Available columns:", centrality_df.columns.tolist())
        raise ValueError(f"No columns found for metric '{metric}' in centrality_df!")

    if all(c.endswith("_" + metric) for c in metric_cols):
        metric_matrix = centrality_df[metric_cols].copy()
        metric_matrix.columns = [c.replace("_" + metric, "") for c in metric_cols]
    else:
        metric_matrix = centrality_df[metric_cols].copy()
        metric_matrix.columns = [c.replace(metric + "_", "") for c in metric_cols]

    periods = metric_matrix.index

    factor_list = []
    for date in periods:
        scores = metric_matrix.loc[date]
        scores = scores.dropna()
        print(f"[HL-Factor] {date.date()}: {len(scores)} assets with valid {metric} centrality.")
        
        # Check if all centralities are zero (no network activity)
        if scores.sum() == 0.0:
            print(f"  [Skip] All {metric} centralities are zero - no network activity")
            factor_list.append(np.nan)
            continue
            
        if len(scores) < top_n + bottom_n:
            print(f"  [Skip] Not enough assets to build H-L factor (needed {top_n + bottom_n}, found {len(scores)})")
            factor_list.append(np.nan)
            continue

        top_assets = scores.sort_values(ascending=False).index[:top_n]
        bot_assets = scores.sort_values(ascending=False).index[-bottom_n:]

        ret = returns_df.loc[date]
        try:
            high = ret[top_assets].mean()
            low = ret[bot_assets].mean()
            factor = high - low
        except Exception as e:
            print(f"  [Error] Problem with returns extraction on {date.date()}: {e}")
            factor = np.nan
        factor_list.append(factor)

    factor_series = pd.Series(factor_list, index=periods, name="NetworkRisk")
    factor_series.to_csv(out_path)
    print(f"[INFO] Saved NetworkRisk factor to {out_path}")
    print(f"[HL-Factor] Non-NaN periods: {(~factor_series.isna()).sum()} / {len(factor_series)}")

    if (~factor_series.isna()).sum() == 0:
        print("[WARNING] No valid factor periods created! Try reducing top_n and bottom_n in config.")

    return factor_series

# For direct script use:
if __name__ == "__main__":
    import sys
    centrality_path = sys.argv[1]     # e.g. outputs/centralities.csv
    returns_path = sys.argv[2]        # e.g. data/monthly_log_returns.csv
    metric = sys.argv[3] if len(sys.argv) > 3 else 'eigenvector'
    top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    bottom_n = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    centrality_df = pd.read_csv(centrality_path, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    build_HL_factor(centrality_df, returns_df, metric=metric, top_n=top_n, bottom_n=bottom_n)
