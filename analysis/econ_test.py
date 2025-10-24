import pandas as pd
import numpy as np
import os
from scipy import stats

def run_econ_test(
    factor_path="outputs/NetworkRisk.csv",
    returns_path="data/monthly_log_returns.csv",
    out_dir="outputs",
    annualization=None,  # e.g. 12 for monthly, 52 for weekly; will guess from dates if None
    print_output=True
):
    # Load factor and returns
    factor = pd.read_csv(factor_path, index_col=0, parse_dates=True).squeeze("columns")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

    # Remove NaN values from factor first (due to lookback window)
    factor = factor.dropna()
    
    # Handle timezone differences - make both timezone-naive for alignment
    if hasattr(factor.index, 'tz') and factor.index.tz is not None:
        factor.index = factor.index.tz_localize(None)
    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    
    # Align on common dates (only where factor is available)
    common_dates = factor.index.intersection(returns.index)
    factor = factor.loc[common_dates]
    returns = returns.loc[common_dates]

    # Guess frequency for Sharpe ratio
    if annualization is None:
        try:
            inferred = pd.infer_freq(factor.index)
        except Exception:
            inferred = None

        if inferred is None:
            # If too few dates, default to daily
            if len(factor.index) < 3:
                annualization = 252
            else:
                # Fallback by median spacing
                median_delta = np.median(
                    np.diff(factor.index.values).astype("timedelta64[D]").astype(int)
                )
                if median_delta <= 2:
                    annualization = 252
                elif median_delta <= 8:
                    annualization = 52
                else:
                    annualization = 12
        elif inferred[0] == "D":
            annualization = 252
        elif inferred[0] == "W":
            annualization = 52
        elif inferred[0] == "M":
            annualization = 12
        else:
            annualization = 12  # Default to monthly

    # --- 1. Factor summary stats ---
    mu = factor.mean()
    sigma = factor.std()
    t_stat = mu / (sigma / np.sqrt(len(factor)))
    sharpe = mu / sigma * np.sqrt(annualization)

    stats_dict = {
        "Mean": mu,
        "Std": sigma,
        "t-stat": t_stat,
        "Sharpe": sharpe,
        "N": len(factor),
        "Annualization": annualization
    }
    stats_df = pd.DataFrame([stats_dict])
    stats_outfile = os.path.join(out_dir, "factor_summary_stats.csv")
    stats_df.to_csv(stats_outfile, index=False)

    # --- 2. Regress each asset's returns on the factor ---
    results = []
    for col in returns.columns:
        y = returns[col].dropna()
        # Align factor with available return data
        x = factor.loc[y.index]
        
        # Double-check: remove any remaining NaN pairs
        valid_idx = ~(x.isna() | y.isna())
        x = x[valid_idx]
        y = y[valid_idx]
        
        # Drop if not enough data
        if len(y) < 10:
            results.append({"Asset": col, "Beta": np.nan, "t-stat": np.nan, "p-value": np.nan, "N": len(y)})
            continue
            
        x_ = np.vstack([np.ones_like(x), x]).T  # columns: [const, factor]
        try:
            alpha, beta = np.linalg.lstsq(x_, y, rcond=None)[0]  # <- alpha first, beta second
            y_pred = alpha + beta * x
            residuals = y - y_pred
            s_err = np.sqrt(np.sum(residuals ** 2) / (len(y) - 2))
            s_beta = s_err / np.sqrt(np.sum((x - x.mean()) ** 2))
            t_beta = beta / s_beta if s_beta > 0 else np.nan
            p_beta = 2 * (1 - stats.t.cdf(abs(t_beta), df=len(y) - 2)) if not np.isnan(t_beta) else np.nan

            results.append({
                "Asset": col,
                "Beta": beta,
                "t-stat": t_beta,
                "p-value": p_beta,
                "N": len(y)
            })
        except Exception as e:
            # This handles LinAlgError or any rare numeric fail
            results.append({"Asset": col, "Beta": np.nan, "t-stat": np.nan, "p-value": np.nan, "N": len(y)})
            print(f"[WARN] Regression failed for {col}: {e}")


    results_df = pd.DataFrame(results)
    results_outfile = os.path.join(out_dir, "cross_sectional_factor_loadings.csv")
    results_df.to_csv(results_outfile, index=False)

    if print_output:
        print("\n=== Network Risk Factor Summary ===")
        print(stats_df)
        print("\n=== Cross-sectional Loadings (Beta, t, p) ===")
        print(results_df.sort_values("p-value").head(10))

    print(f"[INFO] Saved factor stats to {stats_outfile}")
    print(f"[INFO] Saved cross-sectional regression results to {results_outfile}")

    return stats_df, results_df

if __name__ == "__main__":
    run_econ_test()
