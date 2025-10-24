#!/usr/bin/env python3
"""
Test all assets regression to validate the complete fix
"""
import pandas as pd
import numpy as np

def test_all_assets():
    # Load data
    factor = pd.read_csv('outputs/NetworkRisk_daily.csv', index_col=0, parse_dates=True).squeeze("columns")
    returns = pd.read_csv('data/daily_log_returns.csv', index_col=0, parse_dates=True)

    # Apply our fix
    factor = factor.dropna()
    
    # Handle timezone differences
    if hasattr(factor.index, 'tz') and factor.index.tz is not None:
        factor.index = factor.index.tz_localize(None)
    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    
    # Align on common dates
    common_dates = factor.index.intersection(returns.index)
    factor = factor.loc[common_dates]
    returns = returns.loc[common_dates]
    
    print(f"[INFO] Testing {len(returns.columns)} assets with {len(factor)} observations")
    
    # Test regression for all assets
    results = []
    success_count = 0
    
    for col in returns.columns:
        y = returns[col].dropna()
        x = factor.loc[y.index]
        
        # Double-check: remove any remaining NaN pairs
        valid_idx = ~(x.isna() | y.isna())
        x = x[valid_idx]
        y = y[valid_idx]
        
        if len(y) >= 10:
            x_ = np.vstack([np.ones_like(x), x]).T
            try:
                alpha, beta = np.linalg.lstsq(x_, y, rcond=None)[0]
                y_pred = alpha + beta * x
                residuals = y - y_pred
                s_err = np.sqrt(np.sum(residuals ** 2) / (len(y) - 2))
                s_beta = s_err / np.sqrt(np.sum((x - x.mean()) ** 2))
                t_beta = beta / s_beta if s_beta > 0 else np.nan
                
                results.append({
                    "Asset": col,
                    "Beta": beta,
                    "t-stat": t_beta,
                    "N": len(y),
                    "Status": "SUCCESS"
                })
                success_count += 1
                
            except Exception as e:
                results.append({
                    "Asset": col,
                    "Beta": np.nan,
                    "t-stat": np.nan,
                    "N": len(y),
                    "Status": f"FAILED: {e}"
                })
        else:
            results.append({
                "Asset": col,
                "Beta": np.nan,
                "t-stat": np.nan,
                "N": len(y),
                "Status": f"INSUFFICIENT_DATA ({len(y)} < 10)"
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n[SUMMARY] Regression Results:")
    print(f"  - Total assets: {len(results_df)}")
    print(f"  - Successful regressions: {success_count}")
    print(f"  - Failed regressions: {len(results_df) - success_count}")
    
    print(f"\n[RESULTS] Top 5 results by absolute t-stat:")
    successful = results_df[results_df['Status'] == 'SUCCESS'].copy()
    if len(successful) > 0:
        successful['abs_t'] = successful['t-stat'].abs()
        top_results = successful.nlargest(5, 'abs_t')[['Asset', 'Beta', 't-stat', 'N']]
        for _, row in top_results.iterrows():
            print(f"  {row['Asset']}: β={row['Beta']:.4f}, t={row['t-stat']:.3f}, N={row['N']}")
    
    # Show any failures
    failures = results_df[results_df['Status'] != 'SUCCESS']
    if len(failures) > 0:
        print(f"\n[FAILURES] {len(failures)} failed regressions:")
        for _, row in failures.iterrows():
            print(f"  {row['Asset']}: {row['Status']}")
    
    return success_count == len(results_df)

if __name__ == "__main__":
    all_success = test_all_assets()
    if all_success:
        print("\n✅ ALL REGRESSIONS SUCCESSFUL! The fix is working perfectly.")
    else:
        print("\n⚠️  Some regressions failed, but this may be expected for assets with insufficient data.")