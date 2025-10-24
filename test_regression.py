#!/usr/bin/env python3
"""
Simple test script to validate regression fix without scipy dependency
"""
import pandas as pd
import numpy as np

def simple_regression_test():
    # Load data
    factor = pd.read_csv('outputs/NetworkRisk_daily.csv', index_col=0, parse_dates=True).squeeze("columns")
    returns = pd.read_csv('data/daily_log_returns.csv', index_col=0, parse_dates=True)

    print(f"[INFO] Original factor shape: {factor.shape}")
    print(f"[INFO] Original returns shape: {returns.shape}")
    print(f"[INFO] Factor NaN count: {factor.isna().sum()}")

    # Remove NaN values from factor first (due to lookback window)
    factor = factor.dropna()
    print(f"[INFO] Factor shape after dropna: {factor.shape}")
    
    # Handle timezone differences - make both timezone-naive for alignment
    if hasattr(factor.index, 'tz') and factor.index.tz is not None:
        factor.index = factor.index.tz_localize(None)
    if hasattr(returns.index, 'tz') and returns.index.tz is not None:
        returns.index = returns.index.tz_localize(None)
    
    # Align on common dates (only where factor is available)
    common_dates = factor.index.intersection(returns.index)
    factor = factor.loc[common_dates]
    returns = returns.loc[common_dates]
    
    print(f"[INFO] Common dates: {len(common_dates)}")
    print(f"[INFO] Aligned factor shape: {factor.shape}")
    print(f"[INFO] Aligned returns shape: {returns.shape}")
    
    # Test simple regression for first asset
    asset_col = returns.columns[0]
    y = returns[asset_col].dropna()
    x = factor.loc[y.index]
    
    # Double-check: remove any remaining NaN pairs
    valid_idx = ~(x.isna() | y.isna())
    x = x[valid_idx]
    y = y[valid_idx]
    
    print(f"\n[INFO] Testing regression for {asset_col}")
    print(f"[INFO] Valid observations: {len(y)}")
    print(f"[INFO] X (factor) has NaN: {x.isna().any()}")
    print(f"[INFO] Y (returns) has NaN: {y.isna().any()}")
    
    if len(y) >= 10:
        # Build design matrix
        x_ = np.vstack([np.ones_like(x), x]).T  # columns: [const, factor]
        print(f"[INFO] Design matrix shape: {x_.shape}")
        print(f"[INFO] Design matrix has NaN: {np.isnan(x_).any()}")
        print(f"[INFO] Y vector has NaN: {np.isnan(y).any()}")
        
        try:
            # Try the regression
            result = np.linalg.lstsq(x_, y, rcond=None)
            alpha, beta = result[0]
            print(f"[SUCCESS] Regression completed!")
            print(f"[INFO] Alpha (intercept): {alpha:.6f}")
            print(f"[INFO] Beta (factor loading): {beta:.6f}")
            
            # Check residuals and stats
            y_pred = alpha + beta * x
            residuals = y - y_pred
            s_err = np.sqrt(np.sum(residuals ** 2) / (len(y) - 2))
            s_beta = s_err / np.sqrt(np.sum((x - x.mean()) ** 2))
            t_beta = beta / s_beta if s_beta > 0 else np.nan
            
            print(f"[INFO] Standard error of beta: {s_beta:.6f}")
            print(f"[INFO] t-statistic: {t_beta:.6f}")
            
            return True
            
        except np.linalg.LinAlgError as e:
            print(f"[ERROR] Regression failed: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            return False
    else:
        print(f"[WARN] Not enough data points ({len(y)} < 10)")
        return False

if __name__ == "__main__":
    success = simple_regression_test()
    if success:
        print("\n[SUCCESS] Regression fix appears to be working!")
    else:
        print("\n[FAIL] Regression is still failing")