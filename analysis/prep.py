"""
prep.py
-------
Utility functions that take a wide (Date × Ticker) price or return matrix
and return: prices, simple returns, log returns, and a melted long-form DF.
Supports any frequency (daily, weekly, monthly) as provided.

Extended to:
  • Read multiple per-coin CSVs from data/cryptos/*.csv
  • Produce DAILY and WEEKLY panels for prices, returns, and volumes
  • Optionally compute dollar-volume (units × lagged close)
"""

import os, glob, re
import pandas as pd
import numpy as np

# ╔════════════════════════════════════════════════╗
# 0. CSV ingest helpers (new)
# ╚════════════════════════════════════════════════╝

def _read_coin_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV reader:
      - auto-detect delimiter,
      - tries ',', then ';' if needed,
      - handles UTF-8 BOM,
      - forwards to _normalize_crypto_csv with file_hint for diagnostics.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")  # auto-detect delimiter
    except Exception:
        # Fallbacks for tricky files
        try:
            df = pd.read_csv(path, sep=",")
        except Exception:
            df = pd.read_csv(path, sep=";")
    return _normalize_crypto_csv(df, file_hint=path)


def _normalize_crypto_csv(df: pd.DataFrame,
                          date_col: str | None = None,
                          price_col: str | None = None,
                          vol_col: str | None = None,
                          file_hint: str | None = None) -> pd.DataFrame:
    """
    Normalize to ['date','close','volume'(optional)] with permissive header matching.
    Supports many vendor variants (e.g., 'Timestamp', 'snapped_at', 'Price (USD)', 'Adj Close', 'C', etc.).
    """

    # 1) Normalize headers aggressively: lowercase, strip, remove spaces/punct
    def _norm(s: str) -> str:
        s0 = str(s).strip().lower()
        s0 = s0.replace("\ufeff", "")        # strip BOM if present
        s0 = re.sub(r"\s+", "", s0)          # remove spaces
        s0 = re.sub(r"[^a-z0-9]", "", s0)    # keep only [a-z0-9]
        return s0

    norm_to_orig = { _norm(c): c for c in df.columns }
    norm_cols = list(norm_to_orig.keys())

    # 2) Respect explicit overrides if provided
    date_norm  = _norm(date_col)  if date_col  else None
    price_norm = _norm(price_col) if price_col else None
    vol_norm   = _norm(vol_col)   if vol_col   else None

    # 3) Candidate keys for common vendors
    date_keys  = ["date","timestamp","time","snappedat","datetime","unixtime","unix","opentime","closetime","t"]
    close_keys = [
        "close", "adjclose", "price", "prices", "closeprice", "closingprice", "last", "c",
        "closeusd", "closeusdt"
    ]
    vol_keys = [
        "volume", "totalvolume", "totalvolumes", "volumefrom", "volumeto", "basevolume",
        "quotevolume", "v", "volumeusd", "volumeusdt", "vol"
    ]

    def _pick(keys, explicit=None):
        if explicit and explicit in norm_to_orig:
            return norm_to_orig[explicit]
        for k in keys:
            if k in norm_to_orig:
                return norm_to_orig[k]
        return None

    date_pick  = _pick(date_keys,  date_norm)
    close_pick = _pick(close_keys, price_norm)
    vol_pick   = _pick(vol_keys,   vol_norm)

    # 4) Fail with diagnostics if we still cannot match
    if date_pick is None or close_pick is None:
        hint = f" (file: {file_hint})" if file_hint else ""
        raise ValueError(
            "Could not infer date/close columns from CSV headers"
            f"{hint}. Normalized headers detected: {norm_cols}"
        )

    # 5) Canonicalize, parse types, and sanitize
    out = df.rename(columns={
        date_pick:  "date",
        close_pick: "close",
        **({vol_pick: "volume"} if vol_pick is not None else {})
    })
    out["date"]  = pd.to_datetime(out["date"], utc=False, errors="coerce")
    out = out.sort_values("date").dropna(subset=["date"])

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out.loc[out["close"]  <= 0, "close"]  = np.nan
    if "volume" in out.columns:
        out.loc[out["volume"] <= 0, "volume"] = np.nan

    return out[["date", "close"] + (["volume"] if "volume" in out.columns else [])]



def build_crypto_panels_from_folder(folder: str = "data/cryptos",
                                    fill_short_gaps: int = 3):
    """
    Scan folder for *.csv (e.g., bitcoin.csv, litecoin.csv, dogecoin.csv),
    pivot into wide panels:
       prices  : T×N close levels
       volumes : T×N base units (if available)
    Calendar is 7-day 'D'; forward-fill up to `fill_short_gaps` days for closes.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    price_cols, vol_cols = [], []
    for path in files:
        sym = os.path.splitext(os.path.basename(path))[0].upper()
        try:
            df = _read_coin_csv(path).set_index("date")
        except Exception as e:
            raise RuntimeError(f"Failed to parse {path}: {e}") from e
        price_cols.append(df["close"].rename(sym))
        if "volume" in df.columns:
            vol_cols.append(df["volume"].rename(sym))

    prices = pd.concat(price_cols, axis=1).sort_index()
    volumes = pd.concat(vol_cols, axis=1).sort_index() if vol_cols else None

    # full daily grid (crypto trades 7 days/week)
    full_days = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    prices = prices.reindex(full_days).ffill(limit=fill_short_gaps)
    prices.index.name = "Date"  # optional
    if volumes is not None:
        volumes = volumes.reindex(full_days)

    return prices, volumes


def resample_crypto_weekly(prices_D: pd.DataFrame,
                           volumes_D: pd.DataFrame | None,
                           week_ending: str = "SUN"):
    """
    Create weekly panels from daily crypto:
      • Prices: LAST close of each week
      • Volumes: SUM of units across the week
    week_ending ∈ {"SUN","MON","TUE","WED","THU","FRI","SAT"} controls anchor.
    """
    anchor = f"W-{week_ending.upper()}"
    prices_W = prices_D.resample(anchor).last()
    volumes_W = volumes_D.resample(anchor).sum() if volumes_D is not None else None
    return prices_W, volumes_W


# ╔════════════════════════════════════════════════╗
# 1. Return calculations (kept from your version)
# ╚════════════════════════════════════════════════╝

def _final_return_cleanup(ret_df: pd.DataFrame, ffill_limit: int = 3) -> pd.DataFrame:
    """
    Ensure the return matrix is free of NaNs:
      • drop the first row created by diff()
      • forward-fill tiny gaps (≤ ffill_limit)
      • *then* drop any residual rows/cols that still contain NaNs
    """
    ret_df = (
        ret_df.iloc[1:]                         # toss the all-NaN first row
               .ffill(limit=ffill_limit)        # small holes
               .replace([np.inf, -np.inf], np.nan)
               .dropna(axis=0, how="any")       # purge rows with remaining NaNs
               .dropna(axis=1, how="any")       # purge columns with remaining NaNs
    )
    return ret_df


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """R_t = P_t / P_{t-1} - 1   (preserves NaNs where data is missing). Works for any frequency."""
    ret = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return _final_return_cleanup(ret)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln(P_t) - ln(P_{t-1}); works for any frequency."""
    ret = np.log(prices).diff().replace([np.inf, -np.inf], np.nan)
    return _final_return_cleanup(ret)


# ╔════════════════════════════════════════════════╗
# 2. Tidying helpers (kept)
# ╚════════════════════════════════════════════════╝

def wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert wide (Date index, tickers in columns) → long
    with columns: Date | Ticker | <value_name>.
    Robust to unnamed index (so reset_index creates 'Date' directly).
    """
    df = df.copy()
    if df.index.name is None:
        df.index.name = "Date"
    return (
        df.reset_index()
          .melt(id_vars="Date", var_name="Ticker", value_name=value_name)
          .dropna(subset=[value_name])
    )



# ╔════════════════════════════════════════════════╗
# 3. Convenience bundles (extended)
# ╚════════════════════════════════════════════════╝

def prepare_all(prices: pd.DataFrame, sample: int | None = None) -> dict:
    """
    Optionally sub-samples a random set of columns first (for fast smoke-runs).
    Returns a dict with:
        - prices        (input, no further cleaning)
        - ret_simple    (pct returns)
        - ret_log       (log returns)
        - long_prices   (tidy, long-form)
        - long_log_ret  (tidy, long-form)
    """
    if sample and sample < prices.shape[1]:
        keep = np.random.default_rng(42).choice(prices.columns, size=sample, replace=False)
        prices = prices[keep]

    ret_simple = compute_simple_returns(prices)
    ret_log    = compute_log_returns(prices)

    out = {
        "prices":       prices,
        "ret_simple":   ret_simple,
        "ret_log":      ret_log,
        "long_prices":  wide_to_long(prices, "Price"),
        "long_log_ret": wide_to_long(ret_log, "LogRet"),
    }
    return out


# NEW: one-call prep directly from data/cryptos/, with DAILY or WEEKLY frequency
def prepare_crypto_from_folder(folder: str = "data/cryptos",
                               freq: str = "D",
                               week_ending: str = "SUN",
                               dollar_volume: bool = True,
                               sample: int | None = None) -> dict:
    """
    Load per-coin CSVs from `folder`, build DAILY or WEEKLY panels, then returns:
        - prices_{D|W}
        - volumes_units_{D|W} (if available)
        - volumes_dollar_{D|W} (if volume available and dollar_volume=True)
        - ret_simple_{D|W}
        - ret_log_{D|W}
        - long_prices_{D|W}
        - long_log_ret_{D|W}
    """
    prices_D, volumes_D = build_crypto_panels_from_folder(folder)

    # Optional column sampling before resampling to keep shapes consistent across outputs
    if sample and sample < prices_D.shape[1]:
        keep = np.random.default_rng(42).choice(prices_D.columns, size=sample, replace=False)
        prices_D = prices_D[keep]
        if volumes_D is not None:
            volumes_D = volumes_D.reindex(columns=keep)

    if freq.upper().startswith("W"):
        prices, volumes = resample_crypto_weekly(prices_D, volumes_D, week_ending=week_ending)
        tag = "W"
    else:
        prices, volumes = prices_D, volumes_D
        tag = "D"

    # Dollar volume uses yesterday's close (conservative)
    volumes_dollar = None
    if volumes is not None and dollar_volume:
        volumes_dollar = volumes * prices.shift(1)

    # Returns
    ret_simple = compute_simple_returns(prices)
    ret_log    = compute_log_returns(prices)

    out = {
        f"prices_{tag}":            prices,
        f"ret_simple_{tag}":        ret_simple,
        f"ret_log_{tag}":           ret_log,
        f"long_prices_{tag}":       wide_to_long(prices, "Price"),
        f"long_log_ret_{tag}":      wide_to_long(ret_log, "LogRet"),
        f"volumes_units_{tag}":     volumes,
        f"volumes_dollar_{tag}":    volumes_dollar,
    }
    return out


# Convenience wrappers for explicit daily/weekly use in the pipeline
def prepare_crypto_daily(folder: str = "data/cryptos",
                         dollar_volume: bool = True,
                         sample: int | None = None) -> dict:
    return prepare_crypto_from_folder(folder=folder, freq="D",
                                      dollar_volume=dollar_volume, sample=sample)


def prepare_crypto_weekly(folder: str = "data/cryptos",
                          week_ending: str = "SUN",
                          dollar_volume: bool = True,
                          sample: int | None = None) -> dict:
    return prepare_crypto_from_folder(folder=folder, freq="W",
                                      week_ending=week_ending,
                                      dollar_volume=dollar_volume, sample=sample)
