"""
Correlation of sentiment with price movements.

For each asset:
  - Pearson & Spearman correlation between daily mean_sentiment and SAME-day return
    and NEXT-day return.
  - Lag analysis: correlation of sentiment(t) with return(t+lag) for lag in
    [-max_lag .. +max_lag]; positive lag where corr peaks => sentiment LEADS price.

Artifacts (data/processed/):
  - correlation_by_ticker.csv
  - lag_analysis.csv            (long format: ticker, lag, corr)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.features import load_merged
from utils.config import CONFIG, get_path
from utils.logger import get_logger

log = get_logger("correlation")


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(a.corr(b, method=method))


def compute_correlations(merged: pd.DataFrame | None = None):
    merged = load_merged() if merged is None else merged
    if merged.empty:
        log.warning("No merged data for correlation")
        return pd.DataFrame(), pd.DataFrame()

    max_lag = CONFIG["analysis"]["correlation_max_lag"]
    by_ticker, lag_rows = [], []

    for tk, g in merged.groupby("ticker"):
        g = g.sort_values("date")
        sent = g["mean_sentiment"].reset_index(drop=True)
        same = g["daily_return"].reset_index(drop=True)
        nxt = g["next_return"].reset_index(drop=True)

        pear_same = _safe_corr(sent, same, "pearson")
        pear_next = _safe_corr(sent, nxt, "pearson")
        spear_next = _safe_corr(sent, nxt, "spearman")

        # lag scan: corr( sentiment(t), return(t+lag) )
        best_lag, best_corr = 0, 0.0
        for lag in range(-max_lag, max_lag + 1):
            shifted = same.shift(-lag)
            c = _safe_corr(sent, shifted, "pearson")
            lag_rows.append({"ticker": tk, "lag": lag, "corr": c})
            if not np.isnan(c) and abs(c) > abs(best_corr):
                best_lag, best_corr = lag, c

        by_ticker.append({
            "ticker": tk,
            "pearson_same_day": pear_same,
            "pearson_next_day": pear_next,
            "spearman_next_day": spear_next,
            "best_lag": best_lag,
            "best_lag_corr": best_corr,
            "n_days": len(g),
        })

    by_ticker_df = pd.DataFrame(by_ticker).sort_values(
        "pearson_next_day", ascending=False)
    lag_df = pd.DataFrame(lag_rows)

    out = get_path("processed_dir")
    by_ticker_df.to_csv(out / "correlation_by_ticker.csv", index=False)
    lag_df.to_csv(out / "lag_analysis.csv", index=False)
    log.info("Correlation: %d tickers | mean next-day pearson=%.3f",
             len(by_ticker_df), by_ticker_df["pearson_next_day"].mean())
    return by_ticker_df, lag_df
