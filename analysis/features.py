"""
Time-series feature generation.

Turns scored individual posts into per-asset, per-day features:
  - mean_sentiment    : CONFIDENCE-WEIGHTED mean signed sentiment (extra feature B)
  - post_volume       : number of posts that day
  - pct_positive / pct_negative
  - sentiment_momentum: change in mean_sentiment vs previous day
  - volume_change     : pct change in volume vs previous day

These feed correlation, anomaly detection, backtesting and the XGBoost model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.db import read_table


def load_merged() -> pd.DataFrame:
    """Join daily sentiment features with price data; add next-day return + direction.

    Shared by correlation, backtest and the XGBoost model.
    Columns added: daily_return, next_return, next_direction (1 up / 0 down).
    """
    daily = read_table("sentiment_daily")
    prices = read_table("prices")
    if daily.empty or prices.empty:
        return pd.DataFrame()

    daily["date"] = pd.to_datetime(daily["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["next_return"] = prices.groupby("ticker")["daily_return"].shift(-1)

    merged = daily.merge(
        prices[["ticker", "date", "close", "daily_return", "next_return"]],
        on=["ticker", "date"], how="inner",
    )
    merged = merged.dropna(subset=["next_return"])
    merged["next_direction"] = (merged["next_return"] > 0).astype(int)
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_daily_features(posts: pd.DataFrame) -> pd.DataFrame:
    """posts must have: ticker, created_at, sentiment_label, sentiment_score, confidence."""
    if posts.empty:
        return pd.DataFrame()

    df = posts.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at"].dt.date
    # weight = confidence, clipped so a 0.5 floor still contributes
    df["w"] = df["confidence"].fillna(0.5).clip(lower=0.1)
    df["wscore"] = df["sentiment_score"] * df["w"]

    grp = df.groupby(["ticker", "date"])
    daily = grp.agg(
        wscore_sum=("wscore", "sum"),
        w_sum=("w", "sum"),
        post_volume=("sentiment_score", "size"),
        n_pos=("sentiment_label", lambda s: (s == "positive").sum()),
        n_neg=("sentiment_label", lambda s: (s == "negative").sum()),
    ).reset_index()

    daily["mean_sentiment"] = daily["wscore_sum"] / daily["w_sum"].replace(0, np.nan)
    daily["mean_sentiment"] = daily["mean_sentiment"].fillna(0.0)
    daily["pct_positive"] = daily["n_pos"] / daily["post_volume"]
    daily["pct_negative"] = daily["n_neg"] / daily["post_volume"]

    daily = daily.sort_values(["ticker", "date"])
    daily["sentiment_momentum"] = daily.groupby("ticker")["mean_sentiment"].diff().fillna(0.0)
    daily["volume_change"] = (
        daily.groupby("ticker")["post_volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
    )

    return daily[[
        "ticker", "date", "mean_sentiment", "post_volume",
        "pct_positive", "pct_negative", "sentiment_momentum", "volume_change",
    ]]
