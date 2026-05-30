"""
Price collector -> fills the `prices` table.

Strategy:
  1. Try yfinance for real OHLCV (works without API keys; needs internet).
  2. If yfinance is unavailable / offline / returns nothing, fall back to a
     reproducible synthetic geometric-random-walk so the demo ALWAYS builds.

Real prices make the correlation/backtest against the actual market far more
convincing, so we prefer them whenever possible.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from utils.config import CONFIG
from utils.db import write_df, get_engine, prices_table
from utils.logger import get_logger
from utils.tickers import TICKERS

log = get_logger("price_collector")


def _business_days(n_days: int) -> pd.DatetimeIndex:
    """Last `n_days` calendar days mapped to business days, ending today."""
    end = pd.Timestamp(dt.date.today())
    start = end - pd.Timedelta(days=int(n_days * 1.5) + 10)  # buffer for weekends
    days = pd.bdate_range(start=start, end=end)
    return days[-n_days:]


def _synthetic_prices(dates: pd.DatetimeIndex, tickers: list[str], seed: int) -> pd.DataFrame:
    """Geometric random walk per ticker -> realistic-looking OHLCV."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, tk in enumerate(tickers):
        # Per-ticker character: starting price, drift, volatility
        price = float(rng.uniform(20, 400))
        mu = rng.normal(0.0004, 0.0006)        # small daily drift
        sigma = rng.uniform(0.012, 0.045)      # daily volatility
        rets = rng.normal(mu, sigma, len(dates))
        closes = price * np.cumprod(1 + rets)
        for d, c, r in zip(dates, closes, rets):
            o = c / (1 + rng.normal(0, sigma / 3))
            hi = max(o, c) * (1 + abs(rng.normal(0, sigma / 4)))
            lo = min(o, c) * (1 - abs(rng.normal(0, sigma / 4)))
            vol = float(rng.lognormal(15, 0.6))
            rows.append((tk, d.date(), o, hi, lo, c, vol, r))
    df = pd.DataFrame(
        rows,
        columns=["ticker", "date", "open", "high", "low", "close", "volume", "daily_return"],
    )
    return df


def _fetch_yfinance(dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame | None:
    """Return a tidy prices DataFrame from yfinance, or None if it fails."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed -> using synthetic prices")
        return None

    start = dates[0].date().isoformat()
    end = (dates[-1] + pd.Timedelta(days=1)).date().isoformat()
    try:
        raw = yf.download(
            tickers, start=start, end=end, group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:  # network / API issues
        log.warning("yfinance download failed (%s) -> synthetic fallback", e)
        return None

    if raw is None or len(raw) == 0:
        log.warning("yfinance returned no data -> synthetic fallback")
        return None

    frames = []
    for tk in tickers:
        try:
            sub = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        sub = sub.dropna(subset=["Close"]).copy()
        if sub.empty:
            continue
        sub = sub.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        })
        sub["ticker"] = tk
        sub["date"] = pd.to_datetime(sub["date"]).dt.date
        sub["daily_return"] = sub["close"].pct_change().fillna(0.0)
        frames.append(sub[["ticker", "date", "open", "high", "low",
                            "close", "volume", "daily_return"]])

    if not frames:
        log.warning("yfinance gave no usable tickers -> synthetic fallback")
        return None
    out = pd.concat(frames, ignore_index=True)
    log.info("yfinance: fetched %d rows across %d tickers",
             len(out), out["ticker"].nunique())
    return out


def fetch_and_store_prices() -> pd.DataFrame:
    """Fetch (real or synthetic) prices for the full ticker universe and store them."""
    n_days = CONFIG["synthetic"]["days"]
    seed = CONFIG["synthetic"]["seed"]
    dates = _business_days(n_days)

    df = _fetch_yfinance(dates, TICKERS)
    if df is None:
        log.info("Generating synthetic prices for %d tickers x %d days",
                 len(TICKERS), len(dates))
        df = _synthetic_prices(dates, TICKERS, seed)

    # Replace the table contents (idempotent rebuilds)
    engine = get_engine()
    prices_table.drop(engine, checkfirst=True)
    prices_table.create(engine, checkfirst=True)
    write_df(df, "prices", engine=engine, if_exists="append")
    return df


if __name__ == "__main__":
    fetch_and_store_prices()
