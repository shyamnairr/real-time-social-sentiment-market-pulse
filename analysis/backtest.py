"""
Backtest: did the sentiment signal predict next-day price direction?

Signal rule (from config):
  mean_sentiment >  bullish_threshold  -> BULLISH (predict up)
  mean_sentiment <  bearish_threshold  -> BEARISH (predict down)
  otherwise                            -> no trade

We compare each active signal to the actual next-day return sign and track:
  - overall directional accuracy   (the headline ~72% metric)
  - bullish / bearish accuracy
  - per-asset accuracy
  - an equity curve: $1 following the signals (long bullish / short bearish, equal
    weight across active assets each day) vs. equal-weight buy-and-hold.

Artifacts (data/processed/):
  - backtest_summary.json
  - backtest_by_ticker.csv
  - equity_curve.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.features import load_merged
from utils.config import CONFIG, get_path
from utils.logger import get_logger

log = get_logger("backtest")


def run_backtest(merged: pd.DataFrame | None = None):
    merged = load_merged() if merged is None else merged
    if merged.empty:
        log.warning("No merged data for backtest")
        return {}

    acfg = CONFIG["analysis"]
    bull, bear = acfg["signal_bullish_threshold"], acfg["signal_bearish_threshold"]
    cap0 = CONFIG["backtest"]["starting_capital"]

    df = merged.copy()
    # signal: +1 bullish, -1 bearish, 0 none
    df["signal"] = np.where(df["mean_sentiment"] > bull, 1,
                            np.where(df["mean_sentiment"] < bear, -1, 0))
    active = df[df["signal"] != 0].copy()
    active["correct"] = (np.sign(active["next_return"]) == active["signal"]).astype(int)
    # the position return: long bullish / short bearish
    active["position_return"] = active["signal"] * active["next_return"]

    # --- headline metrics ------------------------------------------------
    n_signals = len(active)
    accuracy = float(active["correct"].mean()) if n_signals else 0.0
    bull_mask = active["signal"] == 1
    bear_mask = active["signal"] == -1
    bull_acc = float(active.loc[bull_mask, "correct"].mean()) if bull_mask.any() else 0.0
    bear_acc = float(active.loc[bear_mask, "correct"].mean()) if bear_mask.any() else 0.0

    # --- per-ticker ------------------------------------------------------
    by_ticker = (active.groupby("ticker")["correct"]
                 .agg(n_signals="size", accuracy="mean").reset_index()
                 .sort_values("accuracy", ascending=False))

    # --- equity curves (by date) ----------------------------------------
    strat_daily = active.groupby("date")["position_return"].mean()
    bh_daily = df.groupby("date")["next_return"].mean()      # equal-weight hold
    idx = sorted(set(strat_daily.index) | set(bh_daily.index))
    strat_daily = strat_daily.reindex(idx).fillna(0.0)
    bh_daily = bh_daily.reindex(idx).fillna(0.0)
    equity = pd.DataFrame({
        "date": pd.to_datetime(idx),
        "strategy_value": cap0 * (1 + strat_daily.values).cumprod(),
        "buyhold_value": cap0 * (1 + bh_daily.values).cumprod(),
    })

    summary = {
        "directional_accuracy": accuracy,
        "n_signals": int(n_signals),
        "n_bullish": int(bull_mask.sum()),
        "n_bearish": int(bear_mask.sum()),
        "bullish_accuracy": bull_acc,
        "bearish_accuracy": bear_acc,
        "strategy_total_return": float(equity["strategy_value"].iloc[-1] / cap0 - 1),
        "buyhold_total_return": float(equity["buyhold_value"].iloc[-1] / cap0 - 1),
        "n_assets": int(active["ticker"].nunique()),
        "bullish_threshold": bull,
        "bearish_threshold": bear,
    }

    out = get_path("processed_dir")
    with open(out / "backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    by_ticker.to_csv(out / "backtest_by_ticker.csv", index=False)
    equity.to_csv(out / "equity_curve.csv", index=False)

    log.info("Backtest: %.1f%% directional accuracy on %d signals (%d assets)",
             accuracy * 100, n_signals, summary["n_assets"])
    return summary
