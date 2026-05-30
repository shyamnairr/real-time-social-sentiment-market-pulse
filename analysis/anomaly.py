"""
Anomaly detection on the daily sentiment time-series.

Two detectors per asset (configurable):
  - Z-SCORE on a rolling window: flags when today's value is > threshold standard
    deviations from the recent rolling mean. Applied to BOTH:
      * post_volume      (sudden surge in chatter, e.g. 10x mentions)
      * mean_sentiment   (sharp mood swing)
  - IQR method (global per ticker): value outside [Q1-1.5*IQR, Q3+1.5*IQR].

Each flag records direction (spike/drop), z-score, severity, and the value.

Artifact: data/processed/anomalies.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import CONFIG, get_path
from utils.db import read_table
from utils.logger import get_logger

log = get_logger("anomaly")


def _severity(z: float) -> str:
    a = abs(z)
    if a >= 4:
        return "critical"
    if a >= 3:
        return "high"
    return "medium"


def detect_anomalies(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = read_table("sentiment_daily") if daily is None else daily
    if daily.empty:
        log.warning("No daily data for anomaly detection")
        return pd.DataFrame()

    acfg = CONFIG["analysis"]
    thr = acfg["anomaly_zscore_threshold"]
    win = acfg["anomaly_rolling_window"]

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ticker", "date"])

    rows = []
    for tk, g in daily.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        for metric in ("post_volume", "mean_sentiment"):
            s = g[metric].astype(float)
            roll_mean = s.rolling(win, min_periods=max(5, win // 3)).mean()
            roll_std = s.rolling(win, min_periods=max(5, win // 3)).std()
            z = (s - roll_mean) / roll_std.replace(0, np.nan)

            # IQR bounds (global per ticker) as a second opinion
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            for i in range(len(g)):
                zi = z.iloc[i]
                if pd.isna(zi):
                    continue
                iqr_out = s.iloc[i] < lo or s.iloc[i] > hi
                if abs(zi) >= thr or iqr_out:
                    direction = "spike" if zi > 0 else "drop"
                    rows.append({
                        "ticker": tk,
                        "date": g["date"].iloc[i].date(),
                        "metric": metric,
                        "direction": direction,
                        "value": float(s.iloc[i]),
                        "zscore": float(zi),
                        "iqr_flag": bool(iqr_out),
                        "severity": _severity(zi),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "zscore"], key=lambda c: c if c.name == "date"
                            else c.abs(), ascending=False)
    out = get_path("processed_dir")
    df.to_csv(out / "anomalies.csv", index=False)
    log.info("Anomalies: %d flags (%d volume, %d sentiment)",
             len(df),
             int((df["metric"] == "post_volume").sum()) if not df.empty else 0,
             int((df["metric"] == "mean_sentiment").sum()) if not df.empty else 0)
    return df
