"""
Cached data-access layer for the dashboard.

All DB reads and artifact loads are wrapped in @st.cache_data so page navigation is
instant. `clear_caches()` is called by the Refresh button.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.features import load_merged as _load_merged
from utils.config import CONFIG, ROOT, get_path
from utils.db import read_table, read_query

PROCESSED = get_path("processed_dir")
MODELS_DIR = get_path("models_dir")


def clear_caches():
    st.cache_data.clear()


# --- raw tables -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def posts_total() -> int:
    df = read_query("SELECT COUNT(*) AS n FROM posts")
    return int(df["n"].iloc[0]) if not df.empty else 0


@st.cache_data(show_spinner=False)
def sentiment_daily() -> pd.DataFrame:
    df = read_table("sentiment_daily")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def prices() -> pd.DataFrame:
    df = read_table("prices")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def merged() -> pd.DataFrame:
    return _load_merged()


@st.cache_data(show_spinner=False)
def ticker_texts(ticker: str, limit: int = 4000) -> pd.DataFrame:
    """Recent cleaned post text + label for a ticker (for word-driver analysis)."""
    return read_query(
        f"SELECT clean_text, sentiment_label FROM posts WHERE ticker = '{ticker}' "
        f"ORDER BY created_at DESC LIMIT {limit}"
    )


@st.cache_data(show_spinner=False)
def recent_posts(ticker: str, limit: int = 12) -> pd.DataFrame:
    return read_query(
        f"SELECT created_at, source, text, sentiment_label, sentiment_score, "
        f"confidence, upvotes FROM posts WHERE ticker = '{ticker}' "
        f"ORDER BY created_at DESC LIMIT {limit}"
    )


# --- processed artifacts --------------------------------------------------
def _csv(name: str) -> pd.DataFrame:
    p = PROCESSED / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def correlation_by_ticker() -> pd.DataFrame:
    return _csv("correlation_by_ticker.csv")


@st.cache_data(show_spinner=False)
def lag_analysis() -> pd.DataFrame:
    return _csv("lag_analysis.csv")


@st.cache_data(show_spinner=False)
def anomalies() -> pd.DataFrame:
    df = _csv("anomalies.csv")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def backtest_summary() -> dict:
    return _json(PROCESSED / "backtest_summary.json")


@st.cache_data(show_spinner=False)
def backtest_by_ticker() -> pd.DataFrame:
    return _csv("backtest_by_ticker.csv")


@st.cache_data(show_spinner=False)
def equity_curve() -> pd.DataFrame:
    df = _csv("equity_curve.csv")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def xgboost_metrics() -> dict:
    return _json(PROCESSED / "xgboost_metrics.json")


@st.cache_data(show_spinner=False)
def feature_importance() -> pd.DataFrame:
    return _csv("feature_importance.csv")


@st.cache_data(show_spinner=False)
def model_metrics() -> dict:
    """Trained metrics if available, else the precomputed example fallback."""
    trained = MODELS_DIR / "distilbert_finetuned" / "metrics.json"
    if trained.exists():
        return _json(trained)
    fallback = ROOT / "models" / "precomputed_metrics.json"
    return _json(fallback)


# --- derived KPI helpers --------------------------------------------------
@st.cache_data(show_spinner=False)
def latest_day_kpis() -> dict:
    """KPIs for the most recent day with data."""
    d = sentiment_daily()
    if d.empty:
        return {}
    last = d["date"].max()
    today = d[d["date"] == last]
    prev = d[d["date"] == (d["date"][d["date"] < last].max())] if (d["date"] < last).any() else today

    avg_today = float((today["mean_sentiment"] * today["post_volume"]).sum()
                      / max(today["post_volume"].sum(), 1))
    avg_prev = float((prev["mean_sentiment"] * prev["post_volume"]).sum()
                     / max(prev["post_volume"].sum(), 1))

    ranked = today.sort_values("mean_sentiment", ascending=False)
    top_bull = ranked.head(3)[["ticker", "mean_sentiment"]].values.tolist()
    top_bear = ranked.tail(3)[["ticker", "mean_sentiment"]].values.tolist()[::-1]

    return {
        "date": last,
        "avg_sentiment": avg_today,
        "avg_sentiment_delta": avg_today - avg_prev,
        "posts_today": int(today["post_volume"].sum()),
        "top_bullish": top_bull,
        "top_bearish": top_bear,
        "n_assets": int(today["ticker"].nunique()),
    }


def mode() -> str:
    return CONFIG.get("mode", "demo")
