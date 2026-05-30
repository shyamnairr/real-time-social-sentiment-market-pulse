"""
Database layer (SQLAlchemy Core). SQLite by default; flip config.database.use_postgres
to true (and set postgres_url) to use PostgreSQL with the SAME schema/code.

Tables
------
posts            : every individual scored post/headline
sentiment_daily  : per-asset per-day aggregated sentiment + volume features
prices           : per-asset per-day OHLCV from yfinance

Helper functions wrap pandas <-> SQL so the rest of the code stays clean.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import (
    Column, Date, DateTime, Float, Integer, MetaData, String, Table, Text,
    create_engine,
)

from utils.config import CONFIG, get_path, ensure_dirs
from utils.logger import get_logger

log = get_logger("utils.db")

metadata = MetaData()

posts_table = Table(
    "posts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("post_id", String(64), index=True),       # source id (reddit id / synthetic)
    Column("source", String(32)),                    # reddit_wsb / rss_yahoo / synthetic
    Column("ticker", String(12), index=True),
    Column("created_at", DateTime, index=True),
    Column("text", Text),
    Column("clean_text", Text),
    Column("sentiment_label", String(12)),           # positive/negative/neutral
    Column("sentiment_score", Float),                # signed: -1..+1
    Column("confidence", Float),                     # model softmax max prob
    Column("upvotes", Integer),                      # engagement (0 for rss)
)

sentiment_daily_table = Table(
    "sentiment_daily", metadata,
    Column("ticker", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("mean_sentiment", Float),                 # confidence-weighted mean -1..+1
    Column("post_volume", Integer),
    Column("pct_positive", Float),
    Column("pct_negative", Float),
    Column("sentiment_momentum", Float),             # change vs previous day
    Column("volume_change", Float),                  # pct change in volume vs prev day
)

prices_table = Table(
    "prices", metadata,
    Column("ticker", String(12), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("volume", Float),
    Column("daily_return", Float),                   # close-to-close pct change
)


def get_engine():
    """Create the SQLAlchemy engine for SQLite (default) or PostgreSQL."""
    ensure_dirs()
    db_cfg = CONFIG["database"]
    if db_cfg.get("use_postgres"):
        url = db_cfg["postgres_url"]
        log.info("Using PostgreSQL backend")
    else:
        url = f"sqlite:///{get_path('db_file')}"
    return create_engine(url, future=True)


def init_db(engine=None, drop: bool = False):
    """Create tables (optionally dropping existing ones first)."""
    engine = engine or get_engine()
    if drop:
        metadata.drop_all(engine)
        log.info("Dropped existing tables")
    metadata.create_all(engine)
    log.info("Database schema ready")
    return engine


# --- pandas helpers -------------------------------------------------------
def write_df(df: pd.DataFrame, table: str, engine=None, if_exists: str = "append"):
    """Write a DataFrame to a table."""
    if df is None or df.empty:
        log.warning("write_df: empty frame for table '%s' (skipped)", table)
        return
    engine = engine or get_engine()
    df.to_sql(table, engine, if_exists=if_exists, index=False, chunksize=10000)
    log.info("Wrote %d rows -> %s", len(df), table)


def read_table(table: str, engine=None) -> pd.DataFrame:
    """Read an entire table into a DataFrame (returns empty frame if missing)."""
    engine = engine or get_engine()
    try:
        return pd.read_sql_table(table, engine)
    except ValueError:
        return pd.DataFrame()


def read_query(sql: str, engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    return pd.read_sql_query(sql, engine)
