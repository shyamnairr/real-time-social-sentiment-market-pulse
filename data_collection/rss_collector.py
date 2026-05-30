"""
RSS collector (LIVE mode) via feedparser.

Parses the configured financial RSS feeds (Yahoo Finance, MarketWatch, Seeking
Alpha), extracts tickers from headlines/summaries, and returns one row per
(headline, ticker). No API key needed.
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from analysis.entity_extraction import extract_tickers
from utils.config import CONFIG
from utils.logger import get_logger

log = get_logger("rss_collector")


def _source_name(url: str) -> str:
    u = url.lower()
    if "yahoo" in u:
        return "rss_yahoo"
    if "dowjones" in u or "marketwatch" in u or "mw_" in u:
        return "rss_marketwatch"
    if "seekingalpha" in u:
        return "rss_seekingalpha"
    return "rss_other"


def fetch_rss_posts() -> pd.DataFrame:
    import feedparser

    rows = []
    for url in CONFIG["rss_feeds"]:
        src = _source_name(url)
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            log.warning("RSS parse failed for %s (%s)", url, e)
            continue
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            text = f"{title}. {summary}".strip()
            # published time -> datetime (fallback: now)
            tstruct = entry.get("published_parsed") or entry.get("updated_parsed")
            created = (dt.datetime.utcfromtimestamp(time.mktime(tstruct))
                       if tstruct else dt.datetime.utcnow())
            for tk in extract_tickers(text):
                rows.append({
                    "post_id": f"rss_{abs(hash(entry.get('id', text)))%10**12}",
                    "source": src,
                    "ticker": tk,
                    "created_at": created,
                    "text": text[:1000],
                    "upvotes": 0,
                })
        log.info("%s: parsed %d entries", src, len(feed.entries))

    df = pd.DataFrame(rows)
    log.info("RSS: %d (headline,ticker) rows", len(df))
    return df
