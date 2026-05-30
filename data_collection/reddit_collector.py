"""
Reddit collector (LIVE mode) via PRAW.

Pulls recent submissions from the configured subreddits, extracts tickers, and
returns one row per (post, ticker). Requires REDDIT_* credentials in .env.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from analysis.entity_extraction import extract_tickers
from utils.config import CONFIG, REDDIT_CREDS, has_reddit_creds
from utils.logger import get_logger

log = get_logger("reddit_collector")


def fetch_reddit_posts() -> pd.DataFrame:
    if not has_reddit_creds():
        log.warning("No Reddit credentials in .env -> skipping Reddit collection")
        return pd.DataFrame()

    import praw

    reddit = praw.Reddit(
        client_id=REDDIT_CREDS["client_id"],
        client_secret=REDDIT_CREDS["client_secret"],
        user_agent=REDDIT_CREDS["user_agent"],
    )

    rcfg = CONFIG["reddit"]
    rows = []
    for sub in rcfg["subreddits"]:
        try:
            for s in reddit.subreddit(sub).hot(limit=rcfg["posts_per_subreddit"]):
                text = f"{s.title}. {getattr(s, 'selftext', '') or ''}".strip()
                created = dt.datetime.utcfromtimestamp(s.created_utc)
                for tk in extract_tickers(text):
                    rows.append({
                        "post_id": f"rdt_{s.id}",
                        "source": f"reddit_{sub}",
                        "ticker": tk,
                        "created_at": created,
                        "text": text[:1000],
                        "upvotes": int(s.score),
                    })
            log.info("r/%s collected", sub)
        except Exception as e:
            log.warning("Reddit fetch failed for r/%s (%s)", sub, e)

    df = pd.DataFrame(rows)
    log.info("Reddit: %d (post,ticker) rows", len(df))
    return df
