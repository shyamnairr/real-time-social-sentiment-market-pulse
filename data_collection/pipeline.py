"""
The automated pipeline: collect -> clean -> entity-extract -> score -> aggregate.

run_pipeline()        : clean + score posts already in the DB, then build daily features.
run_live_collection() : (LIVE mode) pull Reddit+RSS, store raw posts, then run_pipeline().

Demo mode populates `posts` via synthetic_generator; live mode populates it via the
collectors here. Either way run_pipeline() finishes the job.
"""
from __future__ import annotations

import time

import pandas as pd

from analysis.features import build_daily_features
from analysis.text_cleaning import clean_series
from models.sentiment_model import get_scorer
from utils.db import read_table, write_df, get_engine, posts_table
from utils.logger import get_logger

log = get_logger("pipeline")


def run_pipeline():
    """Clean + score all posts, then generate daily time-series features."""
    posts = read_table("posts")
    if posts.empty:
        log.warning("No posts to process. Run `generate` (demo) or `collect` (live) first.")
        return

    log.info("Cleaning %d posts...", len(posts))
    posts["clean_text"] = clean_series(posts["text"])

    log.info("Scoring sentiment...")
    t0 = time.time()
    scorer = get_scorer()
    scored = scorer.score_texts(posts["clean_text"].tolist())
    posts["sentiment_label"] = scored["label"].values
    posts["sentiment_score"] = scored["score"].values
    posts["confidence"] = scored["confidence"].values
    log.info("Scored %d posts in %.1fs (backend=%s)",
             len(posts), time.time() - t0, scorer.backend)

    write_df(posts, "posts", if_exists="replace")

    log.info("Building daily features...")
    daily = build_daily_features(posts)
    write_df(daily, "sentiment_daily", if_exists="replace")
    log.info("Pipeline complete: %d daily rows across %d tickers",
             len(daily), daily["ticker"].nunique())


def run_live_collection():
    """LIVE mode: fetch Reddit + RSS, store raw posts, then run the pipeline."""
    from data_collection.reddit_collector import fetch_reddit_posts
    from data_collection.rss_collector import fetch_rss_posts

    frames = []
    rdt = fetch_reddit_posts()
    if not rdt.empty:
        frames.append(rdt)
    rss = fetch_rss_posts()
    if not rss.empty:
        frames.append(rss)

    if not frames:
        log.warning("Live collection returned nothing. Check .env / connectivity. "
                    "Falling back to demo data is recommended (mode: demo).")
        return

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["post_id", "ticker"])
    log.info("Collected %d (post,ticker) rows live", len(raw))

    # append to existing posts (live mode accumulates over refreshes)
    engine = get_engine()
    posts_table.create(engine, checkfirst=True)
    write_df(raw, "posts", engine=engine, if_exists="append")

    run_pipeline()
