"""
Real-Time Social Sentiment Market Pulse - command-line entrypoint.

Typical one-command demo:
    python run.py all          # build demo data -> prices -> analysis (no training)
    python run.py dashboard    # launch the Streamlit app

Individual steps:
    python run.py init-db                 # create database schema
    python run.py generate                # synthetic posts (demo mode)
    python run.py collect                 # real Reddit+RSS (live mode, needs .env)
    python run.py prices                  # pull yfinance price history
    python run.py pipeline                # clean -> entities -> score -> aggregate
    python run.py analyze                 # correlation + anomaly + backtest + xgboost
    python run.py train                   # FINE-TUNE DistilBERT (run on your GPU)

Subcommands import their heavy dependencies lazily so the CLI starts fast.
"""
import argparse
import subprocess
import sys

from utils.config import CONFIG
from utils.logger import get_logger

log = get_logger("run")


def cmd_init_db(_args):
    from utils.db import init_db
    init_db(drop=_args.reset)


def cmd_generate(_args):
    from data_collection.synthetic_generator import generate_and_store
    generate_and_store()


def cmd_collect(_args):
    from data_collection.pipeline import run_live_collection
    run_live_collection()


def cmd_prices(_args):
    from data_collection.price_collector import fetch_and_store_prices
    fetch_and_store_prices()


def cmd_pipeline(_args):
    from data_collection.pipeline import run_pipeline
    run_pipeline()


def cmd_analyze(_args):
    from analysis.run_analysis import run_all_analysis
    run_all_analysis()


def cmd_train(_args):
    from models.train_distilbert import train
    train()


def cmd_all(_args):
    """End-to-end demo build (no model training; uses precomputed/pretrained scoring)."""
    from utils.db import init_db
    init_db(drop=True)
    log.info("=== STEP 1/4: pull price history ===")
    cmd_prices(_args)
    log.info("=== STEP 2/4: generate synthetic posts (sentiment leads real returns) ===")
    cmd_generate(_args)
    log.info("=== STEP 3/4: run pipeline (clean/score/aggregate) ===")
    cmd_pipeline(_args)
    log.info("=== STEP 4/4: analysis (correlation/anomaly/backtest/xgboost) ===")
    cmd_analyze(_args)
    log.info("Build complete. Launch with:  python run.py dashboard")


def cmd_dashboard(_args):
    """Launch Streamlit."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Market Pulse CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="create database schema")
    p.add_argument("--reset", action="store_true", help="drop existing tables first")
    p.set_defaults(func=cmd_init_db)

    for name, fn, helptext in [
        ("generate", cmd_generate, "synthetic posts (demo mode)"),
        ("collect", cmd_collect, "real Reddit+RSS (live mode)"),
        ("prices", cmd_prices, "pull yfinance price history"),
        ("pipeline", cmd_pipeline, "clean/score/aggregate posts"),
        ("analyze", cmd_analyze, "correlation/anomaly/backtest/xgboost"),
        ("train", cmd_train, "fine-tune DistilBERT (GPU)"),
        ("dashboard", cmd_dashboard, "launch Streamlit dashboard"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("all", help="end-to-end demo build")
    sp.set_defaults(func=cmd_all)

    args = parser.parse_args()
    log.info("Mode: %s | command: %s", CONFIG["mode"], args.command)
    args.func(args)


if __name__ == "__main__":
    main()
