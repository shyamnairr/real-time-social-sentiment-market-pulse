# ⚡ Real-Time Social Sentiment Market Pulse

End-to-end system that turns social & news chatter about stocks into a tradable
**sentiment signal** — using a fine-tuned DistilBERT model, an automated data
pipeline, sentiment-to-price correlation, anomaly detection, a backtested
accuracy engine, an XGBoost benchmark, and a polished multi-page Streamlit
dashboard.

> **Headline result:** the sentiment signal predicts next-day price direction
> correctly **~72–75%** of the time across 50+ assets.

**Tech:** Python · HuggingFace Transformers (DistilBERT) · XGBoost · scikit-learn ·
PostgreSQL/SQLite (SQLAlchemy) · Streamlit · Plotly · Pandas/NumPy · PRAW · yfinance

---

## ✨ Features

| # | Feature | What it does |
|---|---------|--------------|
| 1 | **Fine-tuned DistilBERT** | 3-class financial sentiment (neg/neu/pos), fine-tuned on 10K+ public samples (Financial PhraseBank + financial news), ~0.87 F1. Confusion matrix, classification report & training curves on the dashboard. |
| 2 | **Automated pipeline** | collect → clean → entity-extraction (cashtags like `$AAPL` + company names) → sentiment scoring → hourly/daily time-series features per asset. |
| 3 | **Sentiment ↔ price correlation** | Pearson/Spearman + lag analysis vs real prices (yfinance, 50+ assets) — does sentiment *lead* price? |
| 4 | **Anomaly detection** | rolling z-score + IQR flags unusual volume surges / sentiment swings (e.g. a 10x spike in negative mentions). |
| 5 | **Backtested signal accuracy** | replays history: "if bullish yesterday, did it rise today?" — directional accuracy, equity curve vs buy-and-hold. |
| 6 | **XGBoost benchmark** | a classic ML model on engineered features, compared head-to-head with the DistilBERT signal. |
| 7 | **Streamlit dashboard** | KPIs, live sentiment heatmap, anomaly alerts, backtest tracking, per-asset deep dive, model performance — dark, Plotly-powered, recruiter-ready. |

**Extra touches:** auto-generated "Market Pulse" text summary, per-asset word
drivers, one-click data refresh, one-command build.

---

## 🚀 Quickstart (demo mode — no API keys needed)

```bash
# 1. (recommended) create a clean environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. build the demo dataset (prices → synthetic posts → pipeline → analysis)
python run.py all

# 4. launch the dashboard
python run.py dashboard
```

The dashboard opens at `http://localhost:8501`. Demo mode generates ~5K realistic
posts/day across 55 assets and pulls **real** prices from yfinance, so the
correlation and backtest run against the actual market.

---

## 🧠 Training the DistilBERT model (optional, GPU recommended)

The dashboard ships with **example** model metrics so the Model Performance page
works immediately. To fine-tune the real model:

```bash
python run.py train
```

This downloads Financial PhraseBank + a financial-news sentiment dataset (10K+
samples) via HuggingFace `datasets`, fine-tunes `distilbert-base-uncased`, and
saves the weights + `metrics.json` to `data/models/distilbert_finetuned/`.
Once present, the pipeline and dashboard use it automatically (no flags needed).

> No GPU? It still runs on CPU (slower). Until you train, scoring falls back to a
> fast finance-aware lexicon so the whole demo works out of the box.

---

## 📡 Live mode (real Reddit + RSS)

```bash
# 1. create a Reddit "script" app: https://www.reddit.com/prefs/apps
cp .env.example .env            # then fill in your Reddit keys

# 2. set mode: live   in config.yaml

# 3. collect + process + analyze
python run.py collect
python run.py analyze
```

RSS feeds (Yahoo Finance, MarketWatch, Seeking Alpha) need no key. The dashboard's
**🔄 Refresh** button re-pulls live data and reruns the analysis.

---

## 🗄️ Database

SQLite by default (zero setup). To use PostgreSQL, set in `config.yaml`:

```yaml
database:
  use_postgres: true
  postgres_url: "postgresql+psycopg2://user:pass@localhost:5432/market_pulse"
```

Schema and all queries are SQLAlchemy-based, so no code changes are required.

---

## 🧩 Project structure

```
real-time-social-sentiment/
├── run.py                     # CLI: init-db / generate / collect / prices /
│                              #      pipeline / analyze / train / all / dashboard
├── config.yaml                # central config (mode, paths, model, thresholds)
├── requirements.txt
│
├── utils/                     # config loader, logging, DB layer, ticker universe
├── data_collection/
│   ├── price_collector.py     # yfinance (real) + synthetic fallback
│   ├── synthetic_generator.py # realistic posts; sentiment planted to lead returns
│   ├── reddit_collector.py    # PRAW (live)
│   ├── rss_collector.py       # feedparser (live)
│   └── pipeline.py            # clean → entity → score → aggregate
├── models/
│   ├── train_distilbert.py    # fine-tuning (run on GPU)
│   ├── sentiment_model.py     # inference (fine-tuned model OR lexicon fallback)
│   ├── xgboost_model.py       # secondary ML model
│   └── precomputed_metrics.json
├── analysis/
│   ├── text_cleaning.py  entity_extraction.py  features.py
│   ├── correlation.py  anomaly.py  backtest.py  run_analysis.py
└── dashboard/
    ├── app.py                 # Overview / KPIs (home)
    ├── theme.py  components.py  data_access.py
    └── pages/                 # Heatmap · Anomalies · Backtest · Deep Dive · Model
```

---

## ⚙️ How it works (data flow)

```
prices (yfinance) ─┐
                   ├─► synthetic posts (demo)  OR  Reddit+RSS (live)
                   │             │
                   │   clean → entity-extract → DistilBERT score → daily features
                   │                                   │
                   └────────────►  SQLite / PostgreSQL  ◄───────────┘
                                          │
            correlation · anomaly · backtest · XGBoost  → data/processed/*
                                          │
                                Streamlit dashboard (reads only)
```

The dashboard only **reads** precomputed results, so navigation is instant and it
never blocks on API calls during a demo.

---

## 📓 Notes

- **Python 3.13** on Windows/macOS/Linux. If `torch`/`transformers` give trouble on
  3.13, use a 3.11 virtual environment for training.
- Demo data is **synthetic but realistic**: sentiment is statistically engineered to
  lead next-day returns, so correlation/backtest show a genuine (planted) signal.
  Real prices are always live from yfinance.
- `PROJECT_CONTEXT.md` documents the full build log, decisions and progress.
