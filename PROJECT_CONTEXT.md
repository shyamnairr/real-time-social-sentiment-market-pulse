# PROJECT_CONTEXT.md — Real-Time Social Sentiment Market Pulse

> **Living document.** If chat context is ever lost, READ THIS FIRST to resume without
> repeating work. Updated after every major step.

---

## 1. What this project is

A system that ingests social/news text about stocks, scores its sentiment with a
**fine-tuned DistilBERT** model, **correlates sentiment with real price moves**, flags
**anomalous sentiment spikes**, **backtests** whether sentiment predicted next-day
direction (~72% target), compares against an **XGBoost** model, and presents everything
in a polished **multi-page Streamlit dashboard**.

Resume bullet this implements (every feature MUST exist):
- Fine-tuned DistilBERT on 10K+ financial samples, ~0.87 F1, confusion matrix / report.
- Automated pipeline: collect (Reddit PRAW + financial RSS) → clean → entity extraction
  (tickers like $AAPL) → sentiment scoring → time-series feature generation.
- Correlation of sentiment with price (yfinance, 50+ assets), scatter plots, lag analysis.
- Anomaly detection (z-score / IQR) on sentiment + volume spikes.
- Backtested signal accuracy (~72% next-day directional).
- Streamlit dashboard: live heatmaps, anomaly alerts, backtest tracking, per-asset deep
  dive, model performance page, KPI cards.

---

## 2. Approved decisions (from user, locked in)

| Topic | Decision |
|---|---|
| Data mode | **Both**; `demo` (synthetic) is default + `live` (real Reddit/RSS/yfinance) |
| Training | **User HAS a GPU** — real DistilBERT fine-tune script; user runs `python run.py train` |
| Extra features | **All recommended**: (A) word drivers/cloud, (C) refresh button, (D) auto Market Pulse summary, (E) one-command launch; (B) confidence weighting if time |
| Database | **SQLite default, Postgres-ready** via SQLAlchemy (one config switch) |
| Charts | Plotly | 
| "Real-time" | Batch refresh (button / CLI), not streaming |

### Other technical decisions
- **10K+ training samples**: combine Financial PhraseBank + FiQA (+ optional Kaggle set)
  via HuggingFace `datasets`. (Implemented in Phase 3.)
- Dashboard only **reads** precomputed DB/artifacts → fast, never blocks on APIs in demo.
- Sentiment stored as signed score in [-1, +1]; confidence = softmax max prob.
- Daily sentiment is **confidence-weighted** (extra feature B).

---

## 3. File / module structure

```
real-time-social-sentiment/
├── README.md                     # setup + run (Phase 7)
├── PROJECT_CONTEXT.md            # THIS FILE
├── requirements.txt              # [DONE]
├── config.yaml                   # central config: mode, paths, model, analysis [DONE]
├── .env.example                  # Reddit API key template [DONE]
├── .gitignore                    # [DONE]
├── run.py                        # CLI: init-db/generate/collect/prices/pipeline/
│                                 #      analyze/train/all/dashboard [DONE]
│
├── utils/
│   ├── __init__.py               # [DONE]
│   ├── config.py                 # loads config.yaml + .env, paths, reddit creds [DONE]
│   ├── logger.py                 # console logging [DONE]
│   ├── db.py                     # SQLAlchemy schema (posts, sentiment_daily, prices)
│   │                             #   + pandas read/write helpers [DONE]
│   └── tickers.py                # 55-asset universe + name->ticker map [DONE]
│
├── data_collection/
│   ├── __init__.py               # [DONE]
│   ├── synthetic_generator.py    # realistic fake posts w/ embedded sentiment-price link [TODO P2]
│   ├── reddit_collector.py       # PRAW -> raw posts (live) [TODO P4]
│   ├── rss_collector.py          # feedparser -> headlines (live) [TODO P4]
│   ├── price_collector.py        # yfinance -> prices table [TODO P4]
│   └── pipeline.py               # orchestrate clean->entity->score->aggregate [TODO P4]
│
├── models/
│   ├── __init__.py               # [DONE]
│   ├── train_distilbert.py       # fine-tune + save model + metrics.json [TODO P3]
│   ├── sentiment_model.py        # load + batch inference (fallback to pretrained) [TODO P3]
│   └── xgboost_model.py          # secondary ML model + feature importance [TODO P5]
│
├── analysis/
│   ├── __init__.py               # [DONE]
│   ├── text_cleaning.py          # clean/normalize text [TODO P4]
│   ├── entity_extraction.py      # cashtag + name -> ticker [TODO P4]
│   ├── features.py               # daily aggregation features [TODO P4]
│   ├── correlation.py            # Pearson/Spearman + lag analysis [TODO P5]
│   ├── anomaly.py                # z-score / IQR detection [TODO P5]
│   ├── backtest.py               # directional accuracy + equity curve [TODO P5]
│   └── run_analysis.py           # runs all analysis, saves artifacts [TODO P5]
│
└── dashboard/                    # [TODO P6]
    ├── app.py                    # Streamlit entry + theme + nav + Market Pulse summary
    ├── components.py             # KPI cards, reusable charts
    └── pages/                    # Overview, Heatmap, Anomalies, Backtest, DeepDive, Model
```

### Artifacts (generated, gitignored, in `data/`)
- `data/market_pulse.db` — SQLite database.
- `data/models/distilbert_finetuned/` — model weights + `metrics.json` (F1, confusion
  matrix, classification report, training curve).
- `data/processed/*.json|*.csv` — analysis artifacts (correlation, anomalies, backtest,
  xgboost) the dashboard reads.

---

## 4. Data flow

```
demo:  synthetic_generator ─┐
live:  reddit + rss ────────┤→ raw posts → clean → entity extract → DistilBERT score
                            │                                            │
yfinance → prices table     │                          features.py (daily aggregation)
                            │                                            │
                            └────────────→ SQLite (posts, sentiment_daily, prices)
                                                          │
              correlation / anomaly / backtest / xgboost ─┘→ processed artifacts
                                                          │
                                              Streamlit dashboard (reads only)
```

---

## 5. Current progress

**Phase 1 — Scaffold: COMPLETE.** requirements, config.yaml, .env.example, .gitignore,
run.py, utils/ (config, logger, db, tickers). DB schema verified (`python run.py init-db`).

**Phase 2 — Synthetic generator + price collector: COMPLETE.**
- `price_collector.py`: yfinance real OHLCV (verified: 54/55 tickers, 6156 rows) with a
  synthetic geometric-random-walk fallback if offline. NOTE: ticker `SQ` renamed to `XYZ`
  (Block) because SQ is delisted on yfinance.
- `synthetic_generator.py`: generated 667,204 posts across 55 tickers / 114 days in ~38s.
  Sentiment leads next-day returns (per-ticker alpha ~N(0.72,0.07)); ~3% volume-spike
  anomalies injected; weighted sources (wsb dominant). Reads `prices` table for returns.
- Decision: in demo build, prices are fetched FIRST so sentiment is built to lead the
  REAL market (cmd_all order = prices -> generate -> pipeline -> analyze).

**Phase 3 — DistilBERT train + inference: COMPLETE (training runs on user GPU).**
- `train_distilbert.py`: combines financial_phrasebank + twitter-financial-news-sentiment
  (>10K) -> fine-tunes DistilBERT -> saves model + `metrics.json` (F1, confusion matrix,
  classification report, training history). Target ~0.87 F1. Run via `python run.py train`.
- `sentiment_model.py`: auto-selects backend — fine-tuned DistilBERT if present, else a
  finance-aware **lexicon scorer** (NO torch needed). Verified lexicon recovers sentiment
  from synthetic posts. API: `get_scorer().score_texts(list) -> df[label,score,confidence]`.
- `models/precomputed_metrics.json`: dashboard fallback (f1_weighted=0.882) so the Model
  Performance page works before training. Marked source="precomputed_example".

**Phase 4 — Pipeline + collectors: COMPLETE.**
- `analysis/text_cleaning.py` (URLs/markdown/emoji strip), `analysis/entity_extraction.py`
  (cashtags + bare symbols + company names, ambiguous-word guard),
  `analysis/features.py` (daily aggregation + `load_merged()` join helper).
- `data_collection/pipeline.py`: `run_pipeline()` clean->score->aggregate (verified:
  672K posts scored in 2.3s lexicon, 6270 daily rows). `run_live_collection()` for live.
- `reddit_collector.py` (PRAW) + `rss_collector.py` (feedparser) for live mode.
- CALIBRATION: per-ticker alpha lowered to N(0.55,0.08) so daily directional acc lands
  ~0.70 raw / ~0.74 thresholded (matches resume ~72%). Pearson ~0.56.

**Phase 5 — Analysis: COMPLETE.** Artifacts in data/processed/ (verified):
- `correlation.py` -> correlation_by_ticker.csv, lag_analysis.csv (mean next-day pearson 0.558).
- `anomaly.py` -> anomalies.csv (228 flags: rolling z-score + IQR on volume & sentiment).
- `backtest.py` -> backtest_summary.json (74.3% acc, 4748 signals), backtest_by_ticker.csv,
  equity_curve.csv (strategy vs buy&hold).
- `xgboost_model.py` -> xgboost_metrics.json (acc 0.685, AUC 0.771 vs DistilBERT-signal
  0.741), feature_importance.csv. Time-based split. NOTE: XGBoost scores every day while
  the signal is selective -> honest comparison talking point, not a bug.
- `run_analysis.py` orchestrates all four, sharing one load_merged() frame.

**Phase 6 — Streamlit dashboard: COMPLETE.** All 6 pages verified via Streamlit
AppTest (headless) — zero exceptions — and the live server boots (health 200 OK).
- `.streamlit/config.toml` (dark theme), `dashboard/theme.py` (palette, Plotly template,
  CSS, sentiment colorscale), `dashboard/components.py` (page_setup, sidebar + Refresh
  button [feature C], KPI cards, Market Pulse auto-summary [feature D]),
  `dashboard/data_access.py` (cached DB/artifact loaders, latest_day_kpis).
- `dashboard/app.py` = Overview/home (KPI cards, pulse summary, market trend).
- pages/: 1 Sentiment_Heatmap, 2 Anomaly_Alerts, 3 Backtest_and_Signals,
  4 Asset_Deep_Dive (price-vs-sentiment dual axis, scatter+OLS, lag bars, word drivers
  [feature A], recent posts), 5 Model_Performance (F1, confusion matrix, report, curves).
- GOTCHA fixed: each page self-bootstraps sys.path via an inline `__file__` upward search
  (the original `import _bootstrap` failed under Streamlit's multipage runner). Uses
  `width="stretch"` (not deprecated `use_container_width`).
- ENCODING GOTCHA (resolved): a PowerShell bulk Get/Set-Content round-trip mojibak ed all
  emojis; fixed by rewriting the 7 affected files with the Write tool (clean UTF-8).
  LESSON: don't use PS Get-Content|Set-Content for files with non-ASCII; use the Write tool.

**Phase 7 — README + end-to-end test: COMPLETE.**
- `README.md` written (quickstart, GPU training, live mode, Postgres switch, structure).
- `python run.py all` verified end-to-end (~70s): 55 tickers, 696K posts, pipeline,
  analysis. Dashboard server boots, health endpoint 200 OK.
- Removed unused deps (wordcloud/matplotlib); added statsmodels (OLS trendline).

### Phase checklist — ALL COMPLETE ✅
- [x] Phase 1–7 done. Project is fully runnable in demo mode with one command.

### Key numbers (CURRENT — after real-model re-score)
- ~696,751 posts | 55 tickers | 114 days
- **Model weighted F1 = 0.867 (REAL, trained)** | accuracy 86.7%
- **Backtest directional accuracy = 78.0%** (3,889 signals) | pearson 0.543
- XGBoost: acc 0.694, AUC 0.769 vs DistilBERT-signal 0.767 | anomalies 259

### MODEL TRAINED ✅ (2026-05-30, on user's RTX 3060)
- Real fine-tuned DistilBERT saved to data/models/distilbert_finetuned/ with metrics.json
  (`source: "trained"`). **Weighted F1 = 0.867**, accuracy 86.7%, macro F1 0.834.
  Trained on 12,230 / eval 2,159 samples (PhraseBank 4,846 + Twitter-fin 9,543 = 14,389).
  ~2.3 min train time. Dashboard now auto-shows real metrics (no "example" banner).
- After training, re-ran `pipeline` (re-score posts with the real model) + `analyze`.

### Environment gotchas hit during training setup (IMPORTANT for resuming)
- venv is **Python 3.12**. Default `pip install torch` gave a CPU-only build; installed
  GPU build via `pip install torch --index-url https://download.pytorch.org/whl/cu124`
  (torch 2.6.0+cu124). Verify with `torch.cuda.is_available()`.
- **Windows pyarrow/torch load-order crash:** importing `datasets` AFTER torch/transformers
  causes a hard access-violation (exit 0xC0000005, no traceback). FIX (already in code):
  `import datasets` at the TOP of models/train_distilbert.py, before torch. Only matters
  in the training process (pipeline/dashboard don't import datasets).
- **Version pins:** env had grabbed transformers 5.9 / datasets 4.8 (API breaks + the
  crash). Pinned to **transformers==4.46.3, datasets==3.2.0** (tokenizers 0.20.3,
  huggingface_hub 0.36.2). requirements.txt should reflect these caps.
- Dataset label schema mismatch (ClassLabel vs int64) fixed by casting both to Value("int64").

### What's left for the USER to do
1. `pip install -r requirements.txt` in a fresh env (installs torch/transformers/etc.).
2. (optional, GPU) `python run.py train` to produce the REAL fine-tuned DistilBERT +
   metrics.json — dashboard then shows real F1 instead of the precomputed example.
3. (optional) live mode: add Reddit keys to .env, set mode: live in config.yaml.
4. Record the demo video.

### Notes for resuming
- This sandbox installed only light deps (pandas/numpy/sqlalchemy/yfinance/sklearn/
  xgboost/streamlit/plotly/statsmodels). torch/transformers/datasets/praw/feedparser/tqdm
  are NOT installed here but are in requirements.txt.
- Lexicon backend is the demo default; it auto-upgrades to the fine-tuned model once
  data/models/distilbert_finetuned/ exists.

---

## 5b. SESSION 2 (2026-05-30) — post-training polish + GitHub deploy

**UI polish (vivid theme) — COMPLETE.** dashboard/theme.py heavily upgraded:
- Inter font, layered radial-glow background (blue/purple/green), colored+glowing KPI
  cards (tinted by accent color via new `rgba()` helper in theme.py), styled native
  st.metric tiles, gradient h1, accent-bar h4, styled sidebar/buttons/badges, framed
  plotly charts. components.kpi_card now injects per-card tinted style + glow accent.
- app.py KPI colors varied (ACCENT2 for posts, GOLD for assets). Palette brightened
  (BULL #1ed79b, BEAR #ff4d5e, added GOLD #f5b94a, ACCENT2 #a06bf5).
- LESSON: editing imported modules (theme.py) requires a FULL Streamlit restart
  (Ctrl+C + relaunch) + browser hard-refresh; "Rerun" reuses cached modules.

**Bug fixes — COMPLETE.**
- Heatmap x-axis showed 2004/2005 (Plotly misparsed "MM-DD" strings). FIX: pass real
  datetimes + `xaxis=dict(type="date", tickformat="%b %d")` in 1_Sentiment_Heatmap.py.
- matplotlib IS required (pandas Styler.background_gradient on Model page). It was wrongly
  removed earlier; **re-added to requirements.txt** and installed. (Supersedes the Phase-7
  note that said matplotlib was removed.)

**GitHub — DEPLOYED (public).**
- Repo: **https://github.com/shyamnairr/real-time-social-sentiment-market-pulse** (public, MIT).
- `git init` (branch main); `git config user.email shemneyr99@gmail.com`, `user.name shyamnairr`.
- .gitignore extended to exclude `.claude/`, `train_log.txt`, `rescore_log.txt`.
  Added `LICENSE` (MIT, "Shyam Nair", 2026). 43 files tracked (NO data/, NO .venv, NO .env,
  NO trained model — all gitignored; models/precomputed_metrics.json IS tracked as fallback).
- Committed, pushed, then amended to REMOVE the Co-Authored-By trailer (user wanted sole
  authorship) and force-pushed. Current sole author: shyamnairr <shemneyr99@gmail.com>.
- Remote `origin` already configured. Future updates: `git add -A; git commit -m "..."; git push`.

**Status: PROJECT COMPLETE & SHIPPED.** Optional remaining (user choice): repo
description+topics on GitHub, pin to profile, README screenshots, 60-sec video script.

---

## 6. How to run (will expand as built)
```
pip install -r requirements.txt
python run.py all          # build demo data + analysis
python run.py dashboard    # launch dashboard
# (optional, on GPU) python run.py train   # real DistilBERT fine-tune
```

---

## 7. Open notes / gotchas
- Python 3.13 + Anaconda on Windows. Confirm torch/transformers install cleanly on 3.13;
  if not, document a 3.11 venv fallback in README.
- Reddit/RSS historical depth is shallow → backtesting relies on synthetic history.
- `metrics.json` is shipped/precomputed so the Model Performance page works before training.
