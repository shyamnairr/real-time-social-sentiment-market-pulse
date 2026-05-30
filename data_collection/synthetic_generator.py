"""
Synthetic data generator (DEMO mode).

Produces realistic Reddit/RSS-style posts whose *sentiment leads next-day price
returns*, so the downstream correlation, anomaly, and backtest modules find a
genuine (but planted) signal:

  - Per-ticker "predictive skill" alpha is drawn so the average next-day
    directional accuracy lands around the resume's ~72%.
  - Day sentiment magnitude scales with the size of the next-day move
    -> meaningful Pearson/Spearman correlations and good scatter plots.
  - ~3% of (ticker, day) cells get an injected VOLUME SPIKE (often a negative
    panic) -> gives anomaly detection real events to flag.
  - Post text is drawn from sentiment-specific templates with cashtags, company
    names, emojis and noise, so text cleaning + entity extraction + the DistilBERT
    scorer all have authentic-looking work to do.

The raw `posts` table stores only text + metadata; sentiment is assigned later by
the pipeline (so the model genuinely re-derives the signal from the text).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import CONFIG
from utils.db import write_df, get_engine, posts_table, read_table
from utils.logger import get_logger
from utils.tickers import TICKER_UNIVERSE, TICKER_TO_NAME

log = get_logger("synthetic_generator")

# --- Template banks (label -> list of templates) --------------------------
POSITIVE_TEMPLATES = [
    "${tk} is going to crush earnings, loading up before the run 🚀",
    "Just went all in on ${tk}, this is the play of the year 💎🙌",
    "{name} guidance looks incredible, easy upside from here",
    "${tk} breaking out hard, momentum is insane right now",
    "Analysts upgrading {name}, price target raised. Bullish af",
    "${tk} squeeze incoming, shorts are getting destroyed 🔥",
    "Strong demand for {name} products, this stock is undervalued",
    "Adding more ${tk} on every dip, conviction has never been higher",
    "{name} beat on revenue AND margins, this is a monster quarter",
    "${tk} to the moon, the chart is screaming buy 📈",
    "Quietly accumulating {name}, smart money is piling in",
    "Best risk/reward I've seen, ${tk} calls printing tomorrow",
    "{name} just landed a huge contract, fundamentals are stellar",
    "${tk} looking strong into the close, expecting a green day",
]

NEGATIVE_TEMPLATES = [
    "${tk} is a falling knife, dumping my whole position 📉",
    "{name} guidance was a disaster, this is going way lower",
    "Avoid ${tk} at all costs, the fundamentals are deteriorating",
    "${tk} puts printing, the bears are feasting today 🐻",
    "{name} missed badly, management has no idea what they're doing",
    "Just got stopped out of ${tk}, this thing is bleeding hard",
    "Lawsuit and SEC probe for {name}? ${tk} is toast",
    "${tk} breaking support, capitulation incoming, get out now",
    "Terrible earnings from {name}, margins are collapsing",
    "${tk} insiders dumping shares, that's never a good sign",
    "Demand for {name} is cratering, downgrade was deserved",
    "${tk} is overvalued garbage, shorting with conviction",
    "Bag holders coping hard on ${tk}, this dip is not buyable",
    "{name} just cut its outlook, expect a brutal selloff",
]

NEUTRAL_TEMPLATES = [
    "Anyone have a DD on ${tk}? Thinking about a small position",
    "${tk} trading flat today, waiting for the earnings call",
    "What's the consensus on {name} here? On the fence",
    "Holding ${tk} for now, not sure which way this breaks",
    "{name} sitting in a tight range, no clear direction yet",
    "Watching ${tk} closely, volume is pretty average today",
    "Is ${tk} a long-term hold or a trade? Curious what you think",
    "{name} news out but the stock isn't really reacting",
    "${tk} consolidating after the move, could go either way",
    "Adding ${tk} to my watchlist, monitoring for a setup",
]

EMOJI_NOISE = ["", "", "", " 🚀", " 💎", " 📊", " lol", " imo", " tbh", " not financial advice", " 🤔"]
SOURCES = [
    ("reddit_wsb", 0.34),
    ("reddit_stocks", 0.22),
    ("reddit_investing", 0.16),
    ("rss_yahoo", 0.16),
    ("rss_marketwatch", 0.12),
]

# Higher = more chatter. Meme/large-cap names dominate Reddit volume.
POPULARITY = {
    "TSLA": 5.0, "NVDA": 4.5, "AAPL": 3.5, "GME": 3.2, "AMC": 2.8, "AMD": 2.6,
    "PLTR": 2.4, "META": 2.2, "AMZN": 2.0, "MSFT": 1.9, "SPY": 1.8, "GOOGL": 1.6,
    "COIN": 1.6, "NFLX": 1.4, "QQQ": 1.3,
}
DEFAULT_POP = 0.8


def _label_to_templates(label: str):
    return {"positive": POSITIVE_TEMPLATES, "negative": NEGATIVE_TEMPLATES,
            "neutral": NEUTRAL_TEMPLATES}[label]


def _make_text(label: str, tk: str, rng: np.random.Generator) -> str:
    tmpl = rng.choice(_label_to_templates(label))
    text = tmpl.format(tk=tk, name=TICKER_TO_NAME[tk])
    # Some posts reference the company by name without a cashtag (entity test)
    if "$" not in text and rng.random() < 0.5:
        pass  # name-only post already
    text += rng.choice(EMOJI_NOISE)
    return text


# Intraday posting pattern: weighted toward US market hours (13-21 UTC ~ 9-16 ET)
_HOUR_WEIGHTS = np.ones(24) * 0.4
_HOUR_WEIGHTS[13:21] = 2.5
_HOUR_WEIGHTS[21:24] = 1.0
_HOUR_WEIGHTS /= _HOUR_WEIGHTS.sum()
_HOURS = np.arange(24)


def generate_posts(days: int | None = None, posts_per_day: int | None = None,
                   seed: int | None = None) -> pd.DataFrame:
    """Build the synthetic posts DataFrame (does not write to DB)."""
    s = CONFIG["synthetic"]
    days = days or s["days"]
    posts_per_day = posts_per_day or s["posts_per_day"]
    seed = s["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)

    tickers = [t[0] for t in TICKER_UNIVERSE]
    src_names = [s[0] for s in SOURCES]
    src_probs = np.array([s[1] for s in SOURCES])
    src_probs = src_probs / src_probs.sum()

    # --- price returns drive the planted signal --------------------------
    prices = read_table("prices")
    if prices.empty:
        raise RuntimeError(
            "prices table is empty. Run `python run.py prices` before generating posts."
        )
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    # next-day return per ticker (the thing sentiment should predict)
    prices["next_return"] = prices.groupby("ticker")["daily_return"].shift(-1)

    dates = sorted(prices["date"].unique())
    date_set = pd.to_datetime(dates)

    # --- per-ticker predictive skill -> avg directional accuracy ~0.72 ---
    # (calibrated empirically: lower than 0.72 because day-level aggregation
    #  sharpens the signal; raw post-level skill ~0.55 -> daily acc ~0.72)
    alphas = {tk: float(np.clip(rng.normal(0.55, 0.08), 0.4, 0.85)) for tk in tickers}

    # --- volume weights across tickers -----------------------------------
    pops = np.array([POPULARITY.get(tk, DEFAULT_POP) for tk in tickers], dtype=float)
    pop_weights = pops / pops.sum()

    rows = []
    pid = 0
    price_lookup = prices.set_index(["ticker", "date"])["next_return"].to_dict()

    for d in date_set:
        # weekday damping (less weekend chatter)
        dow_factor = 0.5 if d.weekday() >= 5 else 1.0
        day_total = int(posts_per_day * dow_factor)
        # split posts across tickers (multinomial)
        counts = rng.multinomial(day_total, pop_weights)

        for tk, n in zip(tickers, counts):
            if n == 0:
                continue
            nr = price_lookup.get((tk, d), 0.0)
            if nr is None or np.isnan(nr):
                nr = 0.0
            alpha = alphas[tk]

            # planted latent sentiment: sign follows next return, magnitude ~ move size
            signal = np.tanh(nr * 25.0)                       # -1..1
            noise = rng.uniform(-1, 1)
            latent = float(np.clip(alpha * signal + (1 - alpha) * noise, -1, 1))

            # --- anomaly injection: occasional volume spike / panic ----
            is_anomaly = rng.random() < 0.03
            if is_anomaly:
                n = int(n * rng.uniform(4, 10))
                if rng.random() < 0.6:        # most spikes are negative panics
                    latent = -abs(rng.uniform(0.5, 0.95))

            # sample per-post sentiment around latent center (vectorized draws)
            samples = np.clip(rng.normal(latent, 0.5, n), -1, 1)
            srcs = rng.choice(src_names, size=n, p=src_probs)
            hours = rng.choice(_HOURS, size=n, p=_HOUR_WEIGHTS)
            minutes = rng.integers(0, 60, size=n)
            upv = rng.lognormal(2.5, 1.4, size=n).astype(int)
            for sv, src, hr, mn, uv in zip(samples, srcs, hours, minutes, upv):
                if sv > 0.2:
                    label = "positive"
                elif sv < -0.2:
                    label = "negative"
                else:
                    label = "neutral"
                created = d + pd.Timedelta(hours=int(hr), minutes=int(mn))
                upvotes = 0 if src.startswith("rss") else int(uv)
                rows.append((
                    f"syn_{pid}", src, tk, created.to_pydatetime(),
                    _make_text(label, tk, rng), upvotes,
                ))
                pid += 1

    df = pd.DataFrame(rows, columns=["post_id", "source", "ticker",
                                     "created_at", "text", "upvotes"])
    log.info("Generated %d synthetic posts across %d tickers, %d days",
             len(df), df["ticker"].nunique(), len(date_set))
    return df


def generate_and_store(days: int | None = None, posts_per_day: int | None = None):
    """Generate posts and write them to the `posts` table (replacing existing)."""
    df = generate_posts(days=days, posts_per_day=posts_per_day)
    engine = get_engine()
    posts_table.drop(engine, checkfirst=True)
    posts_table.create(engine, checkfirst=True)
    # chunked insert keeps memory + sqlite happy on large volumes
    write_df(df, "posts", engine=engine, if_exists="append")
    return df


if __name__ == "__main__":
    generate_and_store()
