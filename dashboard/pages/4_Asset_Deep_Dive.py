import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve()
while _R.parent != _R and not (_R / "config.yaml").exists():
    _R = _R.parent
if str(_R) not in _sys.path:
    _sys.path.insert(0, str(_R))

import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard import data_access as da
from dashboard.components import page_setup
from dashboard.theme import ACCENT, BEAR, BULL, NEUTRAL
from utils.tickers import TICKER_TO_NAME

page_setup("Asset Deep Dive", "🔍")

st.title("🔍 Per-Asset Deep Dive")

d = da.sentiment_daily()
if d.empty:
    st.warning("No data. Run `python run.py all` first.")
    st.stop()

tickers = sorted(d["ticker"].unique())
default_ix = tickers.index("TSLA") if "TSLA" in tickers else 0
ticker = st.selectbox("Select an asset", tickers, index=default_ix,
                      format_func=lambda t: f"{t} — {TICKER_TO_NAME.get(t, t)}")

# --- merged sentiment + price for this ticker -----------------------------
m = da.merged()
mt = m[m["ticker"] == ticker].sort_values("date") if not m.empty else pd.DataFrame()
corr_row = da.correlation_by_ticker()
corr_row = corr_row[corr_row["ticker"] == ticker]

# --- KPIs -----------------------------------------------------------------
dt = d[d["ticker"] == ticker]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg sentiment", f"{dt['mean_sentiment'].mean():+.2f}")
k2.metric("Total posts", f"{int(dt['post_volume'].sum()):,}")
if not corr_row.empty:
    k3.metric("Sentiment-price corr", f"{corr_row['pearson_next_day'].iloc[0]:+.2f}",
              "next-day Pearson")
    k4.metric("Best lead/lag", f"{int(corr_row['best_lag'].iloc[0]):+d} d",
              f"corr {corr_row['best_lag_corr'].iloc[0]:+.2f}")

st.divider()

# --- price vs sentiment dual-axis -----------------------------------------
st.markdown("#### Price vs Sentiment")
if not mt.empty:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=mt["date"], y=mt["close"], name="Close price",
                             line=dict(color=ACCENT, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=mt["date"], y=mt["mean_sentiment"], name="Sentiment",
                             line=dict(color=BULL, width=1.8)), secondary_y=True)
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL, secondary_y=True)
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment", range=[-1, 1], secondary_y=True)
    fig.update_layout(height=380, xaxis_title="",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

col_l, col_r = st.columns(2)

# --- scatter: sentiment vs next-day return --------------------------------
with col_l:
    st.markdown("#### Sentiment vs Next-Day Return")
    if not mt.empty:
        fig = px.scatter(mt, x="mean_sentiment", y="next_return",
                         trendline="ols", color_discrete_sequence=[ACCENT])
        fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig.add_vline(x=0, line_dash="dot", line_color=NEUTRAL)
        fig.update_layout(height=330, xaxis_title="Daily sentiment",
                          yaxis_title="Next-day return")
        st.plotly_chart(fig, width="stretch")

# --- lag analysis ---------------------------------------------------------
with col_r:
    st.markdown("#### Lag Analysis")
    lag = da.lag_analysis()
    lt = lag[lag["ticker"] == ticker] if not lag.empty else pd.DataFrame()
    if not lt.empty:
        fig = px.bar(lt, x="lag", y="corr",
                     color="corr", color_continuous_scale="RdYlGn", range_color=[-1, 1])
        fig.update_layout(height=330, coloraxis_showscale=False,
                          xaxis_title="Lag (days): sentiment(t) vs return(t+lag)",
                          yaxis_title="Correlation")
        st.plotly_chart(fig, width="stretch")
        st.caption("A peak at a positive lag means sentiment *leads* price.")

st.divider()

# --- word drivers (extra feature A) ---------------------------------------
st.markdown("#### 🗣️ What's Driving the Conversation")
_STOP = set("""the a an and or to of in for on is it that this i you we my your with
at be are was as so im not no but if up down today day stock will would can has have
just like get got going its his her their our about into out over more most very
this thats whats really still even than then them they have here gonna lol imo tbh""".split())
name_tokens = set(TICKER_TO_NAME.get(ticker, "").lower().split())
name_tokens.add(ticker.lower())


def top_words(texts, n=12):
    cnt = Counter()
    for t in texts:
        for w in re.findall(r"[a-z']{3,}", str(t).lower()):
            if w in _STOP or w in name_tokens:
                continue
            cnt[w] += 1
    return cnt.most_common(n)


tx = da.ticker_texts(ticker)
if not tx.empty:
    pos_words = top_words(tx[tx["sentiment_label"] == "positive"]["clean_text"])
    neg_words = top_words(tx[tx["sentiment_label"] == "negative"]["clean_text"])
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**🟢 Bullish drivers**")
        if pos_words:
            pw = pd.DataFrame(pos_words, columns=["word", "count"]).iloc[::-1]
            fig = px.bar(pw, x="count", y="word", orientation="h",
                         color_discrete_sequence=[BULL])
            fig.update_layout(height=320, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
    with w2:
        st.markdown("**🔴 Bearish drivers**")
        if neg_words:
            nw = pd.DataFrame(neg_words, columns=["word", "count"]).iloc[::-1]
            fig = px.bar(nw, x="count", y="word", orientation="h",
                         color_discrete_sequence=[BEAR])
            fig.update_layout(height=320, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")

st.divider()

# --- recent posts ---------------------------------------------------------
st.markdown("#### 📝 Recent Posts")
rp = da.recent_posts(ticker, limit=12)
color = {"positive": BULL, "negative": BEAR, "neutral": NEUTRAL}
for _, r in rp.iterrows():
    c = color.get(r["sentiment_label"], NEUTRAL)
    st.markdown(
        f"<div class='kpi-card' style='margin-bottom:6px;padding:10px 14px'>"
        f"<span style='color:{c};font-weight:600'>{r['sentiment_label'].upper()} "
        f"({r['sentiment_score']:+.2f})</span> "
        f"<span style='color:{NEUTRAL};font-size:0.8rem'>· {r['source']} · "
        f"▲{int(r['upvotes'])}</span><br>{r['text']}</div>",
        unsafe_allow_html=True,
    )
