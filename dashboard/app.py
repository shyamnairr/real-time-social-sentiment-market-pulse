# --- make project root importable regardless of how Streamlit launches us ---
import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve()
while _R.parent != _R and not (_R / "config.yaml").exists():
    _R = _R.parent
if str(_R) not in _sys.path:
    _sys.path.insert(0, str(_R))
# ---------------------------------------------------------------------------

import plotly.express as px
import streamlit as st

from dashboard import data_access as da
from dashboard.components import (assets_inline, kpi_card, market_pulse_summary,
                                  page_setup)
from dashboard.theme import ACCENT, ACCENT2, BEAR, BULL, GOLD, NEUTRAL

page_setup("Overview", "📈")

st.title("⚡ Real-Time Social Sentiment Market Pulse")
st.caption("Fine-tuned DistilBERT · Reddit + financial RSS · 50+ assets · "
           "sentiment-price correlation, anomaly detection & backtested signals")

kpi = da.latest_day_kpis()
if not kpi:
    st.warning("No data found. Build the demo dataset first:\n\n"
               "```\npython run.py all\n```")
    st.stop()

# --- Market Pulse auto-summary (extra feature D) --------------------------
st.markdown(f"<div class='pulse-summary'>📰 &nbsp;{market_pulse_summary()}</div>",
            unsafe_allow_html=True)
st.write("")

# --- KPI cards ------------------------------------------------------------
bt = da.backtest_summary()
anom = da.anomalies()
n_active_anom = 0
if not anom.empty:
    n_active_anom = int((anom["date"] == anom["date"].max()).sum())

avg = kpi["avg_sentiment"]
delta = kpi["avg_sentiment_delta"]
delta_col = BULL if delta >= 0 else BEAR

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Total Posts Processed", f"{da.posts_total():,}", "all sources", ACCENT2)
with c2:
    kpi_card("Avg Sentiment (today)", f"{avg:+.2f}",
             f"{delta:+.2f} vs prev day", delta_col)
with c3:
    acc = bt.get("directional_accuracy", 0) * 100
    kpi_card("Backtest Accuracy", f"{acc:.1f}%", "next-day direction", ACCENT)
with c4:
    kpi_card("Active Anomalies", f"{n_active_anom}", "latest day",
             BEAR if n_active_anom else NEUTRAL)
with c5:
    kpi_card("Assets Tracked", f"{kpi['n_assets']}", "live universe", GOLD)

st.write("")

# --- Top movers -----------------------------------------------------------
left, right = st.columns(2)
with left:
    st.markdown("#### 🟢 Top Bullish Today")
    st.markdown(assets_inline(kpi["top_bullish"], positive=True),
                unsafe_allow_html=True)
with right:
    st.markdown("#### 🔴 Top Bearish Today")
    st.markdown(assets_inline(kpi["top_bearish"], positive=False),
                unsafe_allow_html=True)

st.divider()

# --- Market-wide sentiment trend ------------------------------------------
st.markdown("#### 📈 Market-Wide Sentiment Trend")
d = da.sentiment_daily()
trend = (d.assign(w=lambda x: x["mean_sentiment"] * x["post_volume"])
         .groupby("date").agg(w=("w", "sum"), v=("post_volume", "sum")).reset_index())
trend["market_sentiment"] = trend["w"] / trend["v"].clip(lower=1)
fig = px.area(trend, x="date", y="market_sentiment",
              labels={"market_sentiment": "Avg sentiment", "date": ""})
fig.update_traces(line_color=ACCENT, fillcolor="rgba(91,141,239,0.18)")
fig.add_hline(y=0, line_dash="dash", line_color=NEUTRAL)
fig.update_layout(height=320)
st.plotly_chart(fig, width="stretch")

st.caption("Use the sidebar to explore the heatmap, anomaly alerts, backtest, "
           "per-asset deep dives and model performance.")
