import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve()
while _R.parent != _R and not (_R / "config.yaml").exists():
    _R = _R.parent
if str(_R) not in _sys.path:
    _sys.path.insert(0, str(_R))

import plotly.graph_objects as go
import streamlit as st

from dashboard import data_access as da
from dashboard.components import page_setup
from dashboard.theme import SENTIMENT_SCALE

page_setup("Sentiment Heatmap", "🌡️")

st.title("🌡️ Live Sentiment Heatmap")
st.caption("Each cell = confidence-weighted average sentiment for an asset on a day. "
           "Green = bullish, red = bearish.")

d = da.sentiment_daily()
if d.empty:
    st.warning("No data. Run `python run.py all` first.")
    st.stop()

# --- controls -------------------------------------------------------------
vol_rank = (d.groupby("ticker")["post_volume"].sum()
            .sort_values(ascending=False))
c1, c2 = st.columns([1, 2])
with c1:
    top_n = st.slider("Assets to show (by volume)", 10, len(vol_rank),
                      min(25, len(vol_rank)))
with c2:
    dates = sorted(d["date"].dt.date.unique())
    dr = st.select_slider("Date range", options=dates,
                          value=(dates[max(0, len(dates) - 45)], dates[-1]))

tickers = vol_rank.head(top_n).index.tolist()
mask = (d["ticker"].isin(tickers) & (d["date"].dt.date >= dr[0])
        & (d["date"].dt.date <= dr[1]))
sub = d[mask]

pivot = sub.pivot_table(index="ticker", columns="date",
                        values="mean_sentiment", aggfunc="mean")
# order rows by average sentiment for a clean gradient
pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

fig = go.Figure(go.Heatmap(
    z=pivot.values,
    x=list(pivot.columns),          # real datetimes -> correct year on the axis
    y=pivot.index,
    colorscale=SENTIMENT_SCALE, zmid=0, zmin=-1, zmax=1,
    colorbar=dict(title="Sentiment"),
    hovertemplate="<b>%{y}</b> · %{x|%b %d, %Y}<br>sentiment %{z:.2f}<extra></extra>",
))
fig.update_layout(height=max(420, 18 * len(pivot)),
                  xaxis_title="", yaxis_title="",
                  xaxis=dict(type="date", tickformat="%b %d"))
st.plotly_chart(fig, width="stretch")

st.divider()
cc1, cc2, cc3 = st.columns(3)
cc1.metric("Assets shown", len(pivot))
cc2.metric("Days shown", pivot.shape[1])
cc3.metric("Avg sentiment (window)", f"{sub['mean_sentiment'].mean():+.2f}")
