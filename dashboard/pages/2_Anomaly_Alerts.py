import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve()
while _R.parent != _R and not (_R / "config.yaml").exists():
    _R = _R.parent
if str(_R) not in _sys.path:
    _sys.path.insert(0, str(_R))

import plotly.express as px
import streamlit as st

from dashboard import data_access as da
from dashboard.components import page_setup
from dashboard.theme import BEAR, BULL, NEUTRAL

page_setup("Anomaly Alerts", "🚨")

st.title("🚨 Anomaly Detection Alerts")
st.caption("Unusual spikes/drops in post volume or sentiment, flagged via rolling "
           "z-score and IQR. e.g. a sudden 10x surge in negative mentions.")

anom = da.anomalies()
if anom.empty:
    st.info("No anomalies detected (or analysis not yet run). Try `python run.py analyze`.")
    st.stop()

# --- KPIs -----------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Anomalies", len(anom))
k2.metric("Critical", int((anom["severity"] == "critical").sum()))
k3.metric("Volume Spikes", int(((anom["metric"] == "post_volume") &
                                (anom["direction"] == "spike")).sum()))
k4.metric("Assets Affected", anom["ticker"].nunique())

st.divider()

# --- filters --------------------------------------------------------------
f1, f2 = st.columns(2)
with f1:
    sev = st.multiselect("Severity", ["critical", "high", "medium"],
                         default=["critical", "high"])
with f2:
    met = st.multiselect("Metric", ["post_volume", "mean_sentiment"],
                         default=["post_volume", "mean_sentiment"])
view = anom[anom["severity"].isin(sev) & anom["metric"].isin(met)]

# --- timeline scatter -----------------------------------------------------
st.markdown("#### Anomaly Timeline")
if not view.empty:
    plot = view.copy()
    plot["abs_z"] = plot["zscore"].abs()
    fig = px.scatter(plot, x="date", y="ticker", size="abs_z", color="direction",
                     color_discrete_map={"spike": BULL, "drop": BEAR},
                     hover_data=["metric", "zscore", "severity"], size_max=22)
    fig.update_layout(height=max(380, 16 * plot["ticker"].nunique()),
                      xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, width="stretch")

# --- alert feed -----------------------------------------------------------
st.markdown("#### Alert Feed")
badge = {"critical": "badge-crit", "high": "badge-high", "medium": "badge-med"}
for _, r in view.sort_values("date", ascending=False).head(40).iterrows():
    arrow = "▲" if r["direction"] == "spike" else "▼"
    color = BEAR if (r["metric"] == "mean_sentiment" and r["direction"] == "drop") \
        or (r["metric"] == "post_volume" and r["direction"] == "spike") else BULL
    st.markdown(
        f"<div class='kpi-card' style='margin-bottom:8px;padding:12px 16px'>"
        f"<span class='badge {badge[r['severity']]}'>{r['severity'].upper()}</span> "
        f"&nbsp;<b>{r['ticker']}</b> &nbsp;<span style='color:{NEUTRAL}'>"
        f"{r['date'].date()}</span><br>"
        f"<span style='color:{color}'>{arrow} {r['metric'].replace('_',' ')} "
        f"{r['direction']}</span> &nbsp; z-score = <b>{r['zscore']:.2f}</b>"
        f"{' · IQR outlier' if r['iqr_flag'] else ''}</div>",
        unsafe_allow_html=True,
    )

st.caption(f"Showing {min(40, len(view))} of {len(view)} matching alerts.")
