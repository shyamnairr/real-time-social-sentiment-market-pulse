"""Reusable dashboard UI: page setup, sidebar, KPI cards, Market Pulse summary."""
from __future__ import annotations

import streamlit as st

from dashboard import data_access as da
from dashboard.theme import BULL, BEAR, NEUTRAL, apply_theme, rgba


def page_setup(title: str, icon: str = "📈"):
    st.set_page_config(page_title=f"{title} · Market Pulse", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    apply_theme()
    sidebar()


def sidebar():
    with st.sidebar:
        st.markdown("## ⚡ Market Pulse")
        st.caption("Real-Time Social Sentiment")
        st.divider()
        st.markdown(f"**Mode:** `{da.mode()}`")
        kpi = da.latest_day_kpis()
        if kpi:
            st.markdown(f"**Latest data:** {kpi['date'].date()}")
            st.markdown(f"**Assets tracked:** {kpi['n_assets']}")
        st.divider()
        if st.button("🔄 Refresh data", width="stretch"):
            _refresh()
        st.caption("Demo mode refresh reloads cached data. "
                   "Live mode pulls fresh Reddit/RSS + reruns analysis.")
        st.divider()
        st.caption("Pages — Overview, Heatmap, Anomalies, Backtest, "
                   "Deep Dive, Model Performance")


def _refresh():
    if da.mode() == "live":
        with st.spinner("Collecting live data + re-running analysis..."):
            from data_collection.pipeline import run_live_collection
            from analysis.run_analysis import run_all_analysis
            run_live_collection()
            run_all_analysis()
    da.clear_caches()
    st.success("Data refreshed.")
    st.rerun()


def kpi_card(label: str, value: str, delta: str | None = None,
             delta_color: str = NEUTRAL):
    delta_html = (f"<div class='kpi-delta' style='color:{delta_color}'>{delta}</div>"
                  if delta else "")
    # tint the whole card with the accent color so the row feels alive
    style = (f"background:linear-gradient(155deg, {rgba(delta_color,0.22)}, "
             f"rgba(16,20,30,0.92)); border:1px solid {rgba(delta_color,0.38)}; "
             f"box-shadow:0 8px 26px {rgba(delta_color,0.16)};")
    st.markdown(
        f"<div class='kpi-card' style='{style}'>"
        f"<div class='kpi-accent' style='background:{delta_color};"
        f"box-shadow:0 0 14px {delta_color}'></div>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True,
    )


def assets_inline(pairs, positive=True):
    """Render 'TICKER (+0.42)' chips for top bullish/bearish lists."""
    color = BULL if positive else BEAR
    parts = [f"<span style='color:{color};font-weight:600'>{t}</span> "
             f"<span style='color:{NEUTRAL}'>({v:+.2f})</span>" for t, v in pairs]
    return " &nbsp; ".join(parts)


def market_pulse_summary() -> str:
    """Auto-generated natural-language market summary (extra feature D)."""
    kpi = da.latest_day_kpis()
    bt = da.backtest_summary()
    anom = da.anomalies()
    if not kpi:
        return "No data yet. Run `python run.py all` to build the demo dataset."

    avg = kpi["avg_sentiment"]
    mood = ("broadly bullish" if avg > 0.1 else
            "broadly bearish" if avg < -0.1 else "mixed / neutral")
    lead = kpi["top_bullish"][0][0] if kpi["top_bullish"] else "—"
    lag = kpi["top_bearish"][0][0] if kpi["top_bearish"] else "—"

    # most severe anomaly on the latest day, if any
    flag_txt = ""
    if not anom.empty:
        latest = anom[anom["date"] == anom["date"].max()]
        if not latest.empty:
            top = latest.reindex(
                latest["zscore"].abs().sort_values(ascending=False).index).iloc[0]
            flag_txt = (f" ⚠️ <b>{top['ticker']}</b> flagged for an anomalous "
                        f"{top['metric'].replace('_',' ')} {top['direction']} "
                        f"(z={top['zscore']:.1f}).")

    acc = bt.get("directional_accuracy", 0) * 100
    return (f"Market sentiment is <b>{mood}</b> as of {kpi['date'].date()}, "
            f"with <b>{kpi['posts_today']:,}</b> posts processed across "
            f"{kpi['n_assets']} assets. Sentiment is led by <b>{lead}</b> on the "
            f"bullish side and weighed down by <b>{lag}</b>.{flag_txt} "
            f"The sentiment signal has called next-day direction correctly "
            f"<b>{acc:.0f}%</b> of the time in backtests.")
