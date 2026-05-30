import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve()
while _R.parent != _R and not (_R / "config.yaml").exists():
    _R = _R.parent
if str(_R) not in _sys.path:
    _sys.path.insert(0, str(_R))

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from dashboard import data_access as da
from dashboard.components import kpi_card, page_setup
from dashboard.theme import ACCENT, ACCENT2, BEAR, BULL, NEUTRAL

page_setup("Backtest & Signals", "🎯")

st.title("🎯 Backtested Signal Accuracy")
st.caption("If sentiment was bullish yesterday, did the stock rise today? "
           "We replay history and keep score across all assets.")

bt = da.backtest_summary()
if not bt:
    st.warning("Run `python run.py analyze` to produce backtest results.")
    st.stop()

# --- headline KPIs --------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Directional Accuracy", f"{bt['directional_accuracy']*100:.1f}%",
             "next-day movement", ACCENT)
with c2:
    kpi_card("Signals Tested", f"{bt['n_signals']:,}",
             f"{bt['n_assets']} assets", NEUTRAL)
with c3:
    kpi_card("Bullish Accuracy", f"{bt['bullish_accuracy']*100:.1f}%",
             f"{bt['n_bullish']:,} calls", BULL)
with c4:
    kpi_card("Bearish Accuracy", f"{bt['bearish_accuracy']*100:.1f}%",
             f"{bt['n_bearish']:,} calls", BEAR)

st.write("")

# --- equity curve ---------------------------------------------------------
eq = da.equity_curve()
if not eq.empty:
    st.markdown("#### 💰 Strategy vs Buy & Hold  ($1 invested)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["strategy_value"],
                             name="Sentiment strategy", line=dict(color=BULL, width=2.5)))
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["buyhold_value"],
                             name="Buy & hold", line=dict(color=NEUTRAL, width=2, dash="dash")))
    fig.update_layout(height=330, yaxis_title="Portfolio value ($)", xaxis_title="")
    st.plotly_chart(fig, width="stretch")
    s_ret, b_ret = bt["strategy_total_return"] * 100, bt["buyhold_total_return"] * 100
    cc1, cc2 = st.columns(2)
    cc1.metric("Strategy total return", f"{s_ret:+.1f}%")
    cc2.metric("Buy & hold total return", f"{b_ret:+.1f}%", f"{s_ret-b_ret:+.1f}% vs strategy")

st.divider()

col_l, col_r = st.columns(2)

# --- per-ticker accuracy --------------------------------------------------
with col_l:
    st.markdown("#### Accuracy by Asset (top 15)")
    bts = da.backtest_by_ticker()
    if not bts.empty:
        top = bts.head(15).iloc[::-1]
        fig = px.bar(top, x="accuracy", y="ticker", orientation="h",
                     color="accuracy", color_continuous_scale="Tealgrn",
                     hover_data=["n_signals"])
        fig.add_vline(x=0.5, line_dash="dash", line_color=NEUTRAL)
        fig.update_layout(height=460, xaxis_tickformat=".0%",
                          coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

# --- XGBoost vs DistilBERT comparison -------------------------------------
with col_r:
    st.markdown("#### Model Comparison")
    xgb = da.xgboost_metrics()
    if xgb:
        comp = go.Figure(go.Bar(
            x=["DistilBERT signal", "XGBoost"],
            y=[xgb["distilbert_signal_accuracy"], xgb["xgb_accuracy"]],
            marker_color=[ACCENT, ACCENT2],
            text=[f"{xgb['distilbert_signal_accuracy']*100:.1f}%",
                  f"{xgb['xgb_accuracy']*100:.1f}%"], textposition="outside",
        ))
        comp.add_hline(y=0.5, line_dash="dash", line_color=NEUTRAL)
        comp.update_layout(height=240, yaxis_tickformat=".0%",
                           yaxis_range=[0, 1], title="Test-set accuracy")
        st.plotly_chart(comp, width="stretch")
        st.caption(f"XGBoost ROC-AUC: **{xgb['xgb_roc_auc']:.3f}** · "
                   f"trained on {xgb['n_train']:,} / tested on {xgb['n_test']:,} days. "
                   "Note: XGBoost predicts *every* day; the DistilBERT signal only "
                   "acts on high-conviction days.")

        fi = da.feature_importance()
        if not fi.empty:
            st.markdown("##### XGBoost Feature Importance")
            fig = px.bar(fi.iloc[::-1], x="importance", y="feature",
                         orientation="h", color="importance",
                         color_continuous_scale="Purp")
            fig.update_layout(height=300, coloraxis_showscale=False,
                              xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
