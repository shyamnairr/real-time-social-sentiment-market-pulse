import sys as _sys, pathlib as _pl
_R = _pl.Path(__file__).resolve()
while _R.parent != _R and not (_R / "config.yaml").exists():
    _R = _R.parent
if str(_R) not in _sys.path:
    _sys.path.insert(0, str(_R))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data_access as da
from dashboard.components import kpi_card, page_setup
from dashboard.theme import ACCENT, BULL, NEUTRAL

page_setup("Model Performance", "🧠")

st.title("🧠 Model Performance — Fine-Tuned DistilBERT")

m = da.model_metrics()
if not m:
    st.warning("No metrics found. Train the model (`python run.py train`) on your GPU, "
               "or ensure models/precomputed_metrics.json exists.")
    st.stop()

if m.get("source") == "precomputed_example":
    st.info("📌 Showing **example** metrics (precomputed). Run `python run.py train` "
            "on your GPU to fine-tune DistilBERT and populate real numbers here.")
else:
    st.success("✅ Showing metrics from your fine-tuned DistilBERT.")

labels = m.get("labels", ["negative", "neutral", "positive"])

# --- headline KPIs --------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Weighted F1", f"{m['f1_weighted']:.3f}", "target ~0.87", ACCENT)
with c2:
    kpi_card("Accuracy", f"{m['accuracy']*100:.1f}%", None, BULL)
with c3:
    kpi_card("Macro F1", f"{m.get('f1_macro', 0):.3f}", None, NEUTRAL)
with c4:
    kpi_card("Eval Samples", f"{m.get('n_eval', 0):,}",
             f"trained on {m.get('n_train', 0):,}", NEUTRAL)

st.divider()
col_l, col_r = st.columns(2)

# --- confusion matrix -----------------------------------------------------
with col_l:
    st.markdown("#### Confusion Matrix")
    cm = m.get("confusion_matrix")
    if cm:
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

# --- per-class report -----------------------------------------------------
with col_r:
    st.markdown("#### Classification Report")
    rep = m.get("classification_report", {})
    rows = []
    for lab in labels:
        if lab in rep:
            r = rep[lab]
            rows.append({"class": lab, "precision": r["precision"],
                         "recall": r["recall"], "f1-score": r["f1-score"],
                         "support": int(r["support"])})
    if rows:
        rdf = pd.DataFrame(rows)
        st.dataframe(
            rdf.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                              "f1-score": "{:.3f}"})
            .background_gradient(cmap="Greens", subset=["f1-score"]),
            width="stretch", hide_index=True,
        )
        # per-class F1 bar
        fig = px.bar(rdf, x="class", y="f1-score", color="class",
                     color_discrete_sequence=[ACCENT, NEUTRAL, BULL], text_auto=".3f")
        fig.update_layout(height=240, showlegend=False, yaxis_range=[0, 1],
                          xaxis_title="", yaxis_title="F1")
        st.plotly_chart(fig, width="stretch")

st.divider()

# --- training history -----------------------------------------------------
st.markdown("#### Training History")
hist = m.get("training_history", [])
if hist:
    h = pd.DataFrame(hist)
    fig = go.Figure()
    if "loss" in h:
        tl = h.dropna(subset=["loss"])
        fig.add_trace(go.Scatter(x=tl["epoch"], y=tl["loss"], name="Train loss",
                                 line=dict(color=ACCENT)))
    if "eval_loss" in h:
        el = h.dropna(subset=["eval_loss"])
        fig.add_trace(go.Scatter(x=el["epoch"], y=el["eval_loss"], name="Eval loss",
                                 line=dict(color="#f5a623")))
    if "eval_f1_weighted" in h:
        fl = h.dropna(subset=["eval_f1_weighted"])
        fig.add_trace(go.Scatter(x=fl["epoch"], y=fl["eval_f1_weighted"],
                                 name="Eval F1", line=dict(color=BULL),
                                 yaxis="y2"))
    fig.update_layout(
        height=340, xaxis_title="Epoch", yaxis_title="Loss",
        yaxis2=dict(title="F1", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")

st.caption("Model: DistilBERT fine-tuned on Financial PhraseBank + financial news "
           "sentiment (10K+ samples), 3-class (negative/neutral/positive).")
