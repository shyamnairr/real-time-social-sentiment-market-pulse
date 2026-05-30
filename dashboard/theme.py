"""Shared visual language for the dashboard: colors, Plotly template, CSS."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --- palette --------------------------------------------------------------
BULL = "#1ed79b"      # green  (positive / up)
BEAR = "#ff4d5e"      # red    (negative / down)
NEUTRAL = "#8b90a0"   # gray
ACCENT = "#5b8def"    # blue
ACCENT2 = "#a06bf5"   # purple
GOLD = "#f5b94a"
BG = "#0a0d14"
PANEL = "#141925"
TEXT = "#e8ebf2"

# diverging colorscale for sentiment heatmaps (-1 red .. 0 gray .. +1 green)
SENTIMENT_SCALE = [
    [0.0, BEAR], [0.25, "#c0455a"], [0.5, "#3a3f4d"],
    [0.75, "#21a87f"], [1.0, BULL],
]


def rgba(hex_color: str, alpha: float) -> str:
    """'#1ed79b', 0.2 -> 'rgba(30,215,155,0.2)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def register_plotly_template():
    """A dark Plotly template matching the Streamlit theme."""
    tmpl = go.layout.Template()
    tmpl.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13,
                  family="Inter, Segoe UI, system-ui, sans-serif"),
        colorway=[ACCENT, BULL, BEAR, ACCENT2, GOLD, NEUTRAL],
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)",
                   linecolor="rgba(255,255,255,0.10)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.10)",
                   linecolor="rgba(255,255,255,0.10)"),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=PANEL, bordercolor="rgba(255,255,255,0.15)",
                        font=dict(color=TEXT, family="Inter, sans-serif")),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["pulse"] = tmpl
    pio.templates.default = "pulse"


_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"] {{
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }}

  /* App background: vivid layered glow on deep navy */
  .stApp {{
      background:
        radial-gradient(900px 520px at 8% -6%, rgba(91,141,239,0.28), transparent 55%),
        radial-gradient(900px 520px at 95% -4%, rgba(160,107,245,0.24), transparent 55%),
        radial-gradient(1100px 700px at 50% 115%, rgba(30,215,155,0.16), transparent 60%),
        linear-gradient(180deg, #0e1422 0%, {BG} 60%);
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1500px; }}

  /* Headings */
  h1 {{
      font-weight: 800 !important; letter-spacing: -0.02em;
      background: linear-gradient(95deg, {TEXT} 30%, {ACCENT} 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
  }}
  h2, h3 {{ font-weight: 700 !important; letter-spacing: -0.01em; color: {TEXT}; }}
  /* Section sub-headers (st.markdown("#### ...")) get an accent bar */
  h4 {{
      font-weight: 700 !important; color: {TEXT}; letter-spacing: -0.01em;
      padding-left: 12px; border-left: 3px solid {ACCENT}; margin-top: 0.4rem;
  }}
  h5 {{ color: {TEXT}; font-weight: 600 !important; }}

  /* KPI cards (custom) */
  .kpi-card {{
      position: relative; overflow: hidden;
      background: linear-gradient(160deg, rgba(31,38,54,0.95), rgba(16,20,30,0.95));
      border: 1px solid rgba(255,255,255,0.07); border-radius: 16px;
      padding: 18px 20px; box-shadow: 0 6px 22px rgba(0,0,0,0.40);
      transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
  }}
  .kpi-card:hover {{
      transform: translateY(-3px);
      box-shadow: 0 12px 30px rgba(0,0,0,0.55);
      border-color: rgba(91,141,239,0.45);
  }}
  .kpi-accent {{ position:absolute; top:0; left:0; height:3px; width:100%; opacity:0.9; }}
  .kpi-label {{ color: {NEUTRAL}; font-size: 0.74rem; text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 8px; font-weight: 600; }}
  .kpi-value {{ font-size: 2.0rem; font-weight: 800; color: {TEXT}; line-height: 1.05;
      letter-spacing: -0.02em; }}
  .kpi-delta {{ font-size: 0.84rem; margin-top: 6px; font-weight: 600; }}

  /* Native st.metric -> matching card look */
  [data-testid="stMetric"] {{
      background: linear-gradient(160deg, rgba(91,141,239,0.14), rgba(20,25,37,0.85));
      border: 1px solid rgba(91,141,239,0.22); border-left: 3px solid {ACCENT};
      border-radius: 14px; padding: 14px 18px; box-shadow: 0 6px 20px rgba(0,0,0,0.35);
  }}
  [data-testid="stMetricLabel"] p {{ color: {NEUTRAL} !important; font-size:0.78rem !important;
      text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
  [data-testid="stMetricValue"] {{ font-weight: 800 !important; letter-spacing:-0.01em; }}

  /* Market Pulse summary banner */
  .pulse-summary {{
      background: linear-gradient(135deg, rgba(91,141,239,0.14), rgba(160,107,245,0.12));
      border: 1px solid rgba(91,141,239,0.30); border-left: 4px solid {ACCENT};
      border-radius: 14px; padding: 18px 22px; font-size: 1.04rem; line-height: 1.55;
      box-shadow: 0 4px 18px rgba(0,0,0,0.30);
  }}

  /* Badges */
  .badge {{ display:inline-block; padding:3px 11px; border-radius:999px;
      font-size:0.7rem; font-weight:700; letter-spacing:0.04em; }}
  .badge-crit {{ background:rgba(255,77,94,0.16); color:{BEAR}; border:1px solid {BEAR}; }}
  .badge-high {{ background:rgba(245,185,74,0.15); color:{GOLD}; border:1px solid {GOLD}; }}
  .badge-med  {{ background:rgba(139,144,160,0.15); color:{NEUTRAL}; border:1px solid {NEUTRAL}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #0d1119, #0a0d14);
      border-right: 1px solid rgba(255,255,255,0.06);
  }}
  [data-testid="stSidebar"] .stButton button {{
      border-radius: 10px; border: 1px solid rgba(91,141,239,0.35);
      background: rgba(91,141,239,0.12); color: {TEXT}; font-weight: 600;
      transition: all .15s ease;
  }}
  [data-testid="stSidebar"] .stButton button:hover {{
      background: rgba(91,141,239,0.28); border-color: {ACCENT};
  }}
  /* Sidebar nav links */
  [data-testid="stSidebarNav"] a {{ border-radius: 8px; }}
  [data-testid="stSidebarNav"] a:hover {{ background: rgba(91,141,239,0.10); }}

  /* Dividers a touch softer */
  hr {{ border-color: rgba(255,255,255,0.07) !important; }}

  /* Dataframe rounding */
  [data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}

  /* Tighten plotly chart cards with a subtle frame */
  [data-testid="stPlotlyChart"] {{
      background: rgba(20,25,37,0.45); border: 1px solid rgba(255,255,255,0.05);
      border-radius: 14px; padding: 8px 10px;
  }}
</style>
"""


def apply_theme():
    register_plotly_template()
    st.markdown(_CSS, unsafe_allow_html=True)


def sentiment_color(v: float) -> str:
    if v > 0.15:
        return BULL
    if v < -0.15:
        return BEAR
    return NEUTRAL
