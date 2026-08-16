"""Dark terminal styling. Single source of visual truth for the app."""

import streamlit as st

# Palette — muted terminal, not neon. Positive/negative are colourblind-safe
# (blue/amber rather than green/red) and never the only signal.
BG = "#0d1117"
PANEL = "#161b22"
BORDER = "#272e38"
TEXT = "#c9d1d9"
MUTED = "#7d8590"
POS = "#3fb950"
NEG = "#f85149"
WARN = "#d29922"
ACCENT = "#58a6ff"
AMBER = "#ff9e1b"
MONO = "'JetBrains Mono','SF Mono',Menlo,Consolas,monospace"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="SF Mono, Menlo, Consolas, monospace", size=11, color=TEXT),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    margin=dict(l=48, r=16, t=28, b=32),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

CSS = f"""
<style>
  .stApp {{ background: {BG}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ display: none; }}
  .block-container {{ padding: 0.6rem 1.4rem 2rem; max-width: 1600px; }}

  h1,h2,h3,h4 {{ font-family: {MONO}; letter-spacing: .02em; color: {TEXT}; }}
  h3 {{ font-size: .82rem !important; text-transform: uppercase;
        color: {MUTED}; font-weight: 600; margin: 1.1rem 0 .4rem;
        border-bottom: 1px solid {BORDER}; padding-bottom: .3rem; }}

  /* Horizontal nav built from a radio group */
  div[role="radiogroup"] {{ gap: 0 !important; border-bottom: 1px solid {BORDER};
                            margin-bottom: .8rem; }}
  div[role="radiogroup"] label {{
      font-family: {MONO}; font-size: .74rem; text-transform: uppercase;
      letter-spacing: .06em; padding: .45rem .95rem; margin: 0 !important;
      color: {MUTED}; border-bottom: 2px solid transparent; cursor: pointer; }}
  div[role="radiogroup"] label:hover {{ color: {AMBER}; }}
  div[role="radiogroup"] input {{ display: none; }}
  div[role="radiogroup"] label:has(input:checked) {{
      color: {AMBER}; border-bottom-color: {AMBER}; font-weight: 600; }}

  .hdr {{ display:flex; align-items:center; gap:1.4rem; flex-wrap:wrap;
          font-family:{MONO}; font-size:.74rem; background:{PANEL};
          border:1px solid {BORDER}; border-radius:4px;
          padding:.5rem .9rem; margin-bottom:.5rem; }}
  .hdr .brand {{ font-weight:700; letter-spacing:.12em; color:{TEXT}; }}
  .hdr .k {{ color:{MUTED}; margin-right:.3rem; }}
  .hdr .v {{ color:{TEXT}; }}

  .pill {{ font-family:{MONO}; font-size:.68rem; padding:.16rem .5rem;
           border-radius:3px; border:1px solid; letter-spacing:.05em; }}
  .pill-ok   {{ color:{POS};  border-color:{POS}33;  background:{POS}14; }}
  .pill-warn {{ color:{WARN}; border-color:{WARN}33; background:{WARN}14; }}
  .pill-bad  {{ color:{NEG};  border-color:{NEG}33;  background:{NEG}14; }}
  .pill-mute {{ color:{MUTED};border-color:{BORDER}; background:transparent; }}

  .card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:4px;
           padding:.6rem .75rem; height:100%; }}
  .card .lbl {{ font-family:{MONO}; font-size:.64rem; text-transform:uppercase;
                letter-spacing:.07em; color:{MUTED}; margin-bottom:.25rem; }}
  .card .val {{ font-family:{MONO}; font-size:1.22rem; font-weight:600;
                line-height:1.15; }}
  .card .sub {{ font-family:{MONO}; font-size:.66rem; color:{MUTED};
                margin-top:.15rem; }}
  .v-pos {{ color:{POS}; }} .v-neg {{ color:{NEG}; }} .v-neu {{ color:{TEXT}; }}

  .kv {{ display:flex; justify-content:space-between; font-family:{MONO};
         font-size:.76rem; padding:.24rem 0;
         border-bottom:1px solid {BORDER}44; }}
  .kv .k {{ color:{MUTED}; }} .kv .v {{ color:{TEXT}; font-weight:600; }}

  .logline {{ font-family:{MONO}; font-size:.72rem; padding:.12rem 0;
              white-space:pre-wrap; }}
  .banner {{ font-family:{MONO}; font-size:.76rem; padding:.5rem .8rem;
             border-radius:4px; border:1px solid; margin:.4rem 0; }}

  .stDataFrame {{ font-family:{MONO} !important; font-size:.74rem !important; }}
  div[data-testid="stMetricValue"] {{ font-family:{MONO}; }}
  .stButton>button {{ font-family:{MONO}; font-size:.74rem;
                      border:1px solid {BORDER}; background:{PANEL};
                      color:{TEXT}; border-radius:3px; }}
  .stButton>button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
  .danger .stButton>button {{ border-color:{NEG}66; color:{NEG};
                              background:{NEG}0f; }}
  .danger .stButton>button:hover {{ background:{NEG}22; border-color:{NEG}; }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
