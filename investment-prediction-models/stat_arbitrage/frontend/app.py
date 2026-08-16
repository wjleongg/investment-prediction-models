"""STAT ARB ENGINE — monitoring and control frontend.

Streamlit is the window, never the engine. This app reads materialised state
from Supabase and issues control requests. It computes no signals and makes
no trading decisions.

Run:  streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `contract` and the frontend modules importable regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "frontend"), str(ROOT / "frontend" / "views")]

import components as c  # noqa: E402
import data  # noqa: E402
import theme  # noqa: E402
from views import analysis, control, live, ops  # noqa: E402

st.set_page_config(page_title="STAT ARB ENGINE", page_icon="◧",
                   layout="wide", initial_sidebar_state="collapsed")
theme.apply()

PAGES = {
    "Overview": (live.overview, 1),
    "Strategy": (live.strategy, 5),
    "Research": (analysis.research, None),
    "Performance": (analysis.performance, None),
    "Trades": (ops.trades, None),
    "System Health": (ops.system_health, 5),
    "Configuration": (control.configuration, None),
    "Controls": (control.controls, None),
}


# Pages that can change system state require an authenticated session.
PROTECTED = {"Trades","Configuration", "Controls"}


def sign_in_panel(reason: str) -> None:
    """Inline sign-in shown only when a protected page is opened."""
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        c.banner(reason, "warn")
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", use_container_width=True):
                ok, detail = data.sign_in(email, password)
                if ok:
                    st.rerun()
                else:
                    st.error(f"Sign-in failed: {detail}")
        st.caption("Read-only pages need no sign-in. Authentication is "
                   "required only to change configuration or issue commands.")


def main() -> None:
    header = data.fetch_header_state()
    c.render_header(header)

    nav, auth, actions = st.columns([6, 1.2, 1])
    with nav:
        page = st.radio("Navigation", list(PAGES.keys()), horizontal=True,
                        label_visibility="collapsed", key="nav")
    with auth:
        if data.is_authenticated():
            if st.button("Sign out", use_container_width=True):
                data.sign_out()
                st.rerun()
        else:
            st.markdown(
                "<div style='font-family:monospace;font-size:.68rem;"
                "color:#7d8590;padding-top:.45rem;text-align:center'>"
                "READ ONLY</div>", unsafe_allow_html=True)
    with actions:
        if st.button("↻ Refresh", use_container_width=True):
            data.clear_caches()
            st.rerun()

    if header is None:
        c.banner("No pair is configured. Insert a row into `pairs` and mark it "
                 "active.", "bad")
        return

    if page in PROTECTED and not data.is_authenticated():
        c.section(page)
        sign_in_panel(f"{page} can change live system state. Sign in to "
                      f"continue.")
        return

    render, refresh = PAGES[page]

    if refresh and hasattr(st, "fragment"):
        @st.fragment(run_every=refresh)
        def _live_block():
            render(data.fetch_header_state() or header)

        _live_block()
    else:
        render(header)

    user = st.session_state.get("user_email")
    st.markdown(
        f"<div style='margin-top:2rem;color:#7d8590;font-size:.68rem;"
        f"font-family:monospace;border-top:1px solid #272e38;padding-top:.6rem'>"
        f"{'Signed in as ' + user if user else 'Read-only session'} · "
        f"Streamlit is the monitoring layer; the Python engine is the source "
        f"of truth.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()