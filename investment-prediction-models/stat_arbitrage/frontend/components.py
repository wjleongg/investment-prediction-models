"""Pure render helpers. No data access, no business logic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import streamlit as st

from contract.enums import ConnectionState, RelationshipHealth, SystemStatus
from contract.models import HeaderState

ET_OFFSET_NOTE = "UTC"


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------


def money(v: float | None, dp: int = 2) -> str:
    return "—" if v is None else f"{'-' if v < 0 else ''}${abs(v):,.{dp}f}"


def pct(v: float | None, dp: int = 2) -> str:
    return "—" if v is None else f"{v:+.{dp}f}%"


def num(v: float | None, dp: int = 4) -> str:
    return "—" if v is None else f"{v:,.{dp}f}"


def ts(dt: datetime | None, fmt: str = "%H:%M:%S") -> str:
    return "—" if dt is None else dt.strftime(fmt)


def ts_full(dt: datetime | None) -> str:
    return "—" if dt is None else dt.strftime("%Y-%m-%d %H:%M:%S")


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


def sign_class(v: float | None) -> str:
    if v is None or v == 0:
        return "v-neu"
    return "v-pos" if v > 0 else "v-neg"


# ---------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------


def pill(text: str, tone: str = "mute") -> str:
    return f'<span class="pill pill-{tone}">{text}</span>'


STATUS_TONE = {
    SystemStatus.RUNNING: "ok",
    SystemStatus.STARTING: "warn",
    SystemStatus.PAUSED: "warn",
    SystemStatus.STOPPING: "warn",
    SystemStatus.STOPPED: "mute",
    SystemStatus.ERROR: "bad",
    SystemStatus.KILLED: "bad",
}

HEALTH_TONE = {
    RelationshipHealth.VALID: "ok",
    RelationshipHealth.DEGRADED: "warn",
    RelationshipHealth.INVALID: "bad",
}

CONN_TONE = {
    ConnectionState.CONNECTED: "ok",
    ConnectionState.RECONNECTING: "warn",
    ConnectionState.DISCONNECTED: "bad",
    ConnectionState.ERROR: "bad",
    ConnectionState.UNKNOWN: "mute",
}


def card(label: str, value: str, sub: str = "", tone: str = "v-neu") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val {tone}">{value}</div>{sub_html}</div>')


def card_row(cards: list[str], per_row: int = 4) -> None:
    for i in range(0, len(cards), per_row):
        cols = st.columns(per_row, gap="small")
        for col, html in zip(cols, cards[i:i + per_row]):
            col.markdown(html, unsafe_allow_html=True)


def kv_block(items: Iterable[tuple[str, Any]]) -> None:
    rows = "".join(f'<div class="kv"><span class="k">{k}</span>'
                   f'<span class="v">{v}</span></div>' for k, v in items)
    st.markdown(rows, unsafe_allow_html=True)


def banner(text: str, tone: str = "warn") -> None:
    colours = {"ok": "#3fb950", "warn": "#d29922", "bad": "#f85149",
               "mute": "#7d8590"}
    c = colours.get(tone, colours["mute"])
    st.markdown(
        f'<div class="banner" style="border-color:{c}55;background:{c}12;'
        f'color:{c};">{text}</div>', unsafe_allow_html=True)


def empty_state(what: str, reason: str = "") -> None:
    """Honest empty state. Never fabricate data to fill a chart."""
    detail = f"<br><span style='color:#7d8590'>{reason}</span>" if reason else ""
    st.markdown(
        f'<div class="banner" style="border-color:#272e38;background:#161b22;'
        f'color:#7d8590;text-align:center;padding:1.6rem;">'
        f'NO {what.upper()} YET{detail}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f"### {title}")


# ---------------------------------------------------------------------
# Persistent header
# ---------------------------------------------------------------------


def render_header(header: HeaderState | None) -> None:
    if header is None:
        st.markdown('<div class="hdr"><span class="brand">STAT ARB ENGINE</span>'
                    '<span class="pill pill-bad">NO PAIR CONFIGURED</span></div>',
                    unsafe_allow_html=True)
        return

    hb = header.heartbeat
    status = header.system_status
    src = header.data_source
    now = datetime.now(timezone.utc)

    status_pill = pill(f"● {status.value}", STATUS_TONE.get(status, "mute"))
    src_pill = pill(src.badge, "ok" if src.is_live else "warn")
    mkt_pill = pill(header.market_status.value.replace("_", " "), "mute")

    if hb is None:
        hb_pill = pill("✕ NO ENGINE", "bad")
    else:
        age = hb.age_seconds(now)
        tone = "ok" if age <= 5 else ("warn" if age <= 30 else "bad")
        hb_pill = pill(f"♥ {age:.1f}s", tone)

    st.markdown(
        f'<div class="hdr">'
        f'<span class="brand">STAT ARB ENGINE</span>'
        f'{status_pill}'
        f'<span><span class="k">PAIR</span>'
        f'<span class="v">{header.pair.label}</span></span>'
        f'{src_pill}{mkt_pill}{hb_pill}'
        f'<span style="margin-left:auto"><span class="k">UTC</span>'
        f'<span class="v">{now.strftime("%Y-%m-%d %H:%M:%S")}</span></span>'
        f'</div>', unsafe_allow_html=True)

    if src.is_live is False:
        banner(f"{src.badge} — figures below are not live market data.", "warn")
    if hb is None:
        banner("✕ ENGINE NOT RUNNING — no heartbeat has ever been recorded. "
               "Live state, positions and P&L will be empty until the engine "
               "starts.", "bad")
    elif hb.is_dead(now):
        banner(f"✕ ENGINE UNREACHABLE — last heartbeat "
               f"{hb.age_seconds(now):.0f}s ago. Displayed state is stale.", "bad")
    elif hb.is_stale(now):
        banner(f"⚠ ENGINE STALE — last heartbeat "
               f"{hb.age_seconds(now):.1f}s ago.", "warn")


def live(run_every: int | float):
    """Wrap a render function so Streamlit reruns just that block periodically.

    Falls back to a static render on Streamlit versions without fragments,
    so the app degrades to manual refresh rather than breaking.
    """
    def decorator(fn):
        if hasattr(st, "fragment"):
            return st.fragment(run_every=run_every)(fn)
        return fn
    return decorator
