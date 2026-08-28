"""Data access layer.

The only module that talks to Supabase. Every function returns typed objects
from `contract.models`. Page code never sees a raw dict and never builds a
query, which keeps the fetch -> state -> render boundary intact.

Caching TTLs mirror the refresh cadences in the spec: ~1s for live state,
~5s for diagnostics, longer for research and performance.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import streamlit as st
from supabase import Client, create_client

try:
    # Local development reads .env. On Streamlit Cloud the package may be
    # absent and secrets come from st.secrets instead, so this is optional.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> None:
        return None

from contract.enums import MarketStatus
from contract.models import (
    CointegrationResult,
    Command,
    ComponentStatus,
    EngineHeartbeat,
    EquityPoint,
    Fill,
    HeaderState,
    LiveState,
    LogEntry,
    ModelStatePoint,
    Order,
    Pair,
    PairPosition,
    PerformanceMetrics,
    Position,
    RollingDiagnostic,
    Signal,
    StrategyConfig,
    SystemMetrics,
    Trade,
)

load_dotenv()

# Cache lifetimes, chosen for egress cost as much as freshness.
# live_state is a single row (~1KB) so a 1s TTL is cheap. Chart history is
# hundreds of rows, so refetching it every second would cost ~720MB/hour per
# viewer and exhaust a free-tier egress budget in a single session.
TTL_LIVE = 1        # single-row reads only
TTL_SERIES = 30     # chart history
TTL_DIAG = 5
TTL_SLOW = 60
TTL_RESEARCH = 600

# Hard caps so a long-running engine cannot make a page fetch unbounded rows.
MAX_CHART_POINTS = 1500
MAX_LOG_LINES = 200


# ---------------------------------------------------------------------
# Client and authentication
# ---------------------------------------------------------------------


def _secret(name: str) -> str | None:
    """Prefer Streamlit secrets (cloud), fall back to .env (local)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


@st.cache_resource
def _public_client() -> Client:
    """Anonymous client for public reads.

    Cached, therefore SHARED ACROSS ALL SESSIONS. It must never have a user
    session attached to it — signing in on a cached client would grant every
    other visitor the signed-in user's privileges.
    """
    url, key = _secret("SUPABASE_URL"), _secret("SUPABASE_ANON_KEY")
    if not url or not key:
        st.error("SUPABASE_URL / SUPABASE_ANON_KEY not configured.")
        st.stop()
    return create_client(url, key)


def _session_client() -> Client | None:
    """Per-session authenticated client, created fresh on sign-in."""
    return st.session_state.get("_auth_client")


def get_client() -> Client:
    """Authenticated client when this session has signed in, else anonymous."""
    return _session_client() or _public_client()


def sign_in(email: str, password: str) -> tuple[bool, str]:
    """Authenticate into a session-scoped client, never the cached one."""
    url, key = _secret("SUPABASE_URL"), _secret("SUPABASE_ANON_KEY")
    try:
        client = create_client(url, key)          # deliberately uncached
        res = client.auth.sign_in_with_password(
            {"email": email, "password": password})
        if res.user is None:
            return False, "no user returned"
        st.session_state["_auth_client"] = client
        st.session_state["user_email"] = res.user.email
        return True, res.user.email
    except Exception as e:
        return False, str(e)


def sign_out() -> None:
    client = _session_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    st.session_state.pop("_auth_client", None)
    st.session_state.pop("user_email", None)


def is_authenticated() -> bool:
    return _session_client() is not None


def _t(table: str):
    return get_client().table(table)


# ---------------------------------------------------------------------
# Pair and configuration
# ---------------------------------------------------------------------


@st.cache_data(ttl=TTL_SLOW)
def fetch_pairs() -> list[Pair]:
    rows = _t("pairs").select("*").order("id").execute().data or []
    return [Pair.from_row(r) for r in rows]


@st.cache_data(ttl=TTL_SLOW)
def fetch_active_pair() -> Pair | None:
    pairs = fetch_pairs()
    active = [p for p in pairs if p.is_active]
    return active[0] if active else (pairs[0] if pairs else None)


@st.cache_data(ttl=TTL_DIAG)
def fetch_config(pair_id: int, status: str = "CURRENT") -> StrategyConfig | None:
    rows = (_t("strategy_config").select("*")
            .eq("pair_id", pair_id).eq("status", status)
            .limit(1).execute().data or [])
    return StrategyConfig.from_row(rows[0]) if rows else None


@st.cache_data(ttl=TTL_SLOW)
def fetch_config_history(pair_id: int, limit: int = 20) -> list[StrategyConfig]:
    rows = (_t("strategy_config").select("*").eq("pair_id", pair_id)
            .order("version", desc=True).limit(limit).execute().data or [])
    return [StrategyConfig.from_row(r) for r in rows]


def propose_config(payload: dict[str, Any]) -> tuple[bool, str]:
    """Insert a PENDING config. The engine promotes it to CURRENT."""
    try:
        _t("strategy_config").insert({**payload, "status": "PENDING"}).execute()
        fetch_config.clear()
        return True, "Pending configuration submitted."
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------
# Engine health
# ---------------------------------------------------------------------


@st.cache_data(ttl=TTL_LIVE)
def fetch_heartbeat() -> EngineHeartbeat | None:
    rows = (_t("engine_heartbeat").select("*")
            .order("heartbeat_at", desc=True).limit(1).execute().data or [])
    return EngineHeartbeat.from_row(rows[0]) if rows else None


@st.cache_data(ttl=TTL_DIAG)
def fetch_components() -> list[ComponentStatus]:
    rows = _t("connection_status").select("*").order("component").execute().data or []
    return [ComponentStatus.from_row(r) for r in rows]


@st.cache_data(ttl=TTL_DIAG)
def fetch_system_metrics() -> SystemMetrics | None:
    rows = _t("system_metrics").select("*").limit(1).execute().data or []
    return SystemMetrics.from_row(rows[0]) if rows else None


def fetch_header_state() -> HeaderState | None:
    pair = fetch_active_pair()
    if pair is None:
        return None
    hb = fetch_heartbeat()
    return HeaderState(
        pair=pair,
        heartbeat=hb,
        market_status=hb.market_status if hb else MarketStatus.UNKNOWN,
    )


# ---------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------


@st.cache_data(ttl=TTL_LIVE)
def fetch_live_state(pair_id: int) -> LiveState | None:
    rows = _t("live_state").select("*").eq("pair_id", pair_id).execute().data or []
    if not rows:
        return None
    pair = next((p for p in fetch_pairs() if p.id == pair_id), None)
    return LiveState.from_row(rows[0], pair) if pair else None


@st.cache_data(ttl=TTL_LIVE)
def fetch_positions(pair_id: int) -> list[Position]:
    rows = (_t("positions").select("*").eq("pair_id", pair_id)
            .order("ticker").execute().data or [])
    return [Position.from_row(r) for r in rows]


def fetch_pair_position(pair: Pair) -> PairPosition:
    return PairPosition(pair=pair, legs=fetch_positions(pair.id))


@st.cache_data(ttl=TTL_LIVE)
def fetch_recent_fills(pair_id: int, limit: int = 10) -> list[Fill]:
    rows = (_t("fills").select("*").eq("pair_id", pair_id)
            .order("ts", desc=True).limit(limit).execute().data or [])
    return [Fill.from_row(r) for r in rows]


# ---------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------

# Windows are expressed in wall-clock time so they stay correct when the feed
# moves from daily bars to intraday.
TIMEFRAMES: dict[str, timedelta | None] = {
    "1H": timedelta(hours=1),
    "4H": timedelta(hours=4),
    "1D": timedelta(days=1),
    "1W": timedelta(weeks=1),
    "1M": timedelta(days=30),
    "3M": timedelta(days=91),
    "1Y": timedelta(days=365),
    "ALL": None,
}


@st.cache_data(ttl=TTL_SERIES)
def fetch_model_history(pair_id: int, window: str = "1M",
                        max_points: int = MAX_CHART_POINTS,
                        bar_interval: str | None = None) -> list[ModelStatePoint]:
    q = _t("model_state_history").select("*").eq("pair_id", pair_id)
    if bar_interval:
        # Daily backfill and intraday live data are different timescales and
        # must never be mixed in one series.
        q = q.eq("bar_interval", bar_interval)
    delta = TIMEFRAMES.get(window)
    if delta is not None:
        cutoff = (datetime.now(timezone.utc) - delta).isoformat()
        q = q.gte("ts", cutoff)
    rows = q.order("ts", desc=True).limit(max_points).execute().data or []
    return [ModelStatePoint.from_row(r) for r in reversed(rows)]


@st.cache_data(ttl=TTL_RESEARCH)
def fetch_price_series(pair_id: int, limit: int = 3000) -> list[dict]:
    return (_t("market_data").select("ts,ticker,price")
            .eq("pair_id", pair_id).order("ts", desc=True)
            .limit(limit).execute().data or [])


@st.cache_data(ttl=TTL_DIAG)
def fetch_signals(pair_id: int, limit: int = 50) -> list[Signal]:
    rows = (_t("signals").select("*").eq("pair_id", pair_id)
            .order("ts", desc=True).limit(limit).execute().data or [])
    return [Signal.from_row(r) for r in rows]


# ---------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------


@st.cache_data(ttl=TTL_RESEARCH)
def fetch_cointegration(pair_id: int) -> list[CointegrationResult]:
    rows = (_t("v_latest_cointegration").select("*")
            .eq("pair_id", pair_id).execute().data or [])
    return [CointegrationResult.from_row(r) for r in rows]


@st.cache_data(ttl=TTL_RESEARCH)
def fetch_rolling(pair_id: int, limit: int = 3000) -> list[RollingDiagnostic]:
    rows = (_t("rolling_diagnostics").select("*").eq("pair_id", pair_id)
            .order("ts", desc=True).limit(limit).execute().data or [])
    return [RollingDiagnostic.from_row(r) for r in reversed(rows)]


# ---------------------------------------------------------------------
# Trades and performance
# ---------------------------------------------------------------------


@st.cache_data(ttl=TTL_SLOW)
def fetch_trades(pair_id: int, limit: int = 1000) -> list[Trade]:
    rows = (_t("v_trades").select("*").eq("pair_id", pair_id)
            .order("entry_time", desc=True).limit(limit).execute().data or [])
    return [Trade.from_row(r) for r in rows]


@st.cache_data(ttl=TTL_SLOW)
def fetch_orders(pair_id: int, limit: int = 500) -> list[Order]:
    rows = (_t("orders").select("*").eq("pair_id", pair_id)
            .order("submitted_at", desc=True).limit(limit).execute().data or [])
    return [Order.from_row(r) for r in rows]


@st.cache_data(ttl=TTL_SLOW)
def fetch_equity_curve(pair_id: int, limit: int = 5000) -> list[EquityPoint]:
    rows = (_t("equity_curve").select("*").eq("pair_id", pair_id)
            .order("ts", desc=True).limit(limit).execute().data or [])
    return [EquityPoint.from_row(r) for r in reversed(rows)]


@st.cache_data(ttl=TTL_SLOW)
def fetch_performance(pair_id: int, period: str = "ALL") -> PerformanceMetrics | None:
    rows = (_t("performance_metrics").select("*").eq("pair_id", pair_id)
            .eq("period_label", period).limit(1).execute().data or [])
    return PerformanceMetrics.from_row(rows[0]) if rows else None


# ---------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------


@st.cache_data(ttl=TTL_DIAG)
def fetch_logs(levels: Sequence[str] = (), categories: Sequence[str] = (),
               limit: int = MAX_LOG_LINES) -> list[LogEntry]:
    q = _t("system_logs").select("*")
    if levels:
        q = q.in_("level", list(levels))
    if categories:
        q = q.in_("category", list(categories))
    rows = q.order("ts", desc=True).limit(limit).execute().data or []
    return [LogEntry.from_row(r) for r in rows]


# ---------------------------------------------------------------------
# Control commands
# ---------------------------------------------------------------------


def issue_command(command: str, pair_id: int | None = None,
                  payload: dict | None = None,
                  requested_by: str = "streamlit") -> tuple[bool, str]:
    """Insert a REQUESTED command.

    A successful insert means the request was RECORDED, not executed. The
    caller must poll `fetch_command` and surface the engine's acknowledgement.
    """
    try:
        row = {"command": command, "status": "REQUESTED",
               "requested_by": requested_by, "source": "STREAMLIT"}
        if pair_id is not None:
            row["pair_id"] = pair_id
        if payload:
            row["payload"] = payload
        res = _t("commands").insert(row).execute().data
        return True, (res[0]["id"] if res else "")
    except Exception as e:
        return False, str(e)


def fetch_command(command_id: str) -> Command | None:
    rows = _t("commands").select("*").eq("id", command_id).execute().data or []
    return Command.from_row(rows[0]) if rows else None


@st.cache_data(ttl=TTL_DIAG)
def fetch_recent_commands(limit: int = 15) -> list[Command]:
    rows = (_t("commands").select("*")
            .order("requested_at", desc=True).limit(limit).execute().data or [])
    return [Command.from_row(r) for r in rows]


def clear_caches() -> None:
    for fn in (fetch_pairs, fetch_active_pair, fetch_config, fetch_config_history,
               fetch_heartbeat, fetch_components, fetch_system_metrics,
               fetch_live_state, fetch_positions, fetch_recent_fills,
               fetch_model_history, fetch_price_series, fetch_signals,
               fetch_cointegration, fetch_rolling, fetch_trades, fetch_orders,
               fetch_equity_curve, fetch_performance, fetch_logs,
               fetch_recent_commands):
        try:
            fn.clear()
        except Exception:
            pass