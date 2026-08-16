"""Trades and System Health pages."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

import charts
import components as c
import data
from contract.enums import LogCategory, LogLevel
from contract.models import HeaderState


# =====================================================================
# TRADES
# =====================================================================


def trades(header: HeaderState) -> None:
    pair = header.pair
    all_trades = data.fetch_trades(pair.id)

    if not all_trades:
        c.empty_state("trades", "The engine has not executed any trades yet.")
        _orders_section(pair.id)
        return

    c.section("Filters")
    f1, f2, f3, f4 = st.columns(4)
    direction = f1.selectbox("Direction", ["All", "LONG_SPREAD", "SHORT_SPREAD"])
    status = f2.selectbox("Status", ["All", "OPEN", "CLOSING", "CLOSED", "FAILED"])
    outcome = f3.selectbox("Outcome", ["All", "Profitable", "Losing"])
    days = f4.selectbox("Period", ["All", "7d", "30d", "90d", "1y"])

    rows = all_trades
    if direction != "All":
        rows = [t for t in rows if t.direction.value == direction]
    if status != "All":
        rows = [t for t in rows if t.status.value == status]
    if outcome != "All":
        want = outcome == "Profitable"
        rows = [t for t in rows if t.is_profitable is want]
    if days != "All":
        cutoff = datetime.now(timezone.utc) - timedelta(
            days={"7d": 7, "30d": 30, "90d": 90, "1y": 365}[days])
        rows = [t for t in rows if t.entry_time >= cutoff]

    c.section(f"Trade history — {len(rows)} of {len(all_trades)}")
    if not rows:
        c.empty_state("matching trades", "Adjust the filters above.")
        return

    # Column headers derive from the pair, never hard-coded tickers.
    l1, l2 = pair.leg1_ticker, pair.leg2_ticker
    df = pd.DataFrame([{
        "ID": t.id,
        "Pair": pair.label,
        "Direction": t.direction.value,
        "Entry": c.ts_full(t.entry_time),
        "Exit": c.ts_full(t.exit_time),
        f"{l1} entry": t.leg1_entry_price,
        f"{l2} entry": t.leg2_entry_price,
        f"{l1} exit": t.leg1_exit_price,
        f"{l2} exit": t.leg2_exit_price,
        "Entry Z": t.entry_zscore,
        "Exit Z": t.exit_zscore,
        "P&L": t.net_pnl,
        "Return %": t.return_pct,
        "Holding": c.duration(t.holding_period_seconds),
        "Status": t.status.value,
    } for t in rows])
    st.dataframe(df, use_container_width=True, hide_index=True, height=340)

    # --- Detail -------------------------------------------------------
    c.section("Trade detail")
    selected_id = st.selectbox("Trade", [t.id for t in rows],
                               format_func=lambda i: f"#{i}")
    trade = next(t for t in rows if t.id == selected_id)

    a, b = st.columns(2, gap="medium")
    with a:
        st.caption("Entry")
        c.kv_block([
            ("Time", c.ts_full(trade.entry_time)),
            ("Direction", trade.direction.value),
            ("Z-score", c.num(trade.entry_zscore)),
            ("Hedge ratio", c.num(trade.entry_hedge_ratio, 6)),
            (f"{l1} price", c.num(trade.leg1_entry_price, 2)),
            (f"{l2} price", c.num(trade.leg2_entry_price, 2)),
            (f"{l1} qty", c.num(trade.leg1_quantity, 0)),
            (f"{l2} qty", c.num(trade.leg2_quantity, 0)),
        ])
    with b:
        st.caption("Exit")
        c.kv_block([
            ("Time", c.ts_full(trade.exit_time)),
            ("Reason", trade.exit_reason.value if trade.exit_reason else "—"),
            ("Z-score", c.num(trade.exit_zscore)),
            ("Hedge ratio", c.num(trade.exit_hedge_ratio, 6)),
            (f"{l1} price", c.num(trade.leg1_exit_price, 2)),
            (f"{l2} price", c.num(trade.leg2_exit_price, 2)),
            ("Net P&L", c.money(trade.net_pnl)),
            ("Holding period", c.duration(trade.holding_period_seconds)),
        ])

    if trade.entry_model_state or trade.exit_model_state:
        m1, m2 = st.columns(2, gap="medium")
        m1.caption("Model state at entry")
        m1.json(trade.entry_model_state or {}, expanded=False)
        m2.caption("Model state at exit")
        m2.json(trade.exit_model_state or {}, expanded=False)

    # Mini chart across the trade lifetime
    window_pts = [p for p in data.fetch_model_history(pair.id, "ALL")
                  if p.ts >= trade.entry_time
                  and (trade.exit_time is None or p.ts <= trade.exit_time)]
    if window_pts:
        st.plotly_chart(charts.mini_trade_chart(window_pts, trade),
                        use_container_width=True, key="td_mini")
    else:
        c.empty_state("model history for this trade window")

    _orders_section(pair.id)


def _orders_section(pair_id: int) -> None:
    c.section("Execution history")
    st.caption("Raw broker orders, separate from strategy-level trades.")
    orders = data.fetch_orders(pair_id)
    if not orders:
        c.empty_state("orders")
        return
    st.dataframe(pd.DataFrame([{
        "Order ID": o.id, "Broker ID": o.broker_order_id or "—",
        "Time": c.ts_full(o.submitted_at), "Ticker": o.ticker,
        "Side": o.side.value, "Qty": o.quantity, "Type": o.order_type.value,
        "Limit": o.limit_price, "Fill price": o.avg_fill_price,
        "Filled": o.filled_quantity, "Status": o.status.value,
        "Trade": o.trade_id or "—",
    } for o in orders]), use_container_width=True, hide_index=True, height=280)


# =====================================================================
# SYSTEM HEALTH
# =====================================================================


def system_health(header: HeaderState) -> None:
    hb = header.heartbeat
    now = datetime.now(timezone.utc)

    # --- Heartbeat, deliberately the most prominent element ----------
    c.section("Engine heartbeat")
    if hb is None:
        st.markdown(c.card("Engine", "✕ NO HEARTBEAT",
                           "The engine has never written to engine_heartbeat.",
                           "v-neg"), unsafe_allow_html=True)
    else:
        age = hb.age_seconds(now)
        tone = "v-pos" if age <= 5 else ("v-neu" if age <= 30 else "v-neg")
        c.card_row([
            c.card("Engine", hb.health_label(now).split(" — ")[0],
                   f"last heartbeat {age:.1f}s ago", tone),
            c.card("Reported status", hb.status.value,
                   "as claimed by the engine row"),
            c.card("Data source", hb.data_source.badge, "",
                   "v-pos" if hb.data_source.is_live else "v-neu"),
            c.card("Uptime", c.duration(
                (now - hb.started_at).total_seconds() if hb.started_at else None)),
        ])
        st.caption("Health is derived from heartbeat age, never from the "
                   "reported status field.")

    # --- Connections --------------------------------------------------
    c.section("Connection status")
    comps = data.fetch_components()
    if comps:
        cards = []
        for comp in comps:
            tone = {"CONNECTED": "v-pos", "RECONNECTING": "v-neu",
                    "DISCONNECTED": "v-neg", "ERROR": "v-neg"}.get(
                        comp.state.value, "v-neu")
            cards.append(c.card(comp.component, comp.state.value,
                                f"last OK {c.ts(comp.last_ok_at)}", tone))
        c.card_row(cards, per_row=5)
    else:
        c.empty_state("connection status")

    # --- Metrics ------------------------------------------------------
    c.section("System metrics")
    m = data.fetch_system_metrics()
    if m:
        c.card_row([
            c.card("Market data events", f"{m.market_data_events:,}"),
            c.card("Model calculations", f"{m.model_calculations:,}"),
            c.card("Signals generated", f"{m.signals_generated:,}"),
            c.card("Orders submitted", f"{m.orders_submitted:,}"),
            c.card("Orders filled", f"{m.orders_filled:,}"),
            c.card("State writes", f"{m.state_writes:,}"),
            c.card("Errors", f"{m.error_count:,}", "",
                   "v-neg" if m.error_count else "v-neu"),
            c.card("Warnings", f"{m.warning_count:,}", "",
                   "v-neu" if not m.warning_count else "v-neu"),
        ])
        c.kv_block([
            ("Last market data", c.ts_full(m.last_market_data_at)),
            ("Last model calculation", c.ts_full(m.last_model_calc_at)),
            ("Last state persistence", c.ts_full(m.last_state_write_at)),
            ("Last order event", c.ts_full(m.last_order_event_at)),
        ])
    else:
        c.empty_state("system metrics")

    # --- Logs ---------------------------------------------------------
    c.section("System log")
    f1, f2 = st.columns([1, 1])
    levels = f1.multiselect("Level", [e.value for e in LogLevel],
                            default=[], placeholder="All levels")
    cats = f2.multiselect("Category", [e.value for e in LogCategory],
                          default=[], placeholder="All categories")
    logs = data.fetch_logs(levels, cats, limit=300)
    if logs:
        colours = {"ERROR": "#f85149", "CRITICAL": "#f85149",
                   "WARNING": "#d29922", "INFO": "#c9d1d9", "DEBUG": "#7d8590"}
        html = "".join(
            f'<div class="logline" style="color:{colours.get(l.level.value, "#c9d1d9")}">'
            f'{l.ts:%H:%M:%S}  {l.level.value:<8} {l.category.value:<12} '
            f'{l.message}</div>' for l in logs)
        st.markdown(f'<div style="max-height:340px;overflow-y:auto;'
                    f'background:#161b22;border:1px solid #272e38;'
                    f'border-radius:4px;padding:.6rem">{html}</div>',
                    unsafe_allow_html=True)
    else:
        c.empty_state("log entries", "No rows match the current filters.")
