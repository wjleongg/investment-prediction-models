"""Overview and Strategy pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import components as c
import data
from contract.models import HeaderState
from engine import analytics


# =====================================================================
# OVERVIEW
# =====================================================================


def overview(header: HeaderState) -> None:
    pair = header.pair
    live = data.fetch_live_state(pair.id)
    cfg = data.fetch_config(pair.id)

    if live is None:
        c.banner("Live state has never been written. The Overview reflects the "
                 "engine's current state — it stays empty until the engine runs. "
                 "Backfilled history is on the Research page.", "mute")

    # --- KPI cards ----------------------------------------------------
    c.section("Portfolio")
    p = live.portfolio if live else None
    c.card_row([
        c.card("Total P&L", c.money(p.total_pnl) if p else "—",
               c.pct(p.total_return_pct) if p else "",
               c.sign_class(p.total_pnl if p else None)),
        c.card("Daily P&L", c.money(p.daily_pnl) if p else "—",
               c.pct(p.daily_return_pct) if p else "",
               c.sign_class(p.daily_pnl if p else None)),
        c.card("Realized", c.money(p.realized_pnl) if p else "—", "",
               c.sign_class(p.realized_pnl if p else None)),
        c.card("Unrealized", c.money(p.unrealized_pnl) if p else "—", "",
               c.sign_class(p.unrealized_pnl if p else None)),
        c.card("Exposure", c.money(p.current_exposure) if p else "—"),
        c.card("Capital used", f"{p.capital_utilisation:.1f}%" if p else "—",
               c.money(cfg.capital_allocation, 0) if cfg else ""),
        c.card("Position", live.current_position.value if live else "—",
               f"target {live.target_position.value}" if live else ""),
        c.card("Config", f"v{live.config_version}" if live and live.config_version
               else (f"v{cfg.version}" if cfg else "—")),
    ], per_row=4)

    # --- Active pair + signal ----------------------------------------
    left, right = st.columns([1, 2], gap="medium")

    with left:
        c.section("Active pair")
        if live:
            m = live.model
            c.kv_block([
                (pair.leg1_ticker, c.num(live.leg1_price, 2)),
                (pair.leg2_ticker, c.num(live.leg2_price, 2)),
                ("Spread", c.num(m.spread)),
                ("Hedge ratio", c.num(m.hedge_ratio, 6)),
                ("Z-score", c.num(m.zscore)),
            ])
            z = m.zscore
            tone = c.sign_class(-z if z else None)
            st.markdown(
                c.card("Z-score", c.num(z, 4) if z is not None else "—",
                       f"SIGNAL: {live.current_signal.value}", tone),
                unsafe_allow_html=True)
            st.markdown(
                c.pill(m.health.label, c.HEALTH_TONE.get(m.health, "mute")),
                unsafe_allow_html=True)
            if live.market_data_is_stale():
                c.banner("Market data is stale.", "warn")
        else:
            c.empty_state("live pair state", "Engine has not written live_state.")

    with right:
        c.section("Spread")
        tf = st.radio("Timeframe", list(data.TIMEFRAMES.keys()), index=5,
                      horizontal=True, key="ov_tf", label_visibility="collapsed")
        points = data.fetch_model_history(pair.id, tf)
        trades = data.fetch_trades(pair.id, limit=200)
        if points:
            st.plotly_chart(charts.spread_chart(points, trades),
                            use_container_width=True, key="ov_spread")
            st.plotly_chart(charts.zscore_chart(points, trades),
                            use_container_width=True, key="ov_z")
        else:
            c.empty_state("spread history", f"No model_state_history in {tf}.")

    # --- Model state --------------------------------------------------
    c.section("Model state")
    if live:
        m = live.model
        a, b = st.columns(2, gap="medium")
        with a:
            c.kv_block([
                ("Correlation", c.num(m.correlation)),
                ("Rolling correlation", c.num(m.rolling_correlation)),
                ("Cointegration p-value", c.num(m.cointegration_pvalue)),
                ("Hedge ratio", c.num(m.hedge_ratio, 6)),
            ])
        with b:
            c.kv_block([
                ("Spread mean", c.num(m.spread_mean)),
                ("Spread std", c.num(m.spread_std)),
                ("Half-life", f"{m.half_life:.1f} bars" if m.half_life else "—"),
                ("Last model update", c.ts_full(m.last_model_update_at)),
            ])
    else:
        c.empty_state("model state")

    # --- Risk and exposure -------------------------------------------
    c.section("Risk & exposure")
    positions = data.fetch_positions(pair.id)
    exposure = analytics.exposure_summary(live, positions, cfg)
    if exposure:
        c.card_row([
            c.card("Gross exposure", c.money(exposure["gross_exposure"], 0),
                   f"{exposure['gross_utilisation_pct']:.0f}% of "
                   f"{c.money(exposure['gross_limit'], 0)} limit",
                   "v-neg" if exposure["gross_utilisation_pct"] > 100 else "v-neu"),
            c.card("Net exposure", c.money(exposure["net_exposure"], 0),
                   exposure["market_neutrality"],
                   "v-neu" if exposure["market_neutrality"] in
                   ("MARKET NEUTRAL", "FLAT") else "v-neg"),
            c.card("Leverage", f"{exposure['leverage']:.2f}x",
                   f"{exposure['leverage_headroom_pct']:.0f}% headroom to "
                   f"{exposure['leverage_limit']:.1f}x",
                   "v-neg" if exposure["leverage"] > exposure["leverage_limit"]
                   else "v-neu"),
            c.card("Largest leg", c.money(exposure["largest_leg_notional"], 0),
                   f"{exposure['position_utilisation_pct']:.0f}% of "
                   f"{c.money(exposure['position_limit'], 0)} limit",
                   "v-neg" if exposure["position_utilisation_pct"] > 100
                   else "v-neu"),
        ])
        for breach in analytics.limit_breaches(exposure):
            c.banner(f"{breach}", "bad")
        if exposure["is_flat"]:
            st.caption("Currently flat. Limits shown are the headroom "
                       "available to the next position.")
    else:
        c.empty_state("exposure data", "No CURRENT configuration for this pair.")

    # --- Positions and fills -----------------------------------------
    a, b = st.columns(2, gap="medium")
    with a:
        c.section("Current position")
        pp = data.fetch_pair_position(pair)
        if pp.legs:
            st.dataframe(pd.DataFrame([{
                "Ticker": x.ticker, "Side": x.side.value, "Qty": x.quantity,
                "Avg entry": x.avg_entry_price, "Price": x.current_price,
                "Mkt value": x.market_value, "Unrealized": x.unrealized_pnl,
            } for x in pp.legs]), use_container_width=True, hide_index=True)
            st.markdown(c.card("Pair-level P&L", c.money(pp.pair_pnl), "",
                               c.sign_class(pp.pair_pnl)),
                        unsafe_allow_html=True)
        else:
            c.empty_state("open positions")

    with b:
        c.section("Recent fills")
        fills = data.fetch_recent_fills(pair.id, 10)
        if fills:
            st.dataframe(pd.DataFrame([{
                "Time": c.ts(f.ts), "Ticker": f.ticker, "Side": f.side.value,
                "Qty": f.quantity, "Price": f.price, "Order": f.order_id,
            } for f in fills]), use_container_width=True, hide_index=True)
        else:
            c.empty_state("fills")


# =====================================================================
# STRATEGY
# =====================================================================


def strategy(header: HeaderState) -> None:
    pair = header.pair
    live = data.fetch_live_state(pair.id)
    cfg = data.fetch_config(pair.id)

    c.section("Pair information")
    if live:
        a, b = st.columns(2, gap="medium")
        with a:
            c.kv_block([
                ("Leg 1", pair.leg1_ticker),
                ("Price", c.num(live.leg1_price, 2)),
                ("Bid / Ask", f"{c.num(live.leg1_bid, 2)} / "
                              f"{c.num(live.leg1_ask, 2)}"),
                ("Mid", c.num(live.leg1_mid, 2)),
            ])
        with b:
            c.kv_block([
                ("Leg 2", pair.leg2_ticker),
                ("Price", c.num(live.leg2_price, 2)),
                ("Bid / Ask", f"{c.num(live.leg2_bid, 2)} / "
                              f"{c.num(live.leg2_ask, 2)}"),
                ("Mid", c.num(live.leg2_mid, 2)),
            ])
    else:
        c.empty_state("live pair data", "Engine has not written live_state.")

    c.section("Statistical relationship")
    if live:
        m = live.model
        a, b = st.columns(2, gap="medium")
        with a:
            c.kv_block([
                ("Correlation", c.num(m.correlation)),
                ("Rolling correlation", c.num(m.rolling_correlation)),
                ("Cointegration statistic", c.num(m.cointegration_stat)),
                ("Cointegration p-value", c.num(m.cointegration_pvalue)),
            ])
        with b:
            c.kv_block([
                ("Hedge ratio", c.num(m.hedge_ratio, 6)),
                ("Half-life", f"{m.half_life:.1f} bars" if m.half_life else "—"),
                ("Spread volatility", c.num(m.spread_volatility)),
                ("Relationship", m.health.label),
            ])

    c.section("Signal engine")
    if live and cfg:
        a, b = st.columns([1, 1], gap="medium")
        with a:
            c.kv_block([
                ("Current z-score", c.num(live.model.zscore)),
                ("Entry threshold", f"±{cfg.entry_threshold:g}"),
                ("Exit threshold", f"±{cfg.exit_threshold:g}"),
                ("Stop-loss threshold", f"±{cfg.stop_loss_threshold:g}"),
            ])
        with b:
            c.kv_block([
                ("Current signal", live.current_signal.value),
                ("Signal since", c.ts_full(live.signal_since)),
                ("Current position", live.current_position.value),
                ("Target position", live.target_position.value),
            ])
        explanation = live.signal_explanation or _fallback_explanation(live, cfg)
        c.banner(explanation, "mute")
    else:
        c.empty_state("signal state")

    c.section("Signal timeline")
    signals = data.fetch_signals(pair.id, 50)
    if signals:
        st.dataframe(pd.DataFrame([{
            "Time": c.ts_full(s.ts), "Signal": s.signal.value,
            "Previous": s.previous_signal.value if s.previous_signal else "—",
            "Z-score": s.zscore, "Acted": s.acted_upon,
            "Note": s.explanation or s.suppressed_reason or "",
        } for s in signals]), use_container_width=True, hide_index=True,
            height=300)
    else:
        c.empty_state("signals", "The engine has not generated any signals yet.")


def _fallback_explanation(live, cfg) -> str:
    """Plain-English state description when the engine hasn't supplied one."""
    z = live.model.zscore
    if z is None:
        return "No z-score available."
    pair = live.pair.label
    if abs(z) >= cfg.entry_threshold:
        side = "below" if z < 0 else "above"
        direction = "LONG SPREAD" if z < 0 else "SHORT SPREAD"
        return (f"{pair} spread is {abs(z):.2f} standard deviations {side} its "
                f"estimated mean, exceeding the ±{cfg.entry_threshold:g} entry "
                f"threshold. Relationship health is {live.model.health.value}, "
                f"which {'permits' if live.model.health.value == 'VALID' else 'blocks'} "
                f"a {direction} entry.")
    return (f"{pair} spread is {abs(z):.2f} standard deviations from its mean, "
            f"inside the ±{cfg.entry_threshold:g} entry threshold. No entry "
            f"condition is met.")
