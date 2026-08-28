"""Research and Performance pages — analytical, on-demand refresh."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import charts
import components as c
import data
import insights
from contract.models import HeaderState
from engine import analytics


# =====================================================================
# RESEARCH
# =====================================================================


def research(header: HeaderState) -> None:
    pair = header.pair
    cfg = data.fetch_config(pair.id)

    c.banner("Research reads persisted diagnostics. All estimation is performed "
             "engine-side; this page never runs statistical tests itself.", "mute")

    # --- Price series -------------------------------------------------
    c.section("Price series")
    price_interval = (data.fetch_bar_intervals(pair.id) or ["1d"])[0]
    rows = data.fetch_price_series(pair.id, bar_interval=price_interval)
    if rows:
        df = pd.DataFrame(rows)
        # Backfilled and engine-written rows use different timestamp precision,
        # so an inferred single format fails on the combined set.
        df["ts"] = pd.to_datetime(df["ts"], format="mixed", utc=True,
                                  errors="coerce")
        df = df.dropna(subset=["ts"])
        wide = df.pivot_table(index="ts", columns="ticker",
                              values="price").sort_index().dropna()
        cols = [t for t in (pair.leg1_ticker, pair.leg2_ticker) if t in wide.columns]
        mode = st.radio("Scale", ["Raw", "Normalised", "Log"], index=1,
                        horizontal=True, key="rs_mode",
                        label_visibility="collapsed")
        st.plotly_chart(charts.price_chart(wide[cols], mode),
                        use_container_width=True, key="rs_price")
        st.caption(f"{len(wide):,} aligned bars · "
                   f"{wide.index[0]:%Y-%m-%d} → {wide.index[-1]:%Y-%m-%d}")
    else:
        c.empty_state("price history", "Run scripts/backfill.py to seed market_data.")

    # --- Spread -------------------------------------------------------
    c.section("Spread analysis")
    # Research uses the daily backfill; live pages use the engine's interval.
    points = data.fetch_model_history(pair.id, "ALL", bar_interval="1d")
    if points:
        st.plotly_chart(charts.spread_chart(points, height=340),
                        use_container_width=True, key="rs_spread")
        spreads = [p.spread for p in points]
        vols = [p.spread_std for p in points if p.spread_std]
        c.card_row([
            c.card("Spread mean", c.num(float(np.mean(spreads)))),
            c.card("Spread std", c.num(float(np.std(spreads)))),
            c.card("Rolling vol (last)", c.num(vols[-1] if vols else None)),
            c.card("Observations", f"{len(points):,}"),
        ])
    else:
        c.empty_state("spread history")

    # --- Correlation --------------------------------------------------
    c.section("Correlation")
    rolling = data.fetch_rolling(pair.id)
    corrs = [r.rolling_correlation for r in rolling if r.rolling_correlation is not None]
    if corrs:
        st.plotly_chart(
            charts.rolling_line(rolling, "rolling_correlation",
                                "Rolling correlation",
                                cfg.min_correlation if cfg else None),
            use_container_width=True, key="rs_corr")
        c.card_row([
            c.card("Current", c.num(corrs[-1])),
            c.card("Average", c.num(float(np.mean(corrs)))),
            c.card("Minimum", c.num(float(np.min(corrs)))),
            c.card("Maximum", c.num(float(np.max(corrs)))),
        ])
        st.caption("Correlation is not static — the minimum matters more than "
                   "the average when sizing risk.")
    else:
        c.empty_state("rolling correlation")

    # --- Cointegration ------------------------------------------------
    c.section("Cointegration tests")
    tests = data.fetch_cointegration(pair.id)
    if tests:
        st.dataframe(pd.DataFrame([{
            "Test": t.test.display_name,
            "Statistic": None if t.statistic is None else round(t.statistic, 4),
            "p-value": None if t.pvalue is None else round(t.pvalue, 6),
            "Result": t.verdict,
            "Interpretation": t.interpretation or "",
        } for t in tests]), use_container_width=True, hide_index=True)
        st.caption(f"Computed {c.ts_full(tests[0].as_of)} over "
                   f"{tests[0].lookback_bars} bars.")
    else:
        c.empty_state("cointegration results")

    # --- Rolling cointegration ---------------------------------------
    c.section("Rolling cointegration")
    if rolling:
        st.plotly_chart(
            charts.rolling_line(rolling, "rolling_cointegration_pvalue",
                                "Rolling p-value",
                                cfg.max_cointegration_pvalue if cfg else 0.05),
            use_container_width=True, key="rs_rollcoint")
        valid = [r for r in rolling if r.relationship_valid]
        st.caption(f"{len(valid)}/{len(rolling)} windows "
                   f"({100 * len(valid) / len(rolling):.0f}%) show cointegration "
                   f"at the configured threshold. Sustained rises in p-value "
                   f"indicate decay.")
    else:
        c.empty_state("rolling cointegration")

    # --- Hedge ratio --------------------------------------------------
    c.section("Hedge ratio")
    betas = [r.rolling_hedge_ratio for r in rolling if r.rolling_hedge_ratio is not None]
    if betas:
        st.plotly_chart(
            charts.rolling_line(rolling, "rolling_hedge_ratio", "Rolling hedge ratio"),
            use_container_width=True, key="rs_beta")
        c.card_row([
            c.card("Current", c.num(betas[-1], 6)),
            c.card("Mean", c.num(float(np.mean(betas)), 6)),
            c.card("Volatility", c.num(float(np.std(betas)), 6)),
            c.card("Stability", f"{100 * (1 - np.std(betas) / abs(np.mean(betas))):.1f}%"
                   if np.mean(betas) else "—",
                   "1 − CV, higher is more stable"),
        ])
    else:
        c.empty_state("hedge ratio history")

    # --- Mean reversion -----------------------------------------------
    c.section("Mean reversion diagnostics")
    hls = [r.half_life for r in rolling if r.half_life is not None]
    hursts = [r.hurst_exponent for r in rolling if r.hurst_exponent is not None]
    if hls or hursts:
        c.card_row([
            c.card("Half-life (latest)", f"{hls[-1]:.2f} bars" if hls else "—"),
            c.card("Half-life (median)",
                   f"{float(np.median(hls)):.2f} bars" if hls else "—"),
            c.card("Hurst (latest)", c.num(hursts[-1]) if hursts else "—",
                   "<0.5 mean-reverting"),
            c.card("Hurst (median)",
                   c.num(float(np.median(hursts))) if hursts else "—"),
        ])
        if hls and float(np.median(hls)) < 2:
            c.banner("Half-life is shorter than the sampling interval. The spread "
                     "reverts faster than the data resolves it — consider "
                     "intraday bars before drawing conclusions about "
                     "tradeability.", "warn")
    else:
        c.empty_state("mean reversion diagnostics")


# =====================================================================
# PERFORMANCE
# =====================================================================


SOURCE_LABEL = {
    "BACKTEST": "Backtest — simulated over historical bars",
    "PAPER": "Paper — fills simulated locally against live prices",
    "LIVE": "Live — orders routed to the broker",
}


def performance(header: HeaderState) -> None:
    pair = header.pair
    sources = data.fetch_result_sources(pair.id) or ["BACKTEST"]
    source = st.radio("Result source", sources, horizontal=True, key="pf_src",
                      format_func=lambda s: s.title())
    c.banner(SOURCE_LABEL.get(source, source), "mute")

    trades = data.fetch_trades(pair.id, source=source)
    equity = data.fetch_equity_curve(pair.id, source=source)
    metrics = data.fetch_performance(pair.id, source=source)

    if not trades and not equity:
        c.empty_state("performance data",
                      "Performance is derived from executed trades. Nothing has "
                      "traded yet — run a backtest or start the engine.")
        return

    c.section("Performance")
    if metrics:
        c.card_row([
            c.card("Total return", c.pct(metrics.total_return_pct), "",
                   c.sign_class(metrics.total_return_pct)),
            c.card("Annualised", c.pct(metrics.annualised_return_pct), "",
                   c.sign_class(metrics.annualised_return_pct)),
            c.card("Sharpe", c.num(metrics.sharpe_ratio, 2)),
            c.card("Sortino", c.num(metrics.sortino_ratio, 2)),
            c.card("Max drawdown", c.pct(metrics.max_drawdown_pct), "",
                   "v-neg"),
            c.card("Win rate", f"{metrics.win_rate:.1f}%"
                   if metrics.win_rate is not None else "—"),
            c.card("Profit factor", c.num(metrics.profit_factor, 2)),
            c.card("Trades", f"{metrics.num_trades:,}"
                   if metrics.num_trades else "0"),
            c.card("Avg trade P&L", c.money(metrics.avg_trade_pnl), "",
                   c.sign_class(metrics.avg_trade_pnl)),
            c.card("Avg holding", c.duration(metrics.avg_holding_period_seconds)),
            c.card("Best trade", c.money(metrics.best_trade_pnl), "", "v-pos"),
            c.card("Worst trade", c.money(metrics.worst_trade_pnl), "", "v-neg"),
        ])
    else:
        c.banner("performance_metrics has no row yet — KPIs below are derived "
                 "from the trades table.", "mute")
        _derived_kpis(trades)

    c.section("Equity curve")
    if equity:
        st.plotly_chart(
            charts.equity_chart(equity, True, pair.leg1_ticker, pair.leg2_ticker),
            use_container_width=True, key="pf_equity")
        c.section("Drawdown")
        st.plotly_chart(charts.drawdown_chart(equity),
                        use_container_width=True, key="pf_dd")
    else:
        c.empty_state("equity curve")

    closed = [t for t in trades if t.net_pnl is not None]
    if closed:
        c.section("Return distribution")
        a, b = st.columns(2, gap="medium")
        with a:
            st.plotly_chart(
                charts.histogram([t.return_pct for t in closed if t.return_pct],
                                 "Trade return %"),
                use_container_width=True, key="pf_hist_trade")
        with b:
            st.plotly_chart(
                charts.histogram([t.net_pnl for t in closed], "Trade P&L"),
                use_container_width=True, key="pf_hist_pnl")

        c.section("Performance by signal")
        df = pd.DataFrame([{
            "Direction": t.direction.value,
            "Z-bucket": t.zscore_bucket or "—",
            "Holding": t.holding_period_seconds or 0,
            "P&L": t.net_pnl or 0.0,
            "Return %": t.return_pct or 0.0,
            "Win": bool(t.net_pnl and t.net_pnl > 0),
        } for t in closed])
        a, b = st.columns(2, gap="medium")
        with a:
            st.caption("By direction")
            st.dataframe(_group(df, "Direction"), use_container_width=True)
        with b:
            st.caption("By entry z-score bucket")
            st.dataframe(_group(df, "Z-bucket"), use_container_width=True)
        st.caption("The point is to see whether returns come from a consistent "
                   "source rather than a handful of outliers.")

    _attribution_section(pair, closed, equity, metrics)


def _attribution_section(pair, closed, equity, metrics) -> None:
    """P&L attribution, cost sensitivity and the narrative layer."""
    if not closed:
        return

    attr = analytics.attribution(closed)
    risk = analytics.risk_metrics(equity)
    cost = analytics.cost_sensitivity(closed)
    cfg = data.fetch_config(pair.id)
    positions = data.fetch_positions(pair.id)
    exposure = analytics.exposure_summary(None, positions, cfg)

    # --- Attribution --------------------------------------------------
    c.section("P&L attribution")
    conc = attr["concentration"]
    c.card_row([
        c.card("Total P&L", c.money(attr["total_pnl"]), "",
               c.sign_class(attr["total_pnl"])),
        c.card("Top 5 trades", c.money(conc["top_5_trades_pnl"]),
               f"{conc['top_5_share_of_total_pct']:.0f}% of total",
               "v-neg" if conc["top_5_share_of_total_pct"] > 50 else "v-neu"),
        c.card("Avg win / loss", f"{c.money(conc['avg_win'])} / "
               f"{c.money(conc['avg_loss'])}"),
        c.card("Win:loss ratio", c.num(conc["win_loss_ratio"], 2),
               "average win ÷ average loss"),
    ])

    tabs = st.tabs(["Direction", "Entry z-score", "Holding period",
                    "Exit reason", "Year"])
    for tab, key in zip(tabs, ["by_direction", "by_entry_zscore",
                               "by_holding_period", "by_exit_reason", "by_year"]):
        with tab:
            rows = attr.get(key, [])
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
            else:
                c.empty_state("data for this breakdown")

    # --- Cost sensitivity ---------------------------------------------
    c.section("Transaction cost sensitivity")
    st.caption("The backtest assumes zero cost. This is what each level of "
               "round-trip cost would have done to the same trades.")
    if cost:
        cards = []
        for row in cost:
            cards.append(c.card(
                f"{row['fee_bps']:g} bps", c.money(row["net_pnl"]),
                f"{row['win_rate_pct']:.0f}% profitable",
                "v-pos" if row["still_profitable"] else "v-neg"))
        c.card_row(cards, per_row=4)
        breakeven = next((r for r in cost if not r["still_profitable"]), None)
        if breakeven:
            c.banner(f"Net P&L turns negative at {breakeven['fee_bps']:g} bps "
                     f"of round-trip cost. Any realistic execution cost above "
                     f"that eliminates the edge entirely.", "bad")

    # --- Risk ----------------------------------------------------------
    if risk:
        c.section("Risk")
        c.card_row([
            c.card("Daily volatility", f"{risk['daily_volatility_pct']:.3f}%"),
            c.card("Annualised vol", f"{risk['annualised_volatility_pct']:.2f}%"),
            c.card("95% VaR (daily)", f"{risk['var_95_pct']:.3f}%", "", "v-neg"),
            c.card("95% CVaR (daily)", f"{risk['cvar_95_pct']:.3f}%", "", "v-neg"),
            c.card("Best day", f"{risk['best_day_pct']:.3f}%", "", "v-pos"),
            c.card("Worst day", f"{risk['worst_day_pct']:.3f}%", "", "v-neg"),
            c.card("Positive days", f"{risk['positive_days_pct']:.1f}%"),
            c.card("Current drawdown", f"{risk['current_drawdown_pct']:.3f}%"),
        ])

    # --- Narrative -----------------------------------------------------
    c.section("Analysis")
    facts = analytics.build_fact_pack(
        pair_label=pair.label, config=cfg, performance=metrics, attr=attr,
        risk=risk, exposure=exposure, cost=cost)

    for warning in facts.get("deterministic_warnings", []):
        c.banner(f"⚠ {warning}", "warn")

    providers = insights.available_providers()
    ctrl, btn = st.columns([2, 1])
    provider = ctrl.selectbox(
        "Model", providers + ["rule-based"] if providers else ["rule-based"],
        help="Gemini is free and suited to development. Switch to Anthropic "
             "for live analysis.")
    generate = btn.button("Generate analysis", use_container_width=True)

    question = st.text_input(
        "Optional question", placeholder="e.g. is the edge robust to costs?")

    if generate:
        with st.spinner("Analysing…"):
            text, used = insights.generate(
                facts, None if provider == "rule-based" else provider,
                question or None)
        st.session_state["insight_text"] = text
        st.session_state["insight_provider"] = used

    if st.session_state.get("insight_text"):
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #272e38;"
            f"border-radius:4px;padding:1rem;white-space:pre-wrap;"
            f"font-size:.82rem;line-height:1.55'>"
            f"{st.session_state['insight_text']}</div>",
            unsafe_allow_html=True)
        st.caption(f"Generated by {st.session_state.get('insight_provider')}. "
                   f"All figures are computed by engine/analytics.py; the model "
                   f"only writes narrative over them and cannot derive its own "
                   f"numbers.")

    with st.expander("Fact pack sent to the model"):
        st.json(facts, expanded=False)


def _group(df: pd.DataFrame, key: str) -> pd.DataFrame:
    g = df.groupby(key).agg(
        Trades=("P&L", "size"), Total_PnL=("P&L", "sum"),
        Avg_PnL=("P&L", "mean"), Win_rate=("Win", "mean"))
    g["Win_rate"] = (g["Win_rate"] * 100).round(1)
    return g.round(2)


def _derived_kpis(trades) -> None:
    closed = [t for t in trades if t.net_pnl is not None]
    if not closed:
        c.empty_state("closed trades")
        return
    pnls = [t.net_pnl for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else None
    c.card_row([
        c.card("Total P&L", c.money(sum(pnls)), "", c.sign_class(sum(pnls))),
        c.card("Trades", f"{len(closed):,}"),
        c.card("Win rate", f"{100 * len(wins) / len(closed):.1f}%"),
        c.card("Profit factor", c.num(pf, 2)),
    ])