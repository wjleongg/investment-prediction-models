"""Deterministic analytics — exposure, risk and P&L attribution.

Every number an LLM ever narrates is computed here first. The model receives
finished figures and is instructed not to derive its own, because a language
model asked to compute a Sharpe ratio will produce a plausible one whether or
not it is correct.

Pure functions over typed models. No I/O, no Streamlit, no network.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd


# =====================================================================
# Exposure and risk limits
# =====================================================================


def exposure_summary(live, positions, cfg) -> dict[str, Any]:
    """Current exposure against configured limits.

    Populated even when flat: showing headroom against limits is useful
    before the first trade, and makes the panel meaningful on day one.
    """
    if cfg is None:
        return {}

    legs = list(positions or [])
    gross = sum(abs(p.market_value or 0.0) for p in legs)
    net = sum(p.market_value or 0.0 for p in legs)
    unrealized = sum(p.unrealized_pnl or 0.0 for p in legs)

    capital = float(cfg.capital_allocation)
    leverage = (gross / capital) if capital else 0.0
    limit_gross = min(float(cfg.max_pair_exposure), capital * cfg.max_leverage)

    largest = max((abs(p.market_value or 0.0) for p in legs), default=0.0)

    return {
        "gross_exposure": gross,
        "net_exposure": net,
        "net_exposure_pct_of_gross": (100 * net / gross) if gross else 0.0,
        "unrealized_pnl": unrealized,
        "capital_allocation": capital,
        "leverage": leverage,
        "leverage_limit": float(cfg.max_leverage),
        "leverage_headroom_pct": (
            100 * (1 - leverage / cfg.max_leverage)) if cfg.max_leverage else 0.0,
        "gross_limit": limit_gross,
        "gross_utilisation_pct": (100 * gross / limit_gross) if limit_gross else 0.0,
        "position_limit": float(cfg.max_position_size),
        "largest_leg_notional": largest,
        "position_utilisation_pct": (
            100 * largest / cfg.max_position_size) if cfg.max_position_size else 0.0,
        "open_legs": len(legs),
        "is_flat": all((p.quantity or 0) == 0 for p in legs) if legs else True,
        "market_neutrality": _neutrality_label(net, gross),
    }


def _neutrality_label(net: float, gross: float) -> str:
    if gross == 0:
        return "FLAT"
    ratio = abs(net) / gross
    if ratio < 0.05:
        return "MARKET NEUTRAL"
    if ratio < 0.20:
        return "SLIGHT DIRECTIONAL TILT"
    return "SIGNIFICANT DIRECTIONAL EXPOSURE"


def limit_breaches(exposure: dict[str, Any]) -> list[str]:
    """Hard checks worth surfacing regardless of what any model says."""
    out = []
    if exposure.get("gross_utilisation_pct", 0) > 100:
        out.append(f"Gross exposure is "
                   f"{exposure['gross_utilisation_pct']:.0f}% of its limit.")
    if exposure.get("position_utilisation_pct", 0) > 100:
        out.append(f"A single leg is "
                   f"{exposure['position_utilisation_pct']:.0f}% of the "
                   f"per-position limit.")
    if exposure.get("leverage", 0) > exposure.get("leverage_limit", 99):
        out.append(f"Leverage {exposure['leverage']:.2f}x exceeds the "
                   f"{exposure['leverage_limit']:.2f}x limit.")
    if exposure.get("market_neutrality") == "SIGNIFICANT DIRECTIONAL EXPOSURE":
        out.append("Net exposure is large relative to gross — the position is "
                   "no longer market neutral.")
    return out


# =====================================================================
# Risk metrics from the equity curve
# =====================================================================


def risk_metrics(equity_points: Sequence) -> dict[str, Any]:
    if len(equity_points) < 3:
        return {}
    eq = pd.Series([p.equity for p in equity_points],
                   index=pd.to_datetime([p.ts for p in equity_points]))
    daily = eq.resample("D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 3:
        return {}

    downside = rets[rets < 0]
    dd = pd.Series([p.drawdown_pct for p in equity_points])

    return {
        "daily_volatility_pct": float(rets.std() * 100),
        "annualised_volatility_pct": float(rets.std() * np.sqrt(252) * 100),
        "downside_deviation_pct": float(downside.std() * 100) if len(downside) > 1 else None,
        "var_95_pct": float(np.percentile(rets, 5) * 100),
        "cvar_95_pct": float(rets[rets <= np.percentile(rets, 5)].mean() * 100),
        "best_day_pct": float(rets.max() * 100),
        "worst_day_pct": float(rets.min() * 100),
        "positive_days_pct": float(100 * (rets > 0).sum() / len(rets)),
        "current_drawdown_pct": float(dd.iloc[-1]),
        "max_drawdown_pct": float(dd.min()),
        "observations": int(len(rets)),
    }


# =====================================================================
# P&L attribution
# =====================================================================


def _holding_bucket(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    days = seconds / 86400
    if days < 1:
        return "<1d"
    if days < 3:
        return "1-3d"
    if days < 7:
        return "3-7d"
    if days < 21:
        return "1-3w"
    return ">3w"


def attribution(trades: Sequence) -> dict[str, Any]:
    """Break P&L down by every dimension worth questioning."""
    closed = [t for t in trades if t.net_pnl is not None]
    if not closed:
        return {}

    total = sum(t.net_pnl for t in closed)
    rows = []
    for t in closed:
        rows.append({
            "direction": t.direction.value,
            "z_bucket": t.zscore_bucket or "unknown",
            "holding_bucket": _holding_bucket(t.holding_period_seconds),
            "exit_reason": t.exit_reason.value if t.exit_reason else "unknown",
            "year": t.entry_time.year,
            "pnl": t.net_pnl,
            "return_pct": t.return_pct or 0.0,
            "win": t.net_pnl > 0,
            "holding_days": (t.holding_period_seconds or 0) / 86400,
        })
    df = pd.DataFrame(rows)

    def group(key: str) -> list[dict]:
        g = df.groupby(key).agg(
            trades=("pnl", "size"), total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"), win_rate=("win", "mean"),
            avg_return_pct=("return_pct", "mean")).reset_index()
        g["win_rate"] = g["win_rate"] * 100
        g["contribution_pct"] = (g["total_pnl"] / total * 100) if total else 0.0
        return g.round(3).to_dict("records")

    pnls = sorted((t.net_pnl for t in closed), reverse=True)
    top5 = sum(pnls[:5])
    top10_pct = (100 * sum(pnls[:10]) / total) if total else 0.0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    return {
        "total_pnl": total,
        "trade_count": len(closed),
        "by_direction": group("direction"),
        "by_entry_zscore": group("z_bucket"),
        "by_holding_period": group("holding_bucket"),
        "by_exit_reason": group("exit_reason"),
        "by_year": group("year"),
        "concentration": {
            "top_5_trades_pnl": top5,
            "top_5_share_of_total_pct": (100 * top5 / total) if total else 0.0,
            "top_10_share_of_total_pct": top10_pct,
            "largest_win": max(pnls),
            "largest_loss": min(pnls),
            "avg_win": float(np.mean(wins)) if wins else 0.0,
            "avg_loss": float(np.mean(losses)) if losses else 0.0,
            "win_loss_ratio": (abs(np.mean(wins) / np.mean(losses))
                               if wins and losses else None),
        },
        "avg_holding_days": float(df["holding_days"].mean()),
    }


def concentration_flags(attr: dict[str, Any]) -> list[str]:
    """Deterministic warnings. These are stated as fact, not model opinion."""
    out = []
    conc = attr.get("concentration", {})
    share = conc.get("top_5_share_of_total_pct")
    n = attr.get("trade_count", 0)
    if share is not None and n >= 20 and share > 50:
        if share > 100:
            out.append(f"The top 5 trades contributed more than the entire net "
                       f"result ({share:.0f}% of total P&L) across {n} trades — "
                       f"every other trade nets negative. The result rests "
                       f"entirely on a handful of outliers.")
        else:
            out.append(f"The top 5 trades account for {share:.0f}% of total P&L "
                       f"across {n} trades — returns are concentrated in "
                       f"outliers rather than broadly distributed.")
    for row in attr.get("by_exit_reason", []):
        if row["exit_reason"] == "MAX_HOLDING_PERIOD" and row["trades"] / n > 0.5:
            out.append(f"{row['trades']} of {n} trades exited on the maximum "
                       f"holding period rather than mean reversion — the exit "
                       f"rule is being pre-empted by the time limit.")
        if row["exit_reason"] == "STOP_LOSS" and row["trades"] / n > 0.3:
            out.append(f"{row['trades']} of {n} trades hit the stop-loss — "
                       f"the spread is diverging more often than it reverts.")
    for row in attr.get("by_direction", []):
        if row["trades"] >= 10 and row["total_pnl"] < 0:
            out.append(f"{row['direction']} trades are net negative "
                       f"({row['total_pnl']:,.0f}) across {row['trades']} "
                       f"trades — the edge is one-sided.")
    return out


def cost_sensitivity(trades: Sequence, bps_levels=(0.5, 1.0, 2.0, 5.0)) -> list[dict]:
    """What each level of round-trip cost would have done to net P&L.

    The single most decision-relevant number for a tight-spread pair.
    """
    closed = [t for t in trades if t.net_pnl is not None]
    if not closed:
        return []
    out = []
    for bps in bps_levels:
        total = 0.0
        profitable = 0
        for t in closed:
            notional = ((t.leg1_entry_price or 0) * (t.leg1_quantity or 0)
                        + (t.leg2_entry_price or 0) * (t.leg2_quantity or 0))
            cost = notional * (bps / 10_000) * 2
            net = t.net_pnl - cost
            total += net
            profitable += 1 if net > 0 else 0
        out.append({
            "fee_bps": bps, "net_pnl": round(total, 2),
            "win_rate_pct": round(100 * profitable / len(closed), 1),
            "still_profitable": total > 0,
        })
    return out


# =====================================================================
# Fact pack for the narrative layer
# =====================================================================


def build_fact_pack(*, pair_label: str, config, performance, attr: dict,
                    risk: dict, exposure: dict, cost: list[dict],
                    model_state: dict | None = None) -> dict[str, Any]:
    """Assemble every computed figure the narrative layer is allowed to use."""
    def rounded(d: dict) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items() if v is not None}

    pack: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair": pair_label,
        "data_basis": "Backtest over historical daily bars. Not live trading.",
    }
    if config is not None:
        pack["configuration"] = {
            "entry_threshold": config.entry_threshold,
            "exit_threshold": config.exit_threshold,
            "stop_loss_threshold": config.stop_loss_threshold,
            "max_holding_period_days": (
                config.max_holding_period_seconds / 86400
                if config.max_holding_period_seconds else None),
            "capital_allocation": float(config.capital_allocation),
            "min_correlation": config.min_correlation,
            "max_cointegration_pvalue": config.max_cointegration_pvalue,
        }
    if performance is not None:
        pack["performance"] = rounded({
            "total_return_pct": performance.total_return_pct,
            "annualised_return_pct": performance.annualised_return_pct,
            "sharpe_ratio": performance.sharpe_ratio,
            "sortino_ratio": performance.sortino_ratio,
            "max_drawdown_pct": performance.max_drawdown_pct,
            "win_rate_pct": performance.win_rate,
            "profit_factor": performance.profit_factor,
            "num_trades": performance.num_trades,
            "avg_trade_pnl": performance.avg_trade_pnl,
            "total_pnl": performance.total_pnl,
        })
    if attr:
        pack["attribution"] = attr
    if risk:
        pack["risk"] = rounded(risk)
    if exposure:
        pack["current_exposure"] = rounded(exposure)
    if cost:
        pack["transaction_cost_sensitivity"] = cost
    if model_state:
        pack["model_state"] = model_state

    pack["deterministic_warnings"] = (
        concentration_flags(attr) if attr else []) + limit_breaches(exposure or {})
    return pack
