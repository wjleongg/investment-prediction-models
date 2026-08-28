"""Step 7 — backtest the strategy over backfilled history.

Replays engine/strategy.py bar by bar across model_state_history, then writes
trades, equity_curve and performance_metrics to Supabase so the Trades and
Performance pages have real data.

The signal logic lives in engine/strategy.py and is imported, not duplicated.
When the live engine is built it imports the same module, so backtest and live
behaviour cannot drift apart.

Does NOT write to signals, orders, fills or positions — those record what the
live engine actually did, and filling them from a simulation would make the
Strategy and System Health pages lie.

Usage:
    python scripts/backtest.py --dry-run
    python scripts/backtest.py
    python scripts/backtest.py --fee-bps 1.0
    python scripts/backtest.py --entry 2.5 --exit 0.25   # override config
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.strategy import (  # noqa: E402
    Action,
    Bar,
    ExitCause,
    OpenPosition,
    StrategyParams,
    decide,
    fees_for,
    pnl_for,
    size_position,
)

load_dotenv()

CHUNK = 500
PAGE = 1000          # PostgREST caps a single response at 1000 rows
TRADING_DAYS = 252


# =====================================================================
# Data access
# =====================================================================


def connect():
    return create_client(os.environ["SUPABASE_URL"],
                         os.environ["SUPABASE_SERVICE_KEY"])


def fetch_all(db, table: str, pair_id: int, order_col: str) -> list[dict]:
    """Paginated read — a single select would silently stop at 1000 rows."""
    rows, offset = [], 0
    while True:
        page = (db.table(table).select("*").eq("pair_id", pair_id)
                .order(order_col).range(offset, offset + PAGE - 1)
                .execute().data or [])
        rows.extend(page)
        if len(page) < PAGE:
            return rows
        offset += PAGE


def load_bars(db, pair_id: int) -> list[Bar]:
    rows = fetch_all(db, "model_state_history", pair_id, "ts")
    bars = []
    for r in rows:
        if r["spread"] is None or r["zscore"] is None:
            continue
        bars.append(Bar(
            ts=datetime.fromisoformat(r["ts"].replace("Z", "+00:00")),
            leg1_price=float(r["leg1_price"]), leg2_price=float(r["leg2_price"]),
            spread=float(r["spread"]), zscore=float(r["zscore"]),
            hedge_ratio=_f(r.get("hedge_ratio")),
            spread_mean=_f(r.get("spread_mean")), spread_std=_f(r.get("spread_std")),
            correlation=_f(r.get("correlation")),
            cointegration_pvalue=_f(r.get("cointegration_pvalue")),
            half_life=_f(r.get("half_life")),
        ))
    return bars


def _f(v):
    return None if v is None else float(v)


# =====================================================================
# Backtest loop
# =====================================================================


def run_backtest(bars: list[Bar], p: StrategyParams,
                 fee_bps: float = 0.0) -> tuple[list[dict], pd.DataFrame]:
    """Replay the strategy. Returns (trades, equity curve)."""
    trades: list[dict] = []
    equity_rows: list[dict] = []
    position: OpenPosition | None = None
    realized = 0.0

    for i, bar in enumerate(bars):
        decision = decide(bar, position, p)

        if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            direction = ("LONG_SPREAD" if decision.action == Action.ENTER_LONG
                         else "SHORT_SPREAD")
            q1, q2 = size_position(bar, p)
            if q1 > 0 and q2 > 0:
                position = OpenPosition(
                    direction=direction, entry_ts=bar.ts, entry_bar=bar,
                    leg1_quantity=q1, leg2_quantity=q2,
                    hedge_ratio=bar.hedge_ratio or 1.0)

        elif decision.action == Action.EXIT and position is not None:
            trades.append(_close(position, bar, decision.cause, fee_bps))
            realized += trades[-1]["net_pnl"]
            position = None

        # Mark to market for the equity curve
        unrealized = position.unrealized(bar) if position else 0.0
        equity = p.capital_allocation + realized + unrealized
        equity_rows.append({
            "ts": bar.ts, "equity": equity,
            "cumulative_pnl": realized + unrealized,
            "leg1_price": bar.leg1_price, "leg2_price": bar.leg2_price,
        })

    # Force-close anything still open at the end of the data
    if position is not None and bars:
        trades.append(_close(position, bars[-1], ExitCause.END_OF_DATA, fee_bps))

    eq = pd.DataFrame(equity_rows)
    if not eq.empty:
        eq["cumulative_return_pct"] = (
            (eq["equity"] / p.capital_allocation - 1.0) * 100)
        running_max = eq["equity"].cummax()
        eq["drawdown_pct"] = (eq["equity"] / running_max - 1.0) * 100
        eq["leg1_cumulative_return_pct"] = (
            eq["leg1_price"] / eq["leg1_price"].iloc[0] - 1.0) * 100
        eq["leg2_cumulative_return_pct"] = (
            eq["leg2_price"] / eq["leg2_price"].iloc[0] - 1.0) * 100
    return trades, eq


def _close(pos: OpenPosition, bar: Bar, cause: ExitCause | None,
           fee_bps: float) -> dict:
    gross = pnl_for(pos.direction, pos.leg1_quantity, pos.leg2_quantity,
                    pos.entry_bar.leg1_price, pos.entry_bar.leg2_price,
                    bar.leg1_price, bar.leg2_price)
    fees = fees_for(pos.leg1_quantity, pos.leg2_quantity,
                    pos.entry_bar.leg1_price, pos.entry_bar.leg2_price, fee_bps)
    net = gross - fees
    notional = pos.gross_notional
    held = (bar.ts - pos.entry_ts).total_seconds()
    return {
        "direction": pos.direction, "status": "CLOSED",
        "entry_time": pos.entry_ts, "exit_time": bar.ts,
        "leg1_entry_price": pos.entry_bar.leg1_price,
        "leg2_entry_price": pos.entry_bar.leg2_price,
        "leg1_exit_price": bar.leg1_price, "leg2_exit_price": bar.leg2_price,
        "leg1_quantity": pos.leg1_quantity, "leg2_quantity": pos.leg2_quantity,
        "entry_zscore": pos.entry_bar.zscore, "exit_zscore": bar.zscore,
        "entry_hedge_ratio": pos.entry_bar.hedge_ratio,
        "exit_hedge_ratio": bar.hedge_ratio,
        "entry_model_state": pos.entry_bar.snapshot(),
        "exit_model_state": bar.snapshot(),
        "gross_pnl": gross, "fees": fees, "net_pnl": net,
        "return_pct": (net / notional * 100) if notional else 0.0,
        "holding_period_seconds": int(held),
        "exit_reason": (cause or ExitCause.MEAN_REVERSION).value,
    }


# =====================================================================
# Performance metrics
# =====================================================================


def compute_metrics(trades: list[dict], eq: pd.DataFrame,
                    capital: float) -> dict:
    """Descriptive performance analytics over the backtest result."""
    out: dict = {"num_trades": len(trades)}
    if eq.empty:
        return out

    daily = eq.set_index("ts")["equity"].resample("D").last().dropna()
    rets = daily.pct_change().dropna()

    total_return = (daily.iloc[-1] / capital - 1.0) * 100
    days = max((daily.index[-1] - daily.index[0]).days, 1)
    years = days / 365.25
    ann = ((daily.iloc[-1] / capital) ** (1 / years) - 1.0) * 100 if years > 0 else None

    sharpe = sortino = None
    if len(rets) > 2 and rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS))
        downside = rets[rets < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(rets.mean() / downside.std() * np.sqrt(TRADING_DAYS))

    dd = eq["drawdown_pct"]
    max_dd = float(dd.min())
    # Longest stretch below the prior peak
    under = dd < -1e-9
    longest, run = 0, 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    pnls = [t["net_pnl"] for t in trades]
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    trade_rets = [t["return_pct"] for t in trades]

    out.update({
        "total_return_pct": float(total_return),
        "annualised_return_pct": None if ann is None else float(ann),
        "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd,
        "max_drawdown_duration_days": float(longest),
        "current_drawdown_pct": float(dd.iloc[-1]),
        "total_pnl": float(sum(pnls)) if pnls else 0.0,
    })
    if pnls:
        out.update({
            "win_rate": 100.0 * len(wins) / len(pnls),
            "profit_factor": (sum(wins) / abs(sum(losses))) if losses else None,
            "avg_trade_pnl": float(np.mean(pnls)),
            "avg_holding_period_seconds": int(np.mean(
                [t["holding_period_seconds"] for t in trades])),
            "best_trade_pnl": float(max(pnls)),
            "worst_trade_pnl": float(min(pnls)),
            "return_mean": float(np.mean(trade_rets)),
            "return_median": float(np.median(trade_rets)),
            "return_std": float(np.std(trade_rets)),
            "return_skew": float(pd.Series(trade_rets).skew()),
            "return_kurtosis": float(pd.Series(trade_rets).kurtosis()),
        })
    return out


def monthly_breakdown(trades: list[dict], eq: pd.DataFrame,
                      capital: float) -> list[dict]:
    if eq.empty:
        return []
    daily = eq.set_index("ts")["equity"].resample("D").last().dropna()
    rows = []
    for period, group in daily.groupby(daily.index.to_period("M")):
        if len(group) < 2:
            continue
        month_trades = [t for t in trades
                        if t["exit_time"].strftime("%Y-%m") == str(period)]
        rows.append({
            "period_label": str(period),
            "period_start": group.index[0].isoformat(),
            "period_end": group.index[-1].isoformat(),
            "total_return_pct": float(
                (group.iloc[-1] / group.iloc[0] - 1.0) * 100),
            "total_pnl": float(sum(t["net_pnl"] for t in month_trades)),
            "num_trades": len(month_trades),
        })
    return rows


# =====================================================================
# Persistence
# =====================================================================


def persist(db, pair_id: int, trades: list[dict], eq: pd.DataFrame,
            metrics: dict, monthly: list[dict]) -> None:
    # Only clear this script's own rows. Live execution results live in the
    # same tables under a different source and must never be deleted here.
    for table in ("trades", "equity_curve", "performance_metrics"):
        (db.table(table).delete().eq("pair_id", pair_id)
         .eq("source", "BACKTEST").execute())

    now = datetime.now(timezone.utc).isoformat()

    trade_rows = [{
        **{k: v for k, v in t.items()
           if k not in ("entry_time", "exit_time")},
        "pair_id": pair_id, "source": "BACKTEST",
        "entry_time": t["entry_time"].isoformat(),
        "exit_time": t["exit_time"].isoformat(),
    } for t in trades]
    _insert(db, "trades", trade_rows)
    print(f"  trades               {len(trade_rows)}")

    eq_rows = [{
        "pair_id": pair_id, "source": "BACKTEST",
        "ts": r["ts"].isoformat(),
        "equity": round(float(r["equity"]), 4),
        "cumulative_pnl": round(float(r["cumulative_pnl"]), 4),
        "cumulative_return_pct": float(r["cumulative_return_pct"]),
        "drawdown_pct": float(r["drawdown_pct"]),
        "leg1_cumulative_return_pct": float(r["leg1_cumulative_return_pct"]),
        "leg2_cumulative_return_pct": float(r["leg2_cumulative_return_pct"]),
    } for _, r in eq.iterrows()]
    _insert(db, "equity_curve", eq_rows)
    print(f"  equity_curve         {len(eq_rows)}")

    perf_rows = [{"pair_id": pair_id, "as_of": now, "source": "BACKTEST",
                  "period_label": "ALL",
                  **{k: v for k, v in metrics.items() if v is not None}}]
    perf_rows += [{"pair_id": pair_id, "as_of": now, "source": "BACKTEST", **m}
                  for m in monthly]
    _insert(db, "performance_metrics", perf_rows)
    print(f"  performance_metrics  {len(perf_rows)}")


def _insert(db, table: str, rows: list[dict]) -> None:
    for i in range(0, len(rows), CHUNK):
        db.table(table).insert(rows[i:i + CHUNK]).execute()


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-id", type=int, default=1)
    ap.add_argument("--fee-bps", type=float, default=0.0,
                    help="Round-trip cost in bps of notional (default 0).")
    ap.add_argument("--entry", type=float, help="Override entry threshold.")
    ap.add_argument("--exit", type=float, help="Override exit threshold.")
    ap.add_argument("--stop", type=float, help="Override stop-loss threshold.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = connect()
    pair = db.table("pairs").select("*").eq("id", args.pair_id).execute().data
    if not pair:
        sys.exit(f"No pair with id={args.pair_id}.")
    pair = pair[0]
    cfg = (db.table("strategy_config").select("*").eq("status", "CURRENT")
           .eq("pair_id", args.pair_id).execute().data)
    if not cfg:
        sys.exit("No CURRENT strategy_config.")
    cfg = cfg[0]

    for key, val in (("entry_threshold", args.entry),
                     ("exit_threshold", args.exit),
                     ("stop_loss_threshold", args.stop)):
        if val is not None:
            cfg[key] = val

    params = StrategyParams.from_config_row(cfg)
    print(f"Pair {pair['id']}: {pair['leg1_ticker']} / {pair['leg2_ticker']}")
    print(f"Thresholds: entry ±{params.entry_threshold:g}  "
          f"exit ±{params.exit_threshold:g}  "
          f"stop ±{params.stop_loss_threshold:g}  fees {args.fee_bps:g}bps")

    bars = load_bars(db, args.pair_id)
    if not bars:
        sys.exit("No model_state_history. Run scripts/backfill.py first.")
    tradeable = [b for b in bars if b.is_tradeable()]
    print(f"{len(bars):,} bars loaded ({len(tradeable):,} tradeable), "
          f"{bars[0].ts.date()} → {bars[-1].ts.date()}")

    trades, eq = run_backtest(bars, params, args.fee_bps)
    metrics = compute_metrics(trades, eq, params.capital_allocation)
    monthly = monthly_breakdown(trades, eq, params.capital_allocation)

    # --- Report -------------------------------------------------------
    print(f"\n{'':-<52}")
    print(f"Trades              {metrics.get('num_trades', 0)}")
    if trades:
        longs = [t for t in trades if t["direction"] == "LONG_SPREAD"]
        print(f"  long / short      {len(longs)} / {len(trades) - len(longs)}")
        causes = pd.Series([t["exit_reason"] for t in trades]).value_counts()
        for cause, n in causes.items():
            print(f"  exit {cause:<16} {n}")
    print(f"Total P&L           ${metrics.get('total_pnl', 0):,.2f}")
    print(f"Total return        {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Annualised          {_fmt(metrics.get('annualised_return_pct'))}%")
    print(f"Sharpe              {_fmt(metrics.get('sharpe_ratio'), 2)}")
    print(f"Sortino             {_fmt(metrics.get('sortino_ratio'), 2)}")
    print(f"Max drawdown        {_fmt(metrics.get('max_drawdown_pct'))}%")
    print(f"Win rate            {_fmt(metrics.get('win_rate'), 1)}%")
    print(f"Profit factor       {_fmt(metrics.get('profit_factor'), 2)}")
    print(f"Avg holding         "
          f"{(metrics.get('avg_holding_period_seconds') or 0) / 86400:.1f} days")
    print(f"{'':-<52}")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    print("\nWriting to Supabase...")
    persist(db, args.pair_id, trades, eq, metrics, monthly)
    print("\nBacktest complete. Trades and Performance pages now have data.")


def _fmt(v, dp: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{dp}f}"


if __name__ == "__main__":
    main()