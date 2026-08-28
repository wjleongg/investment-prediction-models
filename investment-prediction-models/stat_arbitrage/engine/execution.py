"""Paper execution.

Simulates fills locally rather than routing to IBKR. Every order, fill,
position and trade is persisted exactly as a live broker path would, so the
Trades and Overview pages exercise the real schema.

Fill model: crosses the spread when bid/ask are known, otherwise fills at
mid. Optional slippage in basis points. This is deliberately pessimistic —
assuming mid fills would flatter a strategy whose entire edge is smaller
than a typical quoted spread.

Swapping to real IBKR paper orders means replacing `_execute` with
ib.placeOrder and reacting to fill events. Nothing else changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from engine.strategy import OpenPosition, StrategyParams, pnl_for, size_position


@dataclass(frozen=True, slots=True)
class Fill:
    ticker: str
    side: Literal["BUY", "SELL"]
    quantity: float
    price: float
    ts: datetime
    commission: float = 0.0


class PaperBroker:
    """Local fill simulation with the same persistence surface as a broker."""

    def __init__(self, store, pair: dict, params: StrategyParams,
                 slippage_bps: float = 0.0,
                 commission_per_share: float = 0.0,
                 min_commission: float = 0.0) -> None:
        self.store = store
        self.pair = pair
        self.pair_id = pair["id"]
        self.leg1 = pair["leg1_ticker"]
        self.leg2 = pair["leg2_ticker"]
        self.params = params
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

        self.position: OpenPosition | None = None
        self.trade_id: int | None = None
        self.realized_pnl = 0.0
        self.daily_pnl = 0.0
        self._session_date = datetime.now(timezone.utc).date()

    # --- helpers ------------------------------------------------------

    @property
    def is_flat(self) -> bool:
        return self.position is None

    def _fill_price(self, mid: float, side: str,
                    bid: float | None, ask: float | None) -> float:
        """Cross the spread where known, then apply slippage."""
        if side == "BUY":
            base = ask if (ask and ask > 0) else mid
            return base * (1 + self.slippage_bps / 10_000)
        base = bid if (bid and bid > 0) else mid
        return base * (1 - self.slippage_bps / 10_000)

    def _commission(self, quantity: float) -> float:
        if self.commission_per_share <= 0:
            return 0.0
        return max(quantity * self.commission_per_share, self.min_commission)

    def _execute(self, ticker: str, side: str, quantity: float, mid: float,
                 bid: float | None, ask: float | None,
                 trade_id: int | None) -> Fill:
        """Simulate an order and persist order plus fill."""
        ts = datetime.now(timezone.utc)
        price = self._fill_price(mid, side, bid, ask)
        commission = self._commission(quantity)

        order_id = self.store.write_order({
            "pair_id": self.pair_id, "trade_id": trade_id, "ticker": ticker,
            "side": side, "quantity": quantity, "order_type": "MARKET",
            "status": "FILLED", "submitted_at": ts.isoformat(),
            "acknowledged_at": ts.isoformat(), "filled_at": ts.isoformat(),
            "filled_quantity": quantity, "avg_fill_price": round(price, 6),
        })
        if order_id is not None:
            self.store.write_fill({
                "order_id": order_id, "pair_id": self.pair_id,
                "ticker": ticker, "side": side, "quantity": quantity,
                "price": round(price, 6), "commission": commission,
                "ts": ts.isoformat(),
            })
        return Fill(ticker, side, quantity, price, ts, commission)

    # --- entry --------------------------------------------------------

    def enter(self, direction: str, bar, snapshot,
              quotes: dict | None = None) -> bool:
        """Open a pair position. Returns False if sizing produced nothing."""
        if self.position is not None:
            return False

        q1, q2 = size_position(bar, self.params)
        if q1 <= 0 or q2 <= 0:
            self.store.log("WARNING", "RISK",
                           f"Sizing produced zero quantity at "
                           f"{bar.leg1_price:.2f}/{bar.leg2_price:.2f}; "
                           f"no position opened", self.pair_id)
            return False

        ts = datetime.now(timezone.utc)
        trade_id = self.store.open_trade({
            "pair_id": self.pair_id, "direction": direction, "status": "OPEN",
            "entry_time": ts.isoformat(),
            "leg1_entry_price": round(bar.leg1_price, 6),
            "leg2_entry_price": round(bar.leg2_price, 6),
            "leg1_quantity": q1, "leg2_quantity": q2,
            "entry_zscore": snapshot.zscore,
            "entry_hedge_ratio": snapshot.hedge_ratio,
            "entry_model_state": snapshot.as_dict(),
        })
        self.trade_id = trade_id

        # LONG_SPREAD = long leg1, short leg2
        side1 = "BUY" if direction == "LONG_SPREAD" else "SELL"
        side2 = "SELL" if direction == "LONG_SPREAD" else "BUY"
        quotes = quotes or {}
        f1 = self._execute(self.leg1, side1, q1, bar.leg1_price,
                           quotes.get("leg1_bid"), quotes.get("leg1_ask"),
                           trade_id)
        f2 = self._execute(self.leg2, side2, q2, bar.leg2_price,
                           quotes.get("leg2_bid"), quotes.get("leg2_ask"),
                           trade_id)

        self.position = OpenPosition(
            direction=direction, entry_ts=ts, entry_bar=bar,
            leg1_quantity=q1, leg2_quantity=q2,
            hedge_ratio=snapshot.hedge_ratio or 1.0)
        self._entry_fills = (f1, f2)
        self._entry_commission = f1.commission + f2.commission

        self._write_positions(bar)
        self.store.log("INFO", "ORDER",
                       f"{direction} opened: {side1} {q1:g} {self.leg1} @ "
                       f"{f1.price:.4f}, {side2} {q2:g} {self.leg2} @ "
                       f"{f2.price:.4f} (z={snapshot.zscore:+.3f})",
                       self.pair_id)
        return True

    # --- exit ---------------------------------------------------------

    def exit(self, bar, snapshot, reason: str,
             quotes: dict | None = None) -> float | None:
        """Close the open pair position. Returns net P&L."""
        if self.position is None:
            return None

        pos = self.position
        side1 = "SELL" if pos.direction == "LONG_SPREAD" else "BUY"
        side2 = "BUY" if pos.direction == "LONG_SPREAD" else "SELL"
        quotes = quotes or {}
        f1 = self._execute(self.leg1, side1, pos.leg1_quantity, bar.leg1_price,
                           quotes.get("leg1_bid"), quotes.get("leg1_ask"),
                           self.trade_id)
        f2 = self._execute(self.leg2, side2, pos.leg2_quantity, bar.leg2_price,
                           quotes.get("leg2_bid"), quotes.get("leg2_ask"),
                           self.trade_id)

        # P&L from actual fill prices, not mid — this is where a tight-spread
        # strategy loses its edge, so it must be measured honestly.
        entry1, entry2 = self._entry_fills
        gross = pnl_for(pos.direction, pos.leg1_quantity, pos.leg2_quantity,
                        entry1.price, entry2.price, f1.price, f2.price)
        fees = (self._entry_commission + f1.commission + f2.commission)
        net = gross - fees

        held = (datetime.now(timezone.utc) - pos.entry_ts).total_seconds()
        notional = (pos.leg1_quantity * entry1.price
                    + pos.leg2_quantity * entry2.price)

        if self.trade_id is not None:
            self.store.close_trade(self.trade_id, {
                "status": "CLOSED",
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "leg1_exit_price": round(bar.leg1_price, 6),
                "leg2_exit_price": round(bar.leg2_price, 6),
                "exit_zscore": snapshot.zscore,
                "exit_hedge_ratio": snapshot.hedge_ratio,
                "exit_model_state": snapshot.as_dict(),
                "gross_pnl": round(gross, 4), "fees": round(fees, 4),
                "net_pnl": round(net, 4),
                "return_pct": (net / notional * 100) if notional else 0.0,
                "holding_period_seconds": int(held),
                "exit_reason": reason,
            })

        self.realized_pnl += net
        self.daily_pnl += net
        self.position = None
        self.trade_id = None
        self.store.clear_positions(self.pair_id)

        self.store.log("INFO", "ORDER",
                       f"Closed {pos.direction} ({reason}): net "
                       f"{net:+.2f} after {fees:.2f} fees, held "
                       f"{held / 60:.1f}m (z={snapshot.zscore:+.3f})",
                       self.pair_id)
        return net

    # --- marking ------------------------------------------------------

    def unrealized(self, bar) -> float:
        if self.position is None:
            return 0.0
        entry1, entry2 = self._entry_fills
        return pnl_for(self.position.direction,
                       self.position.leg1_quantity,
                       self.position.leg2_quantity,
                       entry1.price, entry2.price,
                       bar.leg1_price, bar.leg2_price)

    def _write_positions(self, bar) -> None:
        if self.position is None:
            return
        pos = self.position
        entry1, entry2 = self._entry_fills
        long_first = pos.direction == "LONG_SPREAD"
        ts = datetime.now(timezone.utc).isoformat()

        for ticker, qty, entry, price, is_long in (
                (self.leg1, pos.leg1_quantity, entry1.price,
                 bar.leg1_price, long_first),
                (self.leg2, pos.leg2_quantity, entry2.price,
                 bar.leg2_price, not long_first)):
            signed = qty if is_long else -qty
            self.store.upsert_position({
                "pair_id": self.pair_id, "trade_id": self.trade_id,
                "ticker": ticker, "side": "LONG" if is_long else "SHORT",
                "quantity": signed, "avg_entry_price": round(entry, 6),
                "current_price": round(price, 6),
                "market_value": round(signed * price, 4),
                "unrealized_pnl": round(signed * (price - entry), 4),
                "opened_at": pos.entry_ts.isoformat(), "updated_at": ts,
            })

    def mark_to_market(self, bar) -> None:
        self._write_positions(bar)

    def roll_session(self) -> None:
        """Reset daily P&L at the start of a new UTC day."""
        today = datetime.now(timezone.utc).date()
        if today != self._session_date:
            self._session_date = today
            self.daily_pnl = 0.0

    def exposure(self, bar) -> tuple[float, float]:
        """(gross, net) exposure at current prices."""
        if self.position is None:
            return 0.0, 0.0
        pos = self.position
        v1 = pos.leg1_quantity * bar.leg1_price
        v2 = pos.leg2_quantity * bar.leg2_price
        if pos.direction == "LONG_SPREAD":
            return v1 + v2, v1 - v2
        return v1 + v2, v2 - v1
