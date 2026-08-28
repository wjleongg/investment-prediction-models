"""Live order routing through IBKR.

Places real orders on the connected account — paper or live, determined
entirely by which TWS session is running. This class does not know or care
which; a paper login and a live login present an identical API.

Reuses PaperBroker's trade lifecycle, position tracking and P&L accounting
and replaces only `_execute`, so the persistence surface is identical and
the simulated and live paths cannot drift apart.

Two risks exist here that simulation does not have:

1. Leg risk. A pair trade is two orders. If the first fills and the second
   is rejected, the position is naked directional exposure, not a spread.
   Every entry is therefore transactional: a failed second leg immediately
   unwinds the first.

2. Reconciliation. If the engine restarts while holding a position, IBKR
   still holds it. The engine must adopt what the broker reports rather
   than assuming it is flat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from engine.execution import Fill, PaperBroker


class ExecutionError(RuntimeError):
    """An order did not fill. Carries whatever the broker reported."""

    def __init__(self, message: str, status: str = "", order_id: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.order_id = order_id


TERMINAL_BAD = {"Cancelled", "ApiCancelled", "Inactive"}


class IBKRBroker(PaperBroker):
    """Routes orders to IBKR and waits for real fills."""

    def __init__(self, store, pair: dict, params, source,
                 fill_timeout: float = 30.0, use_limit_orders: bool = False,
                 limit_offset_bps: float = 2.0, **kwargs) -> None:
        super().__init__(store, pair, params, **kwargs)
        self.source = source            # IBKRSource, supplies ib + contracts
        self.fill_timeout = fill_timeout
        self.use_limit_orders = use_limit_orders
        self.limit_offset_bps = limit_offset_bps

    @property
    def _ib(self):
        return self.source._ib

    # --- order placement ----------------------------------------------

    def _place(self, ticker: str, side: str, quantity: float,
               reference_price: float):
        """Submit one order and wait for a terminal state."""
        from ib_async import LimitOrder, MarketOrder

        contract = self.source._contract(ticker)
        qty = int(round(quantity))
        if qty <= 0:
            raise ExecutionError(f"{ticker}: quantity rounded to zero")

        if self.use_limit_orders:
            # Marketable limit: priced through the touch so it fills like a
            # market order but caps the damage if the book is thin.
            offset = reference_price * (self.limit_offset_bps / 10_000)
            limit = (reference_price + offset if side == "BUY"
                     else reference_price - offset)
            order = LimitOrder(side, qty, round(limit, 2))
        else:
            order = MarketOrder(side, qty)

        trade = self._ib.placeOrder(contract, order)

        deadline = time.monotonic() + self.fill_timeout
        while not trade.isDone():
            self._ib.waitOnUpdate(timeout=1.0)
            if time.monotonic() > deadline:
                self._ib.cancelOrder(order)
                self._ib.sleep(1.0)
                raise ExecutionError(
                    f"{ticker} {side} {qty} did not fill within "
                    f"{self.fill_timeout:g}s; order cancelled",
                    trade.orderStatus.status,
                    str(trade.order.orderId))

        status = trade.orderStatus.status
        filled = float(trade.orderStatus.filled or 0)

        if status in TERMINAL_BAD or filled <= 0:
            raise ExecutionError(
                f"{ticker} {side} {qty} was not filled (status={status})",
                status, str(trade.order.orderId))

        if filled < qty:
            # Partial fills break the hedge ratio; record it loudly.
            self.store.log("WARNING", "ORDER",
                           f"{ticker} {side} partially filled {filled:g}/{qty} "
                           f"— hedge ratio is now approximate", self.pair_id)

        avg = float(trade.orderStatus.avgFillPrice or reference_price)

        # IBKR delivers commissionReport asynchronously, shortly AFTER the
        # fill. Reading it immediately usually returns zero, which would
        # silently understate costs. Wait briefly for it to arrive.
        commission = self._collect_commission(trade)

        return trade, filled, avg, commission

    def _collect_commission(self, trade, wait: float = 3.0) -> float:
        """Sum IBKR's own commission reports for this order.

        Never estimated or hard-coded: whatever the broker charges is what
        gets recorded. On a paper account this is typically zero, and zero is
        then the honest number rather than an assumption.
        """
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            reports = [f.commissionReport for f in trade.fills
                       if f.commissionReport is not None]
            if len(reports) == len(trade.fills) and trade.fills:
                break
            self._ib.waitOnUpdate(timeout=0.5)

        total = 0.0
        missing = 0
        for f in trade.fills:
            report = f.commissionReport
            if report is None or report.commission is None:
                missing += 1
                continue
            value = float(report.commission)
            # IBKR uses a sentinel for "not yet computed".
            if value < 1e17:
                total += value
        if missing:
            self.store.log("WARNING", "ORDER",
                           f"{missing} fill(s) had no commission report yet; "
                           f"recorded cost may understate the true fee",
                           self.pair_id)
        return total

    def _execute(self, ticker: str, side: str, quantity: float, mid: float,
                 bid: float | None, ask: float | None,
                 trade_id: int | None) -> Fill:
        """Place a real order and persist the broker's own identifiers."""
        submitted = datetime.now(timezone.utc)
        try:
            trade, filled, avg, commission = self._place(
                ticker, side, quantity, mid)
        except ExecutionError as e:
            self.store.write_order({
                "pair_id": self.pair_id, "trade_id": trade_id,
                "broker_order_id": e.order_id or None, "ticker": ticker,
                "side": side, "quantity": int(round(quantity)),
                "order_type": "LIMIT" if self.use_limit_orders else "MARKET",
                "status": "REJECTED" if e.status not in TERMINAL_BAD
                          else "CANCELLED",
                "submitted_at": submitted.isoformat(),
                "filled_quantity": 0, "error_message": str(e)[:300],
            })
            self.store.log("ERROR", "ORDER", str(e), self.pair_id)
            raise

        now = datetime.now(timezone.utc)
        order_id = self.store.write_order({
            "pair_id": self.pair_id, "trade_id": trade_id,
            "broker_order_id": str(trade.order.orderId), "ticker": ticker,
            "side": side, "quantity": int(round(quantity)),
            "order_type": "LIMIT" if self.use_limit_orders else "MARKET",
            "status": "FILLED" if filled >= round(quantity)
                      else "PARTIALLY_FILLED",
            "submitted_at": submitted.isoformat(),
            "acknowledged_at": submitted.isoformat(),
            "filled_at": now.isoformat(),
            "filled_quantity": filled, "avg_fill_price": round(avg, 6),
        })

        if order_id is not None:
            for f in trade.fills:
                self.store.write_fill({
                    "order_id": order_id, "pair_id": self.pair_id,
                    "broker_exec_id": f.execution.execId,
                    "ticker": ticker, "side": side,
                    "quantity": float(f.execution.shares),
                    "price": round(float(f.execution.price), 6),
                    "commission": float(
                        f.commissionReport.commission or 0)
                    if f.commissionReport else 0.0,
                    "ts": (f.time or now).isoformat(),
                })

        return Fill(ticker, side, filled, avg, now, commission)

    # --- transactional entry ------------------------------------------

    def enter(self, direction: str, bar, snapshot,
              quotes: dict | None = None) -> bool:
        """Open both legs, unwinding the first if the second fails."""
        try:
            return super().enter(direction, bar, snapshot, quotes)
        except ExecutionError as e:
            self.store.log("ERROR", "ORDER",
                           f"Entry failed: {e}. Checking for an orphaned leg.",
                           self.pair_id)
            self._unwind_orphans(bar)
            self.position = None
            self.trade_id = None
            return False

    def _unwind_orphans(self, bar) -> None:
        """Flatten any single-leg exposure left by a failed pair entry."""
        held = self.broker_positions()
        for ticker, qty in held.items():
            if ticker not in (self.leg1, self.leg2) or qty == 0:
                continue
            side = "SELL" if qty > 0 else "BUY"
            price = bar.leg1_price if ticker == self.leg1 else bar.leg2_price
            self.store.log("WARNING", "RISK",
                           f"Unwinding orphaned leg: {side} {abs(qty):g} "
                           f"{ticker}", self.pair_id)
            try:
                self._place(ticker, side, abs(qty), price)
            except ExecutionError as e:
                self.store.log(
                    "CRITICAL", "RISK",
                    f"COULD NOT UNWIND {ticker}: {e}. The account is holding "
                    f"unhedged directional exposure and needs manual "
                    f"intervention.", self.pair_id)

    # --- reconciliation -----------------------------------------------

    def broker_positions(self) -> dict[str, float]:
        """Signed quantities IBKR reports for this pair's tickers."""
        out: dict[str, float] = {}
        try:
            for p in self._ib.positions():
                symbol = p.contract.symbol
                if symbol in (self.leg1, self.leg2):
                    out[symbol] = float(p.position)
        except Exception as e:
            self.store.log("WARNING", "RISK",
                           f"Could not read broker positions: {e}",
                           self.pair_id)
        return out

    def reconcile(self, bar=None) -> dict:
        """Compare the engine's view against the broker's on startup.

        The broker is the source of truth for what is actually held. An
        engine that restarts believing it is flat while IBKR holds a position
        would size its next entry on top of existing exposure.
        """
        held = self.broker_positions()
        q1 = held.get(self.leg1, 0.0)
        q2 = held.get(self.leg2, 0.0)

        if q1 == 0 and q2 == 0:
            self.store.log("INFO", "SYSTEM",
                           "Reconciliation: broker reports flat, matching "
                           "engine state.", self.pair_id)
            self.store.clear_positions(self.pair_id)
            return {"flat": True}

        self.store.log(
            "WARNING", "RISK",
            f"Reconciliation: broker holds {q1:+g} {self.leg1} and "
            f"{q2:+g} {self.leg2} but the engine started flat. These were "
            f"NOT opened by this session.", self.pair_id)

        if (q1 > 0) == (q2 > 0) and q1 != 0 and q2 != 0:
            self.store.log("WARNING", "RISK",
                           "Both legs are on the same side — this is not a "
                           "spread position.", self.pair_id)

        return {"flat": False, self.leg1: q1, self.leg2: q2,
                "action": "adopt_or_flatten"}

    def flatten_broker_positions(self, bar) -> bool:
        """Close whatever IBKR holds for this pair, regardless of engine state."""
        held = self.broker_positions()
        if not any(held.values()):
            return False
        for ticker, qty in held.items():
            if qty == 0:
                continue
            side = "SELL" if qty > 0 else "BUY"
            price = bar.leg1_price if ticker == self.leg1 else bar.leg2_price
            try:
                self._place(ticker, side, abs(qty), price)
                self.store.log("INFO", "ORDER",
                               f"Flattened {side} {abs(qty):g} {ticker}",
                               self.pair_id)
            except ExecutionError as e:
                self.store.log("CRITICAL", "RISK",
                               f"Failed to flatten {ticker}: {e}", self.pair_id)
                return False
        self.position = None
        self.trade_id = None
        self.store.clear_positions(self.pair_id)
        return True

    def account_value(self) -> float | None:
        try:
            for v in self._ib.accountSummary():
                if v.tag == "NetLiquidation":
                    return float(v.value)
        except Exception:
            pass
        return None