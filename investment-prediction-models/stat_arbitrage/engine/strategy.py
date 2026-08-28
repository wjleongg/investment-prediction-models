"""Mean-reversion pairs strategy — pure logic, no I/O.

This module is deliberately free of database and broker calls so the same code
path drives both the historical backtest and the live engine. If these two ever
diverge, backtest results stop meaning anything about live behaviour.

Everything is expressed in terms of leg 1 and leg 2. No ticker appears here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Action(str, Enum):
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"       # long leg1, short leg2
    ENTER_SHORT = "ENTER_SHORT"     # short leg1, long leg2
    EXIT = "EXIT"


class ExitCause(str, Enum):
    MEAN_REVERSION = "MEAN_REVERSION"
    STOP_LOSS = "STOP_LOSS"
    MAX_HOLDING_PERIOD = "MAX_HOLDING_PERIOD"
    RELATIONSHIP_INVALID = "RELATIONSHIP_INVALID"
    END_OF_DATA = "END_OF_DATA"


@dataclass(frozen=True, slots=True)
class StrategyParams:
    """Mirror of the tradeable fields in strategy_config."""

    entry_threshold: float
    exit_threshold: float
    stop_loss_threshold: float
    max_holding_period_seconds: int | None
    min_correlation: float
    max_cointegration_pvalue: float
    capital_allocation: float
    max_position_size: float
    max_pair_exposure: float
    max_leverage: float = 1.0

    @classmethod
    def from_config_row(cls, row: dict) -> StrategyParams:
        return cls(
            entry_threshold=float(row["entry_threshold"]),
            exit_threshold=float(row["exit_threshold"]),
            stop_loss_threshold=float(row["stop_loss_threshold"]),
            max_holding_period_seconds=row.get("max_holding_period_seconds"),
            min_correlation=float(row["min_correlation"]),
            max_cointegration_pvalue=float(row["max_cointegration_pvalue"]),
            capital_allocation=float(row["capital_allocation"]),
            max_position_size=float(row["max_position_size"]),
            max_pair_exposure=float(row["max_pair_exposure"]),
            max_leverage=float(row.get("max_leverage", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class Bar:
    """One observation of the model state. Matches model_state_history."""

    ts: datetime
    leg1_price: float
    leg2_price: float
    spread: float
    zscore: float
    hedge_ratio: float | None = None
    spread_mean: float | None = None
    spread_std: float | None = None
    correlation: float | None = None
    cointegration_pvalue: float | None = None
    half_life: float | None = None

    def is_tradeable(self) -> bool:
        return (self.hedge_ratio is not None
                and self.leg1_price > 0 and self.leg2_price > 0
                and not math.isnan(self.zscore))

    def snapshot(self) -> dict:
        """Model state captured on the trade record."""
        return {
            "zscore": self.zscore, "spread": self.spread,
            "hedge_ratio": self.hedge_ratio, "spread_mean": self.spread_mean,
            "spread_std": self.spread_std, "correlation": self.correlation,
            "cointegration_pvalue": self.cointegration_pvalue,
            "half_life": self.half_life,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: str
    cause: ExitCause | None = None


@dataclass(slots=True)
class OpenPosition:
    direction: str           # LONG_SPREAD | SHORT_SPREAD
    entry_ts: datetime
    entry_bar: Bar
    leg1_quantity: float
    leg2_quantity: float
    hedge_ratio: float

    @property
    def gross_notional(self) -> float:
        return (self.leg1_quantity * self.entry_bar.leg1_price
                + self.leg2_quantity * self.entry_bar.leg2_price)

    def unrealized(self, bar: Bar) -> float:
        return pnl_for(self.direction, self.leg1_quantity, self.leg2_quantity,
                       self.entry_bar.leg1_price, self.entry_bar.leg2_price,
                       bar.leg1_price, bar.leg2_price)


# ---------------------------------------------------------------------
# Relationship gates
# ---------------------------------------------------------------------


def relationship_is_valid(bar: Bar, p: StrategyParams) -> tuple[bool, str]:
    """Entry is blocked unless the statistical relationship still holds."""
    if bar.correlation is not None and bar.correlation < p.min_correlation:
        return False, (f"correlation {bar.correlation:.3f} below minimum "
                       f"{p.min_correlation:.3f}")
    if (bar.cointegration_pvalue is not None
            and bar.cointegration_pvalue > p.max_cointegration_pvalue):
        return False, (f"cointegration p-value {bar.cointegration_pvalue:.4f} "
                       f"above maximum {p.max_cointegration_pvalue:.4f}")
    return True, "relationship valid"


# ---------------------------------------------------------------------
# Signal logic — the single decision function
# ---------------------------------------------------------------------


def decide(bar: Bar, position: OpenPosition | None,
           p: StrategyParams) -> Decision:
    """Given the current bar and open position, return the action to take."""
    if not bar.is_tradeable():
        return Decision(Action.HOLD, "bar not tradeable")

    z = bar.zscore

    if position is None:
        valid, why = relationship_is_valid(bar, p)
        if not valid:
            return Decision(Action.HOLD, f"entry blocked: {why}")
        if z <= -p.entry_threshold:
            return Decision(
                Action.ENTER_LONG,
                f"z-score {z:.2f} at or below -{p.entry_threshold:g} entry "
                f"threshold; spread is cheap relative to its mean")
        if z >= p.entry_threshold:
            return Decision(
                Action.ENTER_SHORT,
                f"z-score {z:.2f} at or above +{p.entry_threshold:g} entry "
                f"threshold; spread is rich relative to its mean")
        return Decision(Action.HOLD, f"z-score {z:.2f} inside entry threshold")

    # --- Position is open: check exits in priority order --------------
    # Stop-loss first. A widening spread against the position is the case
    # that actually loses money, so it must be evaluated before targets.
    adverse = (z <= -p.stop_loss_threshold if position.direction == "LONG_SPREAD"
               else z >= p.stop_loss_threshold)
    if adverse:
        return Decision(
            Action.EXIT,
            f"z-score {z:.2f} breached the ±{p.stop_loss_threshold:g} stop-loss "
            f"against a {position.direction} position",
            ExitCause.STOP_LOSS)

    if p.max_holding_period_seconds:
        held = (bar.ts - position.entry_ts).total_seconds()
        if held >= p.max_holding_period_seconds:
            return Decision(
                Action.EXIT,
                f"held {held / 86400:.1f} days, reaching the maximum holding "
                f"period", ExitCause.MAX_HOLDING_PERIOD)

    valid, why = relationship_is_valid(bar, p)
    if not valid:
        return Decision(Action.EXIT,
                        f"relationship no longer valid: {why}",
                        ExitCause.RELATIONSHIP_INVALID)

    # Directional, not |z| <= exit. A long spread entered at -2.4 is closed
    # once the z-score has risen TO OR PAST the exit band — including an
    # overshoot to +1.5. Testing abs(z) would leave that position open
    # because it is outside the band on the far side, which on fast bars
    # means a profitable trade is held until the stop or the holding limit.
    reverted = (z >= -p.exit_threshold if position.direction == "LONG_SPREAD"
                else z <= p.exit_threshold)
    if reverted:
        overshoot = ((z > p.exit_threshold) if position.direction == "LONG_SPREAD"
                     else (z < -p.exit_threshold))
        detail = (" (overshot through the band)" if overshoot else "")
        return Decision(
            Action.EXIT,
            f"z-score {z:.2f} reverted to the ±{p.exit_threshold:g} exit "
            f"threshold{detail}", ExitCause.MEAN_REVERSION)

    return Decision(Action.HOLD,
                    f"z-score {z:.2f} still outside the exit threshold")


# ---------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------


def size_position(bar: Bar, p: StrategyParams,
                  whole_shares: bool = True) -> tuple[float, float]:
    """Return (leg1_quantity, leg2_quantity), both positive.

    Leg 2 is scaled by the hedge ratio so the pair is approximately market
    neutral. Sizing respects max_position_size per leg and max_pair_exposure
    across the pair, then scales both legs down together so the hedge ratio
    is preserved.
    """
    beta = bar.hedge_ratio or 1.0

    q1 = p.max_position_size / bar.leg1_price
    q2 = q1 * beta

    # Cap leg 2 by its own position limit, scaling leg 1 to match.
    if q2 * bar.leg2_price > p.max_position_size:
        q2 = p.max_position_size / bar.leg2_price
        q1 = q2 / beta if beta else 0.0

    # Cap the pair by gross exposure.
    gross = q1 * bar.leg1_price + q2 * bar.leg2_price
    limit = min(p.max_pair_exposure, p.capital_allocation * p.max_leverage)
    if gross > limit and gross > 0:
        scale = limit / gross
        q1 *= scale
        q2 *= scale

    if whole_shares:
        q1, q2 = float(math.floor(q1)), float(math.floor(q2))

    return max(q1, 0.0), max(q2, 0.0)


def pnl_for(direction: str, q1: float, q2: float,
            entry1: float, entry2: float, exit1: float, exit2: float) -> float:
    """P&L of a two-leg spread position.

    LONG_SPREAD  = long leg 1, short leg 2
    SHORT_SPREAD = short leg 1, long leg 2
    """
    leg1 = q1 * (exit1 - entry1)
    leg2 = q2 * (exit2 - entry2)
    return (leg1 - leg2) if direction == "LONG_SPREAD" else (leg2 - leg1)


def fees_for(q1: float, q2: float, price1: float, price2: float,
             fee_bps: float) -> float:
    """Round-trip transaction cost in currency, as basis points of notional."""
    if fee_bps <= 0:
        return 0.0
    notional = q1 * price1 + q2 * price2
    return notional * (fee_bps / 10_000.0) * 2  # entry + exit