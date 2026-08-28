"""Rolling statistical model for a pair.

Maintains a sliding window of both legs and recomputes the quantities the
strategy depends on. Pure computation — no I/O, no database, no broker.

Two correctness rules are enforced here, both learned from the backtest:

1. The hedge ratio and the spread are computed from the SAME beta. The
   backfill used a static beta for the spread while storing a rolling one,
   which made position sizing inconsistent with the signal.

2. Statistics are computed from the live bar interval only. Daily-bar
   parameters do not describe an intraday spread — the backfilled mean of
   +1.61 was 23 sigma away from the live spread. The model always warms up
   on its own timescale.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np

try:
    from statsmodels.tsa.stattools import coint
    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False


@dataclass(frozen=True, slots=True)
class ModelParams:
    zscore_lookback: int
    correlation_lookback: int
    cointegration_lookback: int
    min_correlation: float
    max_cointegration_pvalue: float
    #: Cointegration is expensive; recompute every N bars rather than every bar.
    coint_every: int = 30
    #: Half-life is cheap but noisy bar to bar.
    half_life_every: int = 10

    @classmethod
    def from_config_row(cls, row: dict, coint_every: int = 30) -> ModelParams:
        return cls(
            zscore_lookback=int(row["zscore_lookback"]),
            correlation_lookback=int(row["correlation_lookback"]),
            cointegration_lookback=int(row["cointegration_lookback"]),
            min_correlation=float(row["min_correlation"]),
            max_cointegration_pvalue=float(row["max_cointegration_pvalue"]),
            coint_every=coint_every,
        )

    @property
    def required_bars(self) -> int:
        """Bars needed before the model can produce a usable signal."""
        return max(self.zscore_lookback, self.correlation_lookback,
                   self.cointegration_lookback)


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """Complete model state at one bar."""

    ts: datetime
    leg1_price: float
    leg2_price: float
    hedge_ratio: float
    spread: float
    spread_mean: float
    spread_std: float
    zscore: float
    correlation: float | None = None
    rolling_correlation: float | None = None
    cointegration_stat: float | None = None
    cointegration_pvalue: float | None = None
    half_life: float | None = None
    spread_volatility: float | None = None
    health: str = "VALID"
    health_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "spread": self.spread, "zscore": self.zscore,
            "hedge_ratio": self.hedge_ratio, "spread_mean": self.spread_mean,
            "spread_std": self.spread_std, "correlation": self.correlation,
            "cointegration_pvalue": self.cointegration_pvalue,
            "half_life": self.half_life,
        }


class PairModel:
    """Sliding-window model over two price series."""

    def __init__(self, params: ModelParams) -> None:
        self.p = params
        maxlen = params.required_bars + 10
        self._ts: deque[datetime] = deque(maxlen=maxlen)
        self._p1: deque[float] = deque(maxlen=maxlen)
        self._p2: deque[float] = deque(maxlen=maxlen)
        self._bar_count = 0

        # Cached expensive statistics
        self._coint_stat: float | None = None
        self._coint_pvalue: float | None = None
        self._half_life: float | None = None
        self._last_coint_bar = -10_000
        self._last_hl_bar = -10_000

    # --- state --------------------------------------------------------

    @property
    def bars(self) -> int:
        return len(self._p1)

    @property
    def is_ready(self) -> bool:
        """True once enough history exists for every configured lookback."""
        return self.bars >= self.p.required_bars

    def warmup_progress(self) -> tuple[int, int]:
        return self.bars, self.p.required_bars

    # --- ingestion ----------------------------------------------------

    def add(self, ts: datetime, price1: float, price2: float) -> None:
        if price1 <= 0 or price2 <= 0:
            return
        if self._ts and ts <= self._ts[-1]:
            return                      # never ingest a stale or repeated bar
        self._ts.append(ts)
        self._p1.append(float(price1))
        self._p2.append(float(price2))
        self._bar_count += 1

    def warmup(self, bars: Sequence[tuple[datetime, float, float]]) -> int:
        for ts, a, b in bars:
            self.add(ts, a, b)
        return self.bars

    # --- statistics ---------------------------------------------------

    def _hedge_ratio(self) -> float:
        """OLS slope of leg1 on leg2 over the correlation lookback.

        This same beta defines the spread, so sizing and signal always agree.
        """
        n = min(self.p.correlation_lookback, self.bars)
        y = np.fromiter(self._p1, float)[-n:]
        x = np.fromiter(self._p2, float)[-n:]
        var = x.var()
        if var <= 0:
            return 1.0
        return float(np.cov(y, x, bias=True)[0, 1] / var)

    def _spread_series(self, beta: float, n: int) -> np.ndarray:
        y = np.fromiter(self._p1, float)[-n:]
        x = np.fromiter(self._p2, float)[-n:]
        return y - beta * x

    def _compute_half_life(self, spread: np.ndarray) -> float | None:
        if len(spread) < 20:
            return None
        lagged = spread[:-1]
        delta = np.diff(spread)
        var = lagged.var()
        if var <= 0:
            return None
        beta = float(np.cov(delta, lagged, bias=True)[0, 1] / var)
        if beta >= 0:
            return None                 # diverging, no half-life
        return float(-math.log(2) / beta)

    def _compute_cointegration(self) -> tuple[float | None, float | None]:
        if not HAS_STATSMODELS:
            return None, None
        n = min(self.p.cointegration_lookback, self.bars)
        if n < 30:
            return None, None
        y = np.fromiter(self._p1, float)[-n:]
        x = np.fromiter(self._p2, float)[-n:]
        try:
            stat, pvalue, _ = coint(y, x)
            return float(stat), float(pvalue)
        except Exception:
            return None, None

    def _assess_health(self, correlation: float | None,
                       pvalue: float | None) -> tuple[str, str]:
        """Relationship validity, with a DEGRADED band before INVALID."""
        problems = []
        if correlation is not None and correlation < self.p.min_correlation:
            margin = self.p.min_correlation - correlation
            problems.append(
                (f"correlation {correlation:.3f} below minimum "
                 f"{self.p.min_correlation:.3f}", margin > 0.05))
        if pvalue is not None and pvalue > self.p.max_cointegration_pvalue:
            margin = pvalue - self.p.max_cointegration_pvalue
            problems.append(
                (f"cointegration p-value {pvalue:.4f} above maximum "
                 f"{self.p.max_cointegration_pvalue:.4f}", margin > 0.05))

        if not problems:
            return "VALID", "correlation and cointegration within limits"
        reason = "; ".join(p[0] for p in problems)
        severe = any(p[1] for p in problems)
        return ("INVALID" if severe else "DEGRADED"), reason

    # --- main entry point ---------------------------------------------

    def snapshot(self) -> ModelSnapshot | None:
        """Current model state, or None if not enough history yet."""
        if self.bars < 3:
            return None

        beta = self._hedge_ratio()

        z_n = min(self.p.zscore_lookback, self.bars)
        z_spread = self._spread_series(beta, z_n)
        mean = float(z_spread.mean())
        std = float(z_spread.std(ddof=1)) if len(z_spread) > 1 else 0.0
        current = float(z_spread[-1])
        zscore = ((current - mean) / std) if std > 1e-12 else 0.0

        c_n = min(self.p.correlation_lookback, self.bars)
        y = np.fromiter(self._p1, float)[-c_n:]
        x = np.fromiter(self._p2, float)[-c_n:]
        correlation = (float(np.corrcoef(y, x)[0, 1])
                       if c_n > 2 and y.std() > 0 and x.std() > 0 else None)

        # Expensive statistics on a slower cadence
        if self._bar_count - self._last_coint_bar >= self.p.coint_every:
            self._coint_stat, self._coint_pvalue = self._compute_cointegration()
            self._last_coint_bar = self._bar_count
        if self._bar_count - self._last_hl_bar >= self.p.half_life_every:
            hl_n = min(self.p.cointegration_lookback, self.bars)
            self._half_life = self._compute_half_life(
                self._spread_series(beta, hl_n))
            self._last_hl_bar = self._bar_count

        health, reason = self._assess_health(correlation, self._coint_pvalue)

        return ModelSnapshot(
            ts=self._ts[-1],
            leg1_price=self._p1[-1], leg2_price=self._p2[-1],
            hedge_ratio=beta, spread=current, spread_mean=mean,
            spread_std=std, zscore=zscore,
            correlation=correlation, rolling_correlation=correlation,
            cointegration_stat=self._coint_stat,
            cointegration_pvalue=self._coint_pvalue,
            half_life=self._half_life,
            spread_volatility=std,
            health=health, health_reason=reason)
