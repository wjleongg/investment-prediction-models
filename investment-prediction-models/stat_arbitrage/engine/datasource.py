"""Market data sources.

One interface, three implementations. The engine depends only on the
interface, so switching from simulated to IBKR changes a single line in the
engine's construction and nothing else.

    MarketDataSource
      ├── IBKRSource        live or delayed, via TWS / IB Gateway
      ├── YFinanceSource    historical bars, for backfill and replay
      └── SimulatedSource   synthetic cointegrated pair, for offline work

Every source reports which DataSource enum it represents, and that value
drives the SIMULATED FEED badge in the frontend. A source must never claim
to be live when it is not.
"""

from __future__ import annotations

import math
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterator, Sequence


@dataclass(frozen=True, slots=True)
class Quote:
    """A single observation for one instrument."""

    ticker: str
    ts: datetime
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None

    @property
    def mid(self) -> float:
        if self.bid is not None and self.ask is not None and self.bid > 0:
            return (self.bid + self.ask) / 2
        return self.price

    @property
    def spread_bps(self) -> float | None:
        """Quoted spread in basis points — the floor on transaction cost."""
        if self.bid and self.ask and self.bid > 0:
            return (self.ask - self.bid) / self.mid * 10_000
        return None


@dataclass(frozen=True, slots=True)
class PairQuote:
    """Both legs observed together. The engine only ever acts on these."""

    ts: datetime
    leg1: Quote
    leg2: Quote

    @property
    def is_complete(self) -> bool:
        return self.leg1.price > 0 and self.leg2.price > 0


class MarketDataSource(ABC):
    """Interface every source implements."""

    #: Value written to engine_heartbeat.data_source
    data_source: str = "DISCONNECTED"

    #: Bar interval label written alongside every persisted row
    bar_interval: str = "1m"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def stream(self, ticker1: str, ticker2: str) -> Iterator[PairQuote]:
        """Yield synchronised quotes for both legs until stopped."""

    def historical(self, ticker: str, lookback: str = "1 D",
                   bar_size: str = "1 min") -> Sequence[Quote]:
        """Optional. Sources that cannot serve history return an empty list."""
        return []

    def describe(self) -> str:
        return f"{type(self).__name__} ({self.data_source}, {self.bar_interval})"


# =====================================================================
# IBKR
# =====================================================================

# TWS paper 7497 / live 7496; IB Gateway paper 4002 / live 4001.
IBKR_PORTS = {"tws-paper": 7497, "tws-live": 7496,
              "gateway-paper": 4002, "gateway-live": 4001}

# reqMarketDataType: 1 live, 2 frozen, 3 delayed, 4 delayed-frozen
MARKET_DATA_LIVE = 1
MARKET_DATA_DELAYED = 3


class IBKRSource(MarketDataSource):
    """Live or delayed quotes from TWS / IB Gateway via ib_async.

    Requires TWS or Gateway running with the API enabled
    (Configuration → API → Settings → Enable ActiveX and Socket Clients,
    and 127.0.0.1 in Trusted IPs).
    """

    data_source = "LIVE_IBKR"

    def __init__(self, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1, bar_seconds: int = 5,
                 use_delayed: bool = False, exchange: str = "SMART",
                 currency: str = "USD", timeout: float = 15.0) -> None:
        self.host, self.port, self.client_id = host, port, client_id
        self.bar_seconds = bar_seconds
        self.use_delayed = use_delayed
        self.exchange, self.currency = exchange, currency
        self.timeout = timeout
        self.bar_interval = f"{bar_seconds}s"
        self._ib = None
        self._contracts: dict[str, object] = {}
        self._stop = threading.Event()

        if use_delayed:
            # Never let delayed data masquerade as live in the UI.
            self.data_source = "DELAYED_IBKR"

    # --- lifecycle ----------------------------------------------------

    def connect(self) -> None:
        try:
            from ib_async import IB
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "ib_async is not installed. `pip install ib_async`. "
                "(ib_insync is unmaintained; ib_async is its successor.)") from e

        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id,
                         timeout=self.timeout)
        self._ib.reqMarketDataType(
            MARKET_DATA_DELAYED if self.use_delayed else MARKET_DATA_LIVE)

    def disconnect(self) -> None:
        self._stop.set()
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def is_connected(self) -> bool:
        return self._ib is not None and self._ib.isConnected()

    # --- contracts ----------------------------------------------------

    def _contract(self, ticker: str):
        from ib_async import Stock
        if ticker not in self._contracts:
            c = Stock(ticker, self.exchange, self.currency)
            qualified = self._ib.qualifyContracts(c)
            if not qualified:
                raise RuntimeError(f"IBKR could not qualify contract {ticker}")
            self._contracts[ticker] = qualified[0]
        return self._contracts[ticker]

    def account_summary(self) -> dict[str, str]:
        if not self.is_connected():
            return {}
        return {v.tag: v.value for v in self._ib.accountSummary()}

    # --- data ---------------------------------------------------------

    def snapshot(self, ticker1: str, ticker2: str,
                 wait: float = 12.0) -> PairQuote | None:
        """One synchronised observation, polling until both legs populate.

        Delayed ticks can take ten seconds to arrive on first request, so a
        fixed short sleep reports a false negative.
        """
        t1 = self._ib.reqMktData(self._contract(ticker1), "", False, False)
        t2 = self._ib.reqMktData(self._contract(ticker2), "", False, False)
        deadline = time.time() + wait
        while time.time() < deadline:
            self._ib.sleep(0.5)
            q1, q2 = self._to_quote(ticker1, t1), self._to_quote(ticker2, t2)
            if q1 is not None and q2 is not None:
                return PairQuote(ts=datetime.now(timezone.utc), leg1=q1, leg2=q2)
        return None

    def detect_data_type(self, ticker: str, wait: float = 8.0) -> str:
        """Report what IBKR is actually sending, not what we asked for.

        Requesting live without an entitlement makes TWS fall back to delayed
        silently, so the requested type is not evidence of what arrived.
        """
        t = self._ib.reqMktData(self._contract(ticker), "", False, False)
        self._ib.sleep(wait)
        mdt = getattr(t, "marketDataType", None)
        resolved = {1: "LIVE_IBKR", 2: "DELAYED_IBKR",
                    3: "DELAYED_IBKR", 4: "DELAYED_IBKR"}.get(mdt)
        if resolved:
            self.data_source = resolved
        return self.data_source

    def raw_tickers(self, ticker1: str, ticker2: str, wait: float = 12.0):
        """Diagnostic: return the raw Ticker objects and their data type."""
        t1 = self._ib.reqMktData(self._contract(ticker1), "", False, False)
        t2 = self._ib.reqMktData(self._contract(ticker2), "", False, False)
        self._ib.sleep(wait)
        return t1, t2

    def on_error(self, handler) -> None:
        """Subscribe to IBKR error events. The error code names the cause."""
        self._ib.errorEvent += handler

    def historical_pair(self, ticker1: str, ticker2: str,
                        lookback: str = "1 D", bar_size: str = "1 min",
                        completed_only: bool = True) -> PairQuote | None:
        """Latest synchronised bar from historical data.

        Fallback feed when streaming quotes are unavailable: reqHistoricalData
        with keepUpToDate semantics still delivers current prices, at bar
        latency rather than tick latency.
        """
        b1 = self.historical(ticker1, lookback, bar_size)
        b2 = self.historical(ticker2, lookback, bar_size)
        if not b1 or not b2:
            return None

        if completed_only:
            # The final bar is the in-progress minute: its close moves until
            # the minute ends. Emitting it would record a partial-minute price
            # that is never corrected, so step back to the last closed bar.
            if len(b1) < 2 or len(b2) < 2:
                return None
            b1, b2 = b1[:-1], b2[:-1]

        # Align on a shared timestamp so the spread is never computed from
        # two different minutes.
        by_ts1 = {q.ts: q for q in b1}
        by_ts2 = {q.ts: q for q in b2}
        common = sorted(set(by_ts1) & set(by_ts2))
        if not common:
            return None
        ts = common[-1]
        return PairQuote(ts=ts, leg1=by_ts1[ts], leg2=by_ts2[ts])

    def stream(self, ticker1: str, ticker2: str) -> Iterator[PairQuote]:
        """Yield a synchronised pair quote on each bar interval.

        Both legs are read from the same tick snapshot so the spread is never
        computed from prices observed at different instants.
        """
        if not self.is_connected():
            raise RuntimeError("not connected to IBKR")

        self._stop.clear()
        t1 = self._ib.reqMktData(self._contract(ticker1), "", False, False)
        t2 = self._ib.reqMktData(self._contract(ticker2), "", False, False)

        while not self._stop.is_set():
            self._ib.sleep(self.bar_seconds)
            q1, q2 = self._to_quote(ticker1, t1), self._to_quote(ticker2, t2)
            if q1 is None or q2 is None:
                continue
            pq = PairQuote(ts=datetime.now(timezone.utc), leg1=q1, leg2=q2)
            if pq.is_complete:
                yield pq

    def historical(self, ticker: str, lookback: str = "1 D",
                   bar_size: str = "1 min") -> Sequence[Quote]:
        bars = self._ib.reqHistoricalData(
            self._contract(ticker), endDateTime="", durationStr=lookback,
            barSizeSetting=bar_size, whatToShow="MIDPOINT",
            useRTH=True, formatDate=2)
        return [Quote(ticker=ticker,
                      ts=b.date if isinstance(b.date, datetime)
                      else datetime.combine(b.date, datetime.min.time(),
                                            tzinfo=timezone.utc),
                      price=float(b.close), volume=int(b.volume or 0))
                for b in bars]

    @staticmethod
    def _to_quote(ticker: str, t) -> Quote | None:
        """Prefer last trade, fall back to midpoint, then close.

        Delayed feeds populate separate delayed* attributes on some ib_async
        versions, so both naming schemes are checked.
        """
        def ok(v) -> bool:
            return (v is not None
                    and not (isinstance(v, float) and math.isnan(v))
                    and v > 0)

        def attr(name):
            return getattr(t, name, None)

        bid = next((v for v in (attr("bid"), attr("delayedBid")) if ok(v)), None)
        ask = next((v for v in (attr("ask"), attr("delayedAsk")) if ok(v)), None)

        try:
            market_price = t.marketPrice()
        except Exception:
            market_price = None

        price = None
        for candidate in (attr("last"), attr("delayedLast"), attr("close"),
                          attr("delayedClose"), market_price):
            if ok(candidate):
                price = float(candidate)
                break
        if price is None and bid and ask:
            price = (bid + ask) / 2
        if price is None:
            return None
        return Quote(ticker=ticker, ts=datetime.now(timezone.utc), price=price,
                     bid=bid, ask=ask,
                     volume=int(t.volume) if ok(t.volume) else None)


class IBKRPollingSource(IBKRSource):
    """IBKR feed built on historical bars rather than streaming quotes.

    IBKR's complimentary real-time data is licensed for TWS display only and
    is not delivered over the API (error 2186). Historical data goes through
    HMDS, which carries no such restriction, so polling the most recent bar
    gives genuine current prices for free.

    Trade-off: latency is one bar rather than one tick, and there are no
    bid/ask quotes, so quoted-spread cost cannot be measured from this feed.
    Everything else about the engine is unchanged.
    """

    data_source = "LIVE_IBKR"

    #: IBKR rejects more than ~60 historical requests per 10 minutes.
    #: Each poll costs one request per leg, so 2 legs at 60s = 20 per 10 min.
    MIN_POLL_SECONDS = 30.0

    def __init__(self, bar_size: str = "1 min", poll_seconds: float = 60.0,
                 lookback: str = "1 D", **kwargs) -> None:
        kwargs.pop("bar_seconds", None)
        super().__init__(**kwargs)
        self.bar_size = bar_size
        if poll_seconds < self.MIN_POLL_SECONDS:
            poll_seconds = self.MIN_POLL_SECONDS
        self.poll_seconds = poll_seconds
        self.lookback = lookback
        self.consecutive_failures = 0
        self.last_error: str = ""
        self.on_feed_error = None       # set by the engine to surface errors
        self.bar_interval = bar_size.replace(" ", "").replace("min", "m")
        self._last_emitted: datetime | None = None

    @property
    def latency_note(self) -> str:
        lag = self.last_bar_lag_seconds
        measured = (f", last bar {lag / 60:.0f} min behind"
                    if lag is not None else "")
        return (f"bar feed ({self.bar_size} polled every "
                f"{self.poll_seconds:g}s){measured}; no bid/ask")

    def warmup(self, ticker1: str, ticker2: str,
               lookback: str = "2 D") -> list[PairQuote]:
        """Historical bars for model warmup, aligned on timestamp.

        The engine needs its own intraday statistics — hedge ratio and spread
        moments computed from daily bars do not describe an intraday spread.
        """
        b1 = {q.ts: q for q in self.historical(ticker1, lookback, self.bar_size)}
        b2 = {q.ts: q for q in self.historical(ticker2, lookback, self.bar_size)}
        common = sorted(set(b1) & set(b2))
        return [PairQuote(ts=ts, leg1=b1[ts], leg2=b2[ts]) for ts in common]

    def stream(self, ticker1: str, ticker2: str) -> Iterator[PairQuote]:
        """Emit each newly completed bar exactly once."""
        if not self.is_connected():
            raise RuntimeError("not connected to IBKR")
        self._stop.clear()

        backoff = 0.0
        while not self._stop.is_set():
            pq = None
            try:
                pq = self.historical_pair(ticker1, ticker2, self.lookback,
                                          self.bar_size, completed_only=True)
                if pq is None:
                    raise RuntimeError(
                        "historical request returned no usable bars for both "
                        "legs (market may have just opened, or IBKR is pacing "
                        "the request)")
                self.consecutive_failures = 0
                backoff = 0.0
            except Exception as e:
                self.consecutive_failures += 1
                self.last_error = str(e)[:300]
                if self.on_feed_error:
                    self.on_feed_error(self.consecutive_failures, self.last_error)
                # Back off on repeated failures: hammering a paced endpoint
                # makes the pacing violation worse.
                backoff = min(60.0 * min(self.consecutive_failures, 5), 300.0)

            if pq is not None and pq.is_complete:
                # Only emit a bar we have not already published, so a poll
                # faster than the bar interval does not duplicate rows.
                if self._last_emitted is None or pq.ts > self._last_emitted:
                    self._last_emitted = pq.ts
                    # Report how far behind the bar is. Without a market data
                    # subscription IBKR delays historical bars too, and that
                    # lag decides whether a signal is actionable or archaeology.
                    lag = (datetime.now(timezone.utc) - pq.ts).total_seconds()
                    if lag > 300 and self._lag_warned is False:
                        self._report("WARNING",
                                     f"Bars arrive {lag / 60:.0f} min behind "
                                     f"real time. Signals reference a market "
                                     f"that has already moved.")
                        self._lag_warned = True
                    self.last_bar_lag_seconds = lag
                    yield pq

            self._ib.sleep(self.poll_seconds + backoff)


# =====================================================================
# yfinance
# =====================================================================


class YFinanceSource(MarketDataSource):
    """Historical bars. Not a live feed — polls at best, delayed at worst."""

    data_source = "HISTORICAL_DATA"

    def __init__(self, interval: str = "1m", poll_seconds: int = 60) -> None:
        self.bar_interval = interval
        self.poll_seconds = poll_seconds
        self._connected = False

    def connect(self) -> None:
        import yfinance  # noqa: F401 — fail fast if missing
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def stream(self, ticker1: str, ticker2: str) -> Iterator[PairQuote]:
        import yfinance as yf
        while self._connected:
            data = yf.download([ticker1, ticker2], period="1d",
                               interval=self.bar_interval, progress=False,
                               auto_adjust=True)
            if not data.empty:
                close = data["Close"].dropna()
                if not close.empty:
                    row = close.iloc[-1]
                    ts = close.index[-1].to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    yield PairQuote(
                        ts=ts,
                        leg1=Quote(ticker1, ts, float(row[ticker1])),
                        leg2=Quote(ticker2, ts, float(row[ticker2])))
            time.sleep(self.poll_seconds)


# =====================================================================
# Simulated
# =====================================================================


class SimulatedSource(MarketDataSource):
    """Synthetic cointegrated pair with a mean-reverting spread.

    For offline development when markets are closed. The spread follows an
    Ornstein-Uhlenbeck process, so entries and exits actually trigger.
    """

    data_source = "SIMULATED_FEED"

    def __init__(self, base_price: float = 680.0, hedge_ratio: float = 1.0,
                 spread_sigma: float = 0.25, reversion: float = 0.08,
                 tick_seconds: float = 1.0, seed: int | None = None) -> None:
        self.base_price = base_price
        self.hedge_ratio = hedge_ratio
        self.spread_sigma = spread_sigma
        self.reversion = reversion
        self.tick_seconds = tick_seconds
        self.bar_interval = f"{int(tick_seconds)}s"
        self._rng = random.Random(seed)
        self._connected = False
        self._level = base_price
        self._spread = 0.0
        self._stop = threading.Event()

    def connect(self) -> None:
        self._connected = True
        self._stop.clear()

    def disconnect(self) -> None:
        self._connected = False
        self._stop.set()

    def is_connected(self) -> bool:
        return self._connected

    def stream(self, ticker1: str, ticker2: str) -> Iterator[PairQuote]:
        while self._connected and not self._stop.is_set():
            # Common factor: both legs move together (market beta)
            self._level *= (1 + self._rng.gauss(0, 0.0002))
            # Idiosyncratic spread: Ornstein-Uhlenbeck around zero
            self._spread += (-self.reversion * self._spread
                             + self._rng.gauss(0, self.spread_sigma))

            p2 = self._level
            p1 = self.hedge_ratio * p2 + self._spread
            ts = datetime.now(timezone.utc)
            half = 0.01
            yield PairQuote(
                ts=ts,
                leg1=Quote(ticker1, ts, round(p1, 4),
                           bid=round(p1 - half, 4), ask=round(p1 + half, 4)),
                leg2=Quote(ticker2, ts, round(p2, 4),
                           bid=round(p2 - half, 4), ask=round(p2 + half, 4)))
            time.sleep(self.tick_seconds)


# =====================================================================
# Factory
# =====================================================================


def build_source(kind: str, **kwargs) -> MarketDataSource:
    """Construct a source by name. The engine's only coupling to a vendor."""
    kind = kind.lower()
    if kind in ("ibkr", "ib", "ibkr-stream"):
        return IBKRSource(**kwargs)
    if kind in ("ibkr-poll", "ibkr-bars"):
        return IBKRPollingSource(**kwargs)
    if kind in ("yfinance", "yf"):
        return YFinanceSource(**kwargs)
    if kind in ("simulated", "sim", "mock"):
        return SimulatedSource(**kwargs)
    raise ValueError(f"unknown market data source: {kind}")