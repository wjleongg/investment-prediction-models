"""Typed data models sitting between Supabase and the Streamlit render layer.

Boundary rule: page code never touches raw dict rows and never writes SQL.
The data-access layer fetches rows, builds these objects, and hands them to
pure render functions.

    fetch(...) -> typed state object -> render(state)

Numeric convention: money is exposed as `float` here because this layer is
display-only. The database stores NUMERIC and the engine should use Decimal
internally for anything that affects an order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, TypeVar

try:  # package import (frontend) or flat import (standalone scripts)
    from .enums import (
    CointegrationTest, CommandStatus, CommandType, ConfigStatus,
    ConnectionState, DataSource, ExitReason, LogCategory, LogLevel,
    MarketStatus, OrderSide, OrderStatus, OrderType, PositionSide,
    PriceField, RelationshipHealth, SignalType, SystemStatus,
    TradeDirection, TradeStatus,
    )
except ImportError:  # pragma: no cover
    from enums import (  # type: ignore
    CointegrationTest, CommandStatus, CommandType, ConfigStatus,
    ConnectionState, DataSource, ExitReason, LogCategory, LogLevel,
    MarketStatus, OrderSide, OrderStatus, OrderType, PositionSide,
    PriceField, RelationshipHealth, SignalType, SystemStatus,
    TradeDirection, TradeStatus,
    )

# ---------------------------------------------------------------------
# Freshness thresholds. The frontend uses these to decide when to stop
# trusting stored state. Tuned to the refresh cadences in the spec.
# ---------------------------------------------------------------------

HEARTBEAT_STALE_SECONDS = 5.0
HEARTBEAT_DEAD_SECONDS = 30.0
# Default for tick feeds. Bar feeds pass their own threshold: a 1-minute bar
# feed is not stale at 15 seconds, it is simply between bars.
MARKET_DATA_STALE_SECONDS = 15.0

#: How many missed intervals count as stale, for bar-based feeds.
STALE_INTERVAL_MULTIPLE = 3.0

BAR_INTERVAL_SECONDS = {"1s": 1, "2s": 2, "5s": 5, "10s": 10, "30s": 30,
                        "1m": 60, "2m": 120, "5m": 300, "15m": 900,
                        "1h": 3600, "1d": 86400}


def stale_threshold(bar_interval: str | None) -> float:
    """Seconds without data before a feed counts as stale."""
    if not bar_interval:
        return MARKET_DATA_STALE_SECONDS
    seconds = BAR_INTERVAL_SECONDS.get(bar_interval)
    if seconds is None:
        return MARKET_DATA_STALE_SECONDS
    return max(seconds * STALE_INTERVAL_MULTIPLE, MARKET_DATA_STALE_SECONDS)
COMMAND_CONFIRM_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------
# Row coercion helpers
# ---------------------------------------------------------------------

E = TypeVar("E")


def _dt(value: Any) -> datetime | None:
    """Parse a timestamptz into an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _f(value: Any) -> float | None:
    return None if value is None else float(value)


def _f0(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _i(value: Any) -> int | None:
    return None if value is None else int(value)


def _enum(cls: type[E], value: Any, default: E | None = None) -> E | None:
    if value is None:
        return default
    if isinstance(value, cls):
        return value
    return cls(str(value))


def _age_seconds(ts: datetime | None, now: datetime | None = None) -> float | None:
    if ts is None:
        return None
    return ((now or datetime.now(timezone.utc)) - ts).total_seconds()


# ---------------------------------------------------------------------
# Pair and configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pair:
    """The only place tickers are named. Everything downstream uses pair_id."""

    id: int
    leg1_ticker: str
    leg2_ticker: str
    is_active: bool = False
    notes: str | None = None

    @property
    def label(self) -> str:
        return f"{self.leg1_ticker} / {self.leg2_ticker}"

    def ticker_for_leg(self, leg: int) -> str:
        if leg not in (1, 2):
            raise ValueError(f"leg must be 1 or 2, got {leg}")
        return self.leg1_ticker if leg == 1 else self.leg2_ticker

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Pair:
        return cls(
            id=int(row["id"]),
            leg1_ticker=row["leg1_ticker"],
            leg2_ticker=row["leg2_ticker"],
            is_active=bool(row.get("is_active", False)),
            notes=row.get("notes"),
        )


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    id: int
    version: int
    status: ConfigStatus
    pair_id: int

    historical_lookback: int
    zscore_lookback: int
    correlation_lookback: int
    cointegration_lookback: int

    entry_threshold: float
    exit_threshold: float
    stop_loss_threshold: float
    max_holding_period_seconds: int | None

    min_correlation: float
    max_cointegration_pvalue: float

    capital_allocation: float
    max_position_size: float
    max_pair_exposure: float
    max_leverage: float
    max_simultaneous_positions: int

    kill_switch_flattens: bool
    requires_restart: bool
    created_at: datetime | None = None
    created_by: str | None = None
    applied_at: datetime | None = None
    rejection_reason: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> StrategyConfig:
        return cls(
            id=int(row["id"]),
            version=int(row["version"]),
            status=_enum(ConfigStatus, row["status"], ConfigStatus.PENDING),
            pair_id=int(row["pair_id"]),
            historical_lookback=int(row["historical_lookback"]),
            zscore_lookback=int(row["zscore_lookback"]),
            correlation_lookback=int(row["correlation_lookback"]),
            cointegration_lookback=int(row["cointegration_lookback"]),
            entry_threshold=float(row["entry_threshold"]),
            exit_threshold=float(row["exit_threshold"]),
            stop_loss_threshold=float(row["stop_loss_threshold"]),
            max_holding_period_seconds=_i(row.get("max_holding_period_seconds")),
            min_correlation=float(row["min_correlation"]),
            max_cointegration_pvalue=float(row["max_cointegration_pvalue"]),
            capital_allocation=float(row["capital_allocation"]),
            max_position_size=float(row["max_position_size"]),
            max_pair_exposure=float(row["max_pair_exposure"]),
            max_leverage=float(row["max_leverage"]),
            max_simultaneous_positions=int(row["max_simultaneous_positions"]),
            kill_switch_flattens=bool(row["kill_switch_flattens"]),
            requires_restart=bool(row.get("requires_restart", True)),
            created_at=_dt(row.get("created_at")),
            created_by=row.get("created_by"),
            applied_at=_dt(row.get("applied_at")),
            rejection_reason=row.get("rejection_reason"),
        )


@dataclass(frozen=True, slots=True)
class ConfigDiff:
    """One changed field between CURRENT and PENDING config."""

    field_name: str
    current_value: Any
    pending_value: Any


def diff_configs(current: StrategyConfig, pending: StrategyConfig) -> list[ConfigDiff]:
    """Field-level diff powering the CURRENT vs PENDING configuration view."""
    skip = {"id", "version", "status", "created_at", "created_by",
            "applied_at", "rejection_reason", "requires_restart"}
    diffs: list[ConfigDiff] = []
    for name in current.__dataclass_fields__:
        if name in skip:
            continue
        cur, pen = getattr(current, name), getattr(pending, name)
        if cur != pen:
            diffs.append(ConfigDiff(name, cur, pen))
    return diffs


# ---------------------------------------------------------------------
# Engine liveness and infrastructure
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EngineHeartbeat:
    engine_id: str
    status: SystemStatus
    data_source: DataSource
    market_status: MarketStatus
    heartbeat_at: datetime
    started_at: datetime | None = None
    engine_version: str | None = None
    host: str | None = None
    active_config_version: int | None = None
    detail: str | None = None

    def age_seconds(self, now: datetime | None = None) -> float:
        return _age_seconds(self.heartbeat_at, now) or 0.0

    def is_stale(self, now: datetime | None = None) -> bool:
        return self.age_seconds(now) > HEARTBEAT_STALE_SECONDS

    def is_dead(self, now: datetime | None = None) -> bool:
        return self.age_seconds(now) > HEARTBEAT_DEAD_SECONDS

    def health_label(self, now: datetime | None = None) -> str:
        """Never derives health from `status`. Only heartbeat age proves life."""
        age = self.age_seconds(now)
        if age > HEARTBEAT_DEAD_SECONDS:
            return f"✕ ENGINE UNREACHABLE — last heartbeat {age:.1f}s ago"
        if age > HEARTBEAT_STALE_SECONDS:
            return f"⚠ ENGINE STALE — last heartbeat {age:.1f}s ago"
        return f"✓ ENGINE HEALTHY — last heartbeat {age:.1f}s ago"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> EngineHeartbeat:
        return cls(
            engine_id=row["engine_id"],
            status=_enum(SystemStatus, row["status"], SystemStatus.ERROR),
            data_source=_enum(DataSource, row["data_source"], DataSource.DISCONNECTED),
            market_status=_enum(MarketStatus, row.get("market_status"), MarketStatus.UNKNOWN),
            heartbeat_at=_dt(row["heartbeat_at"]),
            started_at=_dt(row.get("started_at")),
            engine_version=row.get("engine_version"),
            host=row.get("host"),
            active_config_version=_i(row.get("active_config_version")),
            detail=row.get("detail"),
        )


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    component: str
    state: ConnectionState
    last_ok_at: datetime | None = None
    updated_at: datetime | None = None
    detail: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ComponentStatus:
        return cls(
            component=row["component"],
            state=_enum(ConnectionState, row["state"], ConnectionState.UNKNOWN),
            last_ok_at=_dt(row.get("last_ok_at")),
            updated_at=_dt(row.get("updated_at")),
            detail=row.get("detail"),
        )


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    engine_id: str
    as_of: datetime | None
    market_data_events: int = 0
    model_calculations: int = 0
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    state_writes: int = 0
    error_count: int = 0
    warning_count: int = 0
    last_market_data_at: datetime | None = None
    last_model_calc_at: datetime | None = None
    last_state_write_at: datetime | None = None
    last_order_event_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> SystemMetrics:
        return cls(
            engine_id=row["engine_id"],
            as_of=_dt(row.get("as_of")),
            market_data_events=int(row.get("market_data_events", 0)),
            model_calculations=int(row.get("model_calculations", 0)),
            signals_generated=int(row.get("signals_generated", 0)),
            orders_submitted=int(row.get("orders_submitted", 0)),
            orders_filled=int(row.get("orders_filled", 0)),
            state_writes=int(row.get("state_writes", 0)),
            error_count=int(row.get("error_count", 0)),
            warning_count=int(row.get("warning_count", 0)),
            last_market_data_at=_dt(row.get("last_market_data_at")),
            last_model_calc_at=_dt(row.get("last_model_calc_at")),
            last_state_write_at=_dt(row.get("last_state_write_at")),
            last_order_event_at=_dt(row.get("last_order_event_at")),
        )


# ---------------------------------------------------------------------
# Live state
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_return_pct: float = 0.0
    daily_return_pct: float = 0.0
    current_exposure: float = 0.0
    capital_utilisation: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelState:
    spread: float | None = None
    hedge_ratio: float | None = None
    zscore: float | None = None
    spread_mean: float | None = None
    spread_std: float | None = None
    spread_volatility: float | None = None
    correlation: float | None = None
    rolling_correlation: float | None = None
    cointegration_stat: float | None = None
    cointegration_pvalue: float | None = None
    half_life: float | None = None
    health: RelationshipHealth = RelationshipHealth.VALID
    health_reason: str | None = None
    last_model_update_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class LiveState:
    """Materialised snapshot. One row per pair, read in a single query."""

    pair: Pair
    as_of: datetime

    # Market data
    leg1_price: float | None = None
    leg2_price: float | None = None
    leg1_bid: float | None = None
    leg1_ask: float | None = None
    leg2_bid: float | None = None
    leg2_ask: float | None = None
    leg1_mid: float | None = None
    leg2_mid: float | None = None
    last_market_data_at: datetime | None = None

    model: ModelState = field(default_factory=ModelState)
    portfolio: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)

    current_signal: SignalType = SignalType.NO_SIGNAL
    signal_since: datetime | None = None
    signal_explanation: str | None = None
    current_position: PositionSide = PositionSide.FLAT
    target_position: PositionSide = PositionSide.FLAT

    entry_threshold: float | None = None
    exit_threshold: float | None = None
    stop_loss_threshold: float | None = None

    config_version: int | None = None

    def market_data_is_stale(self, now: datetime | None = None,
                             bar_interval: str | None = None) -> bool:
        age = _age_seconds(self.last_market_data_at, now)
        if age is None:
            return True
        return age > stale_threshold(bar_interval)

    def market_data_age(self, now: datetime | None = None) -> float | None:
        return _age_seconds(self.last_market_data_at, now)

    def price_for_leg(self, leg: int) -> float | None:
        return self.leg1_price if leg == 1 else self.leg2_price

    @classmethod
    def from_row(cls, row: Mapping[str, Any], pair: Pair) -> LiveState:
        return cls(
            pair=pair,
            as_of=_dt(row["as_of"]),
            leg1_price=_f(row.get("leg1_price")),
            leg2_price=_f(row.get("leg2_price")),
            leg1_bid=_f(row.get("leg1_bid")),
            leg1_ask=_f(row.get("leg1_ask")),
            leg2_bid=_f(row.get("leg2_bid")),
            leg2_ask=_f(row.get("leg2_ask")),
            leg1_mid=_f(row.get("leg1_mid")),
            leg2_mid=_f(row.get("leg2_mid")),
            last_market_data_at=_dt(row.get("last_market_data_at")),
            model=ModelState(
                spread=_f(row.get("spread")),
                hedge_ratio=_f(row.get("hedge_ratio")),
                zscore=_f(row.get("zscore")),
                spread_mean=_f(row.get("spread_mean")),
                spread_std=_f(row.get("spread_std")),
                spread_volatility=_f(row.get("spread_volatility")),
                correlation=_f(row.get("correlation")),
                rolling_correlation=_f(row.get("rolling_correlation")),
                cointegration_stat=_f(row.get("cointegration_stat")),
                cointegration_pvalue=_f(row.get("cointegration_pvalue")),
                half_life=_f(row.get("half_life")),
                health=_enum(RelationshipHealth, row.get("health"), RelationshipHealth.VALID),
                health_reason=row.get("health_reason"),
                last_model_update_at=_dt(row.get("last_model_update_at")),
            ),
            portfolio=PortfolioSnapshot(
                total_pnl=_f0(row.get("total_pnl")),
                daily_pnl=_f0(row.get("daily_pnl")),
                realized_pnl=_f0(row.get("realized_pnl")),
                unrealized_pnl=_f0(row.get("unrealized_pnl")),
                total_return_pct=_f0(row.get("total_return_pct")),
                daily_return_pct=_f0(row.get("daily_return_pct")),
                current_exposure=_f0(row.get("current_exposure")),
                capital_utilisation=_f0(row.get("capital_utilisation")),
            ),
            current_signal=_enum(SignalType, row.get("current_signal"), SignalType.NO_SIGNAL),
            signal_since=_dt(row.get("signal_since")),
            signal_explanation=row.get("signal_explanation"),
            current_position=_enum(PositionSide, row.get("current_position"), PositionSide.FLAT),
            target_position=_enum(PositionSide, row.get("target_position"), PositionSide.FLAT),
            entry_threshold=_f(row.get("entry_threshold")),
            exit_threshold=_f(row.get("exit_threshold")),
            stop_loss_threshold=_f(row.get("stop_loss_threshold")),
            config_version=_i(row.get("config_version")),
        )


# ---------------------------------------------------------------------
# Time series points
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarketDataPoint:
    ts: datetime
    ticker: str
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    field_: PriceField = PriceField.LAST
    source: DataSource = DataSource.SIMULATED_FEED


@dataclass(frozen=True, slots=True)
class ModelStatePoint:
    """One row of the spread / z-score chart series."""

    ts: datetime
    spread: float
    zscore: float
    leg1_price: float | None = None
    leg2_price: float | None = None
    hedge_ratio: float | None = None
    spread_mean: float | None = None
    spread_std: float | None = None
    correlation: float | None = None
    cointegration_pvalue: float | None = None
    half_life: float | None = None
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    stop_loss_threshold: float | None = None
    health: RelationshipHealth | None = None
    signal: SignalType | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ModelStatePoint:
        return cls(
            ts=_dt(row["ts"]),
            spread=float(row["spread"]),
            zscore=float(row["zscore"]),
            leg1_price=_f(row.get("leg1_price")),
            leg2_price=_f(row.get("leg2_price")),
            hedge_ratio=_f(row.get("hedge_ratio")),
            spread_mean=_f(row.get("spread_mean")),
            spread_std=_f(row.get("spread_std")),
            correlation=_f(row.get("correlation")),
            cointegration_pvalue=_f(row.get("cointegration_pvalue")),
            half_life=_f(row.get("half_life")),
            entry_threshold=_f(row.get("entry_threshold")),
            exit_threshold=_f(row.get("exit_threshold")),
            stop_loss_threshold=_f(row.get("stop_loss_threshold")),
            health=_enum(RelationshipHealth, row.get("health")),
            signal=_enum(SignalType, row.get("signal")),
        )


# ---------------------------------------------------------------------
# Signals, orders, fills, positions, trades
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Signal:
    id: int
    ts: datetime
    signal: SignalType
    previous_signal: SignalType | None = None
    zscore: float | None = None
    hedge_ratio: float | None = None
    health: RelationshipHealth | None = None
    explanation: str | None = None
    acted_upon: bool = False
    suppressed_reason: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Signal:
        return cls(
            id=int(row["id"]),
            ts=_dt(row["ts"]),
            signal=_enum(SignalType, row["signal"], SignalType.NO_SIGNAL),
            previous_signal=_enum(SignalType, row.get("previous_signal")),
            zscore=_f(row.get("zscore")),
            hedge_ratio=_f(row.get("hedge_ratio")),
            health=_enum(RelationshipHealth, row.get("health")),
            explanation=row.get("explanation"),
            acted_upon=bool(row.get("acted_upon", False)),
            suppressed_reason=row.get("suppressed_reason"),
        )


@dataclass(frozen=True, slots=True)
class Order:
    id: int
    pair_id: int
    ticker: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    status: OrderStatus
    submitted_at: datetime
    broker_order_id: str | None = None
    trade_id: int | None = None
    limit_price: float | None = None
    acknowledged_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    error_message: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Order:
        return cls(
            id=int(row["id"]),
            pair_id=int(row["pair_id"]),
            ticker=row["ticker"],
            side=_enum(OrderSide, row["side"], OrderSide.BUY),
            quantity=float(row["quantity"]),
            order_type=_enum(OrderType, row.get("order_type"), OrderType.MARKET),
            status=_enum(OrderStatus, row["status"], OrderStatus.PENDING),
            submitted_at=_dt(row["submitted_at"]),
            broker_order_id=row.get("broker_order_id"),
            trade_id=_i(row.get("trade_id")),
            limit_price=_f(row.get("limit_price")),
            acknowledged_at=_dt(row.get("acknowledged_at")),
            filled_at=_dt(row.get("filled_at")),
            cancelled_at=_dt(row.get("cancelled_at")),
            filled_quantity=_f0(row.get("filled_quantity")),
            avg_fill_price=_f(row.get("avg_fill_price")),
            error_message=row.get("error_message"),
        )


@dataclass(frozen=True, slots=True)
class Fill:
    id: int
    order_id: int
    ticker: str
    side: OrderSide
    quantity: float
    price: float
    ts: datetime
    commission: float = 0.0
    broker_exec_id: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Fill:
        return cls(
            id=int(row["id"]),
            order_id=int(row["order_id"]),
            ticker=row["ticker"],
            side=_enum(OrderSide, row["side"], OrderSide.BUY),
            quantity=float(row["quantity"]),
            price=float(row["price"]),
            ts=_dt(row["ts"]),
            commission=_f0(row.get("commission")),
            broker_exec_id=row.get("broker_exec_id"),
        )


@dataclass(frozen=True, slots=True)
class Position:
    id: int
    pair_id: int
    ticker: str
    side: PositionSide
    quantity: float
    avg_entry_price: float
    opened_at: datetime
    trade_id: int | None = None
    current_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Position:
        return cls(
            id=int(row["id"]),
            pair_id=int(row["pair_id"]),
            ticker=row["ticker"],
            side=_enum(PositionSide, row["side"], PositionSide.FLAT),
            quantity=float(row["quantity"]),
            avg_entry_price=float(row["avg_entry_price"]),
            opened_at=_dt(row["opened_at"]),
            trade_id=_i(row.get("trade_id")),
            current_price=_f(row.get("current_price")),
            market_value=_f(row.get("market_value")),
            unrealized_pnl=_f(row.get("unrealized_pnl")),
            updated_at=_dt(row.get("updated_at")),
        )


@dataclass(frozen=True, slots=True)
class PairPosition:
    """Both legs plus the combined pair-level P&L shown on the Overview page."""

    pair: Pair
    legs: Sequence[Position]

    @property
    def pair_pnl(self) -> float:
        return sum(p.unrealized_pnl or 0.0 for p in self.legs)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value or 0.0) for p in self.legs)

    @property
    def net_exposure(self) -> float:
        return sum(p.market_value or 0.0 for p in self.legs)

    @property
    def is_flat(self) -> bool:
        return all(p.quantity == 0 for p in self.legs)


@dataclass(frozen=True, slots=True)
class Trade:
    id: int
    pair_id: int
    direction: TradeDirection
    status: TradeStatus
    entry_time: datetime
    leg1_ticker: str | None = None
    leg2_ticker: str | None = None
    exit_time: datetime | None = None
    leg1_entry_price: float | None = None
    leg2_entry_price: float | None = None
    leg1_exit_price: float | None = None
    leg2_exit_price: float | None = None
    leg1_quantity: float | None = None
    leg2_quantity: float | None = None
    entry_zscore: float | None = None
    exit_zscore: float | None = None
    entry_hedge_ratio: float | None = None
    exit_hedge_ratio: float | None = None
    entry_model_state: Mapping[str, Any] | None = None
    exit_model_state: Mapping[str, Any] | None = None
    gross_pnl: float | None = None
    fees: float = 0.0
    net_pnl: float | None = None
    return_pct: float | None = None
    holding_period_seconds: int | None = None
    exit_reason: ExitReason | None = None
    config_version: int | None = None

    @property
    def is_open(self) -> bool:
        return self.status in (TradeStatus.OPEN, TradeStatus.CLOSING)

    @property
    def is_profitable(self) -> bool | None:
        if self.net_pnl is None:
            return None
        return self.net_pnl > 0

    @property
    def zscore_bucket(self) -> str | None:
        """Entry z-score bucket for performance attribution."""
        if self.entry_zscore is None:
            return None
        magnitude = abs(self.entry_zscore)
        if magnitude < 2.0:
            return "<2.0"
        if magnitude < 2.5:
            return "2.0-2.5"
        if magnitude < 3.0:
            return "2.5-3.0"
        return ">=3.0"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Trade:
        return cls(
            id=int(row["id"]),
            pair_id=int(row["pair_id"]),
            direction=_enum(TradeDirection, row["direction"], TradeDirection.LONG_SPREAD),
            status=_enum(TradeStatus, row["status"], TradeStatus.OPEN),
            entry_time=_dt(row["entry_time"]),
            leg1_ticker=row.get("leg1_ticker"),
            leg2_ticker=row.get("leg2_ticker"),
            exit_time=_dt(row.get("exit_time")),
            leg1_entry_price=_f(row.get("leg1_entry_price")),
            leg2_entry_price=_f(row.get("leg2_entry_price")),
            leg1_exit_price=_f(row.get("leg1_exit_price")),
            leg2_exit_price=_f(row.get("leg2_exit_price")),
            leg1_quantity=_f(row.get("leg1_quantity")),
            leg2_quantity=_f(row.get("leg2_quantity")),
            entry_zscore=_f(row.get("entry_zscore")),
            exit_zscore=_f(row.get("exit_zscore")),
            entry_hedge_ratio=_f(row.get("entry_hedge_ratio")),
            exit_hedge_ratio=_f(row.get("exit_hedge_ratio")),
            entry_model_state=row.get("entry_model_state"),
            exit_model_state=row.get("exit_model_state"),
            gross_pnl=_f(row.get("gross_pnl")),
            fees=_f0(row.get("fees")),
            net_pnl=_f(row.get("net_pnl")),
            return_pct=_f(row.get("return_pct")),
            holding_period_seconds=_i(row.get("holding_period_seconds")),
            exit_reason=_enum(ExitReason, row.get("exit_reason")),
            config_version=_i(row.get("config_version")),
        )


# ---------------------------------------------------------------------
# Research and performance
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CointegrationResult:
    test: CointegrationTest
    as_of: datetime
    lookback_bars: int
    statistic: float | None = None
    pvalue: float | None = None
    critical_values: Mapping[str, float] | None = None
    passed: bool | None = None
    interpretation: str | None = None

    @property
    def verdict(self) -> str:
        if self.passed is None:
            return "INCONCLUSIVE"
        return "COINTEGRATION SUPPORTED" if self.passed else "COINTEGRATION NOT SUPPORTED"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> CointegrationResult:
        return cls(
            test=_enum(CointegrationTest, row["test"], CointegrationTest.ENGLE_GRANGER),
            as_of=_dt(row["as_of"]),
            lookback_bars=int(row["lookback_bars"]),
            statistic=_f(row.get("statistic")),
            pvalue=_f(row.get("pvalue")),
            critical_values=row.get("critical_values"),
            passed=row.get("passed"),
            interpretation=row.get("interpretation"),
        )


@dataclass(frozen=True, slots=True)
class RollingDiagnostic:
    ts: datetime
    window_bars: int
    rolling_correlation: float | None = None
    rolling_hedge_ratio: float | None = None
    rolling_cointegration_stat: float | None = None
    rolling_cointegration_pvalue: float | None = None
    rolling_spread_volatility: float | None = None
    half_life: float | None = None
    hurst_exponent: float | None = None
    relationship_valid: bool | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> RollingDiagnostic:
        return cls(
            ts=_dt(row["ts"]),
            window_bars=int(row["window_bars"]),
            rolling_correlation=_f(row.get("rolling_correlation")),
            rolling_hedge_ratio=_f(row.get("rolling_hedge_ratio")),
            rolling_cointegration_stat=_f(row.get("rolling_cointegration_stat")),
            rolling_cointegration_pvalue=_f(row.get("rolling_cointegration_pvalue")),
            rolling_spread_volatility=_f(row.get("rolling_spread_volatility")),
            half_life=_f(row.get("half_life")),
            hurst_exponent=_f(row.get("hurst_exponent")),
            relationship_valid=row.get("relationship_valid"),
        )


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts: datetime
    equity: float
    cumulative_pnl: float
    cumulative_return_pct: float | None = None
    drawdown_pct: float | None = None
    leg1_cumulative_return_pct: float | None = None
    leg2_cumulative_return_pct: float | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> EquityPoint:
        return cls(
            ts=_dt(row["ts"]),
            equity=float(row["equity"]),
            cumulative_pnl=float(row["cumulative_pnl"]),
            cumulative_return_pct=_f(row.get("cumulative_return_pct")),
            drawdown_pct=_f(row.get("drawdown_pct")),
            leg1_cumulative_return_pct=_f(row.get("leg1_cumulative_return_pct")),
            leg2_cumulative_return_pct=_f(row.get("leg2_cumulative_return_pct")),
        )


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    as_of: datetime
    period_label: str = "ALL"
    period_start: datetime | None = None
    period_end: datetime | None = None
    total_return_pct: float | None = None
    annualised_return_pct: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    max_drawdown_pct: float | None = None
    max_drawdown_duration_days: float | None = None
    current_drawdown_pct: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    num_trades: int | None = None
    avg_trade_pnl: float | None = None
    avg_holding_period_seconds: int | None = None
    best_trade_pnl: float | None = None
    worst_trade_pnl: float | None = None
    total_pnl: float | None = None
    return_mean: float | None = None
    return_median: float | None = None
    return_std: float | None = None
    return_skew: float | None = None
    return_kurtosis: float | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> PerformanceMetrics:
        return cls(
            as_of=_dt(row["as_of"]),
            period_label=row.get("period_label", "ALL"),
            period_start=_dt(row.get("period_start")),
            period_end=_dt(row.get("period_end")),
            total_return_pct=_f(row.get("total_return_pct")),
            annualised_return_pct=_f(row.get("annualised_return_pct")),
            sharpe_ratio=_f(row.get("sharpe_ratio")),
            sortino_ratio=_f(row.get("sortino_ratio")),
            max_drawdown_pct=_f(row.get("max_drawdown_pct")),
            max_drawdown_duration_days=_f(row.get("max_drawdown_duration_days")),
            current_drawdown_pct=_f(row.get("current_drawdown_pct")),
            win_rate=_f(row.get("win_rate")),
            profit_factor=_f(row.get("profit_factor")),
            num_trades=_i(row.get("num_trades")),
            avg_trade_pnl=_f(row.get("avg_trade_pnl")),
            avg_holding_period_seconds=_i(row.get("avg_holding_period_seconds")),
            best_trade_pnl=_f(row.get("best_trade_pnl")),
            worst_trade_pnl=_f(row.get("worst_trade_pnl")),
            total_pnl=_f(row.get("total_pnl")),
            return_mean=_f(row.get("return_mean")),
            return_median=_f(row.get("return_median")),
            return_std=_f(row.get("return_std")),
            return_skew=_f(row.get("return_skew")),
            return_kurtosis=_f(row.get("return_kurtosis")),
        )


# ---------------------------------------------------------------------
# Logs and control commands
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogEntry:
    id: int
    ts: datetime
    level: LogLevel
    category: LogCategory
    message: str
    engine_id: str | None = None
    details: Mapping[str, Any] | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> LogEntry:
        return cls(
            id=int(row["id"]),
            ts=_dt(row["ts"]),
            level=_enum(LogLevel, row["level"], LogLevel.INFO),
            category=_enum(LogCategory, row["category"], LogCategory.SYSTEM),
            message=row["message"],
            engine_id=row.get("engine_id"),
            details=row.get("details"),
        )


@dataclass(frozen=True, slots=True)
class Command:
    """A control request and its lifecycle.

    The UI must never report success on insert alone. Only `is_confirmed`
    means the engine has actually spoken.
    """

    id: str
    command: CommandType
    status: CommandStatus
    requested_at: datetime
    expires_at: datetime
    requested_by: str | None = None
    received_at: datetime | None = None
    acknowledged_at: datetime | None = None
    executed_at: datetime | None = None
    engine_id: str | None = None
    payload: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None
    error_message: str | None = None
    source: str = "STREAMLIT"

    @property
    def is_confirmed(self) -> bool:
        return self.status.is_confirmed

    def is_unconfirmed_and_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return self.status == CommandStatus.REQUESTED and now > self.expires_at

    def lifecycle_display(self, now: datetime | None = None) -> str:
        """Renders the REQUESTED -> RECEIVED -> ACKNOWLEDGED -> EXECUTED chain."""
        if self.is_unconfirmed_and_expired(now):
            return "⚠ NOT CONFIRMED BY ENGINE — command may not have been executed"
        stages = [
            ("REQUESTED", self.requested_at),
            ("RECEIVED", self.received_at),
            ("ACKNOWLEDGED", self.acknowledged_at),
            ("EXECUTED", self.executed_at),
        ]
        return " → ".join(
            f"{name}{'' if ts else ' (pending)'}" for name, ts in stages
        )

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Command:
        return cls(
            id=str(row["id"]),
            command=_enum(CommandType, row["command"], CommandType.PAUSE),
            status=_enum(CommandStatus, row["status"], CommandStatus.REQUESTED),
            requested_at=_dt(row["requested_at"]),
            expires_at=_dt(row["expires_at"]),
            requested_by=row.get("requested_by"),
            received_at=_dt(row.get("received_at")),
            acknowledged_at=_dt(row.get("acknowledged_at")),
            executed_at=_dt(row.get("executed_at")),
            engine_id=row.get("engine_id"),
            payload=row.get("payload"),
            result=row.get("result"),
            error_message=row.get("error_message"),
            source=row.get("source", "STREAMLIT"),
        )


# ---------------------------------------------------------------------
# Composite page state — what render() actually receives
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeaderState:
    """Everything the persistent top header needs, from one fetch."""

    pair: Pair
    heartbeat: EngineHeartbeat | None
    market_status: MarketStatus = MarketStatus.UNKNOWN

    @property
    def data_source(self) -> DataSource:
        return self.heartbeat.data_source if self.heartbeat else DataSource.DISCONNECTED

    @property
    def show_simulated_badge(self) -> bool:
        return not self.data_source.is_live

    @property
    def system_status(self) -> SystemStatus:
        """Reports ERROR when the heartbeat is dead, whatever the row claims."""
        if self.heartbeat is None or self.heartbeat.is_dead():
            return SystemStatus.ERROR
        return self.heartbeat.status


@dataclass(frozen=True, slots=True)
class OverviewState:
    """The single typed object the Overview page renders from."""

    header: HeaderState
    live: LiveState
    positions: PairPosition
    recent_fills: Sequence[Fill] = ()
    spread_series: Sequence[ModelStatePoint] = ()
    trade_markers: Sequence[Trade] = ()


@dataclass(frozen=True, slots=True)
class SystemHealthState:
    header: HeaderState
    components: Sequence[ComponentStatus] = ()
    metrics: SystemMetrics | None = None
    logs: Sequence[LogEntry] = ()
    