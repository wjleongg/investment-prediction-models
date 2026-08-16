"""Python mirrors of the Postgres enum types defined in schema.sql.

Every member's value is the exact string stored in the database. Both the
engine and the frontend import from here, so a change to a state machine
happens in one place.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum that round-trips cleanly through JSON and psycopg."""

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


class SystemStatus(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    KILLED = "KILLED"

    @property
    def is_terminal(self) -> bool:
        return self in (SystemStatus.STOPPED, SystemStatus.KILLED)


class DataSource(StrEnum):
    LIVE_IBKR = "LIVE_IBKR"
    SIMULATED_FEED = "SIMULATED_FEED"
    HISTORICAL_DATA = "HISTORICAL_DATA"
    DISCONNECTED = "DISCONNECTED"

    @property
    def is_live(self) -> bool:
        """Drives the SIMULATED FEED badge. Only true IBKR counts as live."""
        return self is DataSource.LIVE_IBKR

    @property
    def badge(self) -> str:
        return {
            DataSource.LIVE_IBKR: "● LIVE IBKR",
            DataSource.SIMULATED_FEED: "⚠ SIMULATED FEED",
            DataSource.HISTORICAL_DATA: "⚠ HISTORICAL DATA",
            DataSource.DISCONNECTED: "✕ DISCONNECTED",
        }[self]


class MarketStatus(StrEnum):
    PRE_MARKET = "PRE_MARKET"
    MARKET_OPEN = "MARKET_OPEN"
    AFTER_HOURS = "AFTER_HOURS"
    MARKET_CLOSED = "MARKET_CLOSED"
    UNKNOWN = "UNKNOWN"


class SignalType(StrEnum):
    LONG_SPREAD = "LONG_SPREAD"
    SHORT_SPREAD = "SHORT_SPREAD"
    EXIT = "EXIT"
    FLAT = "FLAT"
    NO_SIGNAL = "NO_SIGNAL"


class RelationshipHealth(StrEnum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"

    @property
    def label(self) -> str:
        """Textual label so status never depends on colour alone."""
        return {
            RelationshipHealth.VALID: "✓ RELATIONSHIP VALID",
            RelationshipHealth.DEGRADED: "⚠ RELATIONSHIP DEGRADED",
            RelationshipHealth.INVALID: "✕ RELATIONSHIP INVALID",
        }[self]


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @property
    def is_open(self) -> bool:
        return self in (
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
        )


class TradeDirection(StrEnum):
    LONG_SPREAD = "LONG_SPREAD"
    SHORT_SPREAD = "SHORT_SPREAD"


class TradeStatus(StrEnum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    FAILED = "FAILED"


class ExitReason(StrEnum):
    MEAN_REVERSION = "MEAN_REVERSION"
    STOP_LOSS = "STOP_LOSS"
    MAX_HOLDING_PERIOD = "MAX_HOLDING_PERIOD"
    RELATIONSHIP_INVALID = "RELATIONSHIP_INVALID"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"
    END_OF_DAY = "END_OF_DAY"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(StrEnum):
    MARKET_DATA = "MARKET_DATA"
    MODEL = "MODEL"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    RISK = "RISK"
    SYSTEM = "SYSTEM"
    CONTROL = "CONTROL"


class CommandType(StrEnum):
    START = "START"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    RESTART = "RESTART"
    STOP = "STOP"
    CANCEL_ALL_ORDERS = "CANCEL_ALL_ORDERS"
    FLATTEN_ALL_POSITIONS = "FLATTEN_ALL_POSITIONS"
    KILL_SWITCH = "KILL_SWITCH"
    APPLY_CONFIG = "APPLY_CONFIG"

    @property
    def is_destructive(self) -> bool:
        """Destructive commands require explicit confirmation in the UI."""
        return self in (
            CommandType.KILL_SWITCH,
            CommandType.FLATTEN_ALL_POSITIONS,
            CommandType.CANCEL_ALL_ORDERS,
            CommandType.RESTART,
            CommandType.STOP,
        )


class CommandStatus(StrEnum):
    REQUESTED = "REQUESTED"
    RECEIVED = "RECEIVED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"

    @property
    def is_confirmed(self) -> bool:
        """A command is only confirmed once the ENGINE has spoken.

        A successful database insert means nothing on its own.
        """
        return self in (
            CommandStatus.ACKNOWLEDGED,
            CommandStatus.EXECUTED,
        )

    @property
    def is_settled(self) -> bool:
        return self in (
            CommandStatus.EXECUTED,
            CommandStatus.FAILED,
            CommandStatus.EXPIRED,
        )


class ConnectionState(StrEnum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class ConfigStatus(StrEnum):
    CURRENT = "CURRENT"
    PENDING = "PENDING"
    ARCHIVED = "ARCHIVED"
    REJECTED = "REJECTED"


class CointegrationTest(StrEnum):
    ENGLE_GRANGER = "ENGLE_GRANGER"
    ADF = "ADF"
    PHILLIPS_PERRON = "PHILLIPS_PERRON"
    JOHANSEN = "JOHANSEN"
    KPSS = "KPSS"

    @property
    def display_name(self) -> str:
        return {
            CointegrationTest.ENGLE_GRANGER: "Engle-Granger",
            CointegrationTest.ADF: "Augmented Dickey-Fuller",
            CointegrationTest.PHILLIPS_PERRON: "Phillips-Perron",
            CointegrationTest.JOHANSEN: "Johansen",
            CointegrationTest.KPSS: "KPSS",
        }[self]


class PriceField(StrEnum):
    LAST = "LAST"
    BID = "BID"
    ASK = "ASK"
    MID = "MID"
    CLOSE = "CLOSE"
