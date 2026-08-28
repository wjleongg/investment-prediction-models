"""The trading engine.

Owns the state machine, the model loop, execution and persistence. Runs
independently of the frontend: closing Streamlit, losing the browser, or a
database outage must not stop it managing an open position.

State machine
-------------
    STARTING  -> warming up the intraday model
    RUNNING   -> generating signals and trading
    PAUSED    -> still ingesting data and updating state, not trading
    STOPPING  -> flattening, about to stop
    STOPPED   -> idle, still heartbeating
    KILLED    -> emergency stop, will not trade again this session
    ERROR     -> unrecoverable

The heartbeat runs on its own thread every 2s so liveness is reported even
while the main loop is blocked waiting on a bar. This matters because the
frontend treats a heartbeat older than 5s as stale.
"""

from __future__ import annotations

import math
import os
import signal as os_signal
import threading
import time
from datetime import datetime, time as dtime, timezone
from typing import Any

from engine.datasource import MarketDataSource, PairQuote, build_source
from engine.execution import PaperBroker
from engine.model import ModelParams, ModelSnapshot, PairModel
from engine.persistence import Store
from engine.strategy import Action, Bar, ExitCause, StrategyParams, decide

HEARTBEAT_SECONDS = 2.0
COMMAND_POLL_SECONDS = 2.0

# US equity regular trading hours in UTC (approximate; ignores DST shifts
# and holidays, which is acceptable for a paper engine).
RTH_OPEN_UTC = dtime(13, 30)
RTH_CLOSE_UTC = dtime(20, 0)


def market_status(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return "MARKET_CLOSED"
    t = now.time()
    if t < dtime(8, 0):
        return "MARKET_CLOSED"
    if t < RTH_OPEN_UTC:
        return "PRE_MARKET"
    if t < RTH_CLOSE_UTC:
        return "MARKET_OPEN"
    if t < dtime(24, 0):
        return "AFTER_HOURS"
    return "MARKET_CLOSED"


class Engine:
    def __init__(self, pair_id: int = 1, feed: str = "poll",
                 slippage_bps: float = 0.0, commission_per_share: float = 0.0,
                 trade_outside_rth: bool = False,
                 warmup_lookback: str = "auto",
                 broker: str = "paper",
                 max_data_age_seconds: float = 0.0,
                 flatten_before_close_minutes: int = 0,
                 limit_orders: bool = False) -> None:
        self.store = Store()
        self.pair = self.store.load_pair(pair_id)
        self.pair_id = pair_id
        self.leg1 = self.pair["leg1_ticker"]
        self.leg2 = self.pair["leg2_ticker"]

        self.config = self.store.load_config(pair_id)
        self.model_params = ModelParams.from_config_row(self.config)
        self.strategy_params = StrategyParams.from_config_row(self.config)
        self.model = PairModel(self.model_params)

        self.feed_kind = feed
        self.source = self._build_source(feed)
        self.quote_source: MarketDataSource | None = None
        self.warmup_lookback = warmup_lookback
        self.trade_outside_rth = trade_outside_rth

        self.broker_kind = broker
        #: Refuse new entries when the newest bar is older than this. A signal
        #: computed from stale prices describes a dislocation that has already
        #: closed. 0 disables the guard.
        self.max_data_age_seconds = max_data_age_seconds
        self.flatten_before_close_minutes = flatten_before_close_minutes
        if broker == "ibkr":
            from engine.broker import IBKRBroker
            self.broker = IBKRBroker(
                self.store, self.pair, self.strategy_params, self.source,
                use_limit_orders=limit_orders,
                commission_per_share=commission_per_share)
            # getattr, not direct access: a diagnostic log line must never
            # be able to stop the engine starting.
            self.store.log(
                "INFO", "SYSTEM",
                f"Order routing: "
                f"{'marketable limit' if limit_orders else 'market'} orders, "
                f"TIF={getattr(self.broker, 'tif', 'default')}, "
                f"outsideRth={getattr(self.broker, 'outside_rth', False)}",
                self.pair_id)
            if not hasattr(self.broker, "tif"):
                self.store.log(
                    "WARNING", "SYSTEM",
                    "engine/broker.py predates the TIF fix. Orders may be "
                    "cancelled by a TWS order preset (error 10349). Update "
                    "broker.py before trading.", self.pair_id)
        else:
            self.broker = PaperBroker(
                self.store, self.pair, self.strategy_params,
                slippage_bps=slippage_bps,
                commission_per_share=commission_per_share)

        self.state = "STARTING"
        self.started_at = datetime.now(timezone.utc)
        self.current_signal = "NO_SIGNAL"
        self.signal_since: datetime | None = None
        self.last_snapshot: ModelSnapshot | None = None
        self.last_bar: Bar | None = None
        self.last_quotes: dict[str, float | None] = {}
        self.restart_requested = False

        self._first_bar_seen = False
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_events: dict[str, Any] = {}

    # --- construction -------------------------------------------------

    def _build_source(self, feed: str) -> MarketDataSource:
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = int(os.environ.get("IBKR_PORT", 7497))
        client_id = int(os.environ.get("IBKR_CLIENT_ID", 1))
        common = dict(host=host, port=port, client_id=client_id)

        if feed == "poll":
            return build_source("ibkr-poll", bar_size="1 min",
                                poll_seconds=60.0, lookback="1800 S", **common)
        if feed == "stream":
            return build_source("ibkr", bar_seconds=5, use_delayed=True,
                                **common)
        if feed == "sim":
            return build_source("simulated", tick_seconds=2.0)
        raise ValueError(f"unknown feed: {feed}")

    # --- lifecycle ----------------------------------------------------

    def start(self) -> None:
        self.store.log("INFO", "SYSTEM",
                       f"Engine starting for {self.leg1}/{self.leg2}, "
                       f"feed={self.feed_kind}, config v{self.config['version']}",
                       self.pair_id)
        if self.broker_kind == "ibkr" and self.feed_kind == "sim":
            raise RuntimeError(
                "Cannot route real orders on a simulated feed: fills would be "
                "priced against prices that do not exist.")
        self._connect()
        self._start_background()
        try:
            self._warmup()
            self._reconcile()
            self._run_loop()
        except KeyboardInterrupt:
            self.store.log("INFO", "SYSTEM", "Interrupted by operator",
                           self.pair_id)
        except Exception as e:
            self.state = "ERROR"
            self.store.log("CRITICAL", "SYSTEM",
                           f"Engine failed: {type(e).__name__}: {e}",
                           self.pair_id)
            raise
        finally:
            self.shutdown()

    def _connect(self) -> None:
        self.store.connection("DATABASE", "CONNECTED")
        self.store.connection("ENGINE", "CONNECTED",
                              f"pid running, feed={self.feed_kind}, "
                              f"broker={self.broker_kind}")
        self.store.connection("ANALYTICS", "CONNECTED",
                              "computed in-process by the engine")
        try:
            self.source.connect()
            # Report what IBKR actually sends, not what was requested.
            if hasattr(self.source, "detect_data_type"):
                actual = self.source.detect_data_type(self.leg1)
                if actual != self.source.data_source:
                    self.store.log("WARNING", "MARKET_DATA",
                                   f"Feed reports {actual}, not "
                                   f"{self.source.data_source}", self.pair_id)
            self.store.connection("BROKER", "CONNECTED", self.source.describe())
            self.store.connection("MARKET_DATA", "CONNECTED",
                                  getattr(self.source, "latency_note", ""))
            # Feed problems must reach the dashboard, not just the console.
            if hasattr(self.source, "on_status"):
                self.source.on_status = self._on_feed_status
            self._subscribe_ibkr_errors()
            if hasattr(self.source, "on_feed_error"):
                self.source.on_feed_error = self._on_feed_error
            self.store.log("INFO", "SYSTEM",
                           f"Market data connected: {self.source.describe()}",
                           self.pair_id)
        except Exception as e:
            self.store.connection("BROKER", "ERROR", str(e)[:200])
            self.store.connection("MARKET_DATA", "ERROR", str(e)[:200])
            raise

    def _reconcile(self) -> None:
        """Recover any position still running from a previous session.

        A pair position is a live strategy, not a leftover. If the engine
        stopped while the spread had not yet converged, the correct action on
        restart is to resume managing that position — applying the same exit,
        stop-loss and holding-period rules — not to abandon or blindly close
        it.

        Four cases:
          both agree flat          -> nothing to do
          open trade + broker holds -> adopt and continue managing
          open trade, broker flat   -> closed outside the engine; reconcile
          broker holds, no trade    -> genuinely unknown; pause for a human
        """
        trade_row = self.store.open_trade_row(self.pair_id)
        broker_held = (self.broker.broker_positions()
                       if hasattr(self.broker, "broker_positions") else {})
        has_broker_position = any(v != 0 for v in broker_held.values())

        if trade_row is None and not has_broker_position:
            self.store.log("INFO", "SYSTEM",
                           "Startup reconciliation: flat, nothing to recover.",
                           self.pair_id)
            self.store.clear_positions(self.pair_id)
            return

        if trade_row is not None and (has_broker_position
                                      or self.broker_kind == "paper"):
            self.broker.adopt(trade_row)
            self._restore_signal_state(trade_row)
            return

        if trade_row is not None and not has_broker_position:
            self.store.log(
                "WARNING", "RISK",
                f"Trade #{trade_row['id']} is OPEN in the database but the "
                f"broker reports no position. It was closed outside this "
                f"engine. Marking it closed with zero P&L so the record is "
                f"consistent.", self.pair_id)
            self.store.close_trade(trade_row["id"], {
                "status": "CLOSED",
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "exit_reason": "MANUAL", "gross_pnl": 0, "net_pnl": 0,
            })
            self.store.clear_positions(self.pair_id)
            return

        # Broker holds something with no matching trade record.
        self.state = "PAUSED"
        self.store.log(
            "CRITICAL", "RISK",
            f"Broker holds {broker_held} but there is no open trade record. "
            f"The engine cannot compute a cost basis for this and has PAUSED. "
            f"Flatten it or resume deliberately from the Controls page.",
            self.pair_id)

    def _restore_signal_state(self, trade_row: dict) -> None:
        """Reflect an adopted position in the signal shown on the dashboard."""
        self.current_signal = trade_row["direction"]
        entry_ts = datetime.fromisoformat(
            str(trade_row["entry_time"]).replace("Z", "+00:00"))
        self.signal_since = (entry_ts if entry_ts.tzinfo
                             else entry_ts.replace(tzinfo=timezone.utc))

    def _on_feed_error(self, consecutive: int, message: str) -> None:
        """Report feed failures instead of polling silently forever.

        A source that fails quietly is indistinguishable from a quiet market,
        which is the worst possible failure mode for a trading engine.
        """
        self._last_events["feed_error"] = message
        if consecutive == 1:
            self.store.log("WARNING", "MARKET_DATA",
                           f"No bars from the feed: {message}", self.pair_id)
        elif consecutive in (3, 10) or consecutive % 30 == 0:
            self.store.log("ERROR", "MARKET_DATA",
                           f"{consecutive} consecutive feed failures. Latest: "
                           f"{message}", self.pair_id)
            self.store.connection("MARKET_DATA", "ERROR", message[:200])
        if consecutive >= 3 and not self.broker.is_flat:
            self.store.log("WARNING", "RISK",
                           "A position is open while the feed is failing; it "
                           "cannot be evaluated for exit until bars resume.",
                           self.pair_id)

    def _on_feed_status(self, level: str, message: str) -> None:
        self.store.log(level, "MARKET_DATA", message, self.pair_id)
        if level == "ERROR":
            self.store.connection("MARKET_DATA", "ERROR", message[:200])
        elif level == "INFO":
            self.store.connection("MARKET_DATA", "CONNECTED", message[:200])

    def _subscribe_ibkr_errors(self) -> None:
        """Record TWS connectivity events, which explain most feed gaps."""
        ib = getattr(self.source, "_ib", None)
        if ib is None:
            return

        def handler(reqId, code, message, *rest):
            if code == 1100:
                self.store.connection("BROKER", "DISCONNECTED", message[:200])
                self.store.log("ERROR", "SYSTEM",
                               f"[{code}] TWS lost connectivity to IBKR. "
                               f"No bars will arrive until it returns.",
                               self.pair_id)
            elif code in (1101, 1102):
                self.store.connection("BROKER", "CONNECTED", message[:200])
                self.store.log("INFO", "SYSTEM",
                               f"[{code}] TWS connectivity restored.",
                               self.pair_id)
            elif code == 1300:
                self.store.connection("BROKER", "ERROR", message[:200])
                self.store.log("ERROR", "SYSTEM",
                               f"[{code}] TWS socket dropped.", self.pair_id)

        try:
            ib.errorEvent += handler
        except Exception:
            pass

    def _start_background(self) -> None:
        for target, name in ((self._heartbeat_loop, "heartbeat"),
                             (self._command_loop, "commands")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def shutdown(self) -> None:
        self._stop.set()
        if self.state not in ("KILLED", "ERROR"):
            self.state = "STOPPED"
        try:
            self.source.disconnect()
        except Exception:
            pass
        self.store.connection("MARKET_DATA", "DISCONNECTED")
        self.store.connection("BROKER", "DISCONNECTED")
        self.store.connection("ENGINE", "DISCONNECTED", "engine stopped")
        self._write_heartbeat("engine shut down")
        self.store.log("INFO", "SYSTEM",
                       f"Engine stopped. Realized P&L "
                       f"{self.broker.realized_pnl:+.2f}", self.pair_id)

    # --- background threads -------------------------------------------

    def _write_heartbeat(self, detail: str = "") -> None:
        if not detail:
            if self.state == "STARTING":
                have, need = self.model.warmup_progress()
                detail = f"warming up {have}/{need} bars"
            else:
                failures = getattr(self.source, "consecutive_failures", 0)
                if failures:
                    detail = (f"feed failing ({failures} consecutive): "
                              f"{getattr(self.source, 'last_error', '')[:120]}")
                else:
                    detail = getattr(self.source, "latency_note", "")
        self.store.heartbeat(
            status=self.state, data_source=self.source.data_source,
            market_status=market_status(), started_at=self.started_at,
            config_version=self.config.get("version"), detail=detail)

    def _feed_is_stale(self) -> float | None:
        """Seconds since the last bar, or None if none has arrived."""
        last = self._last_events.get("last_market_data_at")
        if last is None:
            return None
        return (datetime.now(timezone.utc) - last).total_seconds()

    def _heartbeat_loop(self) -> None:
        """Liveness is proven here, independently of the trading loop.

        The heartbeat is a small upsert and runs every 2s to stay inside the
        frontend's 5s staleness threshold. Metrics are a wider row and only
        need to move every 10s.
        """
        tick = 0
        while not self._stop.is_set():
            self._write_heartbeat()
            if tick % 5 == 0:
                self.store.metrics(self.started_at, self._last_events)
            tick += 1
            self._stop.wait(HEARTBEAT_SECONDS)

    def _command_loop(self) -> None:
        while not self._stop.is_set():
            for cmd in self.store.poll_commands():
                self._handle_command(cmd)
            self._stop.wait(COMMAND_POLL_SECONDS)

    # --- commands -----------------------------------------------------

    def _handle_command(self, cmd: dict) -> None:
        cid, name = cmd["id"], cmd["command"]
        self.store.mark_command(cid, "RECEIVED")
        self.store.log("INFO", "CONTROL", f"Command received: {name}",
                       self.pair_id)

        valid, reason = self._validate_command(name)
        if not valid:
            self.store.mark_command(cid, "FAILED", error=reason)
            self.store.log("WARNING", "CONTROL",
                           f"Command {name} rejected: {reason}", self.pair_id)
            return

        self.store.mark_command(cid, "ACKNOWLEDGED")
        try:
            result = self._execute_command(name)
            self.store.mark_command(cid, "EXECUTED", result=result)
            self.store.log("INFO", "CONTROL",
                           f"Command {name} executed: {result}", self.pair_id)
        except Exception as e:
            self.store.mark_command(cid, "FAILED", error=str(e)[:300])
            self.store.log("ERROR", "CONTROL",
                           f"Command {name} failed: {e}", self.pair_id)

    def _validate_command(self, name: str) -> tuple[bool, str]:
        if self.state == "KILLED" and name not in ("START", "RESTART"):
            return False, "engine is KILLED; restart required"
        if name == "RESUME" and self.state != "PAUSED":
            return False, f"cannot resume from {self.state}"
        if name == "PAUSE" and self.state not in ("RUNNING", "STARTING"):
            return False, f"cannot pause from {self.state}"
        return True, ""

    def _execute_command(self, name: str) -> dict:
        if name == "PAUSE":
            self.state = "PAUSED"
            return {"state": self.state}

        if name in ("RESUME", "START"):
            self.state = "RUNNING" if self.model.is_ready else "STARTING"
            return {"state": self.state}

        if name == "STOP":
            self.state = "STOPPING"
            self._flatten("MANUAL")
            self.state = "STOPPED"
            self._stop.set()
            return {"state": self.state}

        if name == "RESTART":
            self.restart_requested = True
            self._stop.set()
            return {"restarting": True}

        if name == "CANCEL_ALL_ORDERS":
            if self.broker_kind == "ibkr":
                cancelled = 0
                for order in list(self.source._ib.openOrders()):
                    self.source._ib.cancelOrder(order)
                    cancelled += 1
                return {"cancelled": cancelled}
            return {"cancelled": 0, "note": "simulated fills are immediate"}

        if name == "FLATTEN_ALL_POSITIONS":
            pnl = self._flatten("MANUAL")
            # Also clear anything the broker holds that the engine does not
            # know about, so "flatten all" means exactly that.
            if hasattr(self.broker, "flatten_broker_positions") and self.last_bar:
                self.broker.flatten_broker_positions(self.last_bar)
            return {"flattened": pnl is not None, "net_pnl": pnl}

        if name == "KILL_SWITCH":
            return self._kill()

        if name == "APPLY_CONFIG":
            return self._apply_pending_config()

        raise ValueError(f"unsupported command: {name}")

    def _kill(self) -> dict:
        """Emergency stop, in the order the spec requires."""
        self.state = "STOPPING"
        self.store.log("CRITICAL", "CONTROL",
                       "KILL SWITCH activated", self.pair_id)
        flattened = None
        if self.config.get("kill_switch_flattens", True):
            flattened = self._flatten("KILL_SWITCH")
        self.state = "KILLED"
        self._write_heartbeat("killed by operator")
        return {"state": "KILLED", "flattened": flattened is not None,
                "net_pnl": flattened,
                "positions_left_open": not self.config.get(
                    "kill_switch_flattens", True)}

    def _apply_pending_config(self) -> dict:
        pending = self.store.pending_config(self.pair_id)
        if not pending:
            return {"applied": False, "reason": "no pending configuration"}
        self.store.promote_config(pending["id"], self.config.get("id"))
        self.config = self.store.load_config(self.pair_id)
        self.model_params = ModelParams.from_config_row(self.config)
        self.strategy_params = StrategyParams.from_config_row(self.config)
        self.broker.params = self.strategy_params
        # Lookbacks may have changed, so the window must be rebuilt.
        self.model = PairModel(self.model_params)
        self.state = "STARTING"
        self.store.log("INFO", "CONTROL",
                       f"Applied config v{self.config['version']}; "
                       f"model re-warming", self.pair_id)
        return {"applied": True, "version": self.config["version"]}

    def _flatten(self, reason: str) -> float | None:
        if self.broker.is_flat or self.last_bar is None:
            return None
        return self.broker.exit(self.last_bar, self.last_snapshot, reason,
                                self.last_quotes)

    # --- warmup -------------------------------------------------------

    #: Regular-hours bars per session, by bar size.
    BARS_PER_SESSION = {"1m": 390, "2m": 195, "5m": 78, "5s": 4680, "1s": 23400}

    def _warmup_duration(self, need: int) -> str:
        """Request enough history to satisfy every lookback.

        A fixed duration silently under-fills the window when lookbacks are
        raised, leaving the engine warming up during live hours without
        saying why.
        """
        if self.warmup_lookback != "auto":
            return self.warmup_lookback
        per_session = self.BARS_PER_SESSION.get(self.source.bar_interval)
        if not per_session:
            return "2 D"
        sessions = math.ceil(need / per_session)
        # Calendar days, not sessions: pad for weekends and holidays.
        days = max(2, math.ceil(sessions * 1.6) + 1)
        return f"{min(days, 5)} D"   # IBKR caps 1-minute history at ~1 week

    def _warmup(self) -> None:
        need = self.model_params.required_bars
        self.store.log("INFO", "MODEL",
                       f"Warming up: need {need} bars at "
                       f"{self.source.bar_interval}. Daily backfill parameters "
                       f"do not transfer to this timescale.", self.pair_id)

        interval_seconds = {"1m": 60, "5s": 5, "2s": 2, "1s": 1}.get(
            self.source.bar_interval)
        if interval_seconds:
            minutes = need * interval_seconds / 60
            if minutes > 30:
                self.store.log(
                    "WARNING", "MODEL",
                    f"Warmup needs {need} bars at {self.source.bar_interval} "
                    f"= {minutes:.0f} minutes of data. Lookbacks in "
                    f"strategy_config are counted in BARS and were set for "
                    f"daily bars; consider reducing them for this feed.",
                    self.pair_id)

        if hasattr(self.source, "warmup"):
            duration = self._warmup_duration(need)
            try:
                quotes = self.source.warmup(self.leg1, self.leg2, duration)
                self.model.warmup([(q.ts, q.leg1.price, q.leg2.price)
                                   for q in quotes])
                sessions = len({q.ts.date() for q in quotes})
                self.store.log(
                    "INFO", "MODEL",
                    f"Loaded {len(quotes)} historical bars over {sessions} "
                    f"session(s) from {duration}; model has "
                    f"{self.model.bars}/{need}", self.pair_id)
                if sessions > 1:
                    self.store.log(
                        "INFO", "MODEL",
                        "Window spans a session boundary: the overnight gap "
                        "appears as a single-bar move in the spread and can "
                        "inflate the z-score at the open.", self.pair_id)
            except Exception as e:
                self.store.log("WARNING", "MODEL",
                               f"Historical warmup failed ({e}); will warm up "
                               f"from the live feed", self.pair_id)

        if self.model.is_ready:
            snap = self.model.snapshot()
            self.state = "RUNNING"
            self.store.log("INFO", "MODEL",
                           f"Warmup complete. beta={snap.hedge_ratio:.6f} "
                           f"spread={snap.spread:+.4f} "
                           f"mean={snap.spread_mean:+.4f} "
                           f"std={snap.spread_std:.4f} z={snap.zscore:+.3f} "
                           f"health={snap.health}", self.pair_id)
        else:
            have, _ = self.model.warmup_progress()
            self.store.log("WARNING", "MODEL",
                           f"Only {have}/{need} bars available; continuing to "
                           f"warm up on live data. No signals until ready.",
                           self.pair_id)

    # --- main loop ----------------------------------------------------

    def _run_loop(self) -> None:
        for quote in self.source.stream(self.leg1, self.leg2):
            if self._stop.is_set():
                break
            try:
                self._on_bar(quote)
            except Exception as e:
                self.store.log("ERROR", "MODEL",
                               f"Bar processing failed: {type(e).__name__}: {e}",
                               self.pair_id)

    def _on_bar(self, quote: PairQuote) -> None:
        now = datetime.now(timezone.utc)
        if not self._first_bar_seen:
            self._first_bar_seen = True
            lag = (now - quote.ts).total_seconds()
            self.store.log(
                "INFO" if lag < 300 else "WARNING", "MARKET_DATA",
                f"First bar received: stamped {quote.ts:%H:%M:%S} UTC, "
                f"{lag / 60:.1f} min behind now. "
                f"{quote.leg1.ticker}={quote.leg1.price:.4f} "
                f"{quote.leg2.ticker}={quote.leg2.price:.4f}", self.pair_id)
        self._last_events["last_market_data_at"] = now
        self.broker.roll_session(self.last_bar)

        self.model.add(quote.ts, quote.leg1.price, quote.leg2.price)
        snapshot = self.model.snapshot()
        if snapshot is None:
            return
        self._last_events["last_model_calc_at"] = now
        self.last_snapshot = snapshot

        self.last_quotes = {
            "leg1_bid": quote.leg1.bid, "leg1_ask": quote.leg1.ask,
            "leg2_bid": quote.leg2.bid, "leg2_ask": quote.leg2.ask,
        }

        bar = Bar(
            ts=snapshot.ts, leg1_price=snapshot.leg1_price,
            leg2_price=snapshot.leg2_price, spread=snapshot.spread,
            zscore=snapshot.zscore, hedge_ratio=snapshot.hedge_ratio,
            spread_mean=snapshot.spread_mean, spread_std=snapshot.spread_std,
            correlation=snapshot.correlation,
            cointegration_pvalue=snapshot.cointegration_pvalue,
            half_life=snapshot.half_life)
        self.last_bar = bar

        # Leave STARTING only once every lookback is satisfied.
        if self.state == "STARTING" and self.model.is_ready:
            self.state = "RUNNING"
            self.store.log("INFO", "MODEL",
                           f"Warmup complete on live data "
                           f"({self.model.bars} bars); now RUNNING",
                           self.pair_id)

        signal = self._evaluate(bar, snapshot)
        self._persist(bar, snapshot, signal)

    def _near_close(self, now: datetime | None = None) -> bool:
        """True inside the no-new-positions window before the closing bell."""
        if self.flatten_before_close_minutes <= 0:
            return False
        now = now or datetime.now(timezone.utc)
        close = now.replace(hour=RTH_CLOSE_UTC.hour,
                            minute=RTH_CLOSE_UTC.minute,
                            second=0, microsecond=0)
        remaining = (close - now).total_seconds() / 60
        return 0 <= remaining <= self.flatten_before_close_minutes

    def _evaluate(self, bar: Bar, snapshot: ModelSnapshot) -> str:
        """Decide and act. Returns the signal label for persistence."""
        # A halted broker means the account is in an unknown state. Retrying
        # can only compound it, so stop rather than trade on.
        if getattr(self.broker, "halted", False) and self.state == "RUNNING":
            self.state = "PAUSED"
            self.store.log(
                "CRITICAL", "RISK",
                f"Engine PAUSED: {self.broker.halt_reason}. Close the "
                f"position manually in TWS, then Resume from Controls.",
                self.pair_id)
            return "NO_SIGNAL"

        if self.state != "RUNNING":
            return "NO_SIGNAL"

        outside_hours = (not self.trade_outside_rth
                         and market_status() != "MARKET_OPEN")
        if outside_hours and self.broker.is_flat:
            # No entries outside regular hours: spreads are wide and prints
            # unreliable, so an entry signal here is not trustworthy.
            return "NO_SIGNAL"
        if outside_hours:
            self.broker.mark_to_market(bar)
            # An open position is still managed: if the spread has moved far
            # enough to justify an exit or a stop, act on it.
            decision = decide(bar, self.broker.position, self.strategy_params)
            if decision.action != Action.EXIT:
                return self.current_signal

        # Close before the bell. An intraday mean-reversion position carried
        # overnight takes gap risk the strategy is not compensated for.
        if self._near_close() and not self.broker.is_flat:
            self.broker.exit(bar, snapshot, "END_OF_DAY", self.last_quotes)
            self.store.log("INFO", "ORDER",
                           f"Flattened before the close "
                           f"({self.flatten_before_close_minutes}m to 20:00 "
                           f"UTC)", self.pair_id)
            self.current_signal = "NO_SIGNAL"
            return "EXIT"

        if not outside_hours:
            decision = decide(bar, self.broker.position, self.strategy_params)
        previous = self.current_signal

        if decision.action == Action.HOLD:
            if not self.broker.is_flat:
                self.broker.mark_to_market(bar)
            return self.current_signal if not self.broker.is_flat else "NO_SIGNAL"

        if decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
            if self._near_close():
                return "NO_SIGNAL"      # too close to the bell to open

            age = (datetime.now(timezone.utc) - bar.ts).total_seconds()
            if self.max_data_age_seconds and age > self.max_data_age_seconds:
                self.store.write_signal(
                    self.pair_id, bar.ts,
                    "LONG_SPREAD" if decision.action == Action.ENTER_LONG
                    else "SHORT_SPREAD", previous, snapshot, decision.reason,
                    False, self.config["version"],
                    f"data {age / 60:.1f} min old, above the "
                    f"{self.max_data_age_seconds / 60:.1f} min limit")
                self.store.log(
                    "WARNING", "RISK",
                    f"Entry suppressed: the bar is {age / 60:.1f} min old and "
                    f"the spread half-life is "
                    f"{(snapshot.half_life or 0):.1f} bars, so this signal "
                    f"describes a dislocation that has already closed.",
                    self.pair_id)
                return "NO_SIGNAL"
            direction = ("LONG_SPREAD" if decision.action == Action.ENTER_LONG
                         else "SHORT_SPREAD")
            data_age = (datetime.now(timezone.utc) - bar.ts).total_seconds()
            opened = self.broker.enter(direction, bar, snapshot,
                                       self.last_quotes)
            if opened:
                self.store.log(
                    "INFO", "ORDER",
                    f"Entered on data {data_age / 60:.1f} min old "
                    f"(z={snapshot.zscore:+.3f} as of "
                    f"{bar.ts:%H:%M:%S} UTC)", self.pair_id)
            self.current_signal = direction if opened else "NO_SIGNAL"
            self.signal_since = datetime.now(timezone.utc)
            self.store.write_signal(
                self.pair_id, bar.ts, direction, previous, snapshot,
                decision.reason, opened, self.config["version"],
                None if opened else "position sizing produced zero quantity")
            self._last_events["last_order_event_at"] = datetime.now(timezone.utc)
            return self.current_signal

        if decision.action == Action.EXIT:
            cause = (decision.cause or ExitCause.MEAN_REVERSION).value
            self.broker.exit(bar, snapshot, cause, self.last_quotes)
            self.current_signal = "EXIT"
            self.signal_since = datetime.now(timezone.utc)
            self.store.write_signal(
                self.pair_id, bar.ts, "EXIT", previous, snapshot,
                decision.reason, True, self.config["version"])
            self._last_events["last_order_event_at"] = datetime.now(timezone.utc)
            self.current_signal = "NO_SIGNAL"
            return "EXIT"

        return "NO_SIGNAL"

    # --- persistence --------------------------------------------------

    def _persist(self, bar: Bar, snapshot: ModelSnapshot, signal: str) -> None:
        cfg = self.config
        thresholds = {
            "entry_threshold": cfg["entry_threshold"],
            "exit_threshold": cfg["exit_threshold"],
            "stop_loss_threshold": cfg["stop_loss_threshold"],
        }
        self.store.write_bar(
            self.pair_id, snapshot, self.leg1, self.leg2,
            self.source.bar_interval, self.source.data_source,
            cfg["version"], signal, thresholds)

        unrealized = self.broker.unrealized(bar)
        gross, net = self.broker.exposure(bar)
        capital = float(cfg["capital_allocation"])
        total = self.broker.realized_pnl + unrealized
        position_side = "FLAT"
        if self.broker.position is not None:
            position_side = ("LONG" if self.broker.position.direction
                             == "LONG_SPREAD" else "SHORT")

        now = datetime.now(timezone.utc).isoformat()
        self.store.upsert_live_state({
            "pair_id": self.pair_id, "as_of": now,
            "leg1_price": round(snapshot.leg1_price, 6),
            "leg2_price": round(snapshot.leg2_price, 6),
            "leg1_bid": self.last_quotes.get("leg1_bid"),
            "leg1_ask": self.last_quotes.get("leg1_ask"),
            "leg2_bid": self.last_quotes.get("leg2_bid"),
            "leg2_ask": self.last_quotes.get("leg2_ask"),
            "last_market_data_at": bar.ts.isoformat(),
            "spread": snapshot.spread, "hedge_ratio": snapshot.hedge_ratio,
            "zscore": snapshot.zscore, "spread_mean": snapshot.spread_mean,
            "spread_std": snapshot.spread_std,
            "spread_volatility": snapshot.spread_volatility,
            "correlation": snapshot.correlation,
            "rolling_correlation": snapshot.rolling_correlation,
            "cointegration_stat": snapshot.cointegration_stat,
            "cointegration_pvalue": snapshot.cointegration_pvalue,
            "half_life": snapshot.half_life,
            "health": snapshot.health, "health_reason": snapshot.health_reason,
            "last_model_update_at": now,
            "current_signal": signal,
            "signal_since": (self.signal_since.isoformat()
                             if self.signal_since else None),
            "signal_explanation": self._explain(bar, snapshot),
            "current_position": position_side,
            "target_position": position_side,
            **thresholds,
            "total_pnl": round(total, 4),
            "daily_pnl": round(self.broker.daily_pnl_at(bar), 4),
            "realized_pnl": round(self.broker.realized_pnl, 4),
            "unrealized_pnl": round(unrealized, 4),
            "total_return_pct": (total / capital * 100) if capital else 0.0,
            "daily_return_pct": ((self.broker.daily_pnl_at(bar) / capital * 100)
                                 if capital else 0.0),
            "current_exposure": round(gross, 4),
            "capital_utilisation": (gross / capital * 100) if capital else 0.0,
            "config_version": cfg["version"],
        })
        self._last_events["last_state_write_at"] = datetime.now(timezone.utc)

    def _explain(self, bar: Bar, snapshot: ModelSnapshot) -> str:
        """Plain-English state for the Strategy page."""
        pair = f"{self.leg1}/{self.leg2}"
        z, entry = snapshot.zscore, self.strategy_params.entry_threshold

        if self.state == "STARTING":
            have, need = self.model.warmup_progress()
            return (f"Model is warming up on {self.source.bar_interval} bars "
                    f"({have}/{need}). No signals are generated until every "
                    f"lookback is satisfied.")
        if self.state != "RUNNING":
            return f"Engine is {self.state}. No new signals will be generated."
        if not self.trade_outside_rth and market_status() != "MARKET_OPEN":
            return (f"Market is {market_status().replace('_', ' ').lower()}. "
                    f"State is being tracked but no trades will be placed.")

        if self.broker.position is not None:
            pos = self.broker.position
            held = (datetime.now(timezone.utc) - pos.entry_ts).total_seconds()
            limit = self.strategy_params.max_holding_period_seconds
            remaining = (f", {(limit - held) / 86400:.1f} days before the "
                         f"holding limit" if limit else "")
            return (f"Holding {pos.direction} on {pair}, opened "
                    f"{held / 86400:.1f} days ago{remaining}. Z-score is "
                    f"{z:+.2f}; the position closes when it reverts inside "
                    f"±{self.strategy_params.exit_threshold:g} or breaches "
                    f"±{self.strategy_params.stop_loss_threshold:g}. It is "
                    f"held across sessions while the entry thesis holds.")

        if snapshot.health != "VALID":
            return (f"Entry is blocked: {snapshot.health_reason}. Z-score is "
                    f"{z:+.2f} but the relationship must be valid before a "
                    f"position is opened.")

        if abs(z) >= entry:
            side = "below" if z < 0 else "above"
            return (f"{pair} spread is {abs(z):.2f} standard deviations {side} "
                    f"its mean, exceeding the ±{entry:g} entry threshold.")
        return (f"{pair} spread is {abs(z):.2f} standard deviations from its "
                f"mean, inside the ±{entry:g} entry threshold. No entry "
                f"condition is met.")


def run(pair_id: int = 1, feed: str = "poll", **kwargs) -> bool:
    """Run one engine session. Returns True if a restart was requested."""
    engine = Engine(pair_id=pair_id, feed=feed, **kwargs)

    def handle_signal(signum, frame):
        engine._stop.set()

    for sig in (os_signal.SIGINT, os_signal.SIGTERM):
        try:
            os_signal.signal(sig, handle_signal)
        except (ValueError, AttributeError):
            pass

    engine.start()
    return engine.restart_requested