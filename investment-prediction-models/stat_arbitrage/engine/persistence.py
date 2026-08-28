"""Supabase persistence for the engine.

The engine is the only writer to every table here except `commands`, which
the frontend writes and the engine acknowledges.

Connects with the service role, which bypasses RLS. That key must never
reach the frontend.

Writes are best-effort and never raise into the trading loop: a database
outage must not stop the engine from managing an open position. Failures are
counted and surfaced through the heartbeat instead.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from supabase import create_client


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """All engine database access."""

    def __init__(self, engine_id: str | None = None) -> None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY required")
        self._url, self._key = url, key
        # The engine runs three threads (main loop, heartbeat, commands) and a
        # single Supabase client shares one httpx connection pool between them.
        # Concurrent use produces "Server disconnected", so each thread gets
        # its own client.
        self._local = threading.local()
        self.engine_id = engine_id or f"engine-{socket.gethostname()}"
        self.write_failures = 0
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}

    @property
    def db(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = create_client(self._url, self._key)
            self._local.client = client
        return client

    def _reset_client(self) -> None:
        """Drop this thread's client so the next call reconnects."""
        self._local.client = None

    # --- helpers ------------------------------------------------------

    #: Errors worth one silent retry before being reported.
    TRANSIENT = ("server disconnected", "connection reset", "timed out",
                 "connection aborted", "remote end closed", "eof occurred")

    def _safe(self, label: str, fn, *args, **kwargs):
        """Run a write, absorbing failures so the trading loop survives.

        Retries once on a transient connection error with a fresh client,
        because a dropped keepalive should not be reported as a failure.
        """
        for attempt in (1, 2):
            try:
                result = fn(*args, **kwargs)
                if attempt == 2:
                    print(f"  [store] {label} recovered on retry")
                return result
            except Exception as e:
                message = str(e).lower()
                if attempt == 1 and any(t in message for t in self.TRANSIENT):
                    self._reset_client()
                    time.sleep(0.4)
                    continue
                with self._lock:
                    self.write_failures += 1
                print(f"  [store] {label} failed: {str(e)[:160]}")
                return None

    def bump(self, counter: str, n: int = 1) -> None:
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + n

    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    # --- configuration ------------------------------------------------

    def load_pair(self, pair_id: int) -> dict:
        rows = self.db.table("pairs").select("*").eq("id", pair_id).execute().data
        if not rows:
            raise RuntimeError(f"no pair with id={pair_id}")
        return rows[0]

    def load_config(self, pair_id: int) -> dict:
        rows = (self.db.table("strategy_config").select("*")
                .eq("pair_id", pair_id).eq("status", "CURRENT")
                .execute().data)
        if not rows:
            raise RuntimeError(f"no CURRENT config for pair {pair_id}")
        return rows[0]

    def pending_config(self, pair_id: int) -> dict | None:
        rows = (self.db.table("strategy_config").select("*")
                .eq("pair_id", pair_id).eq("status", "PENDING")
                .execute().data)
        return rows[0] if rows else None

    def promote_config(self, config_id: int, current_id: int | None) -> None:
        """Apply a pending configuration. Only the engine may do this."""
        if current_id is not None:
            self._safe("archive config", lambda: self.db.table("strategy_config")
                       .update({"status": "ARCHIVED"})
                       .eq("id", current_id).execute())
        self._safe("promote config", lambda: self.db.table("strategy_config")
                   .update({"status": "CURRENT", "applied_at": _now(),
                            "applied_by": self.engine_id})
                   .eq("id", config_id).execute())

    # --- heartbeat and health -----------------------------------------

    def heartbeat(self, status: str, data_source: str, market_status: str,
                  started_at: datetime, config_version: int | None,
                  detail: str = "") -> None:
        self._safe("heartbeat", lambda: self.db.table("engine_heartbeat").upsert({
            "engine_id": self.engine_id, "status": status,
            "data_source": data_source, "market_status": market_status,
            "heartbeat_at": _now(), "started_at": started_at.isoformat(),
            "active_config_version": config_version, "detail": detail,
            "host": socket.gethostname(),
        }).execute())

    def connection(self, component: str, state: str, detail: str = "") -> None:
        self._safe("connection", lambda: self.db.table("connection_status").upsert({
            "component": component, "state": state, "updated_at": _now(),
            "last_ok_at": _now() if state == "CONNECTED" else None,
            "detail": detail,
        }).execute())

    def metrics(self, session_started: datetime, last: dict[str, Any]) -> None:
        c = self.counters()
        self._safe("metrics", lambda: self.db.table("system_metrics").upsert({
            "engine_id": self.engine_id, "as_of": _now(),
            "session_started_at": session_started.isoformat(),
            "market_data_events": c.get("market_data", 0),
            "model_calculations": c.get("model_calcs", 0),
            "signals_generated": c.get("signals", 0),
            "orders_submitted": c.get("orders", 0),
            "orders_filled": c.get("fills", 0),
            "state_writes": c.get("state_writes", 0),
            "error_count": c.get("errors", 0) + self.write_failures,
            "warning_count": c.get("warnings", 0),
            **{k: (v.isoformat() if isinstance(v, datetime) else v)
               for k, v in last.items() if v is not None},
        }).execute())

    # --- logging ------------------------------------------------------

    def log(self, level: str, category: str, message: str,
            pair_id: int | None = None, details: dict | None = None) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  {stamp} {level:<7} {category:<12} {message}")
        if level in ("ERROR", "CRITICAL"):
            self.bump("errors")
        elif level == "WARNING":
            self.bump("warnings")
        self._safe("log", lambda: self.db.table("system_logs").insert({
            "ts": _now(), "engine_id": self.engine_id, "pair_id": pair_id,
            "level": level, "category": category, "message": message,
            "details": details,
        }).execute())

    # --- market data and model state ----------------------------------

    def write_bar(self, pair_id: int, snapshot, leg1: str, leg2: str,
                  bar_interval: str, source: str, config_version: int,
                  signal: str, thresholds: dict) -> None:
        """One bar: two market_data rows and one model_state_history row."""
        ts = snapshot.ts.isoformat()
        self._safe("market_data", lambda: self.db.table("market_data").insert([
            {"pair_id": pair_id, "ticker": leg1, "ts": ts,
             "price": round(snapshot.leg1_price, 6), "field": "MID",
             "source": source, "bar_interval": bar_interval},
            {"pair_id": pair_id, "ticker": leg2, "ts": ts,
             "price": round(snapshot.leg2_price, 6), "field": "MID",
             "source": source, "bar_interval": bar_interval},
        ]).execute())

        self._safe("model_state", lambda: self.db.table("model_state_history")
                   .insert({
                       "pair_id": pair_id, "ts": ts,
                       "leg1_price": round(snapshot.leg1_price, 6),
                       "leg2_price": round(snapshot.leg2_price, 6),
                       "spread": snapshot.spread, "zscore": snapshot.zscore,
                       "hedge_ratio": snapshot.hedge_ratio,
                       "spread_mean": snapshot.spread_mean,
                       "spread_std": snapshot.spread_std,
                       "correlation": snapshot.correlation,
                       "cointegration_pvalue": snapshot.cointegration_pvalue,
                       "half_life": snapshot.half_life,
                       "health": snapshot.health, "signal": signal,
                       "config_version": config_version,
                       "bar_interval": bar_interval,
                       **thresholds,
                   }).execute())
        self.bump("market_data", 2)
        self.bump("model_calcs")

    def upsert_live_state(self, row: dict) -> None:
        self._safe("live_state", lambda: self.db.table("live_state")
                   .upsert(row, on_conflict="pair_id").execute())
        self.bump("state_writes")

    # --- signals ------------------------------------------------------

    def write_signal(self, pair_id: int, ts: datetime, signal: str,
                     previous: str | None, snapshot, explanation: str,
                     acted: bool, config_version: int,
                     suppressed: str | None = None) -> None:
        self._safe("signal", lambda: self.db.table("signals").insert({
            "pair_id": pair_id, "ts": ts.isoformat(), "signal": signal,
            "previous_signal": previous, "zscore": snapshot.zscore,
            "hedge_ratio": snapshot.hedge_ratio, "health": snapshot.health,
            "explanation": explanation, "acted_upon": acted,
            "suppressed_reason": suppressed, "config_version": config_version,
        }).execute())
        self.bump("signals")

    # --- execution ----------------------------------------------------

    def open_trade(self, row: dict) -> int | None:
        res = self._safe("open trade", lambda: self.db.table("trades")
                         .insert(row).execute())
        return res.data[0]["id"] if res and res.data else None

    def close_trade(self, trade_id: int, row: dict) -> None:
        self._safe("close trade", lambda: self.db.table("trades")
                   .update(row).eq("id", trade_id).execute())

    def write_order(self, row: dict) -> int | None:
        res = self._safe("order", lambda: self.db.table("orders")
                         .insert(row).execute())
        self.bump("orders")
        return res.data[0]["id"] if res and res.data else None

    def write_fill(self, row: dict) -> None:
        self._safe("fill", lambda: self.db.table("fills").insert(row).execute())
        self.bump("fills")

    def upsert_position(self, row: dict) -> None:
        self._safe("position", lambda: self.db.table("positions")
                   .upsert(row, on_conflict="pair_id,ticker").execute())

    def clear_positions(self, pair_id: int) -> None:
        self._safe("clear positions", lambda: self.db.table("positions")
                   .delete().eq("pair_id", pair_id).execute())

    # --- commands -----------------------------------------------------

    def poll_commands(self) -> list[dict]:
        """Commands awaiting the engine, oldest first.

        Rows past `expires_at` are marked EXPIRED and never executed. A
        control request queued minutes or days ago does not reflect what the
        operator wants now, and acting on it would be worse than ignoring it.
        """
        try:
            rows = (self.db.table("commands").select("*")
                    .eq("status", "REQUESTED")
                    .order("requested_at").limit(20).execute().data or [])
        except Exception as e:
            message = str(e).lower()
            if any(t in message for t in self.TRANSIENT):
                self._reset_client()
            with self._lock:
                self.write_failures += 1
            return []

        now = datetime.now(timezone.utc)
        live = []
        for row in rows:
            expires = row.get("expires_at")
            if expires:
                deadline = datetime.fromisoformat(
                    str(expires).replace("Z", "+00:00"))
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                if now > deadline:
                    self.mark_command(
                        row["id"], "EXPIRED",
                        error=f"not collected before {deadline.isoformat()}")
                    print(f"  [store] expired stale command "
                          f"{row['command']} from {row.get('requested_at')}")
                    continue
            live.append(row)
        return live

    def mark_command(self, command_id: str, status: str,
                     result: dict | None = None, error: str | None = None) -> None:
        field = {"RECEIVED": "received_at", "ACKNOWLEDGED": "acknowledged_at",
                 "EXECUTED": "executed_at", "FAILED": "executed_at"}.get(status)
        payload: dict[str, Any] = {"status": status, "engine_id": self.engine_id}
        if field:
            payload[field] = _now()
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error_message"] = error
        self._safe("command update", lambda: self.db.table("commands")
                   .update(payload).eq("id", command_id).execute())