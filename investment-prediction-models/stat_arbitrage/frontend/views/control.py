"""Configuration and Controls pages.

Both write to Supabase, and both are careful about it: configuration changes
land as PENDING and never touch live parameters, and every control command is
reported by its engine-acknowledged lifecycle rather than by insert success.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import components as c
import data
from contract.enums import CommandStatus, CommandType
from contract.models import COMMAND_CONFIRM_TIMEOUT_SECONDS, HeaderState, diff_configs


# =====================================================================
# CONFIGURATION
# =====================================================================


def configuration(header: HeaderState) -> None:
    pair = header.pair
    current = data.fetch_config(pair.id, "CURRENT")
    pending = data.fetch_config(pair.id, "PENDING")

    if current is None:
        c.banner("No CURRENT configuration for this pair.", "bad")
        return

    # --- Current vs pending ------------------------------------------
    c.section("Configuration state")
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown(c.card("CURRENT CONFIGURATION", f"v{current.version}",
                           f"applied {c.ts_full(current.applied_at)}", "v-pos"),
                    unsafe_allow_html=True)
    with b:
        if pending:
            st.markdown(c.card("PENDING CONFIGURATION", f"v{pending.version}",
                               f"proposed {c.ts_full(pending.created_at)}",
                               "v-neu"), unsafe_allow_html=True)
        else:
            st.markdown(c.card("PENDING CONFIGURATION", "NONE",
                               "no changes awaiting approval"),
                        unsafe_allow_html=True)

    if pending:
        diffs = diff_configs(current, pending)
        if diffs:
            st.dataframe(pd.DataFrame([{
                "Parameter": d.field_name.replace("_", " "),
                "Current": d.current_value, "Pending": d.pending_value,
            } for d in diffs]), use_container_width=True, hide_index=True)
        c.banner("A pending configuration exists. The engine promotes it to "
                 "CURRENT — this frontend cannot apply it directly."
                 + (" Restart required before it takes effect."
                    if pending.requires_restart else ""), "warn")

    # --- Editor -------------------------------------------------------
    c.section("Propose changes")
    st.caption("Edits are submitted as a PENDING configuration. Live strategy "
               "parameters are never changed silently.")

    pairs = data.fetch_pairs()
    with st.form("config_form"):
        st.markdown("**Pair**")
        p1, p2 = st.columns(2)
        pair_choice = p1.selectbox(
            "Trading pair", pairs, index=pairs.index(
                next(p for p in pairs if p.id == current.pair_id)),
            format_func=lambda p: p.label)
        p2.caption("Pairs are database rows. Add one with an INSERT into "
                   "`pairs` — no code change is required.")

        st.markdown("**Lookbacks (bars)**")
        l1, l2, l3, l4 = st.columns(4)
        historical = l1.number_input("Historical", 100, 20000,
                                     current.historical_lookback, 10)
        zscore_lb = l2.number_input("Z-score", 5, 2000,
                                    current.zscore_lookback, 5)
        corr_lb = l3.number_input("Correlation", 5, 2000,
                                  current.correlation_lookback, 5)
        coint_lb = l4.number_input("Cointegration", 20, 5000,
                                   current.cointegration_lookback, 10)

        st.markdown("**Signal thresholds (absolute z-score)**")
        t1, t2, t3, t4 = st.columns(4)
        entry = t1.number_input("Entry", 0.1, 10.0, current.entry_threshold, 0.05)
        exit_t = t2.number_input("Exit", 0.0, 10.0, current.exit_threshold, 0.05)
        stop = t3.number_input("Stop-loss", 0.1, 20.0,
                               current.stop_loss_threshold, 0.05)
        max_hold = t4.number_input("Max holding (s)", 0, 2_592_000,
                                   current.max_holding_period_seconds or 0, 3600)

        st.markdown("**Relationship gates**")
        g1, g2 = st.columns(2)
        min_corr = g1.number_input("Minimum correlation", -1.0, 1.0,
                                   current.min_correlation, 0.01)
        max_p = g2.number_input("Maximum cointegration p-value", 0.001, 0.999,
                                current.max_cointegration_pvalue, 0.005,
                                format="%.3f")

        st.markdown("**Position and risk**")
        r1, r2, r3 = st.columns(3)
        capital = r1.number_input("Capital allocation", 0.0, 1e9,
                                  float(current.capital_allocation), 1000.0)
        max_pos = r2.number_input("Max position size", 0.0, 1e9,
                                  float(current.max_position_size), 1000.0)
        max_exp = r3.number_input("Max pair exposure", 0.0, 1e9,
                                  float(current.max_pair_exposure), 1000.0)
        r4, r5, r6 = st.columns(3)
        max_lev = r4.number_input("Max leverage", 0.1, 10.0,
                                  current.max_leverage, 0.1)
        max_sim = r5.number_input("Max simultaneous positions", 1, 50,
                                  current.max_simultaneous_positions)
        kill_flat = r6.checkbox("Kill switch flattens positions",
                                current.kill_switch_flattens)

        submitted = st.form_submit_button("Submit as PENDING configuration")

    if submitted:
        if not (exit_t < entry < stop):
            st.error("Threshold ordering violated: exit < entry < stop-loss. "
                     "The database enforces this too.")
            return
        payload = {
            "version": current.version + 1, "pair_id": pair_choice.id,
            "historical_lookback": int(historical),
            "zscore_lookback": int(zscore_lb),
            "correlation_lookback": int(corr_lb),
            "cointegration_lookback": int(coint_lb),
            "entry_threshold": float(entry), "exit_threshold": float(exit_t),
            "stop_loss_threshold": float(stop),
            "max_holding_period_seconds": int(max_hold) or None,
            "min_correlation": float(min_corr),
            "max_cointegration_pvalue": float(max_p),
            "capital_allocation": float(capital),
            "max_position_size": float(max_pos),
            "max_pair_exposure": float(max_exp),
            "max_leverage": float(max_lev),
            "max_simultaneous_positions": int(max_sim),
            "kill_switch_flattens": bool(kill_flat),
            "created_by": st.session_state.get("user_email", "streamlit"),
            "requires_restart": True,
        }
        ok, msg = data.propose_config(payload)
        if ok:
            st.success(f"{msg} The engine must promote it to CURRENT.")
            data.clear_caches()
        else:
            st.error(f"Rejected: {msg}")

    # --- History ------------------------------------------------------
    c.section("Configuration history")
    history = data.fetch_config_history(pair.id)
    if history:
        st.dataframe(pd.DataFrame([{
            "Version": h.version, "Status": h.status.value,
            "Entry": h.entry_threshold, "Exit": h.exit_threshold,
            "Stop": h.stop_loss_threshold, "Capital": float(h.capital_allocation),
            "Created": c.ts_full(h.created_at), "By": h.created_by or "—",
            "Applied": c.ts_full(h.applied_at),
        } for h in history]), use_container_width=True, hide_index=True)


# =====================================================================
# CONTROLS
# =====================================================================

CONTROL_BUTTONS = [
    (CommandType.START, "Start Strategy"),
    (CommandType.PAUSE, "Pause Strategy"),
    (CommandType.RESUME, "Resume Strategy"),
    (CommandType.RESTART, "Restart Strategy"),
    (CommandType.STOP, "Stop Strategy"),
]

ORDER_BUTTONS = [
    (CommandType.CANCEL_ALL_ORDERS, "Cancel All Orders"),
    (CommandType.FLATTEN_ALL_POSITIONS, "Flatten All Positions"),
]


def controls(header: HeaderState) -> None:
    pair = header.pair
    hb = header.heartbeat
    now = datetime.now(timezone.utc)

    if hb is None or hb.is_dead(now):
        c.banner("✕ THE ENGINE IS NOT RESPONDING. Commands issued now will be "
                 "recorded but cannot be acknowledged, and must be treated as "
                 "NOT EXECUTED.", "bad")

    c.section("Strategy controls")
    cols = st.columns(len(CONTROL_BUTTONS))
    for col, (cmd, label) in zip(cols, CONTROL_BUTTONS):
        if col.button(label, key=f"btn_{cmd.value}", use_container_width=True):
            _request(cmd, pair.id, confirm=cmd.is_destructive)

    c.section("Order controls")
    cols = st.columns(2)
    for col, (cmd, label) in zip(cols, ORDER_BUTTONS):
        if col.button(label, key=f"btn_{cmd.value}", use_container_width=True):
            _request(cmd, pair.id, confirm=True)

    # --- Kill switch, deliberately separated -------------------------
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='border:1px solid #f8514955;background:#f851490f;"
        "border-radius:4px;padding:1rem'>", unsafe_allow_html=True)
    st.markdown("#### ⚠ EMERGENCY KILL SWITCH")
    cfg = data.fetch_config(pair.id)
    st.caption(
        "Stops signal generation, cancels outstanding orders, "
        + ("flattens open positions, " if cfg and cfg.kill_switch_flattens
           else "leaves positions open (per configuration), ")
        + "moves the engine to KILLED and records the event.")

    confirm = st.text_input("Type KILL to confirm", key="kill_confirm",
                            placeholder="KILL")
    st.markdown('<div class="danger">', unsafe_allow_html=True)
    if st.button("ACTIVATE KILL SWITCH", key="btn_kill",
                 use_container_width=True):
        if confirm.strip().upper() != "KILL":
            st.error("Confirmation text does not match. Nothing was sent.")
        else:
            _request(CommandType.KILL_SWITCH, pair.id, confirm=False)
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.caption("The database is the audit and fallback path. Once the engine "
               "exposes its control endpoint, that becomes the primary route "
               "and this remains the backup.")

    # --- Lifecycle of the last command -------------------------------
    last_id = st.session_state.get("last_command_id")
    if last_id:
        c.section("Last command")
        _render_lifecycle(last_id)

    c.section("Recent commands")
    cmds = data.fetch_recent_commands(15)
    if cmds:
        st.dataframe(pd.DataFrame([{
            "Command": x.command.value,
            "Status": ("UNCONFIRMED" if x.is_unconfirmed_and_expired(now)
                       else x.status.value),
            "Requested": c.ts_full(x.requested_at),
            "Acknowledged": c.ts_full(x.acknowledged_at),
            "Executed": c.ts_full(x.executed_at),
            "By": x.requested_by or "—",
            "Error": x.error_message or "",
        } for x in cmds]), use_container_width=True, hide_index=True)
    else:
        c.empty_state("commands")


def _request(cmd: CommandType, pair_id: int, confirm: bool) -> None:
    if confirm:
        key = f"pending_confirm_{cmd.value}"
        if not st.session_state.get(key):
            st.session_state[key] = True
            st.warning(f"{cmd.value.replace('_', ' ')} is destructive. "
                       f"Press the button again within this session to confirm.")
            return
        st.session_state[key] = False

    ok, result = data.issue_command(
        cmd.value, pair_id,
        requested_by=st.session_state.get("user_email", "streamlit"))
    if not ok:
        st.error(f"Command was not recorded: {result}")
        return
    st.session_state["last_command_id"] = result
    st.info(f"{cmd.value} RECORDED — awaiting engine acknowledgement. "
            f"This is not confirmation of execution.")
    data.fetch_recent_commands.clear()


def _render_lifecycle(command_id: str) -> None:
    """Follow a command until it settles.

    Polls to EXECUTED / FAILED / EXPIRED rather than stopping at the first
    status change. The engine moves through RECEIVED and ACKNOWLEDGED in well
    under a second, so breaking early renders a stale intermediate state and
    makes a completed command look stuck.
    """
    placeholder = st.empty()
    deadline = time.time() + COMMAND_CONFIRM_TIMEOUT_SECONDS
    cmd = None

    while time.time() < deadline:
        cmd = data.fetch_command(command_id)
        if cmd is not None:
            placeholder.markdown(_lifecycle_chips(cmd), unsafe_allow_html=True)
            if cmd.status.is_settled:
                break
        time.sleep(0.5)

    if cmd is None:
        placeholder.error("Command row could not be read back.")
        return

    placeholder.markdown(_lifecycle_chips(cmd), unsafe_allow_html=True)
    now = datetime.now(timezone.utc)

    if cmd.status == CommandStatus.FAILED:
        c.banner(f"✕ ENGINE REPORTED FAILURE: {cmd.error_message or 'no detail'}",
                 "bad")
    elif cmd.status == CommandStatus.EXPIRED:
        c.banner(f"⚠ COMMAND EXPIRED — the engine did not collect it before "
                 f"it timed out. It was NOT executed.", "bad")
    elif cmd.status == CommandStatus.EXECUTED:
        detail = ""
        if cmd.result:
            detail = " · " + ", ".join(f"{k}={v}" for k, v in cmd.result.items())
        c.banner(f"✓ ENGINE EXECUTED{detail}", "ok")
    elif cmd.is_unconfirmed_and_expired(now):
        c.banner(f"⚠ NOT CONFIRMED BY ENGINE after "
                 f"{COMMAND_CONFIRM_TIMEOUT_SECONDS:.0f}s. The command may not "
                 f"have executed. Verify engine state before acting on this.",
                 "bad")
    elif cmd.status == CommandStatus.ACKNOWLEDGED:
        c.banner("⧗ Engine acknowledged the command and is executing it.",
                 "warn")
    else:
        c.banner("⧗ Awaiting engine acknowledgement — not yet confirmed.",
                 "warn")

    data.fetch_recent_commands.clear()


def _lifecycle_chips(cmd) -> str:
    """REQUESTED -> RECEIVED -> ACKNOWLEDGED -> EXECUTED as timestamped chips."""
    stages = [("REQUESTED", cmd.requested_at), ("RECEIVED", cmd.received_at),
              ("ACKNOWLEDGED", cmd.acknowledged_at),
              ("EXECUTED", cmd.executed_at)]
    tone_done = "bad" if cmd.status in (CommandStatus.FAILED,
                                        CommandStatus.EXPIRED) else "ok"
    return " ".join(
        c.pill(f"{name} {c.ts(t) if t else '—'}", tone_done if t else "mute")
        for name, t in stages)