"""Step 5B — verify the anon/publishable key under an authenticated session.

Confirms four things:
  1. The operator account can sign in.
  2. Authenticated reads work (RLS lets them through).
  3. The frontend CAN insert a REQUESTED command.
  4. The frontend CANNOT write to engine-owned tables.

Run:  python test_auth_connection.py
"""

import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_ANON_KEY"],
)

passed, failed = [], []


def check(label: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


# --- 1. Sign in -------------------------------------------------------
try:
    session = supabase.auth.sign_in_with_password({
        "email": os.environ["OPERATOR_EMAIL"],
        "password": os.environ["OPERATOR_PASSWORD"],
    })
    check("sign in", session.user is not None, session.user.email)
except Exception as e:
    check("sign in", False, str(e))
    raise SystemExit("Cannot continue without a session.")

# --- 2. Authenticated reads -------------------------------------------
try:
    pairs = supabase.table("pairs").select("*").execute().data
    check("read pairs", len(pairs) == 1, pairs[0]["label"] if pairs else "no rows")
except Exception as e:
    check("read pairs", False, str(e))

for table in ("live_state", "strategy_config", "trades", "system_logs", "commands"):
    try:
        rows = supabase.table(table).select("*").limit(5).execute().data
        check(f"read {table}", True, f"{len(rows)} row(s)")
    except Exception as e:
        check(f"read {table}", False, str(e))

# --- 3. Allowed write: a REQUESTED command ----------------------------
command_id = None
try:
    inserted = supabase.table("commands").insert({
        "command": "PAUSE",
        "status": "REQUESTED",
        "pair_id": 1,
        "requested_by": "step-5b-test",
    }).execute().data
    command_id = inserted[0]["id"] if inserted else None
    check("insert REQUESTED command", command_id is not None, command_id or "")
except Exception as e:
    check("insert REQUESTED command", False, str(e))

# --- 4. Forbidden writes ----------------------------------------------
try:
    supabase.table("commands").insert({
        "command": "KILL_SWITCH",
        "status": "EXECUTED",          # policy allows REQUESTED only
        "requested_by": "step-5b-test",
    }).execute()
    check("block pre-executed command", False, "insert was allowed")
except Exception:
    check("block pre-executed command", True, "rejected as expected")

try:
    supabase.table("live_state").update({"zscore": 99.0}).eq("pair_id", 1).execute()
    check("block live_state write", False, "update was allowed")
except Exception:
    check("block live_state write", True, "rejected as expected")

try:
    supabase.table("trades").insert({
        "pair_id": 1, "direction": "LONG_SPREAD",
        "status": "OPEN", "entry_time": "2026-01-01T00:00:00Z",
    }).execute()
    check("block trades write", False, "insert was allowed")
except Exception:
    check("block trades write", True, "rejected as expected")

supabase.auth.sign_out()

print(f"\n{len(passed)} passed, {len(failed)} failed")
if command_id:
    print(f"\nClean up the test command in the SQL editor:\n"
          f"  delete from commands where id = '{command_id}';")
if failed:
    raise SystemExit(1)
