"""Run the trading engine.

    python scripts/run_engine.py                    # IBKR 1-min bars (default)
    python scripts/run_engine.py --feed sim         # simulated, offline
    python scripts/run_engine.py --feed stream      # delayed IBKR ticks
    python scripts/run_engine.py --slippage-bps 0.5 --commission 0.005

The engine keeps running when the frontend is closed. Stop it with Ctrl-C,
the Controls page, or the kill switch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from engine.runner import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-id", type=int, default=1)
    ap.add_argument("--feed", choices=["poll", "stream", "sim"],
                    default="poll",
                    help="poll: IBKR 1-min bars, current (default). "
                         "stream: IBKR delayed ticks, ~15 min behind. "
                         "sim: synthetic, no broker needed.")
    ap.add_argument("--broker", choices=["paper", "ibkr"], default="paper",
                    help="paper: fills simulated locally (default). "
                         "ibkr: real orders sent to the connected TWS "
                         "account. Which account that is — paper or live — "
                         "depends entirely on which TWS session is logged in.")
    ap.add_argument("--max-data-age", type=float, default=0.0, metavar="MIN",
                    help="Refuse new entries when the newest bar is older "
                         "than this many minutes. Without a market data "
                         "subscription IBKR bars run ~16 min late, which is "
                         "many half-lives for this strategy. 0 disables the "
                         "guard (default), which is appropriate only when "
                         "validating the pipeline rather than the edge.")
    ap.add_argument("--limit-orders", action="store_true",
                    help="Use marketable limit orders instead of market "
                         "orders when routing to IBKR.")
    ap.add_argument("--flatten-before-close", type=int, default=0,
                    metavar="MIN",
                    help="Optional: close positions this many minutes before "
                         "20:00 UTC. Default 0 (disabled) — positions are held "
                         "across sessions while the entry thesis holds, "
                         "bounded by max_holding_period_seconds.")
    ap.add_argument("--slippage-bps", type=float, default=0.0,
                    help="Slippage applied to every fill, in bps.")
    ap.add_argument("--commission", type=float, default=0.0,
                    help="Commission per share.")
    ap.add_argument("--trade-outside-rth", action="store_true",
                    help="Trade outside 13:30-20:00 UTC. Off by default: "
                         "spreads widen badly outside regular hours.")
    ap.add_argument("--warmup", default="auto",
                    help="Historical duration for model warmup. 'auto' sizes "
                         "it from the configured lookbacks and bar interval; "
                         "or give an IBKR duration such as '3 D'.")
    args = ap.parse_args()

    print("=" * 62)
    print(f"  STAT ARB ENGINE   pair={args.pair_id}  feed={args.feed}  "
          f"broker={args.broker}")
    if args.broker == "ibkr":
        print("  REAL ORDERS will be sent to the connected TWS account.")
        print("  Verify TWS is logged into the PAPER account before trading.")
    print("=" * 62)

    attempt = 0
    while True:
        attempt += 1
        try:
            restart = run(
                pair_id=args.pair_id, feed=args.feed, broker=args.broker,
                max_data_age_seconds=args.max_data_age * 60,
                limit_orders=args.limit_orders,
                flatten_before_close_minutes=args.flatten_before_close,
                slippage_bps=args.slippage_bps,
                commission_per_share=args.commission,
                trade_outside_rth=args.trade_outside_rth,
                warmup_lookback=args.warmup)
        except KeyboardInterrupt:
            print("\nStopped by operator.")
            return
        except Exception as e:
            print(f"\nEngine crashed: {type(e).__name__}: {e}")
            if attempt >= 3:
                print("Three consecutive failures; not restarting.")
                raise
            print(f"Restarting in 10s (attempt {attempt + 1}/3)...")
            time.sleep(10)
            continue

        if not restart:
            return
        print("\nRestart requested; reconnecting in 3s...")
        attempt = 0
        time.sleep(3)


if __name__ == "__main__":
    main()