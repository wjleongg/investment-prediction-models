"""Step 6 — backfill historical pair statistics into Supabase.

Pulls daily history for the active pair, computes the same quantities the live
engine will compute, and writes them to market_data, model_state_history,
cointegration_results and rolling_diagnostics.

This is deliberately pair-agnostic: the tickers come from the `pairs` row and
every lookback comes from the CURRENT strategy_config. Nothing is hard-coded.

Does NOT write trades, equity_curve or performance_metrics. Those require
running the signal logic, which belongs to the engine. Backtesting comes next.

Usage:
    python scripts/backfill.py                 # uses config lookbacks
    python scripts/backfill.py --years 5       # override history length
    python scripts/backfill.py --pair-id 1
    python scripts/backfill.py --dry-run       # compute, print, write nothing
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from supabase import create_client

try:
    from arch.unitroot import PhillipsPerron
    HAS_ARCH = True
except ImportError:  # pragma: no cover
    HAS_ARCH = False

load_dotenv()

# KPSS reports when the statistic falls outside its p-value lookup table.
# Expected on strongly stationary spreads; the pass/fail verdict is unaffected.
warnings.filterwarnings("ignore", message=".*p-value.*", category=UserWarning)
try:
    from statsmodels.tools.sm_exceptions import InterpolationWarning
    warnings.filterwarnings("ignore", category=InterpolationWarning)
except ImportError:  # pragma: no cover
    pass

CHUNK = 500
MARKET_CLOSE_UTC = 20  # ~16:00 US/Eastern


# =====================================================================
# Statistics — mirrors what the live engine must compute
# =====================================================================


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS slope of y on x, with intercept."""
    model = sm.OLS(y.values, sm.add_constant(x.values)).fit()
    return float(model.params[1])


def rolling_hedge_ratio(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def half_life(spread: pd.Series) -> float | None:
    """AR(1) mean-reversion half-life. None if the series isn't reverting."""
    s = spread.dropna()
    if len(s) < 30:
        return None
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    aligned = pd.concat([delta, lagged], axis=1).dropna()
    if aligned.empty:
        return None
    model = sm.OLS(aligned.iloc[:, 0].values,
                   sm.add_constant(aligned.iloc[:, 1].values)).fit()
    lam = float(model.params[1])
    if lam >= 0:
        return None  # diverging, no half-life
    return float(-np.log(2) / lam)


def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float | None:
    """Hurst via the variance-of-lagged-differences method.

    < 0.5 mean-reverting, ~0.5 random walk, > 0.5 trending.
    """
    s = series.dropna().values
    if len(s) < max_lag * 2:
        max_lag = max(10, len(s) // 4)
    if len(s) < 20:
        return None
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        std = np.std(diff)
        tau.append(std if std > 0 else np.nan)
    tau = np.array(tau)
    valid = ~np.isnan(tau)
    if valid.sum() < 5:
        return None
    poly = np.polyfit(np.log(np.array(list(lags))[valid]), np.log(tau[valid]), 1)
    return float(poly[0])


def run_cointegration_tests(y: pd.Series, x: pd.Series, spread: pd.Series,
                            lookback: int) -> list[dict]:
    """Engle-Granger, ADF, Phillips-Perron, Johansen, KPSS."""
    y_w, x_w = y.tail(lookback), x.tail(lookback)
    spread_w = spread.tail(lookback).dropna()
    results: list[dict] = []

    # Engle-Granger on the price pair
    try:
        stat, pval, crit = coint(y_w, x_w)
        results.append({
            "test": "ENGLE_GRANGER", "statistic": float(stat), "pvalue": float(pval),
            "critical_values": {"1%": float(crit[0]), "5%": float(crit[1]),
                                "10%": float(crit[2])},
            "passed": bool(pval < 0.05),
            "interpretation": ("Cointegration supported at 5%." if pval < 0.05
                               else "No cointegration at 5%."),
        })
    except Exception as e:
        results.append({"test": "ENGLE_GRANGER", "interpretation": f"Failed: {e}"})

    # ADF on the spread — stationary spread is the property we actually trade
    try:
        stat, pval, _, _, crit, _ = adfuller(spread_w, autolag="AIC")
        results.append({
            "test": "ADF", "statistic": float(stat), "pvalue": float(pval),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "passed": bool(pval < 0.05),
            "interpretation": ("Spread is stationary at 5%." if pval < 0.05
                               else "Cannot reject a unit root in the spread."),
        })
    except Exception as e:
        results.append({"test": "ADF", "interpretation": f"Failed: {e}"})

    # Phillips-Perron on the spread
    if HAS_ARCH:
        try:
            pp = PhillipsPerron(spread_w)
            results.append({
                "test": "PHILLIPS_PERRON", "statistic": float(pp.stat),
                "pvalue": float(pp.pvalue),
                "critical_values": {k: float(v) for k, v in pp.critical_values.items()},
                "passed": bool(pp.pvalue < 0.05),
                "interpretation": ("Spread is stationary at 5%." if pp.pvalue < 0.05
                                   else "Cannot reject a unit root in the spread."),
            })
        except Exception as e:
            results.append({"test": "PHILLIPS_PERRON", "interpretation": f"Failed: {e}"})
    else:
        results.append({
            "test": "PHILLIPS_PERRON",
            "interpretation": "Skipped — install `arch` to enable this test.",
        })

    # Johansen trace test — no p-value, compare stat to critical values
    try:
        data = pd.concat([y_w, x_w], axis=1).dropna()
        jres = coint_johansen(data.values, det_order=0, k_ar_diff=1)
        trace_stat = float(jres.lr1[0])
        crit_95 = float(jres.cvt[0, 1])
        results.append({
            "test": "JOHANSEN", "statistic": trace_stat, "pvalue": None,
            "critical_values": {"90%": float(jres.cvt[0, 0]), "95%": crit_95,
                                "99%": float(jres.cvt[0, 2])},
            "passed": bool(trace_stat > crit_95),
            "interpretation": (
                f"Trace statistic {trace_stat:.2f} "
                f"{'exceeds' if trace_stat > crit_95 else 'is below'} "
                f"the 95% critical value {crit_95:.2f}."
            ),
        })
    except Exception as e:
        results.append({"test": "JOHANSEN", "interpretation": f"Failed: {e}"})

    # KPSS — null hypothesis is stationarity, so the logic inverts
    try:
        stat, pval, _, crit = kpss(spread_w, regression="c", nlags="auto")
        results.append({
            "test": "KPSS", "statistic": float(stat), "pvalue": float(pval),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "passed": bool(pval > 0.05),
            "interpretation": ("Stationarity not rejected." if pval > 0.05
                               else "Stationarity rejected at 5%."),
        })
    except Exception as e:
        results.append({"test": "KPSS", "interpretation": f"Failed: {e}"})

    return results


def compute_rolling_diagnostics(y: pd.Series, x: pd.Series, spread: pd.Series,
                                window: int, step: int = 5) -> pd.DataFrame:
    """Rolling cointegration and stability, stepped to keep runtime sane."""
    rows = []
    for i in range(window, len(y), step):
        y_w, x_w = y.iloc[i - window:i], x.iloc[i - window:i]
        s_w = spread.iloc[i - window:i].dropna()
        if len(s_w) < window // 2:
            continue
        try:
            stat, pval, _ = coint(y_w, x_w)
        except Exception:
            stat, pval = np.nan, np.nan
        rows.append({
            "ts": y.index[i - 1],
            "window_bars": window,
            "rolling_correlation": float(y_w.corr(x_w)),
            "rolling_hedge_ratio": hedge_ratio(y_w, x_w),
            "rolling_cointegration_stat": None if np.isnan(stat) else float(stat),
            "rolling_cointegration_pvalue": None if np.isnan(pval) else float(pval),
            "rolling_spread_volatility": float(s_w.std()),
            "half_life": half_life(s_w),
            "hurst_exponent": hurst_exponent(s_w),
            "relationship_valid": bool(not np.isnan(pval) and pval < 0.05),
        })
    return pd.DataFrame(rows)


# =====================================================================
# Database
# =====================================================================


def connect():
    return create_client(os.environ["SUPABASE_URL"],
                         os.environ["SUPABASE_SERVICE_KEY"])


def load_pair_and_config(db, pair_id: int) -> tuple[dict, dict]:
    pair = db.table("pairs").select("*").eq("id", pair_id).execute().data
    if not pair:
        sys.exit(f"No pair with id={pair_id}.")
    cfg = (db.table("strategy_config").select("*")
           .eq("status", "CURRENT").eq("pair_id", pair_id).execute().data)
    if not cfg:
        sys.exit(f"No CURRENT strategy_config for pair {pair_id}.")
    return pair[0], cfg[0]


def insert_chunked(db, table: str, rows: list[dict]) -> int:
    for i in range(0, len(rows), CHUNK):
        db.table(table).insert(rows[i:i + CHUNK]).execute()
    return len(rows)


def clear_pair_rows(db, table: str, pair_id: int) -> None:
    db.table(table).delete().eq("pair_id", pair_id).execute()


def iso(ts) -> str:
    dt = pd.Timestamp(ts).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(hour=MARKET_CLOSE_UTC, tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-id", type=int, default=1)
    ap.add_argument("--years", type=float, default=None,
                    help="Override history length (default: from config lookback).")
    ap.add_argument("--rolling-step", type=int, default=5,
                    help="Bars between rolling diagnostic points.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = connect()
    pair, cfg = load_pair_and_config(db, args.pair_id)
    t1, t2 = pair["leg1_ticker"], pair["leg2_ticker"]
    print(f"Pair {pair['id']}: {t1} / {t2}   config v{cfg['version']}")

    years = args.years or (cfg["historical_lookback"] / 252.0)
    start = datetime.now(timezone.utc) - timedelta(days=int(years * 365.25))
    print(f"Downloading {years:.1f}y of daily history from {start.date()}...")

    raw = yf.download([t1, t2], start=start.date(), auto_adjust=True,
                      progress=False, group_by="column")
    if raw.empty:
        sys.exit("yfinance returned no data. Check the tickers.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = close[[t1, t2]].dropna()
    if len(prices) < cfg["cointegration_lookback"]:
        sys.exit(f"Only {len(prices)} bars; need at least "
                 f"{cfg['cointegration_lookback']}.")
    print(f"{len(prices)} aligned bars: {prices.index[0].date()} → "
          f"{prices.index[-1].date()}")

    y, x = prices[t1], prices[t2]

    # --- Model series -------------------------------------------------
    beta = hedge_ratio(y, x)
    spread = y - beta * x
    z_lb = cfg["zscore_lookback"]
    spread_mean = spread.rolling(z_lb).mean()
    spread_std = spread.rolling(z_lb).std()
    zscore = (spread - spread_mean) / spread_std
    corr = y.rolling(cfg["correlation_lookback"]).corr(x)
    roll_beta = rolling_hedge_ratio(y, x, cfg["correlation_lookback"])
    hl = half_life(spread)

    print(f"\nStatic hedge ratio : {beta:.6f}")
    print(f"Spread mean / std  : {spread.mean():.4f} / {spread.std():.4f}")
    print(f"Half-life          : {'n/a' if hl is None else f'{hl:.1f} bars'}")
    print(f"Hurst              : {hurst_exponent(spread):.4f}")
    print(f"Latest z-score     : {zscore.dropna().iloc[-1]:.4f}")

    # --- Cointegration ------------------------------------------------
    print("\nCointegration tests:")
    tests = run_cointegration_tests(y, x, spread, cfg["cointegration_lookback"])
    for t in tests:
        p = t.get("pvalue")
        print(f"  {t['test']:<16} p={'n/a' if p is None else f'{p:.4f}'}  "
              f"{t.get('interpretation', '')}")

    # --- Rolling ------------------------------------------------------
    print(f"\nComputing rolling diagnostics (step={args.rolling_step})...")
    rolling = compute_rolling_diagnostics(
        y, x, spread, cfg["cointegration_lookback"], args.rolling_step)
    print(f"  {len(rolling)} points")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return

    # --- Write --------------------------------------------------------
    pid = pair["id"]
    now = datetime.now(timezone.utc).isoformat()
    print("\nWriting to Supabase...")

    for table in ("market_data", "model_state_history",
                  "cointegration_results", "rolling_diagnostics"):
        clear_pair_rows(db, table, pid)

    md_rows = []
    for ticker in (t1, t2):
        for ts, price in prices[ticker].items():
            md_rows.append({
                "pair_id": pid, "ticker": ticker, "ts": iso(ts),
                "price": round(float(price), 6), "field": "CLOSE",
                "source": "HISTORICAL_DATA",
            })
    print(f"  market_data          {insert_chunked(db, 'market_data', md_rows)}")

    ms_rows = []
    for ts in prices.index:
        if pd.isna(zscore.get(ts)):
            continue
        ms_rows.append({
            "pair_id": pid, "ts": iso(ts),
            "leg1_price": round(float(y[ts]), 6),
            "leg2_price": round(float(x[ts]), 6),
            "spread": float(spread[ts]),
            "zscore": float(zscore[ts]),
            "hedge_ratio": None if pd.isna(roll_beta.get(ts)) else float(roll_beta[ts]),
            "spread_mean": float(spread_mean[ts]),
            "spread_std": float(spread_std[ts]),
            "correlation": None if pd.isna(corr.get(ts)) else float(corr[ts]),
            "half_life": hl,
            "entry_threshold": cfg["entry_threshold"],
            "exit_threshold": cfg["exit_threshold"],
            "stop_loss_threshold": cfg["stop_loss_threshold"],
            "health": "VALID",
            "signal": "NO_SIGNAL",
            "config_version": cfg["version"],
        })
    print(f"  model_state_history  "
          f"{insert_chunked(db, 'model_state_history', ms_rows)}")

    ct_rows = [{
        "pair_id": pid, "as_of": now, "test": t["test"],
        "lookback_bars": cfg["cointegration_lookback"],
        "statistic": t.get("statistic"), "pvalue": t.get("pvalue"),
        "critical_values": t.get("critical_values"), "passed": t.get("passed"),
        "interpretation": t.get("interpretation"),
    } for t in tests]
    print(f"  cointegration_results "
          f"{insert_chunked(db, 'cointegration_results', ct_rows)}")

    rd_rows = []
    for r in rolling.to_dict("records"):
        r["pair_id"] = pid
        r["ts"] = iso(r["ts"])
        rd_rows.append({k: (None if isinstance(v, float) and np.isnan(v) else v)
                        for k, v in r.items()})
    print(f"  rolling_diagnostics  "
          f"{insert_chunked(db, 'rolling_diagnostics', rd_rows)}")

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
