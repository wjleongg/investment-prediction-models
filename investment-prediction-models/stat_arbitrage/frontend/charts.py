"""Plotly chart builders. Each takes typed models and returns a Figure."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from contract.models import EquityPoint, ModelStatePoint, RollingDiagnostic, Trade
from theme import ACCENT, BORDER, MUTED, NEG, PLOTLY_LAYOUT, POS, TEXT, WARN


def _fig(height: int = 320, **kw) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**{**PLOTLY_LAYOUT, "height": height, **kw})
    return fig


def _hline(fig: go.Figure, y: float, colour: str, label: str,
           dash: str = "dash") -> None:
    fig.add_hline(y=y, line=dict(color=colour, width=1, dash=dash),
                  annotation_text=label, annotation_position="right",
                  annotation_font=dict(size=9, color=colour))


def _trade_markers(fig: go.Figure, trades: Sequence[Trade], y_key: str) -> None:
    """Overlay entry/exit markers. y_key is 'entry_zscore' or a price field."""
    entries = [(t.entry_time, getattr(t, "entry_zscore", None)) for t in trades]
    exits = [(t.exit_time, getattr(t, "exit_zscore", None))
             for t in trades if t.exit_time]
    entries = [(x, y) for x, y in entries if x and y is not None]
    exits = [(x, y) for x, y in exits if x and y is not None]
    if entries:
        fig.add_trace(go.Scatter(
            x=[e[0] for e in entries], y=[e[1] for e in entries],
            mode="markers", name="Entry",
            marker=dict(symbol="triangle-up", size=9, color=ACCENT,
                        line=dict(width=1, color=TEXT))))
    if exits:
        fig.add_trace(go.Scatter(
            x=[e[0] for e in exits], y=[e[1] for e in exits],
            mode="markers", name="Exit",
            marker=dict(symbol="x", size=8, color=WARN)))


# ---------------------------------------------------------------------
# Live charts
# ---------------------------------------------------------------------


def spread_chart(points: Sequence[ModelStatePoint],
                 trades: Sequence[Trade] = (), height: int = 380) -> go.Figure:
    fig = _fig(height, yaxis_title="Spread")
    if not points:
        return fig
    x = [p.ts for p in points]
    fig.add_trace(go.Scatter(x=x, y=[p.spread for p in points], name="Spread",
                             line=dict(color=ACCENT, width=1.4)))

    mean = [p.spread_mean for p in points]
    if any(m is not None for m in mean):
        fig.add_trace(go.Scatter(x=x, y=mean, name="Mean",
                                 line=dict(color=MUTED, width=1, dash="dash")))
        # Entry bands expressed in spread units: mean ± k·sigma
        last = points[-1]
        k = last.entry_threshold
        if k and any(p.spread_std for p in points):
            upper = [(m + k * s) if (m is not None and s) else None
                     for m, s in zip(mean, [p.spread_std for p in points])]
            lower = [(m - k * s) if (m is not None and s) else None
                     for m, s in zip(mean, [p.spread_std for p in points])]
            fig.add_trace(go.Scatter(x=x, y=upper, name=f"+{k:g}σ entry",
                                     line=dict(color=NEG, width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=x, y=lower, name=f"−{k:g}σ entry",
                                     line=dict(color=POS, width=1, dash="dot"),
                                     fill="tonexty",
                                     fillcolor="rgba(88,166,255,0.04)"))
    return fig


def zscore_chart(points: Sequence[ModelStatePoint],
                 trades: Sequence[Trade] = (), height: int = 300) -> go.Figure:
    fig = _fig(height, yaxis_title="Z-score")
    if not points:
        return fig
    x = [p.ts for p in points]
    fig.add_trace(go.Scatter(x=x, y=[p.zscore for p in points], name="Z-score",
                             line=dict(color=ACCENT, width=1.4)))
    last = points[-1]
    _hline(fig, 0, MUTED, "mean", "solid")
    if last.entry_threshold:
        _hline(fig, last.entry_threshold, NEG, f"+{last.entry_threshold:g} entry")
        _hline(fig, -last.entry_threshold, POS, f"−{last.entry_threshold:g} entry")
    if last.exit_threshold:
        _hline(fig, last.exit_threshold, MUTED, "exit", "dot")
        _hline(fig, -last.exit_threshold, MUTED, "exit", "dot")
    if last.stop_loss_threshold:
        _hline(fig, last.stop_loss_threshold, WARN, "stop", "longdash")
        _hline(fig, -last.stop_loss_threshold, WARN, "stop", "longdash")
    _trade_markers(fig, trades, "entry_zscore")
    return fig


# ---------------------------------------------------------------------
# Research charts
# ---------------------------------------------------------------------


def price_chart(df: pd.DataFrame, mode: str = "Normalised",
                height: int = 340) -> go.Figure:
    fig = _fig(height, yaxis_title=mode)
    if df.empty:
        return fig
    plot = df.copy()
    if mode == "Normalised":
        plot = plot / plot.iloc[0] * 100
    elif mode == "Log":
        plot = np.log(plot)
    for i, col in enumerate(plot.columns):
        fig.add_trace(go.Scatter(x=plot.index, y=plot[col], name=col,
                                 line=dict(color=[ACCENT, WARN][i % 2], width=1.3)))
    return fig


def rolling_line(points: Sequence[RollingDiagnostic], attr: str, label: str,
                 threshold: float | None = None, height: int = 280,
                 invert_pass: bool = False) -> go.Figure:
    fig = _fig(height, yaxis_title=label)
    pts = [(p.ts, getattr(p, attr)) for p in points if getattr(p, attr) is not None]
    if not pts:
        return fig
    fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts],
                             name=label, line=dict(color=ACCENT, width=1.3)))
    if threshold is not None:
        _hline(fig, threshold, WARN, f"{threshold:g}")
    return fig


def histogram(values: Sequence[float], label: str, height: int = 280,
              bins: int = 40) -> go.Figure:
    fig = _fig(height, xaxis_title=label, yaxis_title="Count", hovermode="closest")
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return fig
    colours = [POS if v >= 0 else NEG for v in vals]
    fig.add_trace(go.Histogram(x=vals, nbinsx=bins,
                               marker=dict(color=ACCENT, line=dict(width=0))))
    fig.add_vline(x=float(np.mean(vals)), line=dict(color=WARN, width=1,
                                                    dash="dash"),
                  annotation_text="mean", annotation_font=dict(size=9))
    return fig


# ---------------------------------------------------------------------
# Performance charts
# ---------------------------------------------------------------------


def equity_chart(points: Sequence[EquityPoint], show_benchmarks: bool = True,
                 leg1: str = "Leg 1", leg2: str = "Leg 2",
                 height: int = 360) -> go.Figure:
    fig = _fig(height, yaxis_title="Cumulative return %")
    if not points:
        return fig
    x = [p.ts for p in points]
    fig.add_trace(go.Scatter(x=x, y=[p.cumulative_return_pct for p in points],
                             name="Strategy", line=dict(color=ACCENT, width=1.6)))
    if show_benchmarks:
        if any(p.leg1_cumulative_return_pct is not None for p in points):
            fig.add_trace(go.Scatter(
                x=x, y=[p.leg1_cumulative_return_pct for p in points],
                name=leg1, line=dict(color=MUTED, width=1, dash="dash")))
        if any(p.leg2_cumulative_return_pct is not None for p in points):
            fig.add_trace(go.Scatter(
                x=x, y=[p.leg2_cumulative_return_pct for p in points],
                name=leg2, line=dict(color=BORDER, width=1, dash="dot")))
    return fig


def drawdown_chart(points: Sequence[EquityPoint], height: int = 260) -> go.Figure:
    fig = _fig(height, yaxis_title="Drawdown %")
    if not points:
        return fig
    fig.add_trace(go.Scatter(
        x=[p.ts for p in points], y=[p.drawdown_pct for p in points],
        name="Drawdown", line=dict(color=NEG, width=1.2),
        fill="tozeroy", fillcolor="rgba(248,81,73,0.12)"))
    return fig


def mini_trade_chart(points: Sequence[ModelStatePoint], trade: Trade,
                     height: int = 220) -> go.Figure:
    """Spread and z-score across a single trade's lifetime."""
    fig = _fig(height, yaxis_title="Z-score")
    if not points:
        return fig
    fig.add_trace(go.Scatter(x=[p.ts for p in points],
                             y=[p.zscore for p in points],
                             name="Z-score", line=dict(color=ACCENT, width=1.4)))
    _hline(fig, 0, MUTED, "", "solid")
    if trade.entry_time and trade.entry_zscore is not None:
        fig.add_trace(go.Scatter(x=[trade.entry_time], y=[trade.entry_zscore],
                                 mode="markers", name="Entry",
                                 marker=dict(symbol="triangle-up", size=11,
                                             color=ACCENT)))
    if trade.exit_time and trade.exit_zscore is not None:
        fig.add_trace(go.Scatter(x=[trade.exit_time], y=[trade.exit_zscore],
                                 mode="markers", name="Exit",
                                 marker=dict(symbol="x", size=10, color=WARN)))
    return fig
