"""Narrative insights over pre-computed analytics.

Provider-agnostic: Gemini 2.5 Flash by default (free tier, for development),
Anthropic available via a one-line toggle for live use.

Design constraint that matters: the model receives a fact pack of finished
numbers and is explicitly forbidden from computing new ones. A language model
asked to derive a Sharpe ratio will produce a confident, plausible, wrong one.
Every figure in the output must trace back to engine/analytics.py.

If no API key is configured, a deterministic rule-based summary is used
instead, so the panel always says something true.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import requests

Provider = Literal["gemini", "anthropic", "none"]

GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/{version}"
GEMINI_VERSIONS = ("v1beta", "v1")
# Preference order when falling back to a discovered model. Newest first:
# older models remain listed by the API long after they stop accepting new
# users, so a listing is not evidence that a model is callable.
GEMINI_FALLBACKS = ("gemini-3.7-flash", "gemini-3.6-flash",
                    "gemini-3.5-flash", "gemini-3-flash",
                    "gemini-flash-latest", "gemini-2.5-flash")

ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

TIMEOUT = 60
GEMINI_MAX_TOKENS = 4000
ANTHROPIC_MAX_TOKENS = 2000


def _redact(text: str) -> str:
    """Strip anything key-shaped out of error text before it reaches a UI."""
    import re
    text = re.sub(r"[?&]key=[^\s&]+", "?key=REDACTED", text)
    text = re.sub(r"\b(AIza|AQ\.)[A-Za-z0-9_\-.]{10,}", "REDACTED", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_\-]{10,}", "REDACTED", text)
    return text


SYSTEM_PROMPT = """\
You are a quantitative analyst reviewing a statistical arbitrage strategy.

You will receive a JSON fact pack of ALREADY-COMPUTED figures.

Absolute rules:
1. Use ONLY numbers present in the fact pack. Never compute, estimate, derive
   or infer a figure that is not explicitly given. If a number you want is
   absent, say it is not available rather than producing one.
2. Never round in a way that changes meaning, and never restate a number
   inaccurately.
3. The `deterministic_warnings` field contains checks that have already been
   verified as true. Treat them as established fact and address them.
4. Distinguish backtest results from live performance. The `data_basis` field
   tells you which this is. Never describe a backtest as realised performance.
5. Be direct about weaknesses. A strategy with a high Sharpe on negligible
   returns is not a good strategy, and near-zero drawdown usually means
   near-zero risk taken rather than skilful risk management. Say so.
6. If transaction cost sensitivity shows the edge disappearing, that is the
   single most important finding. Lead with it.

Write in four short sections with these exact headings:

WHAT HAPPENED
Two or three sentences on the headline result.

WHAT'S DRIVING IT
Where the P&L actually came from, using the attribution figures. Note whether
returns are broad-based or concentrated.

CONCERNS
The problems. Be specific and quantitative. Cover every deterministic warning.

WHAT TO CHECK NEXT
Two to four concrete, actionable next steps.

Total under 350 words. Plain prose, no markdown bullets or bold. Write for
someone who already understands quantitative finance — no definitions.
"""


# =====================================================================
# Provider selection
# =====================================================================


def resolve_provider(preferred: str | None = None) -> Provider:
    """Pick a provider based on preference and which keys are configured."""
    pref = (preferred or os.environ.get("LLM_PROVIDER") or "gemini").lower()
    if pref == "anthropic" and _key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if pref == "gemini" and _key("GEMINI_API_KEY"):
        return "gemini"
    # Fall back to whichever key exists
    if _key("GEMINI_API_KEY"):
        return "gemini"
    if _key("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def _key(name: str) -> str | None:
    """Streamlit secrets first (cloud), then environment (local)."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def available_providers() -> list[str]:
    out = []
    if _key("GEMINI_API_KEY"):
        out.append("gemini")
    if _key("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    return out


# =====================================================================
# Generation
# =====================================================================


RESEARCH_PROMPT = """\
You are a quantitative researcher assessing whether a pair of securities is
suitable for statistical arbitrage.

You will receive a JSON fact pack of ALREADY-COMPUTED figures.

Absolute rules:
1. Use ONLY numbers present in the fact pack. Never compute, estimate or
   infer a figure that is not given. If something is missing, say so.
2. `deterministic_warnings` are checks already verified as true. Treat them
   as established fact.
3. The single most important question is whether the spread's typical move is
   larger than the cost of trading it. A pair can be perfectly cointegrated
   and completely untradeable. If `tradeability` shows costs exceeding the
   gross move, that is the finding — lead with it and do not bury it under
   the statistical results.
4. Distinguish correlation from cointegration explicitly. Correlation is
   about co-movement of changes; cointegration is about levels staying
   tethered. Only the second is what gets traded. Near-perfect correlation is
   not evidence of a good pair and often indicates the opposite, because it
   usually comes with a spread too small to cover costs.
5. Half-life shorter than the bar interval means the spread reverts faster
   than the data resolves it, so measured z-scores understate the true
   dynamics.

Write four short sections with these exact headings:

WHAT THE DATA SHOWS
Two or three sentences on the statistical relationship.

IS IT TRADEABLE
The economics. Gross move per round trip against cost. Be blunt.

WHAT WOULD HAVE TO CHANGE
What thresholds, costs or instruments would make this work, if anything.

WHAT TO LOOK FOR IN OTHER PAIRS
Two or three specific, measurable criteria a better candidate would meet,
drawn from where this one fails.

Under 350 words. Plain prose, no markdown bullets or bold. Written for
someone who already understands cointegration.
"""


def generate(facts: dict[str, Any], provider: str | None = None,
             question: str | None = None,
             kind: str = "performance") -> tuple[str, str]:
    """Return (narrative, provider_used).

    `kind` selects the system prompt: "performance" reviews results,
    "research" assesses whether a pair is worth trading at all.
    """
    chosen = resolve_provider(provider)
    prompt = _build_prompt(facts, question)
    system = RESEARCH_PROMPT if kind == "research" else SYSTEM_PROMPT

    if chosen == "gemini":
        try:
            text = _call_gemini(prompt, system=system)
            resolved = _resolve_gemini_target.__dict__.get("_cache")
            return text, f"gemini/{resolved[1] if resolved else GEMINI_MODEL}"
        except Exception as e:
            return (rule_based(facts, note=f"Gemini unavailable: "
                               f"{_redact(str(e))}", kind=kind), "rule-based")
    if chosen == "anthropic":
        try:
            return (_call_anthropic(prompt, system=system),
                    f"anthropic/{ANTHROPIC_MODEL}")
        except Exception as e:
            return (rule_based(facts, note=f"Anthropic unavailable: "
                               f"{_redact(str(e))}", kind=kind), "rule-based")
    return rule_based(facts, kind=kind), "rule-based"


def _trim(facts: dict[str, Any]) -> dict[str, Any]:
    """Drop the long tail of per-year rows; keep every headline figure.

    A shorter prompt leaves more of the token budget for the answer.
    """
    slim = dict(facts)
    attr = slim.get("attribution")
    if isinstance(attr, dict):
        attr = dict(attr)
        rows = attr.get("by_year") or []
        if len(rows) > 6:
            attr["by_year"] = rows[-6:]
            attr["by_year_note"] = f"showing 6 most recent of {len(rows)} years"
        slim["attribution"] = attr
    return slim


def _build_prompt(facts: dict[str, Any], question: str | None) -> str:
    body = json.dumps(_trim(facts), indent=2, default=str)
    tail = (f"\n\nThe reviewer specifically asks: {question}"
            if question else "")
    return (f"Fact pack:\n```json\n{body}\n```{tail}\n\n"
            f"Write the review now, using only the figures above.")


def list_gemini_models(key: str | None = None) -> list[dict]:
    """Ask the key what it can actually reach. Used for diagnostics."""
    key = key or _key("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    out = []
    for version in GEMINI_VERSIONS:
        try:
            r = requests.get(f"{GEMINI_BASE.format(version=version)}/models",
                             headers={"x-goog-api-key": key}, timeout=TIMEOUT)
            if r.status_code != 200:
                continue
            for m in r.json().get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    out.append({
                        "version": version,
                        "name": m["name"].removeprefix("models/"),
                        "display": m.get("displayName", ""),
                    })
        except requests.RequestException:
            continue
    return out


def _resolve_gemini_target(key: str) -> tuple[str, str]:
    """Find a (version, model) pair this key can actually call.

    ListModels is not sufficient: a model can be listed and still refuse
    generateContent with 404 "no longer available to new users". Every
    candidate is therefore verified with a real call before being cached.
    """
    cached = _resolve_gemini_target.__dict__.get("_cache")
    if cached:
        return cached

    tried: list[str] = []

    def callable_on(version: str, model: str) -> bool:
        url = (f"{GEMINI_BASE.format(version=version)}/models/"
               f"{model}:generateContent")
        try:
            r = requests.post(
                url, headers={"x-goog-api-key": key},
                json={"contents": [{"parts": [{"text": "ping"}]}],
                      "generationConfig": {"maxOutputTokens": 8}},
                timeout=TIMEOUT)
        except requests.RequestException:
            return False
        if r.status_code in (401, 403):
            raise RuntimeError("API key rejected (check the key is enabled "
                               "for the Generative Language API)")
        tried.append(f"{model}@{version}={r.status_code}")
        return r.status_code == 200

    # Configured model first.
    for version in GEMINI_VERSIONS:
        if callable_on(version, GEMINI_MODEL):
            _resolve_gemini_target._cache = (version, GEMINI_MODEL)
            return _resolve_gemini_target._cache

    # Then discovery, verifying each candidate rather than trusting the list.
    available = list_gemini_models(key)
    if not available:
        raise RuntimeError("no models available to this key — enable the "
                           "Generative Language API in your Google project")

    ordered = []
    for preferred in GEMINI_FALLBACKS:
        ordered += [m for m in available if m["name"].startswith(preferred)]
    ordered += [m for m in available if m not in ordered]

    for m in ordered[:8]:
        if callable_on(m["version"], m["name"]):
            _resolve_gemini_target._cache = (m["version"], m["name"])
            return _resolve_gemini_target._cache

    raise RuntimeError(
        f"no listed model accepted a request. Tried: {', '.join(tried[:8])}. "
        f"Models can be listed but closed to new users.")


def _thinking_configs(model: str) -> list[dict | None]:
    """Thinking controls to try, in order, for this model family.

    Gemini 3.x uses thinkingLevel; 2.5 uses thinkingBudget. Both bill their
    reasoning against maxOutputTokens, which truncates the visible answer if
    left unbounded. Unsupported parameters are sometimes ignored rather than
    rejected, so each is tried and the result verified.
    """
    if "gemini-3" in model:
        return [{"thinkingLevel": "low"}, {"thinkingBudget": 0}, None]
    return [{"thinkingBudget": 0}, {"thinkingLevel": "low"}, None]


def _gemini_payload(prompt: str, max_tokens: int,
                    thinking: dict | None,
                    system: str = SYSTEM_PROMPT) -> dict:
    cfg: dict = {"temperature": 0.2, "maxOutputTokens": max_tokens}
    if thinking:
        cfg["thinkingConfig"] = thinking
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": cfg,
    }


def _extract(cand: dict) -> str:
    parts = cand.get("content", {}).get("parts", [])
    # Skip reasoning parts — they are marked and are not the answer.
    return "".join(p.get("text", "") for p in parts
                   if not p.get("thought")).strip()


def gemini_probe(prompt: str = "Reply with exactly: OK") -> list[dict]:
    """Diagnostic: try each thinking config and report what came back."""
    key = _key("GEMINI_API_KEY")
    version, model = _resolve_gemini_target(key)
    url = f"{GEMINI_BASE.format(version=version)}/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "content-type": "application/json"}
    out = []
    for thinking in _thinking_configs(model):
        r = requests.post(url, headers=headers,
                          json=_gemini_payload(prompt, GEMINI_MAX_TOKENS, thinking),
                          timeout=TIMEOUT)
        row = {"thinking": thinking, "http": r.status_code}
        if r.status_code == 200:
            d = r.json()
            cand = (d.get("candidates") or [{}])[0]
            usage = d.get("usageMetadata", {})
            row.update({
                "finishReason": cand.get("finishReason"),
                "prompt_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
                "thought_tokens": usage.get("thoughtsTokenCount"),
                "chars": len(_extract(cand)),
                "text": _extract(cand)[:120],
            })
        else:
            row["error"] = _redact(r.text)[:200]
        out.append(row)
    return out


def _call_gemini(prompt: str, max_tokens: int = GEMINI_MAX_TOKENS,
                 system: str = SYSTEM_PROMPT) -> str:
    key = _key("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")
    version, model = _resolve_gemini_target(key)
    url = f"{GEMINI_BASE.format(version=version)}/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "content-type": "application/json"}

    last_error = "no attempt succeeded"
    for thinking in _thinking_configs(model):
        for budget in (max_tokens, max_tokens * 4):
            resp = requests.post(
                url, headers=headers,
                json=_gemini_payload(prompt, budget, thinking, system),
                timeout=TIMEOUT)

            if resp.status_code == 400:
                last_error = f"400 with {thinking}: {_redact(resp.text)[:160]}"
                break                      # try the next thinking config
            if resp.status_code != 200:
                raise RuntimeError(f"{resp.status_code} from {model} "
                                   f"({version}): {_redact(resp.text)[:300]}")

            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                blocked = data.get("promptFeedback", {}).get("blockReason")
                last_error = f"no candidates{f' (blocked: {blocked})' if blocked else ''}"
                break

            cand = candidates[0]
            finish = cand.get("finishReason")
            usage = data.get("usageMetadata", {})
            text = _extract(cand)

            # A complete answer covers all four required headings.
            complete = text.count("\n") >= 3 or len(text) > 400
            if text and finish in (None, "STOP") and complete:
                return text

            last_error = (
                f"finishReason={finish}, thinking={thinking}, "
                f"budget={budget}, output_tokens={usage.get('candidatesTokenCount')}, "
                f"thought_tokens={usage.get('thoughtsTokenCount')}, "
                f"chars={len(text)}")
            if finish != "MAX_TOKENS":
                break                      # raising the budget will not help

    raise RuntimeError(f"could not obtain a complete response — {last_error}. "
                       f"Run scripts/check_llm.py for a full breakdown.")


def _call_anthropic(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    key = _key("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    resp = requests.post(
        ANTHROPIC_URL,
        headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                 "content-type": "application/json"},
        json={
            "model": ANTHROPIC_MODEL, "max_tokens": ANTHROPIC_MAX_TOKENS, "temperature": 0.2,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {_redact(resp.text)[:300]}")
    blocks = resp.json().get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if not text.strip():
        raise RuntimeError("empty response")
    return text.strip()


# =====================================================================
# Deterministic fallback
# =====================================================================


def rule_based(facts: dict[str, Any], note: str | None = None,
               kind: str = "performance") -> str:
    """Template summary from the fact pack. Always available, never wrong."""
    perf = facts.get("performance", {})
    attr = facts.get("attribution", {})
    risk = facts.get("risk", {})
    cost = facts.get("transaction_cost_sensitivity", [])
    warnings = facts.get("deterministic_warnings", [])

    if kind == "research":
        return _rule_based_research(facts, note)

    lines = ["WHAT HAPPENED"]
    if perf:
        lines.append(
            f"{facts.get('pair', 'The pair')} produced "
            f"{perf.get('total_return_pct', 0):.2f}% total return across "
            f"{perf.get('num_trades', 0)} trades, with a Sharpe of "
            f"{perf.get('sharpe_ratio', 0):.2f} and a maximum drawdown of "
            f"{perf.get('max_drawdown_pct', 0):.2f}%. "
            f"{facts.get('data_basis', '')}")
    else:
        lines.append("No performance metrics are available yet.")

    lines.append("\nWHAT'S DRIVING IT")
    if attr:
        conc = attr.get("concentration", {})
        lines.append(
            f"Total P&L of {attr.get('total_pnl', 0):,.2f} across "
            f"{attr.get('trade_count', 0)} trades. The top five trades account "
            f"for {conc.get('top_5_share_of_total_pct', 0):.0f}% of it. "
            f"Average holding period is "
            f"{attr.get('avg_holding_days', 0):.1f} days.")
        for row in attr.get("by_direction", []):
            lines.append(
                f"  {row['direction']}: {row['trades']} trades, "
                f"{row['total_pnl']:,.2f} P&L, "
                f"{row['win_rate']:.0f}% win rate.")
    else:
        lines.append("No closed trades to attribute.")

    lines.append("\nCONCERNS")
    if warnings:
        lines.extend(f"  {w}" for w in warnings)
    else:
        lines.append("  No deterministic warnings were triggered.")
    if cost:
        breakeven = next((c for c in cost if not c["still_profitable"]), None)
        if breakeven:
            lines.append(
                f"  At {breakeven['fee_bps']}bps round-trip cost, net P&L "
                f"turns negative ({breakeven['net_pnl']:,.2f}).")
        else:
            lines.append(
                f"  Remains profitable through "
                f"{cost[-1]['fee_bps']}bps of round-trip cost.")

    lines.append("\nWHAT TO CHECK NEXT")
    if note:
        lines.append(f"  Narrative model was not reached — {note}")
    else:
        lines.append("  Configure an API key to enable narrative analysis.")
    if risk:
        lines.append(
            f"  Daily volatility is {risk.get('daily_volatility_pct', 0):.3f}% "
            f"with 95% VaR of {risk.get('var_95_pct', 0):.3f}% — confirm "
            f"whether returns are large enough to be distinguishable from "
            f"noise.")
    return "\n".join(lines)


def _rule_based_research(facts: dict[str, Any], note: str | None) -> str:
    """Deterministic pair assessment when no model is reachable."""
    spread = facts.get("spread", {})
    trade = facts.get("tradeability", {})
    tests = facts.get("statistical_tests", [])
    stability = facts.get("stability", {})
    warnings = facts.get("deterministic_warnings", [])

    lines = ["WHAT THE DATA SHOWS"]
    passed = [t["test"] for t in tests if t.get("passed")]
    if tests:
        lines.append(
            f"{len(passed)} of {len(tests)} tests support cointegration "
            f"({', '.join(passed) if passed else 'none'}). Spread mean "
            f"{spread.get('mean', 0):.4f}, standard deviation "
            f"{spread.get('std', 0):.4f} over "
            f"{spread.get('observations', 0):,} observations.")
    if stability.get("windows_cointegrated_pct") is not None:
        lines.append(
            f"  {stability['windows_cointegrated_pct']:.0f}% of rolling "
            f"windows hold cointegration at the configured threshold.")

    lines.append("\nIS IT TRADEABLE")
    if trade:
        lines.append(
            f"  A round trip captures {trade.get('captured_sigma', 0):.1f} "
            f"sigma, worth {trade.get('gross_profit_per_trade', 0):.2f}, "
            f"against {trade.get('total_cost_per_trade', 0):.2f} of cost. "
            f"Net {trade.get('net_profit_per_trade', 0):+.2f} per trade.")
        if not trade.get("is_tradeable", False):
            lines.append(
                f"  Entry would need to reach "
                f"{trade.get('entry_threshold_needed', 0):.2f} sigma to break "
                f"even.")
    else:
        lines.append("  Tradeability has not been computed — enter the quoted "
                     "spreads in the Tradeability section above.")

    lines.append("\nCONCERNS")
    lines.extend(f"  {w}" for w in warnings) if warnings else lines.append(
        "  No deterministic warnings were triggered.")

    lines.append("\nWHAT TO LOOK FOR IN OTHER PAIRS")
    lines.append("  A spread whose standard deviation is large relative to the "
                 "quoted spread of both legs — that ratio, not correlation, "
                 "determines whether a pair can be traded profitably.")
    if note:
        lines.append(f"\n  Narrative model was not reached — {note}")
    return "\n".join(lines)