"""Diagnose LLM provider connectivity.

Asks each configured key what it can actually reach, then runs a real
generation call. Never prints key material.

Usage:
    python scripts/check_llm.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "frontend"))
load_dotenv()

import insights  # noqa: E402


def fingerprint(key: str) -> str:
    """Enough to tell two keys apart, not enough to use one."""
    return f"{key[:6]}…{key[-4:]} ({len(key)} chars)"


def check_gemini() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    print("\n=== GEMINI ===")
    if not key:
        print("  GEMINI_API_KEY not set.")
        return
    print(f"  key: {fingerprint(key)}")

    for version in insights.GEMINI_VERSIONS:
        url = f"{insights.GEMINI_BASE.format(version=version)}/models"
        try:
            r = requests.get(url, headers={"x-goog-api-key": key}, timeout=30)
        except requests.RequestException as e:
            print(f"  {version}: request failed — {insights._redact(str(e))}")
            continue
        print(f"  {version}: HTTP {r.status_code}")
        if r.status_code != 200:
            print(f"    {insights._redact(r.text)[:300]}")
            continue
        models = [m for m in r.json().get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", [])]
        print(f"    {len(models)} models support generateContent:")
        for m in sorted(models, key=lambda x: x["name"])[:25]:
            name = m["name"].removeprefix("models/")
            marker = " <-- configured default" if name == insights.GEMINI_MODEL else ""
            print(f"      {name}{marker}")

    print("\n  Resolving a callable target...")
    try:
        version, model = insights._resolve_gemini_target(key)
        print(f"    will use: {model} on {version}")
    except Exception as e:
        print(f"    FAILED: {insights._redact(str(e))}")
        return

    print("\n  Probing thinking configurations...")
    try:
        for row in insights.gemini_probe(
                "Write four short paragraphs headed WHAT HAPPENED, "
                "WHAT'S DRIVING IT, CONCERNS, WHAT TO CHECK NEXT."):
            print(f"    thinking={row['thinking']}  HTTP {row['http']}")
            if row["http"] == 200:
                print(f"      finishReason : {row['finishReason']}")
                print(f"      tokens       : prompt={row['prompt_tokens']} "
                      f"output={row['output_tokens']} "
                      f"thoughts={row['thought_tokens']}")
                print(f"      chars        : {row['chars']}")
                print(f"      starts       : {row['text'][:90]!r}")
            else:
                print(f"      error: {row.get('error')}")
    except Exception as e:
        print(f"    probe failed: {insights._redact(str(e))}")

    print("\n  Full generation test...")
    try:
        text = insights._call_gemini(
            "Write four short paragraphs headed WHAT HAPPENED, "
            "WHAT'S DRIVING IT, CONCERNS, WHAT TO CHECK NEXT.")
        print(f"    OK — {len(text)} chars, {text.count(chr(10)) + 1} lines")
    except Exception as e:
        print(f"    FAILED: {insights._redact(str(e))}")


def check_anthropic() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    print("\n=== ANTHROPIC ===")
    if not key:
        print("  ANTHROPIC_API_KEY not set (optional).")
        return
    print(f"  key: {fingerprint(key)}")
    try:
        text = insights._call_anthropic("Reply with exactly: OK")
        print(f"  model {insights.ANTHROPIC_MODEL}: {text[:80]}")
    except Exception as e:
        print(f"  FAILED: {insights._redact(str(e))}")
        print("  If this is a model-not-found error, list your available "
              "models at https://docs.claude.com and update ANTHROPIC_MODEL "
              "in frontend/insights.py.")


if __name__ == "__main__":
    print(f"Resolved provider preference: {insights.resolve_provider()}")
    print(f"Providers with keys present: {insights.available_providers()}")
    check_gemini()
    check_anthropic()
    print("\nDone. No key material was printed.")