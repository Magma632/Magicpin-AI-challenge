"""
llm_client.py — provider-agnostic LLM call used by composer.py and
conversation_handlers.py.

Tries, in order:
  1. Gemini (Google AI Studio's free tier — no credit card required) if
     GEMINI_API_KEY or GOOGLE_API_KEY is set.
  2. Anthropic, if ANTHROPIC_API_KEY is set.
  3. Returns None, which both callers treat as "fall back to the
     deterministic rule-based composer".

This means switching providers is a one-line env var change on Render —
no code or requirements changes needed either way (both dependencies are
already in requirements.txt; the unused one just never gets imported).
"""
from __future__ import annotations

import os
from typing import Optional

import requests

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
ANTHROPIC_MODEL = os.environ.get("COMPOSER_MODEL", "claude-sonnet-4-5-20250929")
TIMEOUT_S = 20  # leaves headroom inside the judge's 30s per-call budget


def _call_gemini(system: str, user: str, max_tokens: int) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        resp = requests.post(
            url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def _call_anthropic(system: str, user: str, max_tokens: int) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
            timeout=TIMEOUT_S,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    except Exception:
        return None


def call_llm(system: str, user: str, max_tokens: int = 600) -> Optional[str]:
    """Never raises. Returns the model's raw text, or None if nothing is
    configured or every configured provider failed."""
    text = _call_gemini(system, user, max_tokens)
    if text:
        return text
    return _call_anthropic(system, user, max_tokens)


def active_provider() -> str:
    """For /v1/metadata — reports which path is actually configured."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return f"gemini:{GEMINI_MODEL}"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return f"anthropic:{ANTHROPIC_MODEL}"
    return "rule-based fallback (no LLM key configured)"
