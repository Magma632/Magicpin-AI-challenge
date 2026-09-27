"""
conversation_handlers.py — multi-turn reply logic for /v1/reply.

Implements the "open challenges" from challenge-brief.md §12 and the Phase-4
replay scenarios from challenge-testing-brief.md §4:
  1. Auto-reply detection -> route differently, don't burn turns.
  2. Intent-transition handling -> switch from pitch to action immediately.
  3. Graceful exit -> not-interested, or repeated auto-replies.
  4. Hostile/off-topic -> stay polite and on-mission without escalating.
  5. Anti-repetition -> never resend an identical body in the same conversation.

`respond(state, merchant_message)` is the function named in challenge-brief.md
§7.4's optional `conversation_handlers.py` contract; bot.py's /v1/reply calls it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import composer
import llm_client

# ---------------------------------------------------------------------------
# Heuristic detectors. Kept as plain pattern lists (not an LLM call) so
# detection is instant and doesn't burn a turn or a token budget on every
# inbound message — speed of detection is explicitly what the brief says
# beats production Vera (§2, pain point #1).
# ---------------------------------------------------------------------------

_AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"we('| a)?ll get back to you",
    r"aapki jaankari ke liye.*shukriya",
    r"hamari team tak pahuncha",
    r"i am an? automated (assistant|reply|response)",
    r"main ek automated assistant",
    r"currently (unavailable|away|closed)",
    r"business hours (are|:)",
    r"your message is important to us",
]

# Strong, unambiguous commitment phrases — safe to match anywhere in the message.
_INTENT_YES_STRONG_PATTERNS = [
    r"go ahead", r"let'?s do it", r"\bstart(ing)?\b", r"\bchalo\b", r"karo\b",
    r"shuru kar", r"kar do", r"join karna hai", r"i want to join", r"mujhe.*join",
]
# Bare affirmations — only count as intent when they're ~the whole message, so a
# reply like "ok fine, what were you saying" (a question, not a commitment)
# doesn't get misrouted into action mode.
_INTENT_YES_BARE_AFFIRMATIONS = [r"\byes\b", r"\byeah\b", r"\bsure\b", r"\bok(ay)?\b", r"\bhaan\b"]
_MAX_WORDS_FOR_BARE_AFFIRMATION = 4

_NOT_INTERESTED_PATTERNS = [
    r"not interested", r"\bno thanks?\b", r"^stop$", r"\bplease stop\b",
    r"stop (messaging|contacting|texting) me", r"nahi chahiye",
    r"band karo", r"unsubscribe", r"don'?t (message|contact) me",
]

_HOSTILE_PATTERNS = [
    r"\bidiot\b", r"\bstupid\b", r"\bshut up\b", r"\bnonsense\b",
    r"\bbakwas\b", r"\bbewakoof\b", r"f+u+c+k", r"\bscam\b",
]

_OFFTOPIC_HINTS = [
    r"\bgst\b", r"income tax", r"\bloan\b", r"personal (advice|problem)",
    r"\bvisa\b", r"immigration", r"\bpolitics\b", r"election",
]


def _matches_any(patterns: list[str], text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_auto_reply(message: str, history: list[str]) -> bool:
    if _matches_any(_AUTO_REPLY_PATTERNS, message):
        return True
    # Unknown canned text: verbatim repeat of something already seen = auto-reply.
    norm = message.strip().lower()
    return any(norm == h.strip().lower() for h in history)


def is_intent_yes(message: str) -> bool:
    if _matches_any(_INTENT_YES_STRONG_PATTERNS, message):
        return True
    if "?" in message:
        return False  # a question is not a commitment, even if it starts with "ok"
    word_count = len(message.strip().split())
    if word_count <= _MAX_WORDS_FOR_BARE_AFFIRMATION and _matches_any(_INTENT_YES_BARE_AFFIRMATIONS, message):
        return True
    return False


def is_not_interested(message: str) -> bool:
    return _matches_any(_NOT_INTERESTED_PATTERNS, message)


def is_hostile(message: str) -> bool:
    return _matches_any(_HOSTILE_PATTERNS, message)


def is_offtopic(message: str) -> bool:
    return _matches_any(_OFFTOPIC_HINTS, message)


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    bundle: Optional[dict] = None          # resolved category/merchant/trigger/customer
    sent_bodies: list[str] = field(default_factory=list)
    merchant_messages: list[str] = field(default_factory=list)
    auto_reply_strikes: int = 0
    turn_number: int = 0
    ended: bool = False


def _dedupe(body: str, state: ConversationState) -> str:
    if body not in state.sent_bodies:
        return body
    # Extremely defensive fallback; the LLM/rule paths shouldn't normally repeat.
    return body + " (following up on the above)"


def _llm_reply(state: ConversationState, message: str, intent_hint: str) -> Optional[str]:
    if not state.bundle:
        return None
    system = (
        "You are Vera, magicpin's WhatsApp assistant, mid-conversation with a merchant "
        "(or, if a CustomerContext is present, writing on the merchant's behalf to their "
        "customer). Continue the conversation naturally in ONE short WhatsApp message. "
        f"Situation: {intent_hint}. "
        "Match the category voice and the language mix already used in the conversation. "
        "Never repeat a message already sent. Return ONLY the message text, nothing else."
    )
    context_str = composer.build_user_prompt(state.bundle)
    history_str = "\n".join(f"- {m}" for m in state.merchant_messages[-4:])
    user = f"Context:\n{context_str}\n\nRecent merchant/customer messages:\n{history_str}\n\nLatest message: {message}"
    text = llm_client.call_llm(system, user, max_tokens=300)
    return text.strip() if text else None


def respond(state: ConversationState, merchant_message: str) -> dict:
    """Given the conversation so far + the latest inbound message, decide the next move."""
    state.turn_number += 1
    history = list(state.merchant_messages)  # snapshot BEFORE appending — avoid aliasing the live list
    state.merchant_messages.append(merchant_message)

    # 1) Auto-reply detection — don't burn multiple turns on a canned bot.
    if is_auto_reply(merchant_message, history):
        state.auto_reply_strikes += 1
        if state.auto_reply_strikes == 1:
            body = "Samajh gayi — before this goes to your team, is there 2 minutes to just glance at it yourself? Easy either way."
            body = _dedupe(body, state)
            state.sent_bodies.append(body)
            return {"action": "send", "body": body, "cta": "binary",
                    "rationale": "First auto-reply detected; one lightweight nudge for a human before backing off."}
        state.ended = True
        return {"action": "end",
                "rationale": "Repeated auto-reply confirms this is WhatsApp Business canned text, not a person; exiting to avoid wasting turns."}

    # 2) Hostile message -> stay polite, don't escalate, don't end (Phase-4 hostile/off-topic scenario).
    # Checked before "not interested" since an insult ("stop wasting my time") can superficially
    # match opt-out wording but is a tone signal, not a considered decision to unsubscribe.
    if is_hostile(merchant_message):
        body = "No worries, I'll keep it brief. Happy to drop this if it's not useful — just say the word."
        body = _dedupe(body, state)
        state.sent_bodies.append(body)
        return {"action": "send", "body": body, "cta": "binary",
                "rationale": "Message read as hostile; de-escalating politely without matching tone, offering an easy out."}

    # 3) Explicit, calm not-interested -> graceful exit.
    if is_not_interested(merchant_message):
        state.ended = True
        return {"action": "end", "rationale": "Merchant explicitly opted out or declined; exiting gracefully."}

    # 4) Off-topic ask -> stay on-mission politely.
    if is_offtopic(merchant_message):
        body = "That's outside what I can help with here, but happy to keep going on the profile/offers side whenever you're ready."
        body = _dedupe(body, state)
        state.sent_bodies.append(body)
        return {"action": "send", "body": body, "cta": "open_ended",
                "rationale": "Off-topic request; politely declined and redirected back to mission scope."}

    # 5) Explicit intent ("yes", "let's do it") -> route straight to action, not another qualifying question.
    if is_intent_yes(merchant_message):
        llm_body = _llm_reply(state, merchant_message, "Merchant just said yes/let's-do-it — start the action immediately, do not ask another qualifying question.")
        body = llm_body or "Great — starting now, no more questions needed. I'll confirm here once it's done."
        body = _dedupe(body, state)
        state.sent_bodies.append(body)
        return {"action": "send", "body": body, "cta": "none",
                "rationale": "Detected explicit affirmative intent; switched from qualification to action mode immediately."}

    # 6) Default: continue the conversation naturally.
    llm_body = _llm_reply(state, merchant_message, "Normal conversational reply — acknowledge what they said and advance to the next best step.")
    body = llm_body or "Got it — noted. Let me know if you'd like me to go ahead."
    body = _dedupe(body, state)
    state.sent_bodies.append(body)
    return {"action": "send", "body": body, "cta": "open_ended",
            "rationale": "Standard continuation; acknowledged the reply and proposed the next low-friction step."}
