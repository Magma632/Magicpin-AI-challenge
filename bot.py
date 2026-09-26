"""
bot.py — HTTP server implementing the 5 endpoints from challenge-testing-brief.md §2.

Run:
    export ANTHROPIC_API_KEY=sk-...      # optional; without it, falls back to the
                                          # deterministic composer in composer.py
    uvicorn bot:app --host 0.0.0.0 --port 8080

Design notes (see README.md for the full writeup):
  - context_store.py owns the versioned, idempotent context state.
  - composer.py turns a resolved (category, merchant, trigger, customer?) bundle
    into a message (LLM-first, rule-based fallback).
  - conversation_handlers.py owns multi-turn logic for /v1/reply (auto-reply
    detection, intent handoff, hostile/off-topic handling, graceful exit).
  - This file is just the transport layer + a few operational policies:
    suppression-key dedup, a 3-nudge cap per suppression key, and the 20
    actions/tick cap from the testing brief.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import composer
from context_store import ContextStore
from conversation_handlers import ConversationState, respond as handle_reply

app = FastAPI(title="Vera-challenger bot")
START_TIME = time.time()
STORE = ContextStore()

MAX_ACTIONS_PER_TICK = 20
MAX_NUDGES_PER_SUPPRESSION_KEY = 3  # "knowing when to stop" — brief §12.5

# conversation_id -> ConversationState
CONVERSATIONS: dict[str, ConversationState] = {}
# suppression_key -> number of times we've proactively nudged on it
NUDGE_COUNTS: dict[str, int] = {}


# ---------------------------------------------------------------------------
# 2.4 GET /v1/healthz
# ---------------------------------------------------------------------------
@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": STORE.counts(),
    }


# ---------------------------------------------------------------------------
# 2.5 GET /v1/metadata
# ---------------------------------------------------------------------------
@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Team Claude",
        "team_members": ["Claude"],
        "model": composer.MODEL_NAME,
        "approach": (
            "4-context resolver (context_store.py) feeding a single rubric-aware "
            "composer prompt (composer.py, temperature=0), dispatched by trigger.kind; "
            "deterministic template fallback when no LLM key is configured; "
            "heuristic-first multi-turn handling (conversation_handlers.py) for "
            "auto-reply / intent-handoff / hostile-offtopic / graceful-exit, with an "
            "LLM call only for the genuinely open-ended reply case."
        ),
        "contact_email": "team@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 2.1 POST /v1/context
# ---------------------------------------------------------------------------
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/v1/context")
async def push_context(body: CtxBody):
    result = STORE.push(body.scope, body.context_id, body.version, body.payload)
    if not result["accepted"]:
        return result
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# 2.2 POST /v1/tick
# ---------------------------------------------------------------------------
class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions: list[dict] = []
    for trigger_id in body.available_triggers:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break
        bundle = STORE.resolve_bundle(trigger_id)
        if not bundle:
            continue  # missing category/merchant context — don't compose from partial data
        suppression_key = bundle["trigger"].get("suppression_key", trigger_id)
        if NUDGE_COUNTS.get(suppression_key, 0) >= MAX_NUDGES_PER_SUPPRESSION_KEY:
            continue  # already tried enough times on this — restraint is rewarded (testing brief FAQ)
        if NUDGE_COUNTS.get(suppression_key, 0) >= 1:
            continue  # one proactive send per suppression_key unless the merchant replies first

        composed = composer.compose(bundle)
        merchant_id = bundle["merchant"]["merchant_id"]
        customer_id = bundle["customer"]["customer_id"] if bundle.get("customer") else None
        conversation_id = f"conv_{merchant_id}_{trigger_id}"

        state = ConversationState(
            conversation_id=conversation_id, merchant_id=merchant_id,
            customer_id=customer_id, trigger_id=trigger_id, bundle=bundle,
        )
        state.sent_bodies.append(composed["body"])
        CONVERSATIONS[conversation_id] = state
        NUDGE_COUNTS[suppression_key] = NUDGE_COUNTS.get(suppression_key, 0) + 1

        actions.append({
            "conversation_id": conversation_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trigger_id,
            "template_name": f"vera_{bundle['trigger'].get('kind', 'generic')}_v1",
            "template_params": [bundle["merchant"]["identity"].get("name", "")],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        })
    return {"actions": actions}


# ---------------------------------------------------------------------------
# 2.3 POST /v1/reply
# ---------------------------------------------------------------------------
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: Optional[int] = None


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    state = CONVERSATIONS.get(body.conversation_id)
    if state is None:
        # Judge referenced a conversation we don't recognize (shouldn't normally
        # happen since we mint conversation_ids in /v1/tick) — start minimal state.
        state = ConversationState(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id, customer_id=body.customer_id,
        )
        CONVERSATIONS[body.conversation_id] = state

    if state.ended:
        return {"action": "end", "rationale": "Conversation already concluded."}

    result = handle_reply(state, body.message)
    return result


# ---------------------------------------------------------------------------
# Optional teardown (testing brief §11)
# ---------------------------------------------------------------------------
@app.post("/v1/teardown")
async def teardown():
    STORE.wipe()
    CONVERSATIONS.clear()
    NUDGE_COUNTS.clear()
    return {"status": "wiped"}
