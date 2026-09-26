# Vera Challenger — approach

## Architecture

```
context_store.py   — versioned, idempotent (scope, context_id) store; resolves a
                      trigger_id into the full {category, merchant, trigger,
                      customer?, digest_item} bundle a composer needs, or returns
                      None if anything required is missing (so we skip rather
                      than compose from partial data / hallucinate).

composer.py         — turns a resolved bundle into a ComposedMessage.
                       Primary path: one rubric-aware prompt (the anti-patterns
                       and compulsion levers from the brief are embedded directly
                       in the system prompt) sent to an Anthropic model at
                       temperature=0, dispatched by trigger.kind via a small
                       KIND_HINTS table (cta shape + framing hint per kind) so a
                       single prompt template flexes across all ~25 trigger kinds
                       in the dataset instead of needing per-kind prompts.
                       Fallback path: a deterministic, template-driven composer
                       with no external dependency, used when no API key is
                       configured or the LLM call/parse fails. It pulls concrete
                       facts straight out of the trigger payload / digest item /
                       merchant signals rather than falling back to generic
                       copy, so it never fabricates and never goes fully silent.

conversation_handlers.py — multi-turn logic for /v1/reply. Cheap heuristics run
                       first (regex-based, instant, no LLM round-trip) to catch
                       the four "open challenge" cases the brief calls out:
                         1. Auto-reply detection — known canned-text patterns,
                            or a verbatim repeat of something already seen.
                            First hit: one lightweight nudge for a human.
                            Second hit: exit immediately (beats production
                            Vera's 2-3 turns per the brief's pain point #1).
                         2. Explicit intent ("yes", "let's do it", "chalo karo")
                            — routes straight to action mode, never asks another
                            qualifying question. Deliberately conservative: a
                            bare "ok" only counts if it's ~the whole message and
                            not a question, so "ok, what does that mean?" isn't
                            misread as a commitment.
                         3. Hostile messages — de-escalate without matching
                            tone or ending the conversation.
                         4. Off-topic asks — politely decline and redirect back
                            to mission scope, don't end.
                         5. Explicit not-interested — graceful exit.
                       Only the genuinely open-ended "normal continuation" case
                       calls the LLM (same temperature=0 pattern as composer.py).
                       Anti-repetition is enforced via a per-conversation
                       sent_bodies list.

bot.py               — FastAPI transport layer implementing the 5 required
                       endpoints. Owns two small operational policies that sit
                       above the composer: a suppression_key dedup (one
                       proactive send per suppression_key unless the merchant
                       replies) and a 3-nudge cap per suppression_key ("knowing
                       when to stop" — brief §12.5). Caps actions/tick at 20 per
                       the testing brief's rate limit.

generate_submission.py — loads the dataset straight from disk and calls
                       composer.compose() directly (no HTTP) to produce
                       submission.jsonl for the 30 canonical test pairs.
```

## Design choices / tradeoffs

- **One prompt, dispatched by kind, not 25 bespoke prompts.** The brief's §13
  suggests a routing layer with different prompt variants per trigger kind.
  Instead we use one system prompt (rubric + anti-patterns + compulsion levers,
  written once) and vary only a short per-kind `frame` hint + `cta_shape` in the
  user turn. This is easier to keep consistent across categories and cheaper to
  maintain than N prompt templates, at the cost of slightly less specialized
  phrasing per kind than a bespoke prompt could get.
- **Rule-based fallback is a safety net, not the intended scoring path.**
  Operational penalties in the testing brief (healthz failures, timeouts,
  malformed JSON) are steep, so the bot must never go silent even if the LLM
  call fails or no key is configured. The fallback is deliberately
  fact-anchored (it reads the actual trigger payload / digest item / merchant
  signals) rather than templated boilerplate, but it can't match the LLM path's
  natural code-mixed phrasing or nuanced lever selection. **Set
  `ANTHROPIC_API_KEY` before running for the real submission** — that's the
  path that should actually be scored.
- **Heuristics before LLM in the reply loop.** Auto-reply / intent / hostility /
  off-topic detection are all regex-based, not LLM calls. This is faster
  (no round-trip, well inside the 30s budget) and, per the brief's own
  pain-point framing, *faster detection is explicitly the win condition* over
  production Vera's 2-3-turn auto-reply handling. The tradeoff is recall on
  auto-reply text we haven't seen a pattern for — mitigated by the verbatim-
  repeat fallback (brief §12 hint: "same message verbatim 3+ times = auto-
  reply"; we act on the 2nd occurrence to detect faster).
- **`{ "placeholder": true }` payloads.** `generate_dataset.py`'s generated
  (non-seed) triggers sometimes carry no real payload facts beyond the kind
  name. Per the "don't fabricate" constraint, the fallback composer explicitly
  detects this and falls back to the merchant's real offer/signals rather than
  inventing a plausible-looking number. The LLM path is instructed the same
  way ("if a needed fact is missing, write around it rather than inventing it").
- **Suppression is conservative.** We send at most once per `suppression_key`
  unless the merchant replies first, and cap total nudges per key at 3. This
  favors the "restraint is rewarded" signal in the testing brief's FAQ over
  maximizing message volume.

## What additional context would have helped most

- A real send/receive timestamp history per merchant (not just the last 5
  conversation_history turns) would let the composer reason about *cadence*
  ("last 3 messages were all research digests, vary the format") rather than
  just the single next message.
- Explicit `category.digest` freshness/expiry per item, so the composer can
  tell "topical this week" from "still valid but not urgent" without inferring
  it from `expires_at` on the trigger alone.
- A confidence/quality score on generated (non-seed) merchants/triggers would
  let the composer decide when a `placeholder` payload is worth a lighter-touch
  message rather than a full nudge.

## Running it

```bash
pip install fastapi uvicorn pydantic anthropic
export ANTHROPIC_API_KEY=sk-...        # optional but recommended — see above
uvicorn bot:app --host 0.0.0.0 --port 8080

# regenerate submission.jsonl directly (no HTTP needed):
python generate_submission.py --dataset dataset --out submission.jsonl

# local harness:
export BOT_URL=http://localhost:8080
python judge_simulator.py
```

Tested locally: context push idempotency/versioning, `/v1/tick` composition +
suppression dedup, and four `/v1/reply` scenarios (normal→intent-handoff,
auto-reply-hell, explicit not-interested, hostile→off-topic→normal) all behave
as expected against the rule-based fallback composer (no API key in this
environment). Swapping in `ANTHROPIC_API_KEY` uses the LLM path automatically
with no code changes.
