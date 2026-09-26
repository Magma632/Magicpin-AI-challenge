"""
composer.py — turns (category, merchant, trigger, customer?) into a ComposedMessage.

Two paths:
  1. LLM path (primary): builds a rubric-aware prompt and calls an Anthropic model
     at temperature=0. This is what should be used for the real submission — it's
     what lets the copy hit "Specificity" / "Category fit" / "Engagement compulsion"
     the way the brief's Appendix A/B examples do.
  2. Rule-based path (fallback): a deterministic, template-driven composer that
     needs no API key. It exists so the bot never goes fully silent (operational
     penalties in the testing brief are steep) and so this can be smoke-tested
     without network access. It follows the same anti-patterns / CTA rules.

`compose()` tries the LLM path when ANTHROPIC_API_KEY is set, validates the
shape of what comes back, and falls back to the rule-based path on any failure
(timeout, malformed JSON, empty body). Never raises — always returns a dict.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

MODEL_NAME = os.environ.get("COMPOSER_MODEL", "claude-sonnet-4-5-20250929")
LLM_TIMEOUT_S = 20  # leaves headroom inside the judge's 30s budget

# ---------------------------------------------------------------------------
# Trigger-kind dispatch hints — short framing guidance + CTA shape + urgency
# framing, so one prompt template can flex across all trigger kinds instead of
# needing a bespoke prompt per kind (per brief §13 "routing layer" suggestion).
# ---------------------------------------------------------------------------
KIND_HINTS: dict[str, dict] = {
    "research_digest":        {"cta": "open_ended", "frame": "Share the single most relevant digest item as a peer would flag it to a colleague. Cite the source. Offer to do the next step (pull it / draft something) rather than just informing."},
    "regulation_change":      {"cta": "binary", "frame": "Compliance deadline framing. State the concrete change, the deadline, and what to check. Loss aversion, not alarmism."},
    "category_research_digest_release": {"cta": "open_ended", "frame": "Same as research_digest."},
    "festival_upcoming":      {"cta": "binary", "frame": "Time-boxed opportunity framing tied to the specific festival/date. Offer a ready-to-go action (draft a post/offer)."},
    "weather_heatwave":       {"cta": "open_ended", "frame": "Local, real-time relevance. Tie the weather event to a concrete customer behavior shift relevant to this category."},
    "local_news_event":       {"cta": "open_ended", "frame": "Local disruption/opportunity. Be factual, not sensational."},
    "competitor_opened":      {"cta": "open_ended", "frame": "Competitive awareness framed as useful intel, not fear. Loss aversion + curiosity."},
    "category_trend_movement": {"cta": "open_ended", "frame": "Search/demand trend relevant to this merchant's offer mix. Curiosity + effort externalization (offer to draft something capturing the trend)."},
    "perf_spike":              {"cta": "open_ended", "frame": "Celebrate the concrete number, then pivot to compounding it further. Positive reciprocity tone."},
    "perf_dip":                {"cta": "binary", "frame": "Loss aversion using the exact dropped metric and %. Immediately offer the fix, don't just flag the problem."},
    "milestone_reached":       {"cta": "open_ended", "frame": "Celebratory + social-proof framing (compare to peer_stats). Light touch, no hard CTA needed."},
    "dormant_with_vera":       {"cta": "binary", "frame": "Re-engagement after a silence gap. Single easy re-entry ask, acknowledge the gap without guilt-tripping."},
    "customer_lapsed_soft":    {"cta": "open_ended", "frame": "Internal signal about the merchant's own customer base going soft. Frame as an opportunity for the merchant to reconnect (not directly to the customer)."},
    "appointment_tomorrow":    {"cta": "binary", "frame": "Reminder framing, logistics-forward (time/slot), reassurance tone."},
    "review_theme_emerged":    {"cta": "open_ended", "frame": "Surface the recurring review theme with its exact count/sentiment and offer a concrete fix."},
    "scheduled_recurring":     {"cta": "open_ended", "frame": "Curiosity-driven check-in per lever #7 (ask the merchant something), not a status report."},
    "recall_due":              {"cta": "binary", "frame": "Customer-facing recall reminder. Real slots, real price, warm but efficient. This is the CustomerContext case — Appendix B is the bar."},
    "renewal_due":             {"cta": "binary", "frame": "Subscription renewal. State days remaining and the renewal amount plainly; low-friction confirm."},
    "curious_ask_due":         {"cta": "open_ended", "frame": "Same as scheduled_recurring — ask the merchant something genuinely useful, don't report a status."},
    "winback_eligible":        {"cta": "binary", "frame": "Lapsed-subscription win-back. Loss aversion using the concrete perf dip and lapsed-customer count since expiry."},
    "ipl_match_today":         {"cta": "open_ended", "frame": "Local live-event relevance (a cricket match tonight). Tie to a concrete customer-behavior opportunity for this category tonight."},
    "wedding_package_followup": {"cta": "open_ended", "frame": "Time-boxed opportunity tied to a customer's known upcoming event (wedding). Effort externalization — offer to set up the next step."},
    "active_planning_intent":  {"cta": "none", "frame": "Merchant already showed intent in their last message — do not ask a qualifying question, move straight into drafting/planning the thing they asked about."},
    "seasonal_perf_dip":       {"cta": "open_ended", "frame": "Performance dip that is seasonally expected. Reassure with the seasonal context, then offer one concrete lever to counter it."},
    "customer_lapsed_hard":    {"cta": "binary", "frame": "A specific customer has fully lapsed. Internal signal to the merchant with a concrete win-back angle for that one customer's known history."},
    "trial_followup":          {"cta": "binary", "frame": "A trial/first-session customer is due for their next session. Offer the concrete next slot."},
    "supply_alert":            {"cta": "binary", "frame": "Pharmacy stock/recall alert. State the exact molecule/batch and the action needed, compliance tone, not sales tone."},
    "chronic_refill_due":      {"cta": "binary", "frame": "A specific customer's regular medication is about to run out. Offer the concrete refill/delivery action."},
    "category_seasonal":       {"cta": "open_ended", "frame": "Category-wide seasonal demand shift with concrete trend numbers. Offer a concrete shelf/stock/offer action."},
    "gbp_unverified":          {"cta": "binary", "frame": "Google Business Profile verification gap with an estimated uplift. Concrete, one clear next step."},
    "cde_opportunity":         {"cta": "open_ended", "frame": "A continuing-education / webinar opportunity relevant to the category. Curiosity + effort externalization."},
}

DEFAULT_HINT = {"cta": "open_ended", "frame": "Use the most specific, verifiable fact available across the contexts."}

RUBRIC_ANTI_PATTERNS = """
ANTI-PATTERNS the judge penalizes (never do these):
- Generic offers ("Flat 30% off") when a service+price offer exists in offer_catalog ("Haircut @ Rs99")
- Multiple CTAs in one message (e.g. "Reply YES for X, NO for Y")
- Burying the call-to-action anywhere but the last sentence
- Promotional/hype tone ("AMAZING DEAL!") for categories whose voice.tone is clinical/peer (e.g. dentists)
- Hallucinated data: never cite a source, name a competitor, or state a number that is not present in the
  provided contexts. If a needed fact is missing, write around it rather than inventing it.
- Long preambles ("I hope you're doing well...")
- Re-introducing yourself if conversation_history already shows prior turns
- Ignoring the merchant/customer's language preference (match hi-en code-mix when languages include "hi" or
  language_pref says "hi-en mix"; use hi-en naturally, not forced transliteration of every word)
- Repeating a message verbatim that appears in conversation_history
""".strip()

COMPULSION_LEVERS = """
COMPULSION LEVERS — use one or more, chosen to fit the trigger:
1. Specificity/verifiability (concrete number, date, headline, source citation)
2. Loss aversion ("you're missing X" / "before this window closes")
3. Social proof ("N other {category} in your area did Y this month") — use only if peer_stats/signals support it
4. Effort externalization ("I've drafted X — just say go" / "5-min setup")
5. Curiosity ("want to see who?")
6. Reciprocity ("I noticed Y, thought you'd want to know")
7. Asking the merchant a question (e.g. "what's your most-asked treatment this week?")
8. Single binary commitment (Reply YES / STOP) — only when cta_shape is "binary"
""".strip()

SYSTEM_PROMPT = """You are the composition engine for Vera, magicpin's merchant-facing (and, when
composing on a merchant's behalf, customer-facing) WhatsApp assistant for India's local-commerce
merchants. You are being benchmarked against production Vera and must outperform it.

You will be given four context layers as JSON: category, merchant, trigger, and optionally customer
(plus, if relevant, the resolved digest item the trigger points at). Compose exactly ONE next WhatsApp
message from these contexts alone.

{anti_patterns}

{levers}

CONSTRAINTS:
- No hard length cap, but stay concise and skimmable on a phone.
- Exactly one primary call-to-action. cta_shape tells you which kind: "binary" (phrase it so a one-word/
  one-number reply resolves it, e.g. "Reply YES" or "Reply 1 for Wed, 2 for Thu"), "open_ended" (end with
  one open question or a low-friction offer), or "none" (pure information, no CTA).
- Voice must match category.voice.tone and avoid category.voice.vocab_taboo; category.voice.vocab_allowed
  and tone_examples show the register to hit.
- Prefer offers from category.offer_catalog / merchant.offers over any generic discount language.
- Ground every specific claim (numbers, dates, names) in the provided JSON. Do not invent anything.
- If a CustomerContext is present, you are writing on behalf of the merchant TO their customer: use the
  customer's name, honor language_pref and preferred_slots, and only offer real prices/slots present in
  the trigger payload or merchant.offers.
- Match language: if merchant.identity.languages includes "hi" (or customer.identity.language_pref
  mentions "hi"), write naturally in Hindi-English code-mix (Latin script), the way an Indian professional
  texts a peer — not a literal translation of an English draft.
- Do not repeat, near-verbatim, any body already present in conversation_history.

Return ONLY a JSON object (no markdown fences, no commentary) with exactly these keys:
{{
  "body": "<the WhatsApp message text>",
  "cta": "binary" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "<copy verbatim from the trigger's suppression_key>",
  "rationale": "<one sentence: why this message, what it should achieve>"
}}
""".format(anti_patterns=RUBRIC_ANTI_PATTERNS, levers=COMPULSION_LEVERS)


def _trim_category(cat: dict) -> dict:
    return {
        "slug": cat.get("slug"),
        "voice": cat.get("voice"),
        "offer_catalog": cat.get("offer_catalog"),
        "peer_stats": cat.get("peer_stats"),
        "seasonal_beats": cat.get("seasonal_beats"),
        "trend_signals": cat.get("trend_signals"),
    }


def _trim_merchant(m: dict) -> dict:
    return {
        "merchant_id": m.get("merchant_id"),
        "identity": m.get("identity"),
        "subscription": m.get("subscription"),
        "performance": m.get("performance"),
        "offers": m.get("offers"),
        "conversation_history": (m.get("conversation_history") or [])[-5:],
        "customer_aggregate": m.get("customer_aggregate"),
        "signals": m.get("signals"),
        "review_themes": m.get("review_themes"),
    }


def build_user_prompt(bundle: dict) -> str:
    category, merchant, trigger, customer, digest_item = (
        bundle["category"], bundle["merchant"], bundle["trigger"],
        bundle.get("customer"), bundle.get("digest_item"),
    )
    hint = KIND_HINTS.get(trigger.get("kind"), DEFAULT_HINT)
    payload = {
        "category": _trim_category(category),
        "merchant": _trim_merchant(merchant),
        "trigger": trigger,
        "digest_item_resolved": digest_item,
        "customer": customer,
        "cta_shape": hint["cta"],
        "framing_hint": hint["frame"],
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _valid_shape(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    if not d.get("body") or not isinstance(d["body"], str) or not d["body"].strip():
        return False
    if d.get("cta") not in ("binary", "open_ended", "none"):
        return False
    if d.get("send_as") not in ("vera", "merchant_on_behalf"):
        return False
    return True


def llm_compose(bundle: dict) -> Optional[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=MODEL_NAME,
            max_tokens=600,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(bundle)}],
            timeout=LLM_TIMEOUT_S,
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
        parsed = _extract_json(text)
        if not _valid_shape(parsed):
            return None
        parsed.setdefault("suppression_key", bundle["trigger"].get("suppression_key", ""))
        return parsed
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback — no network, no API key required.
# ---------------------------------------------------------------------------

def _lang_mix(merchant: dict, customer: Optional[dict]) -> bool:
    if customer:
        return "hi" in (customer.get("identity", {}).get("language_pref") or "").lower()
    return "hi" in (merchant.get("identity", {}).get("languages") or [])


def _first_name(merchant: dict) -> str:
    ident = merchant.get("identity", {})
    return ident.get("owner_first_name") or ident.get("name", "there").split()[0]


def _best_offer(category: dict, merchant: dict) -> Optional[str]:
    for o in merchant.get("offers", []):
        if o.get("status") == "active":
            return o.get("title")
    catalog = category.get("offer_catalog", [])
    return catalog[0]["title"] if catalog else None


def _fmt_value(key: str, value: Any) -> Optional[str]:
    """Turn one payload field into a short human-readable fact fragment."""
    if value in (None, "", [], {}) or key in ("placeholder", "category", "top_item_id", "digest_item_id"):
        return None
    key_h = key.replace("_", " ")
    if isinstance(value, bool):
        return f"{key_h}: {'yes' if value else 'no'}"
    if isinstance(value, (int, float)):
        if "pct" in key:
            return f"{key_h.replace(' pct', '')} {value * 100:+.0f}%"
        return f"{key_h} {value}"
    if isinstance(value, str):
        return f"{key_h}: {value.replace('_', ' ')}" if len(value) < 60 else None
    if isinstance(value, list) and value and isinstance(value[0], str):
        return f"{key_h}: {', '.join(v.replace('_', ' ') for v in value[:3])}"
    return None


def _salient_facts(payload: dict, limit: int = 2) -> list[str]:
    facts = []
    for k, v in payload.items():
        f = _fmt_value(k, v)
        if f:
            facts.append(f)
    return facts[:limit]


def rule_based_compose(bundle: dict) -> dict:
    category, merchant, trigger = bundle["category"], bundle["merchant"], bundle["trigger"]
    customer, digest_item = bundle.get("customer"), bundle.get("digest_item")
    kind = trigger.get("kind", "")
    hint = KIND_HINTS.get(kind, DEFAULT_HINT)
    hi_mix = _lang_mix(merchant, customer)
    name = customer["identity"]["name"] if customer else _first_name(merchant)
    payload = trigger.get("payload", {}) or {}

    if payload.get("placeholder"):
        # Generated (non-seed) triggers sometimes carry no real payload facts.
        # Per the anti-fabrication rule, don't invent numbers — fall back to a
        # generic-but-honest framing anchored on the merchant's real offer/signals instead.
        offer = _best_offer(category, merchant)
        signal = next(iter(merchant.get("signals", [])), None)
        anchor = f"noticed {signal.replace('_', ' ').replace(':', ' at ')}" if signal else f"following up on {kind.replace('_', ' ')}"
        body = f"{name}, {anchor}."
        if offer:
            body += f" \"{offer}\" is live — want me to make sure it's front and center?"
        else:
            body += " Want me to take a look and suggest something concrete?"
    elif kind in ("research_digest", "category_research_digest_release") and digest_item:
        body = (
            f"{name}, {digest_item.get('source', 'a recent industry update')} flagged something worth "
            f"a look: {digest_item.get('title', '')}. "
            + (f"({digest_item.get('trial_n')}-sample study.) " if digest_item.get("trial_n") else "")
            + "Want me to pull the full item and draft something you can share or act on?"
        )
    elif kind == "regulation_change" and digest_item:
        title = digest_item.get("title", "a regulation change")
        deadline = payload.get("deadline_iso", "")
        deadline_note = "" if (deadline and deadline[:10] in title) else (f" (effective {deadline})" if deadline else "")
        actionable = digest_item.get("actionable", "")
        actionable_sep = f" {actionable}." if actionable and not actionable.endswith(".") else f" {actionable}"
        body = f"{name}, heads up — {title}{deadline_note}.{actionable_sep} Want me to send a checklist?"
    elif kind == "perf_dip":
        metric = payload.get("metric", "calls")
        delta = payload.get("delta_pct", 0)
        body = (
            f"{name}, your {metric} are down {abs(delta) * 100:.0f}% this {payload.get('window', 'week')} "
            f"vs your baseline of {payload.get('vs_baseline', 'usual')}. I can check your listing for the "
            f"likely cause and fix it — want me to?"
        )
    elif kind == "perf_spike":
        perf = merchant.get("performance", {})
        delta = payload.get("delta_pct") or perf.get("delta_7d", {}).get("views_pct", 0)
        body = (
            f"{name}, nice jump — views are up {delta * 100:.0f}% this week ({perf.get('views', '')} total). "
            f"Want me to draft a post to keep the momentum going while it's hot?"
        )
    elif kind == "milestone_reached":
        peer = category.get("peer_stats", {})
        metric = (payload.get("metric") or "milestone").replace("_", " ")
        value_now = payload.get("value_now") or payload.get("milestone_value")
        peer_avg = peer.get("avg_reviews")
        milestone_txt = f"{value_now} {metric}" if value_now else metric
        peer_txt = f" — category average is {peer_avg}, so you're ahead of most peers" if peer_avg else ""
        body = f"{name}, milestone — {milestone_txt} reached{peer_txt}. Want me to turn it into a post?"
    elif kind == "dormant_with_vera":
        body = (
            f"{name}, haven't heard from you in a while — no pressure, just checking in. "
            f"Anything on your listing or offers you'd like a hand with this week? Reply YES if so, or STOP to pause these check-ins."
        )
    elif kind == "appointment_tomorrow" and customer:
        slot = payload.get("slot_label") or payload.get("time") or "tomorrow"
        body = f"Hi {name}, quick reminder — your appointment is {slot}. Reply 1 to confirm or 2 to reschedule."
    elif kind == "recall_due" and customer:
        slots = payload.get("available_slots", [])
        slot_txt = " ya ".join(s.get("label", "") for s in slots[:2]) if slots else "aapke convenient time pe"
        offer = _best_offer(category, merchant)
        service_due = (payload.get("service_due") or "recall").replace("_", " ")
        body = (
            f"Hi {name}, {merchant['identity']['name']} here. "
            f"Aapka {service_due} due hai. "
            f"Slots: {slot_txt}."
            + (f" {offer}." if offer else "")
            + " Reply 1 or 2, ya koi aur time batayein."
        )
    elif kind == "review_theme_emerged":
        themes = merchant.get("review_themes", [])
        theme = next((t for t in themes if t.get("sentiment") == "neg"), themes[0] if themes else None)
        if theme:
            body = (
                f"{name}, {theme.get('occurrences_30d')} reviews this month mentioned \"{theme.get('theme', '').replace('_', ' ')}\". "
                f"Want me to draft a response template + a fix you can put in place?"
            )
        else:
            body = f"{name}, a review theme came up this month worth a look — want the details?"
    elif kind == "scheduled_recurring":
        body = f"{name}, quick one for the week — what's the most-asked question or service request you've gotten lately? Might be worth a post."
    elif kind in ("festival_upcoming", "weather_heatwave", "local_news_event", "competitor_opened", "category_trend_movement"):
        offer = _best_offer(category, merchant)
        facts = _salient_facts(payload, limit=3)
        detail = "; ".join(facts) if facts else kind.replace("_", " ")
        body = f"{name}, {detail}" + (f" — worth pushing \"{offer}\" this week?" if offer else " — want a quick post drafted for this?")
    else:
        # No bespoke template for this trigger.kind — anchor on whatever concrete
        # facts the payload actually has rather than falling back to a generic line.
        topic = kind.replace("_", " ") if kind else "an update"
        facts = _salient_facts(payload)
        offer = _best_offer(category, merchant)
        if facts:
            body = f"{name}, on {topic}: {'; '.join(facts)}."
        else:
            body = f"{name}, quick note on {topic}."
        if offer and hint["cta"] != "none":
            body += f" Worth pairing with \"{offer}\"?"
        elif hint["cta"] == "open_ended":
            body += " Want me to take the next step on this?"
        elif hint["cta"] == "binary":
            body += " Reply YES if you'd like me to act on this, or STOP to skip."

    send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"
    return {
        "body": body,
        "cta": hint["cta"],
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": f"Rule-based composition for trigger kind '{kind}'; anchored on {'digest item' if digest_item else 'trigger payload'}.",
    }


def compose(bundle: dict) -> dict:
    """Never raises. Tries LLM, validates, falls back to rule-based."""
    result = llm_compose(bundle)
    if result is None:
        result = rule_based_compose(bundle)
    return result
