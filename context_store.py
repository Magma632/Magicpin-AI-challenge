"""
context_store.py — in-memory, idempotent, versioned store for the 4 context
scopes (category / merchant / customer / trigger), per challenge-testing-brief.md §2.1.

Kept deliberately dependency-free so it can be unit tested without spinning up
the HTTP server.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _Entry:
    version: int
    payload: dict


class ContextStore:
    """(scope, context_id) -> versioned payload, with simple secondary indexes."""

    VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[tuple[str, str], _Entry] = {}

    # -- writes -----------------------------------------------------------
    def push(self, scope: str, context_id: str, version: int, payload: dict) -> dict:
        if scope not in self.VALID_SCOPES:
            return {"accepted": False, "reason": "invalid_scope", "details": f"unknown scope '{scope}'"}
        if not context_id:
            return {"accepted": False, "reason": "invalid_scope", "details": "missing context_id"}
        with self._lock:
            key = (scope, context_id)
            cur = self._store.get(key)
            if cur and cur.version >= version:
                return {"accepted": False, "reason": "stale_version", "current_version": cur.version}
            self._store[key] = _Entry(version=version, payload=payload)
        return {"accepted": True}

    def wipe(self) -> None:
        with self._lock:
            self._store.clear()

    # -- reads --------------------------------------------------------------
    def get(self, scope: str, context_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry.payload if entry else None

    def counts(self) -> dict:
        counts = {s: 0 for s in self.VALID_SCOPES}
        with self._lock:
            for (scope, _), _ in self._store.items():
                counts[scope] = counts.get(scope, 0) + 1
        return counts

    def all_triggers(self) -> list[dict]:
        with self._lock:
            return [e.payload for (scope, _), e in self._store.items() if scope == "trigger"]

    # -- convenience resolvers ----------------------------------------------
    def category(self, slug: str) -> Optional[dict]:
        return self.get("category", slug)

    def merchant(self, merchant_id: str) -> Optional[dict]:
        return self.get("merchant", merchant_id)

    def customer(self, customer_id: str) -> Optional[dict]:
        return self.get("customer", customer_id) if customer_id else None

    def trigger(self, trigger_id: str) -> Optional[dict]:
        return self.get("trigger", trigger_id)

    def digest_item(self, category_slug: str, item_id: str) -> Optional[dict]:
        """Resolve a trigger's payload.top_item_id against the category's digest list."""
        cat = self.category(category_slug)
        if not cat or not item_id:
            return None
        for item in cat.get("digest", []):
            if item.get("id") == item_id:
                return item
        return None

    def resolve_bundle(self, trigger_id: str) -> Optional[dict]:
        """Given a trigger_id, resolve category+merchant+trigger(+customer+digest_item).

        Returns None if any required piece is missing (bot should skip the tick
        rather than compose from partial context / hallucinate).
        """
        trg = self.trigger(trigger_id)
        if not trg:
            return None
        merchant_id = trg.get("merchant_id")
        merchant = self.merchant(merchant_id) if merchant_id else None
        if not merchant:
            return None
        category = self.category(merchant.get("category_slug"))
        if not category:
            return None
        customer_id = trg.get("customer_id")
        customer = self.customer(customer_id) if customer_id else None
        digest_item = None
        top_item_id = (trg.get("payload") or {}).get("top_item_id")
        if top_item_id:
            digest_item = self.digest_item(merchant.get("category_slug"), top_item_id)
        return {
            "category": category,
            "merchant": merchant,
            "trigger": trg,
            "customer": customer,
            "digest_item": digest_item,
        }
