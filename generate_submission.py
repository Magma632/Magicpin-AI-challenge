#!/usr/bin/env python3
"""
generate_submission.py — produce submission.jsonl (challenge-brief.md §7.2)
for the 30 canonical (merchant, trigger[, customer]) test pairs, by calling
composer.compose() directly (no HTTP round-trip needed for this step).

Usage:
    python generate_submission.py --dataset ./dataset --out submission.jsonl

Set ANTHROPIC_API_KEY beforehand to use the LLM composer; without it, the
deterministic rule-based fallback in composer.py is used (still produces a
valid, schema-correct submission, just with less adaptive copy).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import composer
from context_store import ContextStore


def load_dataset(store: ContextStore, dataset_dir: Path) -> None:
    for f in (dataset_dir / "categories").glob("*.json"):
        payload = json.loads(f.read_text(encoding="utf-8"))
        store.push("category", payload["slug"], 1, payload)
    for f in (dataset_dir / "merchants").glob("*.json"):
        payload = json.loads(f.read_text(encoding="utf-8"))
        store.push("merchant", payload["merchant_id"], 1, payload)
    for f in (dataset_dir / "customers").glob("*.json"):
        payload = json.loads(f.read_text(encoding="utf-8"))
        store.push("customer", payload["customer_id"], 1, payload)
    for f in (dataset_dir / "triggers").glob("*.json"):
        payload = json.loads(f.read_text(encoding="utf-8"))
        store.push("trigger", payload["id"], 1, payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="submission.jsonl")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    store = ContextStore()
    load_dataset(store, dataset_dir)

    test_pairs = json.loads((dataset_dir / "test_pairs.json").read_text(encoding="utf-8"))["pairs"]

    used_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    lines = []
    for pair in test_pairs:
        test_id = pair["test_id"]
        bundle = store.resolve_bundle(pair["trigger_id"])
        if not bundle:
            lines.append(json.dumps({
                "test_id": test_id, "body": "", "cta": "none", "send_as": "vera",
                "suppression_key": "", "rationale": "SKIPPED: could not resolve category/merchant/trigger from dataset.",
            }, ensure_ascii=False))
            continue
        # test_pairs.json may pin a specific customer_id different from the trigger's own
        if pair.get("customer_id") and not bundle.get("customer"):
            bundle["customer"] = store.customer(pair["customer_id"])
        composed = composer.compose(bundle)
        lines.append(json.dumps({
            "test_id": test_id,
            "body": composed["body"],
            "cta": composed["cta"],
            "send_as": composed["send_as"],
            "suppression_key": composed.get("suppression_key", ""),
            "rationale": composed.get("rationale", ""),
        }, ensure_ascii=False))

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {args.out} (composer path: {'LLM' if used_llm else 'rule-based fallback'})")


if __name__ == "__main__":
    main()
