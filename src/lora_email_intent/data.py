"""Dataset loading and a deterministic, stratified train/val/test split."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .prompts import INTENTS


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("intent") not in INTENTS:
            raise ValueError(f"line {n}: unknown intent {r.get('intent')!r}")
        if not isinstance(r.get("email"), str) or not r["email"].strip():
            raise ValueError(f"line {n}: empty email")
        rows.append(r)
    return rows


def split(rows: list[dict], val_per_class: int = 2, test_per_class: int = 6, seed: int = 42):
    """Stratified split. Same seed -> same split, so train/test never overlap between runs."""
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["intent"]].append(r)
    train, val, test = [], [], []
    rng = random.Random(seed)
    for label in INTENTS:
        items = sorted(by_label[label], key=lambda r: r["id"])
        rng.shuffle(items)
        if len(items) <= val_per_class + test_per_class:
            raise ValueError(f"class {label} has only {len(items)} rows")
        test += items[:test_per_class]
        val += items[test_per_class:test_per_class + val_per_class]
        train += items[test_per_class + val_per_class:]
    seen = [r["email"].strip().lower() for r in train]
    leaks = [r["id"] for r in val + test if r["email"].strip().lower() in seen]
    if leaks:
        raise ValueError(f"duplicate emails across splits: {leaks}")
    return train, val, test
