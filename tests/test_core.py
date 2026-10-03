import json
from pathlib import Path

import pytest

from lora_email_intent.data import load_jsonl, split
from lora_email_intent.encoding import IGNORE, collate, encode_example
from lora_email_intent.metrics import INVALID, evaluate_predictions, majority_baseline
from lora_email_intent.prompts import INTENTS, build_prompt, parse_lenient, parse_prediction

DATA = Path(__file__).parent.parent / "data" / "emails.jsonl"


class CharTok:
    """Tiny stand-in tokenizer: one token per character; eos id 0."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 250 + 1 for c in text]}


def test_dataset_is_balanced_and_valid():
    rows = load_jsonl(DATA)
    assert len(rows) == 120
    assert {i: sum(r["intent"] == i for r in rows) for i in INTENTS} == {i: 20 for i in INTENTS}
    assert len({r["id"] for r in rows}) == 120 and len({r["email"].lower() for r in rows}) == 120


def test_dataset_matches_generator():
    import subprocess, sys
    before = DATA.read_text()
    subprocess.run([sys.executable, str(DATA.parent / "build_dataset.py")], cwd=DATA.parent.parent, check=True, capture_output=True)
    assert DATA.read_text() == before


def test_split_is_stratified_disjoint_and_deterministic():
    rows = load_jsonl(DATA)
    train_rows, val_rows, test_rows = split(rows)
    assert (len(train_rows), len(val_rows), len(test_rows)) == (72, 12, 36)
    ids = [set(r["id"] for r in s) for s in (train_rows, val_rows, test_rows)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert all(sum(r["intent"] == i for r in test_rows) == 6 for i in INTENTS)
    assert [r["id"] for r in split(rows)[2]] == [r["id"] for r in test_rows]
    assert [r["id"] for r in split(rows, seed=1)[2]] != [r["id"] for r in test_rows]


def test_split_detects_leakage_and_small_classes():
    rows = [{"id": f"{i}-{n}", "email": f"mail {n}" if n else "same", "intent": i} for i in INTENTS for n in range(9)]
    with pytest.raises(ValueError):
        split(rows, val_per_class=2, test_per_class=6)           # only 9 rows per class
    rows = [{"id": f"{i}-{n}", "email": "same text", "intent": i} for i in INTENTS for n in range(10)]
    with pytest.raises(ValueError, match="duplicate"):
        split(rows)


def test_load_jsonl_validation(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"id": "a", "email": "hi", "intent": "nope"}) + "\n")
    with pytest.raises(ValueError, match="unknown intent"):
        load_jsonl(p)
    p.write_text(json.dumps({"id": "a", "email": "  ", "intent": "other"}) + "\n")
    with pytest.raises(ValueError, match="empty"):
        load_jsonl(p)


def test_prompt_lists_all_intents_and_ends_with_answer_slot():
    p = build_prompt("  hello  ")
    assert all(i in p for i in INTENTS) and "### Email:\nhello\n" in p and p.endswith("### Intent:\n")


@pytest.mark.parametrize("text,expected", [
    ("pricing_question", "pricing_question"), ("  other\n", "other"), ("bug_or_problem\nextra lines", "bug_or_problem"),
    ("PRICING_QUESTION", "pricing_question"),
    ("The intent is pricing_question", None), ("pricing", None), ("", None), ("other other", None)])
def test_parse_prediction_is_strict(text, expected):
    assert parse_prediction(text) == expected


def test_parse_lenient_finds_first_label():
    assert parse_lenient("The intent is pricing_question, not other") == "pricing_question"
    assert parse_lenient("I do not know") is None


def test_encoding_masks_prompt_and_adds_eos():
    tok = CharTok()
    e = encode_example(tok, "hi", "other")
    n_prompt = len(tok(build_prompt("hi"))["input_ids"])
    assert e["labels"][:n_prompt] == [IGNORE] * n_prompt
    assert e["labels"][n_prompt:] == e["input_ids"][n_prompt:] and e["input_ids"][-1] == 0
    assert len(e["labels"]) - n_prompt == len("other") + 1
    full = encode_example(tok, "hi", "other", train_on_inputs=True)
    assert full["labels"] == full["input_ids"]


def test_encoding_rejects_too_long():
    with pytest.raises(ValueError):
        encode_example(CharTok(), "x" * 500, "other", max_len=384)


def test_collate_pads_labels_with_ignore():
    b = collate([{"input_ids": [1, 2, 3], "labels": [IGNORE, 2, 3], "attention_mask": [1, 1, 1]},
                 {"input_ids": [4], "labels": [4], "attention_mask": [1]}], pad_id=9)
    assert b["input_ids"][1] == [4, 9, 9] and b["labels"][1] == [4, IGNORE, IGNORE] and b["attention_mask"][1] == [1, 0, 0]


def test_metrics_perfect_and_invalid():
    gold = ["other", "pricing_question", "other"]
    m = evaluate_predictions(gold, gold)
    assert m["accuracy"] == 1 and m["invalid_rate"] == 0
    m = evaluate_predictions(gold, ["other", None, "pricing_question"])
    assert m["accuracy"] == pytest.approx(1 / 3) and m["invalid_rate"] == pytest.approx(1 / 3)
    assert m["confusion"]["pricing_question"][INVALID] == 1
    assert m["per_class"]["other"]["recall"] == 0.5 and m["per_class"]["other"]["precision"] == 1.0
    with pytest.raises(ValueError):
        evaluate_predictions([], [])


def test_majority_baseline():
    assert majority_baseline(["a", "a", "b"], ["a", "b", "a", "a"]) == 0.75
