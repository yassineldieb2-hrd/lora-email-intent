"""Turn (email, intent) into model inputs with the loss restricted to the answer.

alpaca-lora defaults to `train_on_inputs=True` (loss over prompt + answer). For classification
that spends most of the gradient on re-learning the prompt text, so here the prompt tokens are
masked with -100 by default and only the label (+EOS) is learned.
"""
from __future__ import annotations

from .prompts import build_prompt

IGNORE = -100


def encode_example(tokenizer, email: str, intent: str, max_len: int = 384, train_on_inputs: bool = False) -> dict:
    prompt_ids = tokenizer(build_prompt(email), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(intent, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(answer_ids) > max_len:
        raise ValueError(f"example needs {len(prompt_ids) + len(answer_ids)} tokens, max_len is {max_len}")
    input_ids = prompt_ids + answer_ids
    labels = (input_ids[:] if train_on_inputs else [IGNORE] * len(prompt_ids) + answer_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad to the longest example. Returns plain lists; the trainer wrapper converts to tensors."""
    longest = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = longest - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [IGNORE] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return out
