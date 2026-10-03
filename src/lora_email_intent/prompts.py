"""Intent labels, the prompt template, and strict parsing of the model's answer.

The '### Section' prompt style is the one popularised by Alpaca / alpaca-lora; the content is new.
"""
from __future__ import annotations

import re

INTENTS = ["meeting_request", "invoice_or_payment", "bug_or_problem",
           "pricing_question", "cancellation_refund", "other"]

TEMPLATE = (
    "Classify the customer email into exactly one intent.\n"
    "Intents: {intents}\n\n"
    "### Email:\n{email}\n\n"
    "### Intent:\n"
)


def build_prompt(email: str) -> str:
    return TEMPLATE.format(intents=", ".join(INTENTS), email=email.strip())


def parse_prediction(generated: str) -> str | None:
    """Return the intent if the generation starts with exactly one valid label, else None.
    Anything else (extra words, unknown label, empty) counts as an invalid answer, not a guess."""
    m = re.match(r"\s*([a-z_]+)\s*$", generated.strip().split("\n")[0].lower()) if generated.strip() else None
    if m and m.group(1) in INTENTS:
        return m.group(1)
    return None


def parse_lenient(generated: str) -> str | None:
    """Earliest valid label mentioned anywhere in the first 200 characters, else None.
    Used to give the *untuned* model a fair chance: it may know the answer but not the output format."""
    text = generated.strip().lower()[:200]
    hits = [(text.find(i), i) for i in INTENTS if i in text]
    return min(hits)[1] if hits else None
