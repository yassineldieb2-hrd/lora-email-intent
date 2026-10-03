"""Compare the untuned base model (zero-shot) with the LoRA-tuned model on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_jsonl, split
from .metrics import evaluate_predictions, format_report, majority_baseline
from .prompts import build_prompt, parse_lenient, parse_prediction


def generate(model, tokenizer, email: str, max_new_tokens: int = 12) -> str:
    import torch
    ids = tokenizer(build_prompt(email), return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate_model(model, tokenizer, rows: list[dict]) -> dict:
    generations = [generate(model, tokenizer, r["email"]) for r in rows]
    gold = [r["intent"] for r in rows]
    return {
        "strict": evaluate_predictions(gold, [parse_prediction(g) for g in generations]),
        "lenient": evaluate_predictions(gold, [parse_lenient(g) for g in generations]),
        "examples": [{"id": r["id"], "gold": r["intent"], "generated": g} for r, g in zip(rows, generations)],
    }


def main(argv=None):
    import torch
    import transformers
    from peft import PeftModel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="outputs/lora")
    p.add_argument("--model", default=None, help="base model; default: the one recorded in the adapter's training summary")
    p.add_argument("--data", default="data/emails.jsonl")
    p.add_argument("--results", default="outputs/eval_results.json")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    summary = json.loads((Path(a.adapter) / "training_summary.json").read_text())
    base_name = a.model or summary["base_model"]
    train_rows, _, test_rows = split(load_jsonl(a.data), seed=a.seed)
    tok = transformers.AutoTokenizer.from_pretrained(a.adapter)
    base = transformers.AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32)
    if torch.cuda.is_available():
        base = base.to("cuda")
    base.eval()

    results = {"base_model": base_name, "n_test": len(test_rows),
               "majority_class_baseline_accuracy": majority_baseline([r["intent"] for r in train_rows],
                                                                     [r["intent"] for r in test_rows])}
    results["base_zero_shot"] = evaluate_model(base, tok, test_rows)
    tuned = PeftModel.from_pretrained(base, a.adapter).eval()
    results["lora_tuned"] = evaluate_model(tuned, tok, test_rows)

    print(f"majority-class baseline accuracy: {results['majority_class_baseline_accuracy']:.3f}")
    for name in ("base_zero_shot", "lora_tuned"):
        print(format_report(f"{name} [strict parse]", results[name]["strict"]))
        print(f"   lenient-parse accuracy: {results[name]['lenient']['accuracy']:.3f}\n")
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    Path(a.results).write_text(json.dumps(results, indent=2))
    print(f"saved {a.results}")


if __name__ == "__main__":
    main()
