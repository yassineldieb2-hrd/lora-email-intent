"""LoRA fine-tuning with Hugging Face Transformers + PEFT.

Structure follows tloen/alpaca-lora's finetune.py (Apache-2.0): load base model, wrap with a LoRA
config (r=8, alpha=16, dropout 0.05), tokenize prompts, train with the Trainer, save only the adapter.
Modernised and trimmed for a classification task: no int8/bitsandbytes, no wandb, current PEFT API,
prompt tokens masked from the loss, a val split with eval loss, config saved next to the adapter.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import load_jsonl, split
from .encoding import collate, encode_example

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def train(model_name: str = DEFAULT_MODEL, data_path: str = "data/emails.jsonl", output_dir: str = "outputs/lora",
          epochs: int = 5, lr: float = 2e-4, batch_size: int = 4, grad_accum: int = 2,
          lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05,
          target_modules: tuple[str, ...] = DEFAULT_TARGETS, max_len: int = 384,
          train_on_inputs: bool = False, seed: int = 42, model=None, tokenizer=None, max_steps: int = -1) -> dict:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    transformers.set_seed(seed)
    random.seed(seed)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        # fp32 weights + fp16 autocast: simplest setting that is stable on a free Colab T4 for a 0.5B model
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)

    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                     target_modules=list(target_modules), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.3f}%)")

    train_rows, val_rows, _ = split(load_jsonl(data_path), seed=seed)
    enc = lambda rows: [encode_example(tokenizer, r["email"], r["intent"], max_len, train_on_inputs) for r in rows]

    class DS(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    def collator(batch):
        return {k: torch.tensor(v) for k, v in collate(batch, tokenizer.pad_token_id).items()}

    steps_per_epoch = -(-len(train_rows) // (batch_size * grad_accum))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * epochs
    warmup_steps = max(1, int(0.1 * total_steps))  # integer steps: portable across transformers versions

    args = transformers.TrainingArguments(
        output_dir=output_dir, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, num_train_epochs=epochs, max_steps=max_steps,
        learning_rate=lr, warmup_steps=warmup_steps, lr_scheduler_type="cosine", logging_steps=5,
        eval_strategy="epoch", save_strategy="no", report_to="none", seed=seed,
        fp16=torch.cuda.is_available(), use_cpu=not torch.cuda.is_available(), remove_unused_columns=False)
    trainer = transformers.Trainer(model=model, args=args, train_dataset=DS(enc(train_rows)),
                                   eval_dataset=DS(enc(val_rows)), data_collator=collator)
    trainer.train()

    out = Path(output_dir)
    model.save_pretrained(out)  # adapter weights only
    tokenizer.save_pretrained(out)
    summary = {"base_model": model_name, "n_train": len(train_rows), "n_val": len(val_rows),
               "trainable_parameters": trainable, "total_parameters": total,
               "hyperparameters": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
                                   "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
                                   "target_modules": list(target_modules), "max_len": max_len,
                                   "train_on_inputs": train_on_inputs, "seed": seed},
               "log_history": trainer.state.log_history}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default="data/emails.jsonl")
    p.add_argument("--out", default="outputs/lora")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--train-on-inputs", action="store_true", help="also compute loss on the prompt (alpaca-lora default)")
    a = p.parse_args(argv)
    train(a.model, a.data, a.out, a.epochs, a.lr, lora_r=a.lora_r, lora_alpha=2 * a.lora_r,
          train_on_inputs=a.train_on_inputs)


if __name__ == "__main__":
    main()
