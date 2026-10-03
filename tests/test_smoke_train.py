"""End-to-end smoke test of the training + evaluation CODE PATH on a tiny random-weight model.

This proves the pipeline runs (tokenise -> LoRA wrap -> Trainer -> save adapter -> reload -> generate -> parse
-> metrics). It says NOTHING about model quality: the model is random and offline.
Skipped automatically when torch/transformers/peft are not installed.
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from lora_email_intent.data import load_jsonl, split
from lora_email_intent.evaluate import evaluate_model
from lora_email_intent.train import train

DATA = Path(__file__).parent.parent / "data" / "emails.jsonl"


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    rows = load_jsonl(DATA)
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    tk.train_from_iterator([r["email"] for r in rows] + ["### Email: ### Intent: Classify"] + [r["intent"] for r in rows],
                           trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<eos>"],
                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512)
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(cfg), tok, tmp_path_factory.mktemp("lora")


def test_train_saves_adapter_and_summary_then_reload_and_evaluate(tiny):
    model, tok, out = tiny
    cfg = model.config
    summary = train(data_path=str(DATA), output_dir=str(out), model=model, tokenizer=tok,
                    epochs=1, max_steps=3, batch_size=4, grad_accum=1, max_len=512)
    assert (out / "adapter_config.json").exists() and (out / "adapter_model.safetensors").exists()
    saved = json.loads((out / "training_summary.json").read_text())
    assert saved["n_train"] == 72 and 0 < saved["trainable_parameters"] < saved["total_parameters"]
    assert any("eval_loss" in h for h in saved["log_history"]) and summary["hyperparameters"]["train_on_inputs"] is False

    from peft import PeftModel
    # reload the saved adapter onto a fresh copy of the same random base
    base = transformers.AutoModelForCausalLM.from_config(cfg)
    reloaded = PeftModel.from_pretrained(base, str(out)).eval()
    _, _, test_rows = split(load_jsonl(DATA))
    res = evaluate_model(reloaded, tok, test_rows[:6])
    assert res["strict"]["n"] == 6 and len(res["examples"]) == 6
    assert 0.0 <= res["strict"]["accuracy"] <= 1.0 and 0.0 <= res["strict"]["invalid_rate"] <= 1.0


def test_evaluate_cli_end_to_end_on_local_tiny_base(tiny, tmp_path):
    """Runs lora_email_intent.evaluate.main exactly as a user would, against a locally saved tiny base model."""
    from lora_email_intent import evaluate as ev
    _, tok, _ = tiny
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512)
    base = transformers.AutoModelForCausalLM.from_config(cfg)
    base_dir, out_dir, results = tmp_path / "base", tmp_path / "lora", tmp_path / "res.json"
    base.save_pretrained(base_dir)
    train(model_name=str(base_dir), data_path=str(DATA), output_dir=str(out_dir), model=base, tokenizer=tok,
          epochs=1, max_steps=2, batch_size=4, grad_accum=1, max_len=512)
    ev.main(["--adapter", str(out_dir), "--model", str(base_dir), "--data", str(DATA), "--results", str(results)])
    r = json.loads(results.read_text())
    assert r["n_test"] == 36 and set(r) >= {"base_zero_shot", "lora_tuned", "majority_class_baseline_accuracy"}
    assert r["lora_tuned"]["strict"]["n"] == 36 and 0 <= r["majority_class_baseline_accuracy"] <= 1
