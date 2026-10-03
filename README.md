# lora-email-intent

LoRA fine-tuning of a small open LLM (**Qwen2.5-0.5B-Instruct**) to classify customer emails into six intents, with an
evaluation that compares the untuned model, the tuned model and a majority-class baseline on a held-out test set.

## The problem
Routing support and sales emails by intent is a narrow, well-defined task where a small fine-tuned model can be cheap and
consistent. The harder part is measuring honestly what fine-tuning changed. This project trains a LoRA adapter and evaluates it
with strict output parsing, baselines, per-class metrics and a leakage-checked split.

Intents: `meeting_request`, `invoice_or_payment`, `bug_or_problem`, `pricing_question`, `cancellation_refund`, `other`

## Results
One run on a free Colab T4 GPU (seed 42, 5 epochs, 45 optimizer steps, about 18 seconds of training). Test set: 36 held-out emails (6 per intent).
Condensed numbers: [results/results_summary.json](results/results_summary.json).

| Model | Strict accuracy | Lenient accuracy | Macro-F1 (strict) | Invalid answers |
|---|---|---|---|---|
| Majority-class baseline | 16.7% | – | – | – |
| Qwen2.5-0.5B-Instruct, zero-shot | 33.3% (12/36) | 33.3% | 0.258 | 5.6% (2/36) |
| + LoRA | **94.4% (34/36)** | 94.4% | **0.943** | 0.0% |

- 1,081,344 trainable parameters out of 495,114,112 (0.22%). Validation loss fell from 2.28 after epoch 1 to 0.14 after epoch 5.
- The untuned model put `meeting_request` (the first label in the prompt's list) on its first line for 28 of 36 emails regardless of content, then kept writing an explanation. Its score therefore reflects this small model with this prompt, not zero-shot prompting in general.
- The tuned model's only errors: 2 of 6 `invoice_or_payment` emails labelled `bug_or_problem`. Every other class was 6/6.
- With 36 test emails one email is about 2.8 points, so treat small differences as noise.

## Features
- Original dataset: 120 hand-written fictional emails (20 per intent), no scraped, personal or machine-generated data
- Deterministic stratified split: 72 train / 12 validation / 36 test, with a duplicate-leakage check
- LoRA through PEFT (r=8, alpha=16, dropout 0.05, attention projections `q/k/v/o_proj`), loss on the answer tokens only
- Strict output parsing: an answer that is not exactly one valid label counts as **invalid**, not as a guess; a lenient parser gives the untuned model a fair second view
- Evaluation: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, invalid-answer rate, majority-class baseline, base vs tuned
- 21 tests (the torch-dependent ones are skipped if torch is not installed)

## Architecture
```
data/build_dataset.py -> data/emails.jsonl -> data.split() -> train / val / test
                                                |
prompts.build_prompt  --> encoding.encode_example (prompt tokens masked with -100, answer + EOS learned)
                                                |
train.py: base model --(peft LoraConfig)--> Trainer --> outputs/lora (adapter + training_summary.json)
                                                |
evaluate.py: test set -> generate (greedy) -> parse_prediction -> metrics
             base model (zero-shot)  vs  base + adapter  (+ majority baseline)
```
Stack: Python 3.10+, PyTorch, Hugging Face Transformers, PEFT, pytest.

## Key technical decisions
- **Answer-only loss.** Prompt tokens are masked with `-100`, so all gradient goes to producing the label and its end-of-sequence token.
- **Strict parsing with an explicit invalid class.** An unusable answer is a failure, never silently mapped to a label.
- **Baselines before claims.** Majority-class and zero-shot baselines are reported next to the tuned model.
- **Deterministic split with a leakage check.** The same seed always gives the same split, and no email text appears in more than one split.
- **Small base model, fp32 weights with fp16 autocast.** Trains on a free Colab T4 in well under a minute.
- **Offline smoke test.** The full train, save, reload, generate and evaluate path is tested on a tiny random-weight model without downloads.

## How to run
Tests (CPU, about 10 s; torch tests are skipped if torch is not installed):
```bash
pip install -e ".[dev]"          # core + pytest
pip install -e ".[train]"        # torch, transformers, peft, accelerate (for the smoke test and real training)
python -m pytest
```
Training and evaluation on a GPU machine:
```bash
python -m lora_email_intent.train --epochs 5 --out outputs/lora
python -m lora_email_intent.evaluate --adapter outputs/lora
```
Or open [notebooks/colab_train_and_evaluate.ipynb](notebooks/colab_train_and_evaluate.ipynb) in Google Colab, choose a T4 GPU and run all cells.
The base model is downloaded from Hugging Face on first use.

## Limitations
- Small, synthetic, English-only dataset written by one author; the emails are short and fairly prototypical, so real inboxes will be harder and the score here will not transfer
- Single seed and single split, no hyperparameter search; 36 test emails give noisy estimates
- Strict and lenient parsing depend on exact label strings; there is no calibration or confidence estimate
- Results can vary slightly with library versions and hardware

## License
Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party attributions: [SOURCES.md](SOURCES.md).
