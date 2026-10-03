# lora-email-intent

Fine-tune a small open LLM (**Qwen2.5-0.5B-Instruct**) with **LoRA** to classify customer emails into six intents, and measure
honestly what fine-tuning changed by comparing the untuned model with the tuned one on a held-out test set.

Based in part on [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) (Apache-2.0). See [SOURCES.md](SOURCES.md) and [NOTICE](NOTICE).

> **Status:** trained and evaluated on a Colab T4 GPU (numbers below, from one run, one seed). The dataset is small and synthetic,
> so treat the results as a learning exercise, not a benchmark. The test suite additionally exercises the whole pipeline on a tiny random model.

## Why I built it
To learn what LoRA/PEFT actually does, what an evaluation of a fine-tune should look like (baselines, strict vs lenient parsing, per-class
metrics), and how to avoid common mistakes such as train/test leakage and unreported license terms of training data.

## Intents
`meeting_request`, `invoice_or_payment`, `bug_or_problem`, `pricing_question`, `cancellation_refund`, `other`

## Features
- Original dataset: 120 hand-written fictional emails (20 per intent) - no scraped, personal or machine-generated data
- Deterministic stratified split: 72 train / 12 validation / 36 test, with a duplicate-leakage check
- LoRA via PEFT (r=8, alpha=16, dropout 0.05, attention projections), loss on answer tokens only
- Strict output parsing: an answer that is not exactly one valid label counts as **invalid**, not as a guess
- Evaluation: accuracy, macro-F1, per-class P/R/F1, confusion matrix, invalid-answer rate, majority-class baseline, base-vs-tuned
- 22 tests; no GPU needed to run them

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

## Tech stack
Python 3.10+, PyTorch, Hugging Face Transformers, PEFT, pytest. Colab T4 for training.

## How to run
Tests (CPU, ~10 s; torch tests are skipped if torch isn't installed):
```bash
pip install -e ".[dev]"          # core + pytest
pip install -e ".[train]"        # torch, transformers, peft, accelerate (for the smoke test and real training)
python -m pytest
```
Real training and evaluation, easiest on free Colab: open [notebooks/colab_train_and_evaluate.ipynb](notebooks/colab_train_and_evaluate.ipynb),
choose a T4 GPU, run all cells. Or on any machine with a GPU:
```bash
python -m lora_email_intent.train --epochs 5 --out outputs/lora
python -m lora_email_intent.evaluate --adapter outputs/lora
```

## Results
One run on a free Colab T4 (seed 42, 5 epochs, 45 optimizer steps, about 18 seconds of training). Test set: 36 held-out emails (6 per intent).
Condensed numbers are in [results/results_summary.json](results/results_summary.json).

| Model | Strict accuracy | Lenient accuracy | Macro-F1 (strict) | Invalid answers |
|---|---|---|---|---|
| Majority-class baseline | 16.7% | – | – | – |
| Qwen2.5-0.5B-Instruct, zero-shot | 33.3% (12/36) | 33.3% | 0.258 | 5.6% (2/36) |
| + LoRA (this repo) | **94.4% (34/36)** | 94.4% | **0.943** | 0.0% |

- 1,081,344 trainable parameters out of 495,114,112 (0.22%). Validation loss fell from 2.28 after epoch 1 to 0.14 after epoch 5.
- **What the untuned model did:** it put `meeting_request` (the first label in the prompt's list) on its first line for 28 of 36 emails, regardless of content,
  then kept writing an explanation. That is why its accuracy sits only a little above the majority baseline. It is a statement about this particular
  small model and prompt, not about zero-shot prompting in general (a better prompt or a larger model would do much better).
- **What the tuned model got wrong:** 2 of 6 `invoice_or_payment` emails were labelled `bug_or_problem`; every other class was 6/6.
- **Caveats:** only 36 test emails (one email is ~2.8 points), one seed, and short, fairly prototypical emails written by one person.
  The high score shows the tuning works on this data; it does not predict performance on real inboxes.

## What I changed / added vs. alpaca-lora
See [SOURCES.md](SOURCES.md): new task and dataset, answer-only loss, modern PEFT/Transformers API, split with leakage check, evaluation harness with baselines, tests, notebook.

## Key Technical Concepts
- **Fine-tuning vs. prompting**: fine-tuning changes weights; prompting changes inputs. For narrow formats/classification, a small tuned model can be consistent and cheap.
- **LoRA**: freeze the base weights and learn a low-rank update `ΔW = B·A` (rank r) for selected layers. Only the small matrices are trained and saved (here roughly 1 million trainable parameters versus ~500M in the base model). `alpha/r` scales the update.
- **Target modules**: which layers get adapters (`q_proj,k_proj,v_proj,o_proj` here; alpaca-lora used `q_proj,v_proj`). More modules = more capacity and memory.
- **Loss masking**: `-100` labels are ignored by the cross-entropy; masking the prompt makes the model learn the answer, not the prompt text.
- **Prompt template**: the same template must be used in training and inference.
- **Overfitting and splits**: a validation split for eval loss and a separate test split for the final numbers; deterministic splits so runs are comparable; check for duplicates across splits.
- **Evaluation design**: baselines (zero-shot, majority class), strict vs lenient parsing, per-class metrics, confusion matrix; small test sets give noisy results.
- **Mixed precision**: fp32 weights with fp16 autocast on GPU is a simple stable choice for a 0.5B model.
- **Data licensing**: training data and base models have their own licenses; this project avoids third-party data entirely.

## Limitations
- Tiny, synthetic, English-only dataset written by one person; emails are short and fairly prototypical, so real-world emails will be harder and a high score here would not transfer.
- Strict and lenient parsing both depend on exact label strings; no calibration or confidence.
- Single seed, single split; no hyperparameter search.
- Model and tokenizer download needs Hugging Face access; the numbers above were produced once and may vary slightly with library versions and GPU.

## Future improvements
Multiple seeds with mean/std; more and noisier data (typos, multilingual, long threads); QLoRA for larger models; compare ranks and target modules; compare against a TF-IDF + logistic regression baseline; serve the adapter behind a small API.
