# Interview notes - lora-email-intent

Answers describe what this repository actually does. Numbers quoted below come from one real run on a Colab T4 (see README and `results/results_summary.json`); always mention that it is one seed on 36 test emails.

## 11 likely questions
1. **What does the project do?** It fine-tunes Qwen2.5-0.5B-Instruct with LoRA to label customer emails with one of six intents and evaluates the tuned model against the untuned model and a majority-class baseline on 36 held-out emails.
2. **What is LoRA?** Instead of updating all weights, it freezes them and adds trainable low-rank matrices (rank 8 here, alpha 16) to the attention projections. The adapter is small and saved separately; `get_nb_trainable_parameters` prints the trainable fraction at the start of training.
3. **Why LoRA instead of full fine-tuning?** Far less memory and a tiny artifact, so it fits a free Colab GPU. Trade-off: less capacity than full fine-tuning.
4. **What did you take from alpaca-lora and what's different?** The recipe: base model + PEFT LoRA config + prompt template + Trainer + save adapter. Different: my own dataset instead of Alpaca data, answer-only loss, current PEFT/Transformers API without int8/wandb, a leakage-checked split, and a proper evaluation.
5. **What is `train_on_inputs` and why is it false?** It controls whether the loss covers the prompt tokens. alpaca-lora defaults to true. For classification the answer is a few tokens, so with prompt masking all gradient goes to producing the label. I did not run the `--train-on-inputs` comparison, so I can't claim it measurably helps here.
6. **How do you evaluate fairly?** Same prompt and greedy decoding for both models; a held-out test split never used for training or validation; strict parsing (exactly one valid label) plus a lenient parse so the base model isn't penalised only for formatting; a majority-class baseline; per-class F1 and a confusion matrix.
7. **What are the weaknesses of the evaluation?** 36 test emails (±2.8 points per email), one seed, synthetic hand-written data from a single author, prototypical short emails. High accuracy here would not imply real-world performance.
8. **How did you avoid data leakage?** Deterministic stratified split by seed and a check that no email text appears in more than one split. Test examples are never used in training or validation.
9. **How do you know the training code works without a GPU?** A smoke test trains for a few steps on a tiny random-weight Qwen2 model with a locally trained tokenizer, saves the adapter, reloads it with `PeftModel`, runs generation and metrics, and also runs `evaluate.main`. It validates the plumbing, not quality. (It also caught an API change in Transformers 5 - `warmup_ratio` was removed.)
10. **What were the results?** On 36 held-out emails: majority baseline 16.7%, untuned Qwen2.5-0.5B 33.3%, LoRA-tuned 94.4% strict accuracy (macro-F1 0.943, no invalid answers). The only errors were 2 of 6 invoice emails labelled as bug reports. The untuned model mostly answered with the first label in the prompt, so its score mostly reflects that small model and my prompt, not zero-shot LLMs in general. One seed and a small synthetic set, so I treat it as a learning result, not a benchmark.
11. **How would you improve it?** Run several seeds, add noisier and multilingual data, compare ranks/target modules, add a TF-IDF + logistic regression baseline (it may be competitive on this task, and a good fine-tuning story should compare against it), and consider QLoRA for larger models.

## 5 areas to study further
- The math of LoRA (low-rank decomposition, initialisation of A and B, why `B` starts at zero)
- QLoRA and quantisation (4-bit NF4, double quantisation, paged optimizers)
- Evaluation of LLM classifiers: confidence intervals, McNemar test, calibration
- Overfitting dynamics with tiny datasets, learning-rate schedules, regularisation
- Training-data licensing and safe dataset construction (synthetic data pitfalls)

## Design decisions
- **Original dataset:** avoids third-party data licenses and gives full control, at the cost of realism.
- **Small base model:** reproducible on a free Colab GPU.
- **Strict parsing with invalid class:** an unusable answer is a failure, not silently mapped to a label.
- **Lenient parsing as a second view:** fair to the untuned model.
- **Smoke test on random model:** verifies the pipeline in CI-like conditions with no downloads.
- **Notebook shipped without outputs:** no results are claimed until a real run exists.
