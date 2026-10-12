# Reference-Free MT Quality Estimation for German → Upper Sorbian

How good is a machine translation when **there is no human reference translation** to compare against, and the target language is **low-resource**? This project builds and compares reference-free quality-estimation (QE) methods for German → Upper Sorbian (DE→HSB) and checks how well they agree with established metrics.

## Approaches

| Family | What it does | Code |
|---|---|---|
| **LLM translation** | Zero-shot, few-shot and chain-of-thought prompting to produce DE→HSB translations | `translation/translate_llm.py` |
| **LLM-as-a-judge** | An LLM scores each (source, translation) pair 0–100 with a calibrated rubric | `evaluation/judge_llm.py`, `evaluation/evaluate_llm_judge.py` |
| **Fine-tuned neural QE (MetricX-style)** | Gemma-3 backbone + regression head, trained with **QLoRA** (4-bit NF4) on MSE + pairwise margin-ranking loss over z-normalised scores | `models/`, `training/train_metricx.py`, `scripts/predict_metricx.py` |
| **COMET-QE + LoRA** | LoRA fine-tuning of COMET's reference-less regression model | `training/train_comet_lora.py` |
| **Reference-based baselines** | sacreBLEU, BERTScore, COMET (wmt22-comet-da) | `scripts/run_baseline_extraction.py` |

**Synthetic training data.** `scripts/generate_qe_data.py` corrupts translations at controlled noise levels (word deletion, swaps, span masking) so the QE models learn that more damage means a lower score.

## Evaluation

`scripts/run_meta_evaluation.py` and `evaluation/compare_evaluators.py` compute **Pearson** and **Spearman** correlations between every QE candidate and the reference-based metrics, and `visualization/` produces score distributions, noise-level violin plots and comparison charts.

## Key results

On the 4,000-segment WMT 2025 DE→HSB development set (GPT-4o-mini few-shot output), with **COMET-22-DA as the anchor** (the closest available stand-in for human judgement in Upper Sorbian):

| Metric | Pearson r | Spearman ρ | Kendall τ |
|---|---|---|---|
| BLEU | 0.366 | 0.320 | 0.216 |
| chrF++ | 0.572 | 0.551 | 0.387 |
| BERTScore (XLM-R) | **0.593** | **0.553** | **0.387** |
| COMETKiwi (reference-free) | 0.508 | 0.477 | 0.333 |
| **MetricX-style Gemma-3 12B QE (ours, reference-free)** | 0.457 | 0.440 | 0.305 |
| Best LLM judge (GPT-5-mini, few-shot) | 0.312 | 0.300 | – |

- Few-shot prompting beats zero-shot translation (chrF++ 34.95 vs 33.76).
- Our QE model, trained on **749,080 synthetic pseudo-labelled examples**, clearly beats BLEU without seeing any reference, but still trails COMETKiwi. The likely reasons: a single training stage with no calibration stage, and labels distilled entirely from COMET.
- LLM judges look strong overall (r up to 0.76) but drop 40–65 % once you condition on the MT system. Their apparent agreement comes mostly from separating systems, not individual segments.

Full write-up: *Reference-Free Quality Estimation and Meta-Evaluation for German-to-Upper-Sorbian Machine Translation* — Shivam Singh Chauhan & Paul Hagner, TUM.

## Repository layout

```
translation/     LLM translation strategies
evaluation/      LLM judge, neural evaluation, evaluator comparison
models/          MetricX-style dataset + model
training/        QLoRA / LoRA training scripts
scripts/         data generation, baselines, inference, meta-evaluation
visualization/   plots
tests/           unit tests
```

## Running

```bash
pip install torch transformers peft bitsandbytes lightning unbabel-comet sacrebleu bert-score scipy pandas seaborn openai python-dotenv
# for the LLM steps, create a .env file with OPENAI_BASE_URL and OPENAI_API_KEY
```

Fine-tuning the Gemma-3 regressor requires a CUDA GPU (4-bit quantisation via bitsandbytes).

Run scripts as modules **from the repository root**, so the `models/` and `training/` packages resolve:

```bash
python -m training.train_metricx ...     # fine-tune the QE model
python -m scripts.predict_metricx --checkpoint <ckpt> --src <src.txt> ...
python -m unittest tests.test_metricx_local
```

## Context

Built in the TUM course *Projects in Natural Language Processing*, on the WMT 2025 DE→HSB data.
