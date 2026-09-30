# Language-model-derived training data for polymer glass transition prediction: a controlled benchmark

**Rostislav Rusev**

[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15009353-b31b1b.svg)](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15009353/v2)
[![Data](https://img.shields.io/badge/real%20Tg%20data-10.5281%2Fzenodo.15789599-blue.svg)](https://doi.org/10.5281/zenodo.15789599)
[![License: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

**Preprint:** [ChemRxiv](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15009353/v2)

This repository contains the code, data and per-run outputs that accompany the preprint. Every number
in the manuscript is generated from the files here.

---

## Abstract

Experimental measurements of the glass transition temperature (Tg) of polymers are scarce relative to
the size of polymer chemical space, which limits data-driven property prediction. We ask whether data
written by a large language model (LLM) can replace or supplement experimental measurements for this
task. Using Claude Sonnet 5, we construct two synthetic datasets: a **generated** set (7,912 polymers),
in which the model invents both the repeat unit and its Tg, and a **labeled** set (9,925 polymers), in
which it assigns a Tg to hypothetical structures from [PI1M](https://github.com/RUIMINMA1996/PI1M).
We benchmark both against real measurements on a common held-out test set, along a learning curve of
250 to 5,297 real polymers, with controls that separate the value of the LLM's Tg values from the value
of merely exposing the model to more structures.

The synthetic data act as a **supplement, not a substitute, and only when measurements are scarce**.
MoLFormer trained on synthetic data alone (36.8–38.5 °C MAE) does not beat the LLM's own zero-shot
predictions (37.3 °C) and is far from a model trained on all real data (25.2 °C). As a first
training stage, however, synthetic data reduce error by 3–4 °C at 250 real polymers relative to a
real-only baseline trained to convergence, by about 2 °C at 500, inconsistently at 1,000, and not at all
at 5,297. Shuffling the teacher's Tg values eliminates the gain, showing that it derives from the
LLM's Tg knowledge rather than from structural exposure. We further find that 8.3 % of "generated"
polymers are memorized copies of real database entries, and that the LLM's labels are rounded to a
small set of values and biased low. Both datasets together cost USD 10.28 to produce.

**Keywords:** glass transition temperature · polymer informatics · large language models ·
synthetic data · transfer learning · MoLFormer

---

## 1. Introduction

Machine-learning models for polymer Tg are limited less by architecture than by the number of reliable
measurements available for training. Large language models have absorbed a substantial amount of
chemical literature and can produce plausible structure–property pairs at negligible cost, which
raises a practical question: can such output serve as training data?

This study addresses that question with a controlled design. Rather than reporting a single
comparison, it measures the effect of LLM-derived data across a learning curve of real-data sizes,
against baselines trained to convergence, and with ablations that isolate where any benefit comes
from.

**Research questions.**

1. Can a model trained only on LLM-written data match a model trained on real measurements?
2. Does pre-training on LLM-written data improve a model subsequently trained on scarce real data, and
   how does this effect change as real data grow?
3. Is any improvement due to the LLM's Tg values, or simply to exposure to additional structures?
4. How faithful are the synthetic data: are they novel, and how are their labels distributed?

## 2. Methods

### 2.1 Real data

Measured Tg values are taken from the curated collection of Kunchapu and Jablonka
([10.5281/zenodo.15789599](https://doi.org/10.5281/zenodo.15789599), CC BY 4.0). After cleaning, the
data are divided into fixed training, validation and test splits (`data/train_real.csv`,
`val_real.csv`, `test_real.csv`). All models in this study are evaluated on the same held-out test set.

### 2.2 Synthetic data

Two synthetic datasets were produced with Claude Sonnet 5 via the batch API:

- **Generated set.** The model proposes both a polymer repeat unit and its Tg (7,912 polymers after
  cleaning).
- **Labeled set.** The model assigns a Tg to hypothetical structures sampled from PI1M (10,000
  structures sent; 9,925 retained after cleaning).

In addition, the model's **zero-shot** Tg predictions on the real test set serve as a reference point.
All raw responses are stored in `data/claude_responses/`, and the prompts are in `pipeline/common/`.

### 2.3 Polymer identity

Because the same chain can be written with different repeat-unit cuts (`*CCO*` = `*COC*`), two polymers
are considered identical if they match on *either* the canonical SMILES *or* a cut-invariant
ring-closure key ([`pipeline/common/cleaning.py`](pipeline/common/cleaning.py)). Exact matching alone
would have missed almost a third of the test polymers that the LLM reproduced verbatim. This definition
underlies both the cleaning step and the memorization analysis.

### 2.4 Models and training protocol

The primary model is MoLFormer, with a random forest as a secondary baseline. Each configuration is
trained with 5 random seeds. The learning curve varies the number of real training polymers from 250 to
5,297 and compares three configurations: real only, generated set followed by real data, and labeled
set followed by real data (two-stage training).

The main protocol uses early stopping with a patience of 3 epochs. Because an impatient baseline can
inflate the apparent benefit of pre-training, the 30 low-data runs were repeated with a patience of 10
and up to 40 epochs (the **patience ablation**).

### 2.5 Controls

- **Shuffled labels:** the teacher's Tg values are permuted across structures, preserving the
  structures but destroying the structure–Tg relationship.
- **Self-training:** the LLM's labels are replaced by pseudo-labels from a model trained on real data.
- **Noise floor and inference noise:** seed-to-seed variation and the variation from MoLFormer's
  stochastic linear attention are quantified so that differences can be judged against them.
- **Seed-matched comparisons:** differences between configurations are computed per seed.

### 2.6 Compute

120 MoLFormer runs, 40 random-forest runs and the 30-run patience ablation required about 29 GPU-hours
on a single RTX 3050 laptop GPU and 3 CPU-hours.

## 3. Results

### 3.1 Main comparison

**Table 1.** Test mean absolute error (mean ± SD over 5 seeds).

| MoLFormer trained on | Test MAE (°C) |
|---|---|
| 5,297 real polymers | **25.2** ± 1.2 |
| Generated set alone (7,912) | 36.8 ± 0.6 |
| Labeled set alone (9,925) | 38.5 ± 0.7 |
| *Claude Sonnet 5, zero-shot* | *37.3* |
| 250 real only, patience 3 (main protocol) | 39.1 ± 1.4 |
| 250 real only, patience 10 (trained to convergence) | 37.7 ± 0.7 |
| 250 real, after a synthetic first stage (patience 3) | 34.3 ± 0.7 (labeled), 34.6 ± 0.6 (generated) |

### 3.2 Learning curve

![Learning curve](results/figures/fig_learning_curve.png)

**Figure 1** (Figure 3 in the paper). MAE on the real test set against the number of real training
polymers (mean ± SD over 5 seeds; series offset slightly sideways for legibility). Open symbols show the
same three configurations trained to convergence (patience 10); the green dashed line and band show
Claude Sonnet 5's zero-shot MAE and its 95 % bootstrap interval.

Against a real-only baseline trained to convergence, a synthetic first stage improves MAE by 3–4 °C at
250 real polymers on every seed. At 500 real polymers the labeled set improves MAE by about 2 °C on
every seed, while the generated set's 1.5 °C improvement is within seed-to-seed noise. At 1,000 real
polymers the gain is about 2 °C and inconsistent across seeds; at 5,297 it disappears.

Measured against the main protocol's impatient baseline (patience 3), the same gains read 3–5 °C. The
patience ablation thus shows that 16–23 % of the originally observed gain at 250 real polymers, and
39–48 % at 500, was an artifact of stopping the baseline too early.

### 3.3 Source of the gain

Shuffling the teacher's Tg values eliminates the gain, so the benefit comes from the LLM's Tg knowledge
and not from exposure to additional structures. Self-training pseudo-labels recover about 40 % of the
gain.

### 3.4 Properties of the synthetic data

- **Memorization.** 8.3 % of the "generated" polymers are copies of entries in the real database.
- **Label quality.** The LLM's Tg values are concentrated on a small number of rounded values and are
  biased low relative to measurements.
- **Cost.** Both datasets together cost USD 10.28 (USD 11.26 including zero-shot evaluation of the test
  set).

A plain-language walkthrough of every result is given in [`results/SUMMARY.md`](results/SUMMARY.md).

## 4. Discussion and limitations

LLM-derived Tg data are cheap and carry real, if coarse, information: they help a model when only a
few hundred measurements are available, and that help vanishes once real data are plentiful. They
cannot stand in for measurements, since models trained on them alone do no better than the LLM's own
zero-shot predictions. The patience ablation also illustrates a methodological point for similar
studies: pre-training benefits should be measured against a baseline trained to convergence, or they
will be overstated.

The conclusions are limited to one LLM (Claude Sonnet 5), one property (Tg), one primary model
architecture (MoLFormer) and one real-data collection. Memorization means part of the "generated" set is
not new chemistry, and the rounding and low bias of the labels bound how much information the synthetic
data can carry. Because MoLFormer's linear attention redraws random features on every forward pass,
results reproduce statistically rather than bit for bit.

## 5. Reproducibility

### 5.1 Running the pipeline

```bash
pip install -r requirements.txt          # pinned to the versions used for the paper (Python 3.14)
cp .env.example .env                     # only needed to re-query the API; add ANTHROPIC_API_KEY

python pipeline/run.py                   # every step that is missing or out of date, in dependency order
python pipeline/run.py --list            # the steps and whether each is up to date
python pipeline/run.py analyze figures   # only these steps
python -m pytest                         # the fast tests; `-m slow` instead runs the one-epoch CPU smoke test
```

[`pipeline/run.py`](pipeline/run.py) holds the dependency order as a plain table of steps, inputs and
outputs, and runs each step as a subprocess. Every step is one script that can also be run by hand from
any directory (`python pipeline/<script>.py`, `python report/<script>.py`), in this order:

| Step (`run.py` name) | Script | Writes |
|---|---|---|
| download | `pipeline/download_data.py` | `data/raw/` (Zenodo file, md5-checked; PI1M) |
| prepare | `pipeline/prepare_real_data.py` | `data/real_clean.csv`, `train_real.csv`, `val_real.csv`, `test_real.csv` |
| api (only when named) | `pipeline/generate_with_claude.py`, `label_pi1m_with_claude.py`, `zero_shot_with_claude.py` | `data/*_raw.csv`, `results/zero_shot_predictions.csv` |
| clean | `pipeline/clean_synthetic_data.py` | `data/generated_clean.csv`, `data/labeled_clean.csv` |
| describe, memorization, score_zero_shot | `pipeline/describe_split.py`, `memorization.py`, `score_zero_shot.py` | `results/test_similarity.csv`, `memorization.csv`, `zero_shot_metrics.json`, `api_cost.csv` |
| train_molformer, train_random_forest | `pipeline/train_molformer.py`, `train_random_forest.py` | `results/runs/*.json`, `results/predictions/*.csv`, `results/stage1_weights/` |
| inference_noise, patience_ablation | `pipeline/inference_noise.py`, `patience_ablation.py` | `results/tables/inference_noise.csv`, `results/patience_ablation.jsonl` |
| analyze | `pipeline/analyze_results.py` | `results/results.csv`, `results_summary.csv`, `results/tables/*.csv` |
| tables, figures | `report/make_tables.py`, `report/make_figures.py` | `results/tables/latex/*.tex`, `results/figures/*.pdf` and `.png` |

The API scripts are **not** needed to reproduce the results: every raw response is already in
`data/claude_responses/`, and a request whose id has a saved response is never re-sent. All settings
live in `pipeline/config.py`. `inference_noise.py` and `patience_ablation.py` need the stage-1 weights
in `results/stage1_weights/`, which are not in the repository (180 MB per model) and are written by
`train_molformer.py`; their outputs are committed.

Training runs are resumable: a run whose record exists is skipped, but only if the record's
`settings_hash` matches the current settings; otherwise the script stops with a message
(`pipeline/common/resume.py`). `run.py` reruns the training steps only when their outputs are missing,
never because a data file is merely newer, so a regenerated data file cannot trigger a 29-GPU-hour rerun
by accident. `train_molformer.py --patience 10 --max-epochs 40` is the setting of the patience ablation;
`patience_ablation.py` runs exactly the 30 low-data runs of the paper with it.

`results/run_info.json` records the dataset checksum, prompts, request counts, costs and software
versions.

### 5.2 Mapping of paper tables and figures to scripts

All numbers in the manuscript are macros from `results/tables/latex/numbers.tex`; nothing is typed by
hand. Figure 1 (the pipeline overview) is drawn in TikZ in the manuscript source.

| In the paper | Script | Output file |
|---|---|---|
| Table 1, run matrix | `report/make_tables.py` | `results/tables/latex/run_matrix.tex` |
| Table 2, cleaning counts | `report/make_tables.py` | `results/tables/latex/cleaning_counts.tex` |
| Table 3, Tg statistics | `report/make_tables.py` | `results/tables/latex/dataset_stats.tex` |
| Table 4, learning curve | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/results_summary.csv`, `tables/patience_ablation.csv` → `latex/results_main.tex` |
| Table 5, controls | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/results_summary.csv` → `latex/controls.tex` |
| Figure 2, Tg distributions | `report/make_figures.py` | `results/figures/fig_tg_distributions.pdf` |
| Figure 3, learning curve | `report/make_figures.py` | `results/figures/fig_learning_curve.pdf` |
| Figure 4, controls | `report/make_figures.py` | `results/figures/fig_controls.pdf` |
| Figure 5, predicted vs measured | `report/make_figures.py` | `results/figures/fig_predicted_vs_true.pdf` |
| Table S1, settings | `report/make_tables.py` (from `config.py`) | `latex/hyperparameters.tex` |
| Table S2, API cost | `pipeline/score_zero_shot.py` → `report/make_tables.py` | `results/api_cost.csv` → `latex/cost_full.tex` |
| Table S3, top label values | `report/make_tables.py` | `latex/labeled_top_values.tex` |
| Table S4, zero-shot by Tg band | `report/make_tables.py` | `latex/zero_shot_bands.tex` |
| Table S5, memorization | `pipeline/memorization.py` → `report/make_tables.py` | `results/memorization.csv` → `latex/memorization.tex` |
| Table S6, all metrics | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/results_summary.csv` → `latex/results_full.tex` |
| Table S7, per-seed MAE | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/results.csv` → `latex/per_seed_molformer.tex` |
| Table S8, best epochs | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/results.csv` → `latex/best_epochs.tex` |
| Table S9, student vs teacher | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/tables/student_vs_teacher.csv` → `latex/student_vs_teacher.tex` |
| Table S10, noise floor | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/tables/noise_floor.csv` → `latex/noise_floor.tex` |
| Table S11, MAE by Tg band | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/tables/mae_by_tg_bin.csv` → `latex/mae_by_tg_bin.tex` |
| Table S12, MAE by similarity | `pipeline/describe_split.py` → `analyze_results.py` → `report/make_tables.py` | `results/tables/mae_by_similarity_bin.csv` → `latex/mae_by_similarity.tex` |
| Table S13, inference noise | `pipeline/inference_noise.py` → `report/make_tables.py` | `results/tables/inference_noise.csv` → `latex/inference_noise.tex` |
| Table S14, all seed-matched comparisons | `pipeline/analyze_results.py` → `report/make_tables.py` | `results/tables/seed_matched_comparisons.csv` → `latex/paired_differences_full.tex` |
| Table S15, patience ablation | `pipeline/patience_ablation.py` → `analyze_results.py` → `report/make_tables.py` | `results/patience_ablation.jsonl` → `tables/patience_ablation.csv` → `latex/patience_ablation.tex` |
| Figure S1, repetition and rounding | `report/make_figures.py` | `results/figures/fig_si_repetition_rounding.pdf` |
| Figure S2, MAE vs similarity | `pipeline/analyze_results.py` → `report/make_figures.py` | `results/figures/fig_si_similarity.pdf` |
| Figure S3, zero-shot error | `report/make_figures.py` | `results/figures/fig_si_zero_shot_error.pdf` |

### 5.3 Repository structure

```
pipeline/        the study, one script per step, and run.py to run them in order; config.py holds every setting
  common/        Claude batch API, the two prompts, SMILES cleaning and the polymer keys,
                 the MoLFormer training loop, the resume check
report/          the tables (LaTeX + number macros) and figures of the paper, drawn from results/ and data/
tests/           pytest: polymer keys and cleaning rules, the runner; a slow CPU smoke test of the training code
data/            the two synthetic sets, the cleaned real data and the three splits
  claude_responses/   every raw LLM response, one JSON line per request
results/         predictions/ and runs/ (one CSV + one JSON per training run), tables/ (CSV) and
                 tables/latex/ (the paper's tables and number macros), figures/,
                 pseudo_labels/ (the self-training labels), logs/ (run logs),
                 patience_ablation.jsonl, run_info.json, SUMMARY.md
```

## Data availability

Real Tg data: the curated collection of Kunchapu and Jablonka,
[10.5281/zenodo.15789599](https://doi.org/10.5281/zenodo.15789599), CC BY 4.0.
PI1M: [RUIMINMA1996/PI1M](https://github.com/RUIMINMA1996/PI1M).

Both full datasets are downloaded by `pipeline/download_data.py` and are not redistributed here. What
`data/` does contain is derived from them: the cleaned real collection and its splits (from the Zenodo
file, under its CC BY 4.0 terms) and `pi1m_sample.csv`, the 10,000 PI1M structures that were sent to the
LLM (under the terms of the PI1M repository). See [`LICENSE`](LICENSE).

## Code availability

All code is in this repository under the MIT license. A tagged release will be archived on Zenodo;
**TODO: add the Zenodo DOI here once minted.**

## Citation

If you use this code or data, please cite:

```bibtex
@article{rusev2026llmtg,
  title   = {Language-model-derived training data for polymer glass transition
             prediction: a controlled benchmark},
  author  = {Rusev, Rostislav},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15009353/v2},
  url     = {https://chemrxiv.org/doi/full/10.26434/chemrxiv.15009353/v2}
}
```

## License

Code: [MIT](LICENSE). Derived data produced for this study (the synthetic datasets, raw LLM responses
and everything under `results/`): CC BY 4.0. The real Tg data remain CC BY 4.0 under the terms of their
original Zenodo record; PI1M under the terms of its repository. See the notes at the end of
[`LICENSE`](LICENSE).
