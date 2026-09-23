# Language-model-derived training data for polymer glass transition prediction

Code, data and per-run outputs for the paper.

[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv.15009353%2Fv1-orange.svg)](https://doi.org/10.26434/chemrxiv.15009353/v1)

Can data written by a large language model replace or supplement experimental measurements for
predicting the glass transition temperature (Tg) of polymers? We ask Claude Sonnet 5 for two kinds of
synthetic data - a **generated** set, where it invents both the repeat unit and the Tg, and a
**labeled** set, where it assigns a Tg to hypothetical [PI1M](https://github.com/RUIMINMA1996/PI1M)
structures - and benchmark both against real measurements on a common held-out test set.

**Answer: supplement, not substitute, and only when measurements are scarce.**

| MoLFormer trained on | Test MAE (°C) |
|---|---|
| 5,297 real polymers | **25.2** ± 1.2 |
| generated set alone (7,912) | 36.8 ± 0.6 |
| labeled set alone (9,925) | 38.5 ± 0.7 |
| *Claude Sonnet 5, zero-shot* | *37.3* |
| 250 real only | 39.1 ± 1.4 |
| 250 real, after a synthetic first stage | 34.3 ± 0.7 |

A synthetic first stage is worth 3–5 °C at 250–500 real polymers, ~2 °C at 1,000, and nothing at
5,297. Shuffling the teacher's Tg values destroys the gain, so it comes from its Tg knowledge and not
from seeing more structures. 8.3 % of the "generated" polymers turn out to be memorised copies of real
database entries. The two datasets cost USD 11.26 to produce.

![Learning curve](results/figures/learning_curve.png)

Plain-language walkthrough of every result: [`results/SUMMARY.md`](results/SUMMARY.md).

## Reproduce

```bash
pip install -r requirements.txt          # pinned to the versions used for the paper
cp .env.example .env                     # only needed to re-query the API; add ANTHROPIC_API_KEY

python pipeline/download_data.py         # Zenodo (md5-checked) + PI1M
python pipeline/prepare_real_data.py     # clean, dedupe, split 5,297 / 589 / 1,471
python pipeline/clean_synthetic_data.py  # six rules; test+validation polymers removed
python pipeline/train_molformer.py       # 120 runs, 5 seeds, resumable
python pipeline/train_random_forest.py
python pipeline/analyze_results.py       # every table in results/tables/
python pipeline/make_figures.py
```

Run everything from the repository root. The API scripts (`generate_with_claude.py`,
`label_pi1m_with_claude.py`, `zero_shot_with_claude.py`) are **not** needed to reproduce the results:
every raw response is already in `data/claude_responses/`, and a request whose id has a saved response
is never re-sent. All settings live in `pipeline/config.py`.

160 training runs, plus a 30-run early-stopping ablation (`pipeline/patience_ablation.py`):
about 29 GPU-hours on one RTX 3050 laptop GPU.

## What is here
