# Can cheap LLM data replace or supplement real Tg data?

A small pipeline that compares three training sets for polymer glass transition
temperature (Tg) prediction:

- **real**: experimental PolyInfo data (Kaggle `fridaycode/tg-smiles-pid-polyinfo-class`)
- **generated**: polymers *and* Tg values written by Claude Sonnet 5
- **labeled**: hypothetical PI1M polymers with Tg values *predicted* by Claude Sonnet 5

All models are evaluated on the same held-out 20 % of the real data.

## Layout

```
pipeline/
    config.py                every setting: paths, model name, data seed (42), training seeds
                             (42-46), sample sizes, budget cap, PILOT flag, hyperparameters
    common/
        claude_batches.py    talk to the Message Batches API, save raw responses, track cost
        prompts.py           the two prompts and the parsing of Claude's JSON answers
        cleaning.py          SMILES canonicalization and the cleaning rules
        run_info.py          append entries to results/run_info.json
    download_data.py         ┐
    prepare_real_data.py     │
    generate_with_claude.py  │
    label_pi1m_with_claude.py│  the pipeline, run in this order
    zero_shot_with_claude.py │
    clean_synthetic_data.py  │
    train_and_evaluate.py    │
    make_figures.py          ┘
data/raw/                    the original downloaded files
data/claude_responses/       every raw Claude response (JSONL, one line per request)
data/*.csv                   intermediate tables, one per step
results/                     results.csv (one row per experiment, model and seed),
                             results_summary.csv (mean and SD over seeds), predictions/
                             (one file per run), train_times.csv, figures/, run_info.json
```

## Setup

```
pip install -r requirements.txt
copy .env.example .env       # then paste your key into .env
```

`.env` contains one line, `ANTHROPIC_API_KEY=sk-ant-...`. The Claude scripts load it
with python-dotenv; a key already set in the environment takes precedence. `.env` is in
`.gitignore` and the key never appears in code.

## The pipeline, in order

Run every script from the project folder (the one that contains this README), for example
`python pipeline/prepare_real_data.py`. The scripts use the relative paths `data/` and
`results/`, and look for `.env` in the project folder.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `download_data.py` | moves the Kaggle CSV into `data/raw/`, downloads PI1M from GitHub | | `data/raw/*.csv` |
| `prepare_real_data.py` | canonicalizes SMILES, merges duplicates (median Tg), random 80/20 split | `data/raw/` | `data/train_real.csv`, `data/test_real.csv` |
| `generate_with_claude.py` | asks Claude for ~50 polymer/Tg pairs per request, in rounds, until 10,000 clean pairs or 400 requests | `test_real.csv` | `data/claude_responses/generated.jsonl`, `data/generated_raw.csv` |
| `label_pi1m_with_claude.py` | samples 10,000 PI1M polymers and asks Claude for a Tg, one per request | `PI1M.csv` | `data/pi1m_sample.csv`, `data/claude_responses/labeled.jsonl`, `data/labeled_raw.csv` |
| `zero_shot_with_claude.py` | same prompt on the real test set (zero-shot baseline) | `test_real.csv` | `data/claude_responses/zero_shot.jsonl`, `results/zero_shot_predictions.csv` |
| `clean_synthetic_data.py` | six cleaning steps with per-step counts (numeric Tg, RDKit-valid, exactly two `*`, Tg range, duplicates, test-set overlap), keeps at most 10,000 generated pairs | `data/*_raw.csv` | `data/generated_clean.csv`, `data/labeled_clean.csv` |
| `train_and_evaluate.py` | 8 experiments x (random forest + MolFormer) x 5 training seeds, plus the zero-shot row; a finished run is never repeated, so the script can be interrupted and restarted | `data/*_clean.csv`, `train/test_real.csv` | `results/results.csv`, `results/results_summary.csv`, `results/predictions/`, `results/train_times.csv` |
| `make_figures.py` | MAE bar chart, predicted-vs-true scatter plots, Tg histograms | `results/` | `results/figures/*.png` |

Every script can be rerun on its own; each one only reads the CSVs of the previous step.

## Budget safety

- `PILOT = True` in `pipeline/config.py` sends only a handful of requests (2 generation requests,
  100 labeling requests, 100 zero-shot requests). The released `config.py` has
  `PILOT = False`, the full run of the paper; set it to `True` first when you try the
  scripts with a new key.
- Every Claude script prints the number of requests and an estimated cost, then waits
  for you to type `yes`. Passing `--yes` on the command line skips the prompt.
- `BUDGET_LIMIT_USD` in `config.py` is a hard stop: a batch is refused if it would push
  the total spend above it.
- `USE_BATCH_API = True` sends everything through the Message Batches API (50 % cheaper,
  but a batch can sit in Anthropic's queue for hours). `False` sends requests one by one
  at full price; useful for a quick pilot, too expensive for the full run.
- Every raw response is stored in `data/claude_responses/*.jsonl`. A request whose id
  already has a saved response is never sent again, and a batch that was submitted but
  not yet collected is resumed from `*.jsonl.pending` on the next run, so a crash never
  costs twice.
- Spend so far is the sum of `cost_usd` over the saved responses; every script prints it.

## Notes on the Claude API settings

- Model: `claude-sonnet-5`, extended thinking disabled explicitly.
- Sonnet 5 rejects `temperature` / `top_p` / `top_k`, so no sampling parameters are set.
  Diversity in the generated dataset comes from rotating polymer classes, focus hints,
  and a variant sentence in the prompt (see `pipeline/common/prompts.py` and `pipeline/config.py`).
- The two scripts that label polymers use the same short prompt from `pipeline/common/prompts.py`.

## Reproducibility

`results/run_info.json` collects the model name, prompts, thinking setting, dates,
request counts, dataset sizes after every cleaning step, training settings, and the total
API cost. Every script appends its own entries.
