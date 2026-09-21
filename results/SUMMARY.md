# Results: LLM-generated and LLM-labeled Tg data against real data

Written 2026-09-20 from the files in this folder. Every number below is copied from `tables/*.csv`,
`memorization.csv`, `zero_shot_metrics.json`, `api_cost.csv` and `run_info.json`; the full tables are
in `tables/all_tables.md`. MAE, RMSE and SD are in °C. "± x" is the standard deviation over the five
training seeds 42–46. "n = 500" means 500 real training polymers.

## 1. Design in one table

| | |
|---|---|
| Real data | Curated collection of the Jablonka group, Zenodo 10.5281/zenodo.15789599 (CC-BY 4.0), 7,357 polymers after cleaning |
| Split | 80/20 (train+validation)/test, then 90/10 train/validation; 5,297 / 589 / 1,471 |
| Duplicate and overlap matching | exact SMILES **or** ring-closure key ("either key"), everywhere |
| Held-out removal from synthetic data | test and validation |
| Training | early stopping on the validation MAE (patience 3), best weights kept |
| Combining real and synthetic data | two-stage (synthetic, then real) as the main protocol; concatenation for comparison |
| Real subset sizes | 250, 500, 1,000, full (nested) |
| Controls | self-training pseudo-labels; shuffled labels |
| Synthetic data | generated 7,912; labeled 9,925 (after cleaning) |
| Zero-shot query | the labeling prompt on every test polymer (USD 0.98) |

## 2. The real dataset

- Zenodo record 14980914 redirects to the latest version **15789599** (published 2025-07-02), whose
  only file is `LAMALAB_CURATED_Tg_structured_polymerclass_with_embeddings.csv` (md5 `31400a5f499394e3a6e0658488a464bb`;
  7,367 rows, 115 columns, of which 8 are used; Tg is given in kelvin).
- 7,170 of the 7,367 rows come from source "GREA" (Liu et al., KDD 2022), whose polymers were drawn
  from PoLyInfo; the rest come from five smaller sources.
- Cleaning with the rules of the synthetic sets removed nothing (all rows parse, all have exactly two
  `*`, all Tg are within −150 to 500 °C, no exact duplicates); the ring key merged 10 rows → 7,357.
  The ring key cannot be computed for 1,358 polymers (18.5 %, vinyl-type units whose two `*`
  neighbours are bonded); those are matched by exact SMILES, which is already cut-independent for them.
- **Reliability flag.** black 7,088, gold 143, yellow 132, red 4. The colours encode the number of data
  points (black = 1, yellow = 2, gold = 3–6, red = more), not their agreement: gold includes SDs up
  to 145 K. Only 288 polymers have two or more measurements. **No reliability filter was applied**;
  the proposal was a sensitivity analysis on the polymers whose repeated measurements have SD > 30 °C
  (26 polymers, 7 of them in the test set), reported in section 9.
- Split sizes and Tg (mean ± SD): train 5,297 (144.0 ± 113.6), validation 589 (147.0 ± 109.0),
  test 1,471 (141.8 ± 110.9). No polymer is in two splits under either key (asserted in the script).
- Similarity of the test polymers to the training set (maximum Tanimoto, Morgan r = 2, 2048 bits):
  median 0.81, quartiles 0.67 and 0.94; 41 % at or above 0.85, 11.5 % below 0.55; 252 test polymers
  (17 %) have a training polymer with an identical fingerprint (homologues, not duplicates).

## 3. Synthetic data after re-cleaning

| Set | Raw pairs | Unique, exact SMILES | Unique, either key | Test matches removed | Validation matches removed | Final |
|---|---|---|---|---|---|---|
| Generated | 20,086 | 8,501 | 8,077 | 111 (79 exact + 32 ring) | 54 (27 + 27) | **7,912** |
| Labeled | 10,000 | 9,933 | 9,929 | 3 (1 + 2) | 1 (1 + 0) | **9,925** |

Memorization (`memorization.csv`): 668 of the 8,077 unique generated polymers (8.3 %) are polymers of
the real collection by either key; 503 (6.2 %) are in the training set (319 exact, 184 ring) and were
kept. For those 503 the teacher's Tg against the real value has MAE 29.1, median absolute error 19.0,
Spearman ρ 0.91 and mean signed error −10.5. The removed test matches look the same (MAE 29.4), as do
the validation matches (MAE 25.6). Of the labeled set, 11 polymers (0.1 %) match a real polymer.

## 4. Zero-shot teacher on the new test set

MAE **37.3** (95 % bootstrap CI 35.7–39.1), RMSE 50.0, R² 0.797, ρ 0.908, mean signed error −3.4.
99.3 % of the answers are multiples of 5 °C (63.9 % multiples of 10) and only 90 distinct values occur.
By band of the measured Tg: MAE 31.1 below 0 °C (bias +11.8), 30.6 for 0–100, 42.3 for 100–200,
37.6 for 200–300, and 50.3 at and above 300 °C (bias −48.0). 12 of 1,471 answers hit the 200-token
limit and were re-asked at 600 tokens; all 1,471 polymers received a number.

## 5. Main results (MoLFormer, test MAE, mean ± SD over 5 seeds)

| Configuration | n = 250 | n = 500 | n = 1,000 | n = 5,297 (full) |
|---|---|---|---|---|
| Real only | 39.1 ± 1.4 | 35.9 ± 1.4 | 32.3 ± 2.5 | 25.2 ± 1.2 |
| Two-stage: generated → real | 34.6 ± 0.6 | 33.2 ± 0.9 | 30.5 ± 1.0 | 24.9 ± 1.0 |
| Two-stage: labeled → real | 34.3 ± 0.7 | 32.5 ± 0.8 | 30.2 ± 0.7 | 24.8 ± 2.0 |
| Concatenation: real + generated | | 33.0 ± 0.9 | | 24.1 ± 0.7 |
| Concatenation: real + labeled | | 32.8 ± 0.7 | | 26.1 ± 1.3 |

Synthetic data alone (the stage-1 models): generated **36.8 ± 0.6**, labeled **38.5 ± 0.7**; the
zero-shot teacher 37.3. Random forest: real full 26.5 ± 0.1, generated only 47.5, labeled only 47.1,
real + generated 27.3, real + labeled 27.6, 500 real 40.4 ± 0.5, 500 + generated 39.9, 500 + labeled 40.3.

### Seed-matched comparisons (difference in MAE, negative = first is better)

| Comparison | ΔMAE ± SD | Range | Same sign | Paired t p | Wilcoxon p |
|---|---|---|---|---|---|
| Two-stage generated vs real only, n = 250 | −4.5 ± 1.5 | −6.9 to −2.9 | 5/5 | 0.002 | 0.0625 |
| Two-stage labeled vs real only, n = 250 | −4.8 ± 1.6 | −7.5 to −3.6 | 5/5 | 0.003 | 0.0625 |
| Two-stage generated vs real only, n = 500 | −2.7 ± 1.4 | −4.5 to −1.2 | 5/5 | 0.012 | 0.0625 |
| Two-stage labeled vs real only, n = 500 | −3.4 ± 1.6 | −6.1 to −2.2 | 5/5 | 0.009 | 0.0625 |
| Two-stage generated vs real only, n = 1,000 | −1.8 ± 2.9 | −6.7 to +0.3 | 4/5 | 0.23 | 0.19 |
| Two-stage labeled vs real only, n = 1,000 | −2.1 ± 2.1 | −5.8 to −0.4 | 5/5 | 0.09 | 0.0625 |
| Two-stage generated vs real only, full | −0.3 ± 0.3 | −0.6 to +0.2 | 4/5 | 0.09 | 0.13 |
| Two-stage labeled vs real only, full | −0.4 ± 2.0 | −2.9 to +2.5 | 4/5 | 0.64 | 0.44 |
| Concatenation generated vs real only, n = 500 | −2.9 ± 2.0 | −4.8 to +0.4 | 4/5 | 0.035 | 0.13 |
| Concatenation labeled vs real only, n = 500 | −3.1 ± 1.2 | −5.0 to −2.1 | 5/5 | 0.004 | 0.0625 |
| Concatenation generated vs real only, full | −1.1 ± 1.3 | −2.6 to +0.9 | 4/5 | 0.13 | 0.19 |
| Concatenation labeled vs real only, full | +0.9 ± 1.2 | −0.9 to +2.4 | 4/5 | 0.18 | 0.31 |
| Two-stage vs concatenation, generated, n = 500 | +0.1 ± 1.5 | | 4/5 | 0.85 | 0.63 |
| Two-stage vs concatenation, labeled, n = 500 | −0.3 ± 0.5 | | 4/5 | 0.23 | 0.31 |
| Two-stage vs concatenation, generated, full | +0.8 ± 1.3 | | 4/5 | 0.23 | 0.31 |
| Two-stage vs concatenation, labeled, full | −1.3 ± 2.8 | | 4/5 | 0.35 | 0.44 |
| Labeled only vs generated only (stage 1) | +1.7 ± 0.7 | +1.0 to +2.8 | 5/5 | 0.005 | 0.0625 |
| Random forest, real + generated vs real, full | +0.8 ± 0.1 | | 5/5 | 0.0001 | 0.0625 |
| Random forest, real + labeled vs real, full | +1.1 ± 0.1 | | 5/5 | <0.0001 | 0.0625 |

Reading: with 250 to 500 real polymers, either synthetic set reduces the MAE by 3 to 5 °C on every
seed; at 1,000 the gain is about 2 °C and no longer consistent; with the full training set it is below
1 °C and not distinguishable from zero. Two-stage and concatenation give the same result within the
seed noise. The random forest never benefits, and is slightly hurt with the full set.

## 6. Controls at n = 500 (two-stage protocol)

| Configuration | Test MAE | Against | ΔMAE ± SD | Same sign | Paired t p |
|---|---|---|---|---|---|
| Real only | 35.9 ± 1.4 | | | | |
| LLM-labeled → real | 32.5 ± 0.8 | real only | −3.4 ± 1.6 | 5/5 | 0.009 |
| Self-training labels → real | 34.5 ± 1.0 | LLM-labeled | +2.0 ± 0.7 | 5/5 | 0.003 |
| | | real only | −1.4 ± 1.3 | 5/5 | 0.09 |
| Labeled, Tg shuffled → real | 39.5 ± 5.8 | LLM-labeled | +7.0 ± 6.2 | 5/5 | 0.06 |
| Generated → real | 33.2 ± 0.9 | real only | −2.7 ± 1.4 | 5/5 | 0.012 |
| Generated, Tg shuffled → real | 54.5 ± 16.7 | generated | +21.3 ± 16.8 | 5/5 | 0.047 |

- **The LLM's labels carry information beyond the structures.** Shuffling the Tg values destroys the
  gain and makes the model worse than real data alone (the shuffled stage-1 models alone have MAE
  97 and 105). The generated set with shuffled labels is far more harmful (54.5, one seed at 81)
  than the labeled set with shuffled labels (39.5).
- **The LLM's labels beat free pseudo-labels.** Self-training (the seed's own 500-polymer model
  labelling the same 9,925 PI1M structures) gives 34.5, better than real only by 1.4 but worse
  than the LLM labels by 2.0 on every seed. The self-training stage-1 model alone reaches 36.2,
  close to the LLM-labeled stage-1 model (38.5) although its labels come from a 500-polymer model.

## 7. Stage-1 students against the teacher

Paired bootstrap over the 1,471 test polymers (`tables/student_vs_teacher.csv`): the generated-set
student differs from the teacher by −0.5 on average (−1.15 to +0.40 over seeds; the interval never
excludes zero), the labeled-set student is worse by +1.2 (+0.47 to +2.15; the interval excludes zero
on 2 of 5 seeds). A student trained on the teacher's output lands at the teacher's accuracy.

## 8. Noise floor

54 test polymers have two or more measurements. Their experimental SD has median 5.7 and mean 17.3 °C,
so two independent measurements of the same polymer would differ by about 6.5 °C (median-based) or
19.5 °C (mean-based; a few polymers with wildly different literature values dominate the mean). On
these 54 polymers the models reach: real only full 20.9, random forest 21.3, two-stage full 21.6 /
23.7, zero-shot teacher 26.0, generated only 29.6, labeled only 29.8, real only n = 500 34.1. On the
other 1,417 polymers: real only full 25.4, teacher 37.7. Two remarks: the labels are medians of the
listed values, so a model's MAE against the label is not the same quantity as the difference between
two measurements; and the teacher is much better on the polymers with repeated measurements (26.0
against 37.7), which are the well-known polymers, another sign that its zero-shot answers are partly
recall.

## 9. Stratified errors (mean over seeds; all models in `tables/mae_by_*.csv`)

- **By measured Tg.** At and above 300 °C (125 polymers): real only full 35.7, teacher 50.3, labeled
  only 53.3, generated only 72.8. The generated-set student compresses the high end most.
- **By similarity to the training set.** Real only full goes from 38.6 (similarity < 0.4, 34 polymers)
  to 20.7 (≥ 0.85, 608 polymers); the random forest from 51.6 to 18.8; the teacher from 42.2 to 31.7.
  The ordering real < synthetic ≈ teacher holds in every band except the lowest, where the labeled-only
  student (38.6) ties the real-data model on 34 polymers.
- **By reliability flag.** black (1,418): real only full 25.4, teacher 37.7; gold (23): 21.5 and 25.6;
  yellow (28): 20.0 and 25.3. The single red and black/red polymers are not informative.
- **Sensitivity to disagreeing measurements.** Dropping the 7 test polymers with SD > 30 °C changes
  every MAE by at most 0.1.

## 10. The main findings in one paragraph

Synthetic data alone give a student at the teacher's accuracy (36.8 / 38.5 against 37.3) and far from
the real-data student (25.2). As a first stage before 250 or 500 real polymers they reduce the MAE by
3 to 5 C on every seed; at 1,000 the gain is about 2 C and inconsistent; with the full training set
it is below 1 C and not distinguishable from zero. Shuffling the labels abolishes the gain, so it comes
from the teacher's Tg values; self-training pseudo-labels give about half of it. Two-stage and
concatenation are equivalent. MoLFormer beats the random forest on real data (25.2 against 26.5), and
the random forest never benefits from synthetic data.

## 11. Surprises worth knowing

- The reliability colours count data points rather than agreement (section 2).
- **MoLFormer's inference is stochastic.** The checkpoint has `deterministic_eval = False`: the
  linear-attention feature map draws new random projections at every forward pass, in eval mode
  too. Ten passes of the same stage-1 weights over the test set give MAEs with SD 0.13 (per-polymer
  prediction SD 5.0 °C); on the 589 validation polymers the SD is about 0.35.
  It was left unchanged (`tables/inference_noise.csv`); results are reproducible statistically, not bit by bit.
- The shuffled-generated control has a huge seed spread (42 to 81): permuted labels on 7,912 mostly
  real polymer structures can push the network into a state that 500 real polymers do not fully repair.
- The teacher is far better on polymers that the literature measured repeatedly (section 8).

## 12. Caveats

- **Early stopping with patience 3 may stop the small-n real-only runs early.** The validation MAE
  swings by 2 to 5 °C from epoch to epoch (589 polymers, plus the inference noise above), and the
  real-only runs at n = 250 / 500 / 1,000 ran for 8.4 / 8.4 / 11.6 epochs on average (best epoch
  5.4 / 5.4 / 8.6); none reached the cap of 30. The gain from synthetic data at small n may
  therefore be, in part, a gain in optimization that longer or more patient training of the
  real-only baseline would also give.
- **Patience ablation (`pipeline/patience_ablation.py`, `tables/patience_ablation.csv`, 30 runs,
  2.6 GPU-hours).** The real-only and two-stage runs at n = 250 and 500 were repeated with patience 10
  (maximum 40 epochs), same seeds, subsets and stage-1 weights; the first epochs reproduce the main
  runs exactly. Real only improves from 39.0 to 37.7 (n = 250) and from 35.8 to 34.1 (n = 500);
  the two-stage runs improve by only 0.2 to 0.6. The gain of a synthetic first stage shrinks
  to -3.4 (generated) and -3.9 (labeled) at n = 250, both 5/5 seeds, and to -1.5 (4/5 seeds,
  p = 0.06) and -2.1 (5/5, p = 0.008) at n = 500. So a fifth to a quarter of the gain at 250 and
  about 40 % of it at 500 came from the baseline stopping early; the conservative estimate is 3 to 4 °C
  at 250 and about 2 °C at 500. The controls and n >= 1,000 were not repeated with patience 10.
- Target standardization: stage 1 uses the synthetic set's own mean and SD, stage 2 the real subset's,
  and the output layer is rescaled at the handover so that the network predicts exactly the same
  temperatures before and after (checked to 3e-5 C).
- The self-training control labels the 9,925 PI1M polymers of the cleaned labeled set.
- Multi-fragment SMILES (a repeat unit accompanied by a small molecule) were kept; the real data
  contain none.
- Concatenation runs use the 30-epoch cap of runs containing real data.
- The stage-1 weights of the three controls were deleted after their stage 2 (used once).
- The random forest ran on half the CPU cores alongside the GPU grid (`RF_JOBS`); this does not
  change its result.
- No scaffold or cluster split was used; the similarity table (section 9) quantifies how close the
  test polymers are to the training set.

## 13. Compute and cost

- MoLFormer: 120 runs, 26.0 GPU-hours on an RTX 3050 laptop GPU (6 GB). Random forest: 40 runs, 3.2 CPU-hours.
- API: 11,883 requests, USD 11.26 in total (generation 5.01, labeling 5.27, zero-shot 0.98); 13 requests
  at full price, the rest at the batch rate.
- Software: Python 3.14.4, PyTorch 2.13.0+cu130, Transformers 5.17.0, RDKit 2026.03.3, scikit-learn
  1.9.0, pandas 3.0.3, NumPy 2.4.6.

## 14. Files

`results.csv` (one row per run), `results_summary.csv`, `tables/` (every table above as CSV and
`all_tables.md`), `figures/` (learning_curve, controls_n500, predicted_vs_true, tg_histograms),
`predictions/` and `runs/` (one CSV and one JSON per run), `pseudo_labels/` (the self-training labels),
`memorization.csv`, `zero_shot_predictions.csv`, `zero_shot_metrics.json`, `zero_shot_by_tg_bin.csv`,
`api_cost.csv`, `test_similarity.csv`, `logs/`, `run_info.json`. `stage1_weights/` (1.8 GB) is a cache
and is listed in `.gitignore`.
