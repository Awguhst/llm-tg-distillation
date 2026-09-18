"""
Clean the two synthetic datasets with the rules in common/cleaning.py (step 4).

1. drop rows with a missing or non-numeric Tg (invalid JSON never produced a row)
2. drop SMILES RDKit cannot parse; canonicalize the rest
3. keep only SMILES with exactly two * connection points (a real repeat unit)
4. drop Tg outside the plausible range
5. merge duplicate SMILES (median Tg)
6. remove every polymer that appears in test_real.csv
For the generated set we also report (but keep) how many polymers are in the real
training set, and keep exactly GENERATION_TARGET rows if there are more.
"""
import pandas as pd

import config
from common import run_info
from common.cleaning import clean_dataset, print_counts

test_smiles = set(pd.read_csv(config.TEST_REAL_CSV)["smiles"])
train_smiles = set(pd.read_csv(config.TRAIN_REAL_CSV)["smiles"])
log = {}

# ---------------------------------------------------------------- generated dataset
generated_raw = pd.read_csv(config.GENERATED_RAW_CSV)
generated, counts = clean_dataset(generated_raw, test_smiles)
print_counts(counts, "generated")

# Informative only: how much did the model reproduce from memory?
n_in_train = generated["smiles"].isin(train_smiles).sum()
print(f"  generated polymers that also appear in the real TRAIN set: {n_in_train} "
      f"({100 * n_in_train / max(len(generated), 1):.1f}%) - kept")

if len(generated) > config.GENERATION_TARGET:
    generated = generated.sample(n=config.GENERATION_TARGET, random_state=config.SEED).reset_index(drop=True)
    print(f"  randomly kept exactly {config.GENERATION_TARGET} rows (seed {config.SEED})")
generated.to_csv(config.GENERATED_CLEAN_CSV, index=False)
print(f"  saved {len(generated)} rows -> {config.GENERATED_CLEAN_CSV}")
print(f"  raw pairs needed for these clean pairs: {counts['0_raw']}")

log["generated_cleaning_counts"] = counts
log["generated_in_real_train"] = int(n_in_train)
log["generated_final_size"] = int(len(generated))

# ---------------------------------------------------------------- labeled dataset
labeled_raw = pd.read_csv(config.LABELED_RAW_CSV)
labeled, counts = clean_dataset(labeled_raw[["smiles", "tg_celsius"]], test_smiles)
print_counts(counts, "labeled")
labeled.to_csv(config.LABELED_CLEAN_CSV, index=False)
print(f"  saved {len(labeled)} rows -> {config.LABELED_CLEAN_CSV}")

log["labeled_cleaning_counts"] = counts
log["labeled_final_size"] = int(len(labeled))

# ---------------------------------------------------------------- Tg statistics side by side
train = pd.read_csv(config.TRAIN_REAL_CSV)
print("\nTg (C) summary:")
for name, df in [("real train", train), ("generated", generated), ("labeled", labeled)]:
    tg = df["tg_celsius"]
    print(f"  {name:<11} n={len(tg):>6}  mean={tg.mean():7.1f}  std={tg.std():6.1f}  "
          f"min={tg.min():7.1f}  max={tg.max():7.1f}")

run_info.update(log)
