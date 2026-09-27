"""
Clean the two synthetic datasets with the rules in common/cleaning.py.

Reads the parsed responses (data/generated_raw.csv, data/labeled_raw.csv):
1. drop rows with a missing or non-numeric Tg
2. drop SMILES RDKit cannot parse; canonicalize the rest
3. keep only SMILES with exactly two * connection points
4. drop Tg outside the plausible range
5. merge duplicates on either key (exact canonical SMILES or ring-closure key), median Tg
6. remove every polymer matching the real TEST or VALIDATION set on either key
Polymers that also occur in the real training set are counted (memorization.py) but kept.
"""
import pandas as pd

import config
from common import run_info
from common.cleaning import KeySet, clean_dataset, print_counts


def main():
    held_out = {"test": KeySet(pd.read_csv(config.TEST_REAL_CSV)["smiles"]),
                "validation": KeySet(pd.read_csv(config.VAL_REAL_CSV)["smiles"])}
    log = {}

    for name, raw_csv, clean_csv in [("generated", config.GENERATED_RAW_CSV, config.GENERATED_CLEAN_CSV),
                                     ("labeled", config.LABELED_RAW_CSV, config.LABELED_CLEAN_CSV)]:
        raw = pd.read_csv(raw_csv)
        clean, counts = clean_dataset(raw[["smiles", "tg_celsius"]], held_out)
        print_counts(counts, name)
        if len(clean) > config.GENERATION_TARGET:      # never the case in the published run
            clean = clean.sample(n=config.GENERATION_TARGET, random_state=config.SEED).reset_index(drop=True)
        clean.to_csv(clean_csv, index=False)
        print(f"  saved {len(clean)} rows -> {clean_csv}")
        log[f"{name}_cleaning_counts"] = counts
        log[f"{name}_final_size"] = int(len(clean))

    print("\nTg (C) summary:")
    for name, path in [("real train", config.TRAIN_REAL_CSV), ("generated", config.GENERATED_CLEAN_CSV),
                       ("labeled", config.LABELED_CLEAN_CSV)]:
        tg = pd.read_csv(path)["tg_celsius"]
        print(f"  {name:<11} n={len(tg):>6}  mean={tg.mean():7.1f}  std={tg.std():6.1f}  min={tg.min():7.1f}  max={tg.max():7.1f}")
    run_info.update(log)


if __name__ == "__main__":
    main()
