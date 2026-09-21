"""
Memorization analysis of the generated set.

How many of the polymers the teacher "generated" are polymers of the real collection, and how
good are the Tg values it attached to them? Matching uses both keys (common/cleaning.py):
"exact" = same canonical SMILES, "ring" = same polymer with the repeat unit cut elsewhere.
- Matches to the real TRAINING set stay in the generated set (counted here).
- Matches to the TEST and VALIDATION sets were removed by clean_synthetic_data.py; they are
  recovered here from the raw pairs to score the teacher's Tg for them.
The same counts are given for the labeled set (PI1M structures, few matches expected).
Writes results/memorization.csv.
"""
import os

import pandas as pd
from scipy.stats import spearmanr

import config
from common import run_info
from common.cleaning import KeySet, canonical_smiles, merge_duplicates, polymer_id

real = {"train": pd.read_csv(config.TRAIN_REAL_CSV), "validation": pd.read_csv(config.VAL_REAL_CSV),
        "test": pd.read_csv(config.TEST_REAL_CSV)}


def unique_before_removal(raw_csv):
    """Rules 1-5 of the cleaning: the unique synthetic polymers before test and validation matches are removed."""
    df = pd.read_csv(raw_csv)[["smiles", "tg_celsius"]]
    df["tg_celsius"] = pd.to_numeric(df["tg_celsius"], errors="coerce")
    df["smiles"] = df["smiles"].map(canonical_smiles)
    df = df.dropna()
    df = df[df["smiles"].str.count(r"\*") == 2]
    df = df[(df["tg_celsius"] >= config.TG_MIN_CELSIUS) & (df["tg_celsius"] <= config.TG_MAX_CELSIUS)]
    return merge_duplicates(df)


rows = []
for set_name, raw_csv in [("generated", config.GENERATED_RAW_CSV), ("labeled", config.LABELED_RAW_CSV)]:
    synthetic = unique_before_removal(raw_csv)
    synthetic["polymer_id"] = synthetic["smiles"].map(polymer_id)
    n_unique = len(synthetic)
    for split, real_part in real.items():
        match = synthetic["smiles"].map(KeySet(real_part["smiles"]).match_type)
        real_tg = real_part.assign(polymer_id=real_part["smiles"].map(polymer_id)).set_index("polymer_id")["tg_celsius"]
        for kind in ["exact", "ring", "either"]:
            part = synthetic[match.notna()] if kind == "either" else synthetic[match == kind]
            row = {"synthetic_set": set_name, "real_split": split, "match": kind, "n": len(part),
                   "percent_of_unique_synthetic": round(100 * len(part) / n_unique, 2)}
            if len(part) >= 3:
                error = part["tg_celsius"].values - real_tg.loc[part["polymer_id"]].values
                row.update({"teacher_mae": abs(error).mean(), "teacher_median_abs_error": pd.Series(abs(error)).median(),
                            "teacher_spearman": spearmanr(part["tg_celsius"], real_tg.loc[part["polymer_id"]]).statistic,
                            "teacher_mean_signed_error": error.mean()})
            rows.append(row)
    any_real = synthetic["smiles"].map(KeySet(pd.concat(real.values())["smiles"]).match_type).notna().sum()
    print(f"{set_name}: {n_unique} unique polymers before removal, {any_real} "
          f"({100 * any_real / n_unique:.1f} %) match a polymer of the real collection on either key")
    run_info.update({f"{set_name}_unique_before_removal": n_unique, f"{set_name}_matching_any_real_polymer": int(any_real)})

table = pd.DataFrame(rows).round(2)
table.to_csv(os.path.join(config.RESULTS_DIR, "memorization.csv"), index=False)
print(table.to_string(index=False))
