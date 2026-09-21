"""
Clean the real Tg dataset and split it into training, validation and test sets.

The file is the curated Tg collection of the Jablonka group (Zenodo, see config.py).
- Inspect the file: columns, dtypes, row count, reliability flags (nothing is assumed).
- Canonicalize the PSMILES with RDKit (attachment points [*] become *, which are kept).
- Apply the same rules as for the synthetic sets: RDKit-valid, exactly two *, Tg within
  the plausible range, duplicates merged by median Tg. Duplicates are matched on either key
  (common/cleaning.py), so no polymer can end up in two splits under different cuts.
- Shuffle with the fixed seed: 20 % test; of the rest, 10 % validation, 90 % training.

The test set is used only for the final evaluation and for the zero-shot query
(zero_shot_with_claude.py). Every choice made during training uses the validation set.
"""
import ast
import hashlib

import numpy as np
import pandas as pd

import config
from common import run_info
from common.cleaning import canonical_smiles, polymer_id, polymer_keys

# ---------------------------------------------------------------- 1. load and inspect
md5 = hashlib.md5(open(config.REAL_TG_CSV, "rb").read()).hexdigest()
if md5 != config.REAL_TG_MD5:
    raise SystemExit(f"md5 of {config.REAL_TG_CSV} is {md5}, expected {config.REAL_TG_MD5}")

header = pd.read_csv(config.REAL_TG_CSV, nrows=0).columns
c = config.REAL_COLUMNS
raw = pd.read_csv(config.REAL_TG_CSV, usecols=list(c.values()))
print(f"Loaded {len(raw)} rows; the file has {len(header)} columns, of which {len(raw.columns)} are used "
      f"(the rest are precomputed side-chain / backbone / full-polymer features and text embeddings).")
print(raw.dtypes.to_string())
print(raw.head(3).T.to_string())
print("\nReliability flag:\n" + raw[c["reliability"]].value_counts().to_string())
print("\nData source:\n" + raw[c["source"]].value_counts().to_string())

real = pd.DataFrame({
    "smiles": raw[c["psmiles"]].map(canonical_smiles),
    "tg_celsius": raw[c["tg_kelvin"]] - 273.15,
    "n_points": raw[c["n_points"]],
    "reliability": raw[c["reliability"]],
    "source": raw[c["source"]],
    "polymer_class": raw[c["polymer_class"]],
})
# The individual measurements of a polymer, in Celsius: the listed values, or the one value.
real["tg_values"] = [
    [v - 273.15 for v in ast.literal_eval(values)] if isinstance(values, str) else [tg]
    for values, tg in zip(raw[c["tg_values"]], real["tg_celsius"])
]

# ---------------------------------------------------------------- 2. the cleaning rules of the synthetic sets
counts = {"0_raw": len(real)}
real = real.dropna(subset=["smiles"])
counts["2_valid_smiles"] = len(real)
real = real[real["smiles"].str.count(r"\*") == 2]
counts["3_two_stars"] = len(real)
real = real[(real["tg_celsius"] >= config.TG_MIN_CELSIUS) & (real["tg_celsius"] <= config.TG_MAX_CELSIUS)]
counts["4_tg_in_range"] = len(real)
counts["5a_unique_exact_smiles"] = real["smiles"].nunique()

# Merge duplicates on either key: median Tg; the measurements of the merged rows are pooled.
real["polymer_id"] = real["smiles"].map(polymer_id)
real = real.sort_values("smiles").groupby("polymer_id", as_index=False).agg(
    smiles=("smiles", "first"),
    tg_celsius=("tg_celsius", "median"),
    n_points=("n_points", "sum"),
    tg_values=("tg_values", "sum"),
    reliability=("reliability", lambda x: "/".join(sorted(set(x)))),
    source=("source", lambda x: "/".join(sorted(set(x)))),
    polymer_class=("polymer_class", "first"),
    n_rows_merged=("smiles", "size"),
)
counts["5b_unique_either_key"] = len(real)
# Sample SD of the individual measurements (as in the file's own column); empty for one measurement.
real["tg_std"] = [np.std(v, ddof=1) if len(v) > 1 else np.nan for v in real["tg_values"]]
real["ring_key"] = [polymer_keys(s)[1] for s in real["smiles"]]

print("\nCleaning of the real data (rows kept after each rule):")
previous = None
for step, n in counts.items():
    print(f"  {step:<26} {n:>6}" + ("" if previous is None else f"  (removed {previous - n})"))
    previous = n
n_no_ring = int(real["ring_key"].isna().sum())
print(f"Ring key not computable for {n_no_ring} of {len(real)} polymers ({100 * n_no_ring / len(real):.1f} %); "
      f"these are matched by exact canonical SMILES only.")

# ---------------------------------------------------------------- 3. random split: test, then validation
columns = ["smiles", "tg_celsius", "n_points", "tg_std", "reliability", "source", "polymer_class", "ring_key"]
real = real.sort_values("smiles")[columns]                      # a fixed order before shuffling
real.to_csv(config.REAL_CLEAN_CSV, index=False)

real = real.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
n_test = int(round(len(real) * config.TEST_FRACTION))
test, rest = real.iloc[:n_test], real.iloc[n_test:]
rest = rest.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
n_val = int(round(len(rest) * config.VAL_FRACTION))
val, train = rest.iloc[:n_val], rest.iloc[n_val:]

# No polymer may sit in two splits under either key.
ids = [set(part["smiles"].map(polymer_id)) for part in (train, val, test)]
assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])

stats = {}
print("\nSplit (seed %d):" % config.SEED)
for name, part, path in [("train", train, config.TRAIN_REAL_CSV), ("validation", val, config.VAL_REAL_CSV),
                         ("test", test, config.TEST_REAL_CSV)]:
    part.to_csv(path, index=False)
    tg = part["tg_celsius"]
    stats[name] = {"n": len(part), "tg_mean": round(float(tg.mean()), 1), "tg_sd": round(float(tg.std()), 1),
                   "tg_min": round(float(tg.min()), 1), "tg_median": round(float(tg.median()), 1),
                   "tg_max": round(float(tg.max()), 1), "n_with_2plus_points": int((part["n_points"] >= 2).sum())}
    print(f"  {name:<11} {stats[name]}  -> {path}")

run_info.update({
    "real_dataset": {"zenodo_record": config.REAL_TG_ZENODO_RECORD, "doi": config.REAL_TG_DOI,
                     "file": config.REAL_TG_FILE, "md5": md5, "licence": "CC-BY 4.0",
                     "rows": counts["0_raw"], "columns_in_file": len(header)},
    "real_cleaning_counts": counts,
    "real_reliability_counts_raw": raw[c["reliability"]].value_counts().to_dict(),
    "real_source_counts_raw": raw[c["source"]].value_counts().to_dict(),
    "real_ring_key_not_computable": n_no_ring,
    "split": {"method": "random; cleaned polymers deduplicated on either key, shuffled (pandas sample, "
                        "frac=1), first TEST_FRACTION = test; the rest shuffled again, first VAL_FRACTION = validation",
              "seed": config.SEED, "test_fraction": config.TEST_FRACTION, "val_fraction": config.VAL_FRACTION,
              "sizes_and_tg": stats},
})
