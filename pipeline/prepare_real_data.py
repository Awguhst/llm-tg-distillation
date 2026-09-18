"""
Clean the real Tg dataset and split it into train and test.

- Canonicalize SMILES with RDKit (the * connection points are kept).
- Drop rows RDKit cannot parse.
- Merge duplicate canonical SMILES by taking the median Tg.
- Random 80/20 split with the fixed seed.

The test set is used only for the final evaluation and never appears in a prompt.
"""
import pandas as pd

import config
from common.cleaning import canonical_smiles

real = pd.read_csv(config.REAL_TG_CSV)
print(f"Loaded {len(real)} rows with columns {list(real.columns)}")

# Rename to the column names used everywhere else in the pipeline.
real = real.rename(columns={"SMILES": "smiles", "Tg": "tg_celsius", "Polymer Class": "polymer_class", "PID": "pid"})

# The Kaggle file has no column marking experimental vs computed values, so there is
# nothing to report there. Every value is treated as experimental (PolyInfo is experimental).

# Canonicalize and drop what RDKit cannot parse.
real["smiles"] = real["smiles"].map(canonical_smiles)
n_before = len(real)
real = real.dropna(subset=["smiles"])
print(f"Dropped {n_before - len(real)} rows that RDKit could not parse -> {len(real)} rows")

# Merge duplicates: same canonical SMILES -> median Tg. Keep the first class and PID
# only as information for the reader; they are not used for training.
n_before = len(real)
real = real.groupby("smiles", as_index=False).agg(
    tg_celsius=("tg_celsius", "median"),
    polymer_class=("polymer_class", "first"),
    pid=("pid", "first"),
)
print(f"Merged {n_before - len(real)} duplicate structures -> {len(real)} unique polymers")

# Random split. sample(frac=1) shuffles the rows with the fixed seed.
real = real.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
n_test = int(round(len(real) * config.TEST_FRACTION))
test = real.iloc[:n_test]
train = real.iloc[n_test:]

train.to_csv(config.TRAIN_REAL_CSV, index=False)
test.to_csv(config.TEST_REAL_CSV, index=False)
print(f"Saved {len(train)} train rows -> {config.TRAIN_REAL_CSV}")
print(f"Saved {len(test)} test rows  -> {config.TEST_REAL_CSV}")
print(f"Tg range: {real['tg_celsius'].min():.1f} to {real['tg_celsius'].max():.1f} C, "
      f"mean {real['tg_celsius'].mean():.1f} C")
