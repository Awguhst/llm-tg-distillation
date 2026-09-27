"""
Describe the split: how similar is every test polymer to the training set?

For each test polymer, the maximum Tanimoto similarity to any training polymer (Morgan
fingerprints, radius 2, 2048 bits, the fingerprints of the random forest). This is reported
for the Limitations section and used later to stratify the test error. It does not change
the split and uses no Tg value.
Writes results/test_similarity.csv (one row per test polymer).
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

import config
from common import run_info

os.makedirs(config.RESULTS_DIR, exist_ok=True)
generator = rdFingerprintGenerator.GetMorganGenerator(radius=config.FINGERPRINT_RADIUS, fpSize=config.FINGERPRINT_BITS)


def fingerprints(smiles_list):
    return [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def main():
    train = pd.read_csv(config.TRAIN_REAL_CSV)
    test = pd.read_csv(config.TEST_REAL_CSV)
    train_fps = fingerprints(train["smiles"])
    similarity = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in fingerprints(test["smiles"])])

    pd.DataFrame({"smiles": test["smiles"], "max_tanimoto_to_train": similarity.round(4)}).to_csv(
        config.TEST_SIMILARITY_CSV, index=False)

    bins = pd.cut(similarity, config.SIMILARITY_BINS, right=False)
    table = pd.Series(bins).value_counts().sort_index()
    quantiles = {f"q{int(q * 100)}": round(float(np.quantile(similarity, q)), 3) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}
    print(f"Maximum Tanimoto similarity of the {len(test)} test polymers to the {len(train)} training polymers")
    print(f"  quantiles: {quantiles}")
    for interval, n in table.items():
        print(f"  {str(interval):<14} {n:>5}  ({100 * n / len(test):.1f} %)")
    n_identical = int((similarity >= 0.9999).sum())
    print(f"  test polymers with a training polymer of identical fingerprint (similarity 1.0): {n_identical} "
          f"({100 * n_identical / len(test):.1f} %). These are not duplicates (both keys differ); they are mostly homologues.")

    run_info.update({"test_similarity_to_train": {
        "fingerprint": f"Morgan radius {config.FINGERPRINT_RADIUS}, {config.FINGERPRINT_BITS} bits, Tanimoto",
        "quantiles": quantiles, "bins": {str(k): int(v) for k, v in table.items()},
        "n_identical_fingerprint": n_identical}})


if __name__ == "__main__":
    main()
