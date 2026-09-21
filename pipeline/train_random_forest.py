"""
Train and evaluate the random-forest baseline.

500 trees on 2048-bit Morgan fingerprints of radius 2; no early stopping, no two-stage
protocol and no use of the validation set. Eight training sets per seed: real (full), generated,
labeled, real + generated, real + labeled, 500 real, 500 + generated, 500 + labeled. The 500 real
polymers are the same nested subset as in train_molformer.py, and mixed sets are deduplicated
on either key with the real label kept. Outputs as for MoLFormer: one prediction CSV and one JSON
per run under results/ (a run whose JSON exists is skipped).
"""
import json
import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from common.cleaning import KeySet

os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
os.makedirs(config.RUNS_DIR, exist_ok=True)

train_real = pd.read_csv(config.TRAIN_REAL_CSV)[["smiles", "tg_celsius"]]
test = pd.read_csv(config.TEST_REAL_CSV)[["smiles", "tg_celsius"]]
synthetic = {"generated": pd.read_csv(config.GENERATED_CLEAN_CSV), "labeled": pd.read_csv(config.LABELED_CLEAN_CSV)}

generator = rdFingerprintGenerator.GetMorganGenerator(radius=config.FINGERPRINT_RADIUS, fpSize=config.FINGERPRINT_BITS)
cache = {}


def fingerprints(smiles_list):
    """Binary Morgan fingerprints as a matrix. The * atoms are ordinary atoms to RDKit."""
    for s in smiles_list:
        if s not in cache:
            cache[s] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.uint8)
    return np.array([cache[s] for s in smiles_list])


def mixed(real, synthetic_set):
    keep = ~KeySet(real["smiles"]).matches(synthetic_set["smiles"])
    return pd.concat([real, synthetic_set[keep][["smiles", "tg_celsius"]]])


X_test = fingerprints(test["smiles"])
for seed in config.SEEDS:
    order = np.random.RandomState(seed).permutation(len(train_real))      # the same subset as train_molformer.py
    small = train_real.iloc[order[:config.CONTROL_SIZE]]
    n = config.CONTROL_SIZE
    runs = [("real_nfull", train_real, len(train_real)), ("synthetic_generated", synthetic["generated"], 0),
            ("synthetic_labeled", synthetic["labeled"], 0),
            ("concat_generated_nfull", mixed(train_real, synthetic["generated"]), len(train_real)),
            ("concat_labeled_nfull", mixed(train_real, synthetic["labeled"]), len(train_real)),
            (f"real_n{n}", small, n), (f"concat_generated_n{n}", mixed(small, synthetic["generated"]), n),
            (f"concat_labeled_n{n}", mixed(small, synthetic["labeled"]), n)]
    for name, train, n_real in runs:
        json_path = os.path.join(config.RUNS_DIR, f"rf_{name}_seed{seed}.json")
        if os.path.exists(json_path):
            continue
        start_time = time.time()
        model = RandomForestRegressor(n_estimators=config.RF_TREES, random_state=seed, n_jobs=config.RF_JOBS)
        model.fit(fingerprints(train["smiles"]), train["tg_celsius"])
        y_pred = model.predict(X_test)
        y_true = test["tg_celsius"].values
        metrics = {"mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                   "r2": float(r2_score(y_true, y_pred)), "spearman": float(spearmanr(y_true, y_pred).statistic)}
        pd.DataFrame({"smiles": test["smiles"], "tg_true": y_true, "tg_pred": y_pred}).to_csv(
            os.path.join(config.PREDICTIONS_DIR, f"rf_{name}_seed{seed}.csv"), index=False)
        seconds = time.time() - start_time
        print(f"seed {seed} | {name:<24} {len(train):>6} rows | test MAE = {metrics['mae']:.2f}  R2 = {metrics['r2']:.3f}  ({seconds:.0f} s)", flush=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"run": name, "group": "random forest", "seed": seed, "model": "random_forest", "n_real": n_real,
                       "n_train_rows": len(train), "trees": config.RF_TREES, "fingerprint_bits": config.FINGERPRINT_BITS,
                       "fingerprint_radius": config.FINGERPRINT_RADIUS, "test": metrics, "n_test": len(test),
                       "train_seconds": round(seconds), "date": time.strftime("%Y-%m-%d")}, f, indent=2)
