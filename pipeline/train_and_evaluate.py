"""
Train and evaluate (step 5).

For each of the 8 experiments and each seed in config.SEEDS we train two models on the
experiment's training data and evaluate both on the same test_real.csv:
- a random forest on Morgan fingerprints (fast sanity baseline)
- MolFormer (ibm/MoLFormer-XL-both-10pct) fine-tuned for regression
The seed changes the model initialisation, the shuffling, the random forest, and the
500-polymer real subset; the train/test split and the synthetic data never change.
We also score the zero-shot Claude predictions (zero_shot_with_claude.py) on the same test set.

Outputs:
- results/predictions/exp{n}_{model}_seed{seed}.csv   one file per run (a finished run is never repeated)
- results/results.csv                                  one row per run (experiment, model, seed)
- results/results_summary.csv                          mean and standard deviation over seeds
"""
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from common import claude_batches, run_info

os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)
TIMES_CSV = os.path.join(config.RESULTS_DIR, "train_times.csv")   # wall-clock seconds per run

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"GPU found: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. MolFormer fine-tuning on CPU takes roughly 30-60 minutes per 5,000 "
          "training rows and epoch. Consider lowering EPOCHS in config.py or using Google Colab.")


# ---------------------------------------------------------------- metrics

def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).statistic,
    }


# ---------------------------------------------------------------- random forest baseline

def morgan_fingerprints(smiles_list):
    """Binary Morgan fingerprint for every SMILES as a numpy matrix. The * atoms are ordinary atoms to RDKit."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=config.FINGERPRINT_RADIUS, fpSize=config.FINGERPRINT_BITS)
    X = np.zeros((len(smiles_list), config.FINGERPRINT_BITS), dtype=np.uint8)
    for i, smiles in enumerate(smiles_list):
        X[i] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
    return X


def train_random_forest(train, test, seed):
    model = RandomForestRegressor(n_estimators=config.RF_TREES, random_state=seed, n_jobs=-1)
    model.fit(morgan_fingerprints(train["smiles"]), train["tg_celsius"])
    return model.predict(morgan_fingerprints(test["smiles"]))


# ---------------------------------------------------------------- MolFormer fine-tuning

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tokenizer, smiles_list):
    """Tokenize a list of SMILES into padded tensors."""
    return tokenizer(list(smiles_list), padding=True, truncation=True,
                     max_length=config.MAX_SMILES_TOKENS, return_tensors="pt")


def train_molformer(train, test, seed):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(config.MOLFORMER_NAME, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MOLFORMER_NAME, num_labels=1, trust_remote_code=True).to(DEVICE)

    # Standardize the targets so the regression head starts at a sensible scale.
    tg_mean, tg_std = train["tg_celsius"].mean(), train["tg_celsius"].std()
    y = torch.tensor(((train["tg_celsius"] - tg_mean) / tg_std).values, dtype=torch.float32)
    smiles = list(train["smiles"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = torch.nn.MSELoss()

    model.train()
    for epoch in range(config.EPOCHS):
        order = np.random.permutation(len(smiles))   # new shuffle every epoch
        total_loss = 0.0
        for start in range(0, len(order), config.BATCH_SIZE):
            idx = order[start:start + config.BATCH_SIZE]
            batch = encode(tokenizer, [smiles[i] for i in idx]).to(DEVICE)
            prediction = model(**batch).logits.squeeze(-1)
            loss = loss_function(prediction, y[idx].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"    epoch {epoch + 1}/{config.EPOCHS}  train MSE (standardized) = {total_loss / len(smiles):.4f}", flush=True)

    # Predict the test set in batches and undo the standardization.
    model.eval()
    predictions = []
    test_smiles = list(test["smiles"])
    with torch.no_grad():
        for start in range(0, len(test_smiles), config.BATCH_SIZE):
            batch = encode(tokenizer, test_smiles[start:start + config.BATCH_SIZE]).to(DEVICE)
            predictions.append(model(**batch).logits.squeeze(-1).cpu())
    predictions = torch.cat(predictions).numpy() * tg_std + tg_mean

    del model
    torch.cuda.empty_cache()
    return predictions


# ---------------------------------------------------------------- experiments

train_real = pd.read_csv(config.TRAIN_REAL_CSV)
test = pd.read_csv(config.TEST_REAL_CSV)
generated = pd.read_csv(config.GENERATED_CLEAN_CSV)
labeled = pd.read_csv(config.LABELED_CLEAN_CSV)
y_true = test["tg_celsius"].values
columns = ["smiles", "tg_celsius"]   # only the two columns every dataset shares are used


def experiments_for_seed(seed):
    """The eight training sets. The 500-polymer subset is drawn with the run's seed."""
    small_real = train_real.sample(n=config.SMALL_REAL_SUBSET, random_state=seed)
    return [
        (1, "real", train_real[columns]),
        (2, "generated", generated[columns]),
        (3, "labeled", labeled[columns]),
        (4, "real + generated", pd.concat([train_real[columns], generated[columns]])),
        (5, "real + labeled", pd.concat([train_real[columns], labeled[columns]])),
        (6, f"small real ({config.SMALL_REAL_SUBSET})", small_real[columns]),
        (7, "small real + generated", pd.concat([small_real[columns], generated[columns]])),
        (8, "small real + labeled", pd.concat([small_real[columns], labeled[columns]])),
    ]


def prediction_path(number, model_name, seed):
    return os.path.join(config.PREDICTIONS_DIR, f"exp{number}_{model_name}_seed{seed}.csv")


# Train every run whose prediction file does not exist yet.
for seed in config.SEEDS:
    for number, name, train in experiments_for_seed(seed):
        train = train.reset_index(drop=True)
        for model_name, train_function in [("random_forest", train_random_forest), ("molformer", train_molformer)]:
            path = prediction_path(number, model_name, seed)
            if os.path.exists(path):
                continue
            print(f"\n=== seed {seed} | experiment {number}: {name} ({len(train)} rows) | {model_name} ===", flush=True)
            start_time = time.time()
            y_pred = train_function(train, test, seed)
            seconds = time.time() - start_time
            metrics = compute_metrics(y_true, y_pred)
            print(f"    MAE={metrics['mae']:.1f}  RMSE={metrics['rmse']:.1f}  R2={metrics['r2']:.3f}  "
                  f"Spearman={metrics['spearman']:.3f}  ({seconds:.0f} s)", flush=True)
            pd.DataFrame({"smiles": test["smiles"], "tg_true": y_true, "tg_pred": y_pred}).to_csv(path, index=False)
            new_file = not os.path.exists(TIMES_CSV)
            with open(TIMES_CSV, "a") as f:
                if new_file:
                    f.write("experiment,model,seed,train_seconds\n")   # compute_facts.py reads these names
                f.write(f"{number},{model_name},{seed},{round(seconds)}\n")

# ---------------------------------------------------------------- collect all runs from the prediction files
rows = []
names = {number: name for number, name, _ in experiments_for_seed(config.SEED)}
sizes = {number: len(train) for number, _, train in experiments_for_seed(config.SEED)}
for seed in config.SEEDS:
    for number in range(1, 9):
        for model_name in ["random_forest", "molformer"]:
            pred = pd.read_csv(prediction_path(number, model_name, seed))
            row = {"experiment": number, "training_data": names[number], "n_train": sizes[number],
                   "model": model_name, "seed": seed, "n_test": len(pred)}
            row.update(compute_metrics(pred["tg_true"], pred["tg_pred"]))
            rows.append(row)

# The zero-shot Claude predictions have no seed; they appear once.
zero_shot = pd.read_csv(config.ZERO_SHOT_CSV)
usable = zero_shot.dropna(subset=["tg_pred"])
row = {"experiment": 0, "training_data": "none (zero-shot Claude)", "n_train": 0, "model": config.LLM_MODEL,
       "seed": None, "n_test": len(usable)}
row.update(compute_metrics(usable["tg_true"], usable["tg_pred"]))
rows.append(row)

results = pd.DataFrame(rows).sort_values(["experiment", "model", "seed"])
results.to_csv(config.RESULTS_CSV, index=False)

# Mean and standard deviation over seeds for every experiment and model.
metrics = ["mae", "rmse", "r2", "spearman"]
summary = (results[results["experiment"] > 0]
           .groupby(["experiment", "training_data", "n_train", "model"])[metrics]
           .agg(["mean", "std"]).reset_index())
summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
summary["n_seeds"] = len(config.SEEDS)
summary.to_csv(os.path.join(config.RESULTS_DIR, "results_summary.csv"), index=False)

print(f"\nSaved {config.RESULTS_CSV} ({len(results)} rows) and results_summary.csv")
print(summary.round(3).to_string(index=False))
print(f"\nZero-shot {config.LLM_MODEL}: MAE={row['mae']:.1f}  RMSE={row['rmse']:.1f}  R2={row['r2']:.3f}  "
      f"Spearman={row['spearman']:.3f}  ({len(usable)} of {len(zero_shot)} test polymers had a usable prediction)")

run_info.update({
    "training_model": config.MOLFORMER_NAME,
    "training_epochs": config.EPOCHS,
    "training_learning_rate": config.LEARNING_RATE,
    "training_batch_size": config.BATCH_SIZE,
    "training_seeds": config.SEEDS,
    "training_device": DEVICE,
    "training_date": time.strftime("%Y-%m-%d"),
    "dataset_sizes": {"train_real": len(train_real), "test_real": len(test),
                      "generated_clean": len(generated), "labeled_clean": len(labeled),
                      "small_real": config.SMALL_REAL_SUBSET},
    "total_api_cost_usd": round(claude_batches.total_cost_so_far(), 4),
})
