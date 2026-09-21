"""
Train and evaluate MoLFormer.

Every run fine-tunes ibm/MoLFormer-XL-both-10pct with fixed settings (AdamW, learning
rate 3e-5, batch size 16, 128 tokens) and EARLY STOPPING ON THE VALIDATION MAE: the real
validation set is scored after every epoch, the best weights are kept, and training stops after
config.PATIENCE epochs without improvement. The test set is predicted once, with the best
weights, after training has ended; no choice of any kind looks at it.

Runs per seed (config.SEEDS); n is the number of real training polymers, drawn as nested subsets
250 < 500 < 1000 < full from one seeded shuffle of the training set:
  A  synthetic only        stage 1 on the generated / labeled set. Weights are saved and reused.
  B  learning curve        real only; two-stage generated -> real; two-stage labeled -> real   (every n)
  C  concatenation         real + generated; real + labeled, real label kept for duplicates    (n = 500, full)
  D  controls (n = 500)    two-stage with self-training pseudo-labels (the seed's own 500-real model
                           labels the PI1M polymers of the labeled set); two-stage with the Tg values
                           of each synthetic set randomly permuted
Targets are standardized with the mean and SD of the run's own training data (the synthetic set in
stage 1, the real subset in stage 2). When stage 2 starts from stage-1 weights, the output layer is
rescaled so that the network predicts exactly the same temperatures under the new standardization.

Outputs, one pair per run, under results/ (a run whose JSON exists is skipped):
  predictions/<run>_seed<seed>.csv   smiles, tg_true, tg_pred for every test polymer
  runs/<run>_seed<seed>.json         settings, best epoch, validation history, test metrics, seconds

Usage:  python pipeline/train_molformer.py [--seeds 42 43] [--sizes 500]
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from common.cleaning import KeySet

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
parser.add_argument("--sizes", nargs="+", default=None, help="real subset sizes to run, e.g. 500 full (default: all)")
args = parser.parse_args()

PSEUDO_DIR = os.path.join(config.RESULTS_DIR, "pseudo_labels")
for folder in [config.PREDICTIONS_DIR, config.RUNS_DIR, config.WEIGHTS_DIR, PSEUDO_DIR]:
    os.makedirs(folder, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU (slow)'}")

tokenizer = AutoTokenizer.from_pretrained(config.MOLFORMER_NAME, trust_remote_code=True)
train_real = pd.read_csv(config.TRAIN_REAL_CSV)[["smiles", "tg_celsius"]]
val = pd.read_csv(config.VAL_REAL_CSV)[["smiles", "tg_celsius"]]
test = pd.read_csv(config.TEST_REAL_CSV)[["smiles", "tg_celsius"]]
synthetic = {"generated": pd.read_csv(config.GENERATED_CLEAN_CSV), "labeled": pd.read_csv(config.LABELED_CLEAN_CSV)}


# ---------------------------------------------------------------- helpers

def compute_metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(spearmanr(y_true, y_pred).statistic)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(smiles_list):
    return tokenizer(list(smiles_list), padding=True, truncation=True,
                     max_length=config.MAX_SMILES_TOKENS, return_tensors="pt")


def predict(model, smiles_list, mean, std):
    """Predicted Tg in Celsius for a list of SMILES."""
    model.eval()
    smiles_list = list(smiles_list)
    out = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), config.EVAL_BATCH_SIZE):
            batch = encode(smiles_list[start:start + config.EVAL_BATCH_SIZE]).to(DEVICE)
            out.append(model(**batch).logits.squeeze(-1).float().cpu())
    return torch.cat(out).numpy() * std + mean


def load_model(stage1_weights=None, new_mean=None, new_std=None):
    """
    The pretrained checkpoint with a fresh regression head, or a stage-1 model. A stage-1 model was
    trained on targets standardized with its own mean and SD; its output layer is rescaled so that
    it predicts the same temperatures under the standardization of stage 2 (new_mean, new_std).
    """
    model = AutoModelForSequenceClassification.from_pretrained(config.MOLFORMER_NAME, num_labels=1, trust_remote_code=True)
    if stage1_weights is not None:
        saved = torch.load(stage1_weights, map_location="cpu")
        model.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            layer = model.classifier.out_proj
            layer.weight.mul_(saved["std"] / new_std)
            layer.bias.copy_((layer.bias * saved["std"] + saved["mean"] - new_mean) / new_std)
    return model.to(DEVICE)


def fit(model, train, mean, std, max_epochs):
    """
    Fine-tune with early stopping on the validation MAE. Returns the history (one entry per epoch)
    and leaves the best weights in the model.
    """
    y = torch.tensor(((train["tg_celsius"] - mean) / std).values, dtype=torch.float32)
    smiles = list(train["smiles"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = torch.nn.MSELoss()
    history, best_mae, best_state, since_best = [], float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = np.random.permutation(len(smiles))   # new shuffle every epoch
        total_loss = 0.0
        for start in range(0, len(order), config.BATCH_SIZE):
            idx = order[start:start + config.BATCH_SIZE]
            batch = encode([smiles[i] for i in idx]).to(DEVICE)
            loss = loss_function(model(**batch).logits.squeeze(-1), y[idx].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        val_mae = float(np.abs(predict(model, val["smiles"], mean, std) - val["tg_celsius"].values).mean())
        history.append({"epoch": epoch, "train_mse_standardized": total_loss / len(smiles), "val_mae": val_mae})
        print(f"    epoch {epoch:>2}  train MSE = {total_loss / len(smiles):.4f}  validation MAE = {val_mae:.2f}", flush=True)
        if val_mae < best_mae:
            best_mae, since_best = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= config.PATIENCE:
                break
    model.load_state_dict(best_state)
    return history


def run_path(name, seed, kind):
    folder, ext = {"json": (config.RUNS_DIR, "json"), "csv": (config.PREDICTIONS_DIR, "csv"),
                   "weights": (config.WEIGHTS_DIR, "pt")}[kind]
    return os.path.join(folder, f"{name}_seed{seed}.{ext}")


def run(name, seed, group, train, max_epochs, n_real, stage1=None, save_weights=False, pseudo_label=None, extra=None):
    """
    One training run. stage1 = name of the stage-1 run whose weights to start from (None = pretrained
    checkpoint). pseudo_label = SMILES to label with the finished model (self-training control).
    """
    if os.path.exists(run_path(name, seed, "json")):
        return
    print(f"\n=== seed {seed} | {name} | {len(train)} training rows | start: {stage1 or 'pretrained checkpoint'} ===", flush=True)
    start_time = time.time()
    set_seed(seed)
    train = train.reset_index(drop=True)
    mean, std = float(train["tg_celsius"].mean()), float(train["tg_celsius"].std())
    model = load_model(run_path(stage1, seed, "weights") if stage1 else None, mean, std)
    history = fit(model, train, mean, std, max_epochs)
    best = min(history, key=lambda h: h["val_mae"])

    y_pred = predict(model, test["smiles"], mean, std)
    metrics = compute_metrics(test["tg_celsius"].values, y_pred)
    pd.DataFrame({"smiles": test["smiles"], "tg_true": test["tg_celsius"], "tg_pred": y_pred}).to_csv(
        run_path(name, seed, "csv"), index=False)
    if save_weights:
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, run_path(name, seed, "weights"))
    if pseudo_label is not None:
        pd.DataFrame({"smiles": pseudo_label, "tg_celsius": predict(model, pseudo_label, mean, std)}).to_csv(
            os.path.join(PSEUDO_DIR, f"selftrain_seed{seed}.csv"), index=False)
    seconds = time.time() - start_time
    print(f"    best epoch {best['epoch']} (validation MAE {best['val_mae']:.2f}) | test MAE = {metrics['mae']:.2f}  "
          f"R2 = {metrics['r2']:.3f}  ({seconds:.0f} s)", flush=True)

    record = {"run": name, "group": group, "seed": seed, "model": "molformer", "n_real": n_real, "n_train_rows": len(train),
              "start_weights": stage1 or config.MOLFORMER_NAME, "target_mean": mean, "target_std": std,
              "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE, "max_tokens": config.MAX_SMILES_TOKENS,
              "max_epochs": max_epochs, "patience": config.PATIENCE, "epochs_run": len(history),
              "best_epoch": best["epoch"], "best_val_mae": best["val_mae"], "history": history,
              "optimizer_steps_to_best": best["epoch"] * int(np.ceil(len(train) / config.BATCH_SIZE)),
              "test": metrics, "n_test": len(test), "train_seconds": round(seconds), "device": DEVICE,
              "date": time.strftime("%Y-%m-%d")}
    record.update(extra or {})
    with open(run_path(name, seed, "json"), "w", encoding="utf-8") as f:   # written last: marks the run as complete
        json.dump(record, f, indent=2)
    del model
    torch.cuda.empty_cache()


def without_real_duplicates(real, synthetic_set):
    """Concatenation protocol: real + synthetic, a polymer in both (either key) keeps only its real row."""
    keep = ~KeySet(real["smiles"]).matches(synthetic_set["smiles"])
    return pd.concat([real, synthetic_set[keep][["smiles", "tg_celsius"]]]), int((~keep).sum())


# ---------------------------------------------------------------- the runs

sizes = [str(n) for n in config.SUBSET_SIZES] + ["full"]
wanted = args.sizes or sizes

for seed in args.seeds:
    order = np.random.RandomState(seed).permutation(len(train_real))     # nested subsets: the first n of one shuffle
    real = {str(n): train_real.iloc[order[:n]] for n in config.SUBSET_SIZES}
    real["full"] = train_real

    # A. stage 1 = synthetic only
    for name, data in synthetic.items():
        run(f"synthetic_{name}", seed, "A synthetic only", data[["smiles", "tg_celsius"]], config.MAX_EPOCHS_STAGE1,
            n_real=0, save_weights=True)

    # B. learning curve
    for n in sizes:
        if n not in wanted:
            continue
        is_control_size = n == str(config.CONTROL_SIZE)
        run(f"real_n{n}", seed, "B learning curve", real[n], config.MAX_EPOCHS_REAL, n_real=len(real[n]),
            pseudo_label=synthetic["labeled"]["smiles"] if is_control_size else None)
        for name in synthetic:
            run(f"twostage_{name}_n{n}", seed, "B learning curve", real[n], config.MAX_EPOCHS_REAL, n_real=len(real[n]),
                stage1=f"synthetic_{name}")

    # C. concatenation
    for n in [str(config.CONTROL_SIZE), "full"]:
        if n not in wanted:
            continue
        for name, data in synthetic.items():
            mixed, n_dropped = without_real_duplicates(real[n], data)
            run(f"concat_{name}_n{n}", seed, "C concatenation", mixed, config.MAX_EPOCHS_REAL, n_real=len(real[n]),
                extra={"synthetic_rows_dropped_as_duplicates_of_real": n_dropped})

    # D. controls at n = 500, two-stage protocol
    n = str(config.CONTROL_SIZE)
    if n in wanted:
        controls = {"selftrain": pd.read_csv(os.path.join(PSEUDO_DIR, f"selftrain_seed{seed}.csv"))}
        for name, data in synthetic.items():
            permutation = np.random.RandomState(seed).permutation(len(data))   # fixed per seed
            controls[f"shuffled_{name}"] = pd.DataFrame({"smiles": data["smiles"].values,
                                                         "tg_celsius": data["tg_celsius"].values[permutation]})
        for name, data in controls.items():
            final = f"twostage_{name}_n{n}"
            if os.path.exists(run_path(final, seed, "json")):
                continue
            run(f"synthetic_{name}", seed, "D controls (stage 1)", data, config.MAX_EPOCHS_STAGE1, n_real=0, save_weights=True)
            run(final, seed, "D controls", real[n], config.MAX_EPOCHS_REAL, n_real=len(real[n]), stage1=f"synthetic_{name}")
            os.remove(run_path(f"synthetic_{name}", seed, "weights"))    # used once; about 180 MB each

print("\nAll requested runs are complete.")
