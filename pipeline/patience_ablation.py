"""
Ablation of the early-stopping patience in the low-data regime.

With config.PATIENCE = 3 the real-only runs on 250 and 500 polymers stop after about eight epochs
(roughly 100 to 250 optimizer steps), because the validation MAE of so small a training set
fluctuates from epoch to epoch. Is the gain of the two-stage protocol robust to a more patient
baseline? This script repeats the real-only and the two-stage runs at 250 and 500 real polymers
with the same seeds, subsets, settings and stage-1 weights as train_molformer.py, but stops only
after ABLATION_PATIENCE epochs without improvement (at most ABLATION_MAX_EPOCHS).

The test MAE is recorded after every epoch with the random state restored afterwards, so that
training is not disturbed: the first epochs reproduce the published run exactly, and the result
for any smaller patience can be read from the same history. The epoch is always chosen on the
validation MAE; the test MAE is never used for a choice.

Needs results/stage1_weights/ (written by train_molformer.py). Resumable: a finished run is skipped.
Writes results/patience_ablation.jsonl, one line per run; analyze_results.py makes the table.
"""
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

ABLATION_PATIENCE = 10
ABLATION_MAX_EPOCHS = 40
SIZES = [250, 500]
ARMS = ["real", "generated", "labeled"]      # real only; two-stage from the stage-1 model of that synthetic set
OUT = os.path.join(config.RESULTS_DIR, "patience_ablation.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(config.MOLFORMER_NAME, trust_remote_code=True)
train_real = pd.read_csv(config.TRAIN_REAL_CSV)[["smiles", "tg_celsius"]]
val = pd.read_csv(config.VAL_REAL_CSV)[["smiles", "tg_celsius"]]
test = pd.read_csv(config.TEST_REAL_CSV)[["smiles", "tg_celsius"]]


# The next four functions are those of train_molformer.py.

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(smiles_list):
    return tokenizer(list(smiles_list), padding=True, truncation=True,
                     max_length=config.MAX_SMILES_TOKENS, return_tensors="pt")


def predict(model, smiles_list, mean, std):
    model.eval()
    smiles_list = list(smiles_list)
    out = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), config.EVAL_BATCH_SIZE):
            batch = encode(smiles_list[start:start + config.EVAL_BATCH_SIZE]).to(DEVICE)
            out.append(model(**batch).logits.squeeze(-1).float().cpu())
    return torch.cat(out).numpy() * std + mean


def load_model(stage1_weights, new_mean, new_std):
    model = AutoModelForSequenceClassification.from_pretrained(config.MOLFORMER_NAME, num_labels=1, trust_remote_code=True)
    if stage1_weights is not None:
        saved = torch.load(stage1_weights, map_location="cpu")
        model.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            layer = model.classifier.out_proj
            layer.weight.mul_(saved["std"] / new_std)
            layer.bias.copy_((layer.bias * saved["std"] + saved["mean"] - new_mean) / new_std)
    return model.to(DEVICE)


done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        done = {(r["seed"], r["n"], r["arm"]) for r in map(json.loads, f)}

for arm in ARMS:                 # the real-only runs first
    for seed in config.SEEDS:
        order = np.random.RandomState(seed).permutation(len(train_real))     # the subsets of train_molformer.py
        for n in SIZES:
            if (seed, n, arm) in done:
                continue
            start_time = time.time()
            train = train_real.iloc[order[:n]].reset_index(drop=True)
            set_seed(seed)
            mean, std = float(train["tg_celsius"].mean()), float(train["tg_celsius"].std())
            weights = None if arm == "real" else os.path.join(config.WEIGHTS_DIR, f"synthetic_{arm}_seed{seed}.pt")
            model = load_model(weights, mean, std)
            y = torch.tensor(((train["tg_celsius"] - mean) / std).values, dtype=torch.float32)
            smiles = list(train["smiles"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
            loss_function = torch.nn.MSELoss()
            history, best_mae, since_best = [], float("inf"), 0
            for epoch in range(1, ABLATION_MAX_EPOCHS + 1):
                model.train()
                batch_order = np.random.permutation(len(smiles))
                total_loss = 0.0
                for start in range(0, len(batch_order), config.BATCH_SIZE):
                    idx = batch_order[start:start + config.BATCH_SIZE]
                    batch = encode([smiles[i] for i in idx]).to(DEVICE)
                    loss = loss_function(model(**batch).logits.squeeze(-1), y[idx].to(DEVICE))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(idx)
                val_mae = float(np.abs(predict(model, val["smiles"], mean, std) - val["tg_celsius"].values).mean())
                # The test pass draws random attention features; restoring the random state keeps training unchanged.
                cpu_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state() if DEVICE == "cuda" else None
                test_mae = float(np.abs(predict(model, test["smiles"], mean, std) - test["tg_celsius"].values).mean())
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state)
                history.append({"epoch": epoch, "train_mse": total_loss / len(smiles), "val_mae": val_mae, "test_mae": test_mae})
                if val_mae < best_mae:
                    best_mae, since_best = val_mae, 0
                else:
                    since_best += 1
                    if since_best >= ABLATION_PATIENCE:
                        break
            best = min(history, key=lambda h: h["val_mae"])
            print(f"seed {seed} | n {n} | {arm:<9} | {len(history):>2} epochs, best {best['epoch']:>2} "
                  f"(validation MAE {best['val_mae']:.2f}) | test MAE {best['test_mae']:.2f}", flush=True)
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps({"seed": seed, "n": n, "arm": arm, "history": history,
                                    "seconds": round(time.time() - start_time)}) + "\n")
            del model, optimizer
            torch.cuda.empty_cache()
