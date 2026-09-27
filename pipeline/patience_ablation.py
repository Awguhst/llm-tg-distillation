"""
Ablation of the early-stopping patience in the low-data regime.

With config.PATIENCE = 3 the real-only runs on 250 and 500 polymers stop after about eight epochs
(roughly 100 to 250 optimizer steps), because the validation MAE of so small a training set
fluctuates from epoch to epoch. Is the gain of the two-stage protocol robust to a more patient
baseline? This script repeats the real-only and the two-stage runs at 250 and 500 real polymers
with the same seeds, subsets, settings and stage-1 weights as train_molformer.py, but stops only
after --patience epochs without improvement (at most --max-epochs); the defaults are the published
ablation, patience 10 and 40 epochs.

The test MAE is recorded after every epoch with the random state restored afterwards
(common/molformer.py, fit with test=...), so that training is not disturbed: the first epochs
reproduce the published run exactly, and the result for any smaller patience can be read from the
same history. The epoch is always chosen on the validation MAE; the test MAE is never used for a choice.

Needs results/stage1_weights/ (written by train_molformer.py, not in the repository). Resumable: a
finished run is skipped when its settings hash matches (common/resume.py). Writes
results/patience_ablation.jsonl, one line per run: seed, n, arm, history (epoch, train_mse, val_mae,
test_mae), seconds, settings_hash. analyze_results.py makes the table.

Usage:  python pipeline/patience_ablation.py [--patience 10] [--max-epochs 40]
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

import config
from common import molformer, resume

ABLATION_PATIENCE = 10
ABLATION_MAX_EPOCHS = 40
SIZES = [250, 500]
ARMS = ["real", "generated", "labeled"]      # real only; two-stage from the stage-1 model of that synthetic set


def finished_runs(path, settings):
    """The (seed, n, arm) keys already in the JSONL file, each checked against the current settings."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for record in map(json.loads, f):
                resume.check_record(record, settings, resume.PUBLISHED_ABLATION, f"{path} (seed {record['seed']}, n {record['n']}, {record['arm']})")
                done.add((record["seed"], record["n"], record["arm"]))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patience", type=int, default=ABLATION_PATIENCE)
    parser.add_argument("--max-epochs", type=int, default=ABLATION_MAX_EPOCHS)
    args = parser.parse_args()
    settings = molformer.training_settings(args.patience, args.max_epochs)

    # the stage-1 weights are not in the repository, so say why rather than fail inside torch.load
    if not os.path.isdir(config.WEIGHTS_DIR) or not os.listdir(config.WEIGHTS_DIR):
        raise SystemExit(f"{config.WEIGHTS_DIR} is empty or missing. This ablation starts from the stage-1 "
                         f"weights; run 'python pipeline/train_molformer.py' to produce them. Its own output, "
                         f"results/patience_ablation.jsonl, is committed.")

    molformer.get_tokenizer()
    train_real = pd.read_csv(config.TRAIN_REAL_CSV)[["smiles", "tg_celsius"]]
    val = pd.read_csv(config.VAL_REAL_CSV)[["smiles", "tg_celsius"]]
    test = pd.read_csv(config.TEST_REAL_CSV)[["smiles", "tg_celsius"]]
    done = finished_runs(config.PATIENCE_ABLATION_JSONL, settings)

    for arm in ARMS:                 # the real-only runs first
        for seed in config.SEEDS:
            subsets = molformer.nested_subsets(train_real, seed)     # the subsets of train_molformer.py
            for n in SIZES:
                if (seed, n, arm) in done:
                    print(f"skipped (exists): seed {seed} | n {n} | {arm}", flush=True)
                    continue
                start_time = time.time()
                train = subsets[str(n)].reset_index(drop=True)
                molformer.set_seed(seed)
                mean, std = float(train["tg_celsius"].mean()), float(train["tg_celsius"].std())
                weights = None if arm == "real" else os.path.join(config.WEIGHTS_DIR, f"synthetic_{arm}_seed{seed}.pt")
                model = molformer.load_model(weights, mean, std)
                history = molformer.fit(model, train, val, mean, std, args.max_epochs, args.patience, test=test, verbose=False)
                # the record format of the published file: train_mse, not train_mse_standardized
                history = [{"epoch": h["epoch"], "train_mse": h["train_mse_standardized"], "val_mae": h["val_mae"],
                            "test_mae": h["test_mae"]} for h in history]
                best = min(history, key=lambda h: h["val_mae"])
                print(f"seed {seed} | n {n} | {arm:<9} | {len(history):>2} epochs, best {best['epoch']:>2} "
                      f"(validation MAE {best['val_mae']:.2f}) | test MAE {best['test_mae']:.2f}", flush=True)
                with open(config.PATIENCE_ABLATION_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"seed": seed, "n": n, "arm": arm, "history": history,
                                        "seconds": round(time.time() - start_time),
                                        "settings_hash": resume.settings_hash(settings)}) + "\n")
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
