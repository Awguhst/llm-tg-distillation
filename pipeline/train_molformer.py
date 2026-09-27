"""
Train and evaluate MoLFormer.

Every run fine-tunes ibm/MoLFormer-XL-both-10pct with the fixed settings of common/molformer.py and
early stopping on the validation MAE (patience config.PATIENCE). The test set is predicted once, with
the best weights, after training has ended; no choice of any kind looks at it.

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

Outputs, one pair per run, under results/:
  predictions/<run>_seed<seed>.csv   smiles, tg_true, tg_pred for every test polymer
  runs/<run>_seed<seed>.json         settings, best epoch, validation history, test metrics, seconds
A run whose JSON exists is skipped if it was made with the same settings (common/resume.py); with
different settings the script stops.

Usage:  python pipeline/train_molformer.py [--seeds 42 43] [--sizes 500] [--patience 3] [--max-epochs 30]
--patience and --max-epochs (the cap for every run containing real data; stage 1 keeps
config.MAX_EPOCHS_STAGE1) default to the published settings.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import config
from common import molformer, resume
from common.cleaning import KeySet


def run_path(name, seed, kind):
    folder, ext = {"json": (config.RUNS_DIR, "json"), "csv": (config.PREDICTIONS_DIR, "csv"),
                   "weights": (config.WEIGHTS_DIR, "pt")}[kind]
    return os.path.join(folder, f"{name}_seed{seed}.{ext}")


def already_done(name, seed, patience, max_epochs):
    """True if this run's JSON exists and was made with the current settings; stops on a mismatch."""
    path = run_path(name, seed, "json")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    # A record without a hash is from the published run: stage 1 (no real data) had max_epochs 10, everything else 30.
    published = dict(resume.PUBLISHED_MOLFORMER, max_epochs=resume.PUBLISHED_MAX_EPOCHS["stage1" if record["n_real"] == 0 else "real"])
    return resume.check_record(record, molformer.training_settings(patience, max_epochs), published, path)


def run(name, seed, group, train, val, test, max_epochs, patience, n_real, stage1=None, save_weights=False,
        pseudo_label=None, extra=None):
    """
    One training run. stage1 = name of the stage-1 run whose weights to start from (None = pretrained
    checkpoint). pseudo_label = SMILES to label with the finished model (self-training control).
    """
    if already_done(name, seed, patience, max_epochs):
        print(f"skipped (exists): {name} seed {seed}", flush=True)
        return
    # A finished stage-1 run is skipped on its JSON alone, but its weights are gitignored (and the
    # control stage-1 weights are deleted after use), so they can be absent while the JSON is there.
    # Say so, instead of failing later inside torch.load.
    if stage1 is not None and not os.path.exists(run_path(stage1, seed, "weights")):
        raise SystemExit(f"{name} seed {seed} starts from {stage1}, but {run_path(stage1, seed, 'weights')} "
                         f"is missing (stage-1 weights are not in the repository). Delete "
                         f"{run_path(stage1, seed, 'json')} so stage 1 is retrained, then run again.")
    print(f"\n=== seed {seed} | {name} | {len(train)} training rows | start: {stage1 or 'pretrained checkpoint'} ===", flush=True)
    start_time = time.time()
    molformer.set_seed(seed)
    train = train.reset_index(drop=True)
    mean, std = float(train["tg_celsius"].mean()), float(train["tg_celsius"].std())
    model = molformer.load_model(run_path(stage1, seed, "weights") if stage1 else None, mean, std)
    history = molformer.fit(model, train, val, mean, std, max_epochs, patience)
    best = min(history, key=lambda h: h["val_mae"])

    y_pred = molformer.predict(model, test["smiles"], mean, std)
    metrics = molformer.compute_metrics(test["tg_celsius"].values, y_pred)
    pd.DataFrame({"smiles": test["smiles"], "tg_true": test["tg_celsius"], "tg_pred": y_pred}).to_csv(
        run_path(name, seed, "csv"), index=False)
    if save_weights:
        torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, run_path(name, seed, "weights"))
    if pseudo_label is not None:
        pd.DataFrame({"smiles": pseudo_label, "tg_celsius": molformer.predict(model, pseudo_label, mean, std)}).to_csv(
            os.path.join(config.PSEUDO_LABELS_DIR, f"selftrain_seed{seed}.csv"), index=False)
    seconds = time.time() - start_time
    print(f"    best epoch {best['epoch']} (validation MAE {best['val_mae']:.2f}) | test MAE = {metrics['mae']:.2f}  "
          f"R2 = {metrics['r2']:.3f}  ({seconds:.0f} s)", flush=True)

    record = {"run": name, "group": group, "seed": seed, "model": "molformer", "n_real": n_real, "n_train_rows": len(train),
              "start_weights": stage1 or config.MOLFORMER_NAME, "target_mean": mean, "target_std": std,
              "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE, "max_tokens": config.MAX_SMILES_TOKENS,
              "max_epochs": max_epochs, "patience": patience, "epochs_run": len(history),
              "best_epoch": best["epoch"], "best_val_mae": best["val_mae"], "history": history,
              "optimizer_steps_to_best": best["epoch"] * int(np.ceil(len(train) / config.BATCH_SIZE)),
              "test": metrics, "n_test": len(test), "train_seconds": round(seconds), "device": molformer.DEVICE,
              "date": time.strftime("%Y-%m-%d"),
              "settings_hash": resume.settings_hash(molformer.training_settings(patience, max_epochs))}
    record.update(extra or {})
    with open(run_path(name, seed, "json"), "w", encoding="utf-8") as f:   # written last: marks the run as complete
        json.dump(record, f, indent=2)
    del model
    torch.cuda.empty_cache()


def without_real_duplicates(real, synthetic_set):
    """Concatenation protocol: real + synthetic, a polymer in both (either key) keeps only its real row."""
    keep = ~KeySet(real["smiles"]).matches(synthetic_set["smiles"])
    return pd.concat([real, synthetic_set[keep][["smiles", "tg_celsius"]]]), int((~keep).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--sizes", nargs="+", default=None, help="real subset sizes to run, e.g. 500 full (default: all)")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="epochs without improvement before stopping")
    parser.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS_REAL,
                        help="epoch cap for runs containing real data (stage 1 always uses config.MAX_EPOCHS_STAGE1)")
    args = parser.parse_args()

    for folder in [config.PREDICTIONS_DIR, config.RUNS_DIR, config.WEIGHTS_DIR, config.PSEUDO_LABELS_DIR]:
        os.makedirs(folder, exist_ok=True)
    print(f"Device: {torch.cuda.get_device_name(0) if molformer.DEVICE == 'cuda' else 'CPU (slow)'}")
    molformer.get_tokenizer()
    train_real = pd.read_csv(config.TRAIN_REAL_CSV)[["smiles", "tg_celsius"]]
    val = pd.read_csv(config.VAL_REAL_CSV)[["smiles", "tg_celsius"]]
    test = pd.read_csv(config.TEST_REAL_CSV)[["smiles", "tg_celsius"]]
    synthetic = {"generated": pd.read_csv(config.GENERATED_CLEAN_CSV), "labeled": pd.read_csv(config.LABELED_CLEAN_CSV)}

    sizes = [str(n) for n in config.SUBSET_SIZES] + ["full"]
    wanted = args.sizes or sizes
    unknown = [n for n in wanted if n not in sizes]
    if unknown:      # otherwise the run would quietly do nothing and report success
        raise SystemExit(f"--sizes {' '.join(unknown)}: unknown size(s). Choose from: {', '.join(sizes)}")
    patience, max_epochs = args.patience, args.max_epochs
    stage1_epochs = config.MAX_EPOCHS_STAGE1

    for seed in args.seeds:
        real = molformer.nested_subsets(train_real, seed)

        # A. stage 1 = synthetic only
        for name, data in synthetic.items():
            run(f"synthetic_{name}", seed, "A synthetic only", data[["smiles", "tg_celsius"]], val, test, stage1_epochs, patience,
                n_real=0, save_weights=True)

        # B. learning curve
        for n in sizes:
            if n not in wanted:
                continue
            is_control_size = n == str(config.CONTROL_SIZE)
            run(f"real_n{n}", seed, "B learning curve", real[n], val, test, max_epochs, patience, n_real=len(real[n]),
                pseudo_label=synthetic["labeled"]["smiles"] if is_control_size else None)
            for name in synthetic:
                run(f"twostage_{name}_n{n}", seed, "B learning curve", real[n], val, test, max_epochs, patience,
                    n_real=len(real[n]), stage1=f"synthetic_{name}")

        # C. concatenation
        for n in [str(config.CONTROL_SIZE), "full"]:
            if n not in wanted:
                continue
            for name, data in synthetic.items():
                mixed, n_dropped = without_real_duplicates(real[n], data)
                run(f"concat_{name}_n{n}", seed, "C concatenation", mixed, val, test, max_epochs, patience, n_real=len(real[n]),
                    extra={"synthetic_rows_dropped_as_duplicates_of_real": n_dropped})

        # D. controls at n = 500, two-stage protocol
        n = str(config.CONTROL_SIZE)
        if n in wanted:
            controls = {"selftrain": pd.read_csv(os.path.join(config.PSEUDO_LABELS_DIR, f"selftrain_seed{seed}.csv"))}
            for name, data in synthetic.items():
                permutation = np.random.RandomState(seed).permutation(len(data))   # fixed per seed
                controls[f"shuffled_{name}"] = pd.DataFrame({"smiles": data["smiles"].values,
                                                             "tg_celsius": data["tg_celsius"].values[permutation]})
            for name, data in controls.items():
                final = f"twostage_{name}_n{n}"
                if already_done(final, seed, patience, max_epochs):
                    print(f"skipped (exists): {final} seed {seed}", flush=True)
                    continue
                run(f"synthetic_{name}", seed, "D controls (stage 1)", data, val, test, stage1_epochs, patience, n_real=0, save_weights=True)
                run(final, seed, "D controls", real[n], val, test, max_epochs, patience, n_real=len(real[n]), stage1=f"synthetic_{name}")
                os.remove(run_path(f"synthetic_{name}", seed, "weights"))    # used once; about 180 MB each

    print("\nAll requested runs are complete.")


if __name__ == "__main__":
    main()
