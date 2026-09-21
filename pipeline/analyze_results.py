"""
Analysis of all runs. Reads results/runs/*.json and results/predictions/*.csv
and writes every table to results/tables/ as CSV, plus tables/all_tables.md for reading.

1. results.csv (one row per run), results_summary.csv (mean and SD over seeds), per-seed MAE, best epochs
2. seed-matched comparisons: difference in MAE (mean, SD, range, same-sign count), paired t-test, Wilcoxon
3. each stage-1 (synthetic-only) model against the zero-shot teacher, paired bootstrap over test polymers
4. noise floor: experimental SD of the test polymers with 2+ measurements, and model MAE on them
5. MAE by reliability flag, by band of the measured Tg, by band of the similarity to the training set
6. patience ablation: the low-data runs with early stopping that waits 3, 5 or 10 epochs
7. the total training time and software versions
Nothing here feeds back into training; the choices were made on the validation set.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import config
from common import run_info

TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)
markdown = []


def save(table, name, title):
    """Write a table as CSV and keep a markdown copy for all_tables.md."""
    table.to_csv(os.path.join(TABLES_DIR, f"{name}.csv"), index=False)
    cells = table.astype(object).where(table.notna(), "").astype(str)
    lines = ["| " + " | ".join(cells.columns.astype(str)) + " |", "|" + "---|" * len(cells.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.values]
    markdown.append(f"### {title}\n\n" + "\n".join(lines) + "\n")
    print(f"\n--- {title}\n{table.to_string(index=False)}")


# ---------------------------------------------------------------- 1. all runs
rows = []
for path in sorted(glob.glob(os.path.join(config.RUNS_DIR, "*.json"))):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    name = ("rf_" if r["model"] == "random_forest" else "") + r["run"]
    rows.append({"run": name, "group": r["group"], "model": r["model"], "seed": r["seed"], "n_real": r["n_real"],
                 "n_train_rows": r["n_train_rows"], "best_epoch": r.get("best_epoch"), "epochs_run": r.get("epochs_run"),
                 "best_val_mae": r.get("best_val_mae"), **r["test"], "train_seconds": r["train_seconds"]})
results = pd.DataFrame(rows).sort_values(["model", "group", "run", "seed"])
results.to_csv(os.path.join(config.RESULTS_DIR, "results.csv"), index=False)
seeds = sorted(results["seed"].unique())
complete = results.groupby("run")["seed"].nunique()
print(f"{len(results)} runs, seeds {seeds}; runs without all seeds: {list(complete[complete < len(seeds)].index)}")

metrics = ["mae", "rmse", "r2", "spearman"]
summary = results.groupby(["model", "group", "run", "n_real"], sort=False)[metrics + ["best_val_mae", "best_epoch"]].agg(["mean", "std"])
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
summary["n_seeds"] = summary["run"].map(complete)
summary.to_csv(os.path.join(config.RESULTS_DIR, "results_summary.csv"), index=False)

shown = summary[["run", "n_real", "n_seeds"]].copy()
for m in metrics:
    digits = 1 if m in ("mae", "rmse") else 3
    shown[m] = [f"{a:.{digits}f} ± {b:.{digits}f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
shown["val MAE"] = [f"{a:.1f}" if pd.notna(a) else "" for a in summary["best_val_mae_mean"]]
shown["best epoch"] = [f"{a:.1f}" if pd.notna(a) else "" for a in summary["best_epoch_mean"]]
save(shown, "summary", "Test metrics, mean ± SD over seeds (MAE and RMSE in °C)")

mae = results.pivot(index="run", columns="seed", values="mae")
save(mae.round(2).reset_index(), "per_seed_mae", "Test MAE (°C) per seed")
epochs = results[results["model"] == "molformer"].pivot(index="run", columns="seed", values="best_epoch")
save(epochs.reset_index(), "best_epochs", "Best epoch (early stopping on the validation MAE) per seed")

# ---------------------------------------------------------------- 2. seed-matched comparisons
n500 = f"n{config.CONTROL_SIZE}"
pairs = []
for n in [f"n{k}" for k in config.SUBSET_SIZES] + ["nfull"]:
    for s in ["generated", "labeled"]:
        pairs.append((f"two-stage {s} vs real only, {n}", f"twostage_{s}_{n}", f"real_{n}"))
for n in [n500, "nfull"]:
    for s in ["generated", "labeled"]:
        pairs.append((f"two-stage vs concatenation, {s}, {n}", f"twostage_{s}_{n}", f"concat_{s}_{n}"))
        pairs.append((f"concatenation {s} vs real only, {n}", f"concat_{s}_{n}", f"real_{n}"))
pairs += [(f"self-training vs LLM-labeled (two-stage, {n500})", f"twostage_selftrain_{n500}", f"twostage_labeled_{n500}"),
          (f"self-training vs real only, {n500}", f"twostage_selftrain_{n500}", f"real_{n500}"),
          (f"shuffled vs unshuffled generated (two-stage, {n500})", f"twostage_shuffled_generated_{n500}", f"twostage_generated_{n500}"),
          (f"shuffled vs unshuffled labeled (two-stage, {n500})", f"twostage_shuffled_labeled_{n500}", f"twostage_labeled_{n500}"),
          (f"shuffled generated vs real only, {n500}", f"twostage_shuffled_generated_{n500}", f"real_{n500}"),
          (f"shuffled labeled vs real only, {n500}", f"twostage_shuffled_labeled_{n500}", f"real_{n500}"),
          ("labeled only vs generated only (stage 1)", "synthetic_labeled", "synthetic_generated")]
for n in [n500, "nfull"]:
    for s in ["generated", "labeled"]:
        pairs.append((f"random forest: real + {s} vs real only, {n}", f"rf_concat_{s}_{n}", f"rf_real_{n}"))

rows = []
for label, a, b in pairs:
    if a not in mae.index or b not in mae.index:
        continue
    d = (mae.loc[a] - mae.loc[b]).dropna()
    if len(d) < 2:
        continue
    sign = np.sign(d.mean())
    different = d.abs().max() > 0
    rows.append({"comparison (first minus second)": label, "seeds": len(d), "dMAE mean": d.mean(), "SD": d.std(),
                 "min": d.min(), "max": d.max(), "same sign": f"{int((np.sign(d) == sign).sum())}/{len(d)}",
                 "paired t p": stats.ttest_rel(mae.loc[a, d.index], mae.loc[b, d.index]).pvalue if different else np.nan,
                 "Wilcoxon p": stats.wilcoxon(d).pvalue if different else np.nan})
comparisons = pd.DataFrame(rows).round({"dMAE mean": 2, "SD": 2, "min": 2, "max": 2, "paired t p": 4, "Wilcoxon p": 4})
save(comparisons, "seed_matched_comparisons",
     "Seed-matched differences in test MAE (°C); negative = the first configuration is better. "
     "With 5 pairs the Wilcoxon p-value cannot be below 0.0625")

# ---------------------------------------------------------------- per-polymer errors of the main models
test = pd.read_csv(config.TEST_REAL_CSV)
zero_shot = pd.read_csv(config.ZERO_SHOT_CSV)
assert (zero_shot["smiles"].values == test["smiles"].values).all()
zs_abs = (zero_shot["tg_pred"] - zero_shot["tg_true"]).abs().values


def predictions(run, seed):
    p = pd.read_csv(os.path.join(config.PREDICTIONS_DIR, f"{run}_seed{seed}.csv"))
    assert (p["smiles"].values == test["smiles"].values).all()
    return p["tg_pred"].values


def abs_error(run):
    """Absolute error per test polymer, averaged over the seeds of the run."""
    done = mae.loc[run].dropna().index
    return np.mean([np.abs(predictions(run, s) - test["tg_celsius"].values) for s in done], axis=0)


main_models = {"zero-shot teacher": zs_abs}
for label, run in [("real only, full", "real_nfull"), (f"real only, {n500}", f"real_{n500}"),
                   ("generated only", "synthetic_generated"), ("labeled only", "synthetic_labeled"),
                   (f"two-stage generated, {n500}", f"twostage_generated_{n500}"), (f"two-stage labeled, {n500}", f"twostage_labeled_{n500}"),
                   ("two-stage generated, full", "twostage_generated_nfull"), ("two-stage labeled, full", "twostage_labeled_nfull"),
                   ("random forest, real full", "rf_real_nfull")]:
    if run in mae.index:
        main_models[label] = abs_error(run)

# ---------------------------------------------------------------- 3. stage-1 students against the teacher
rng = np.random.RandomState(config.SEED)
resamples = rng.randint(0, len(test), (config.BOOTSTRAP_RESAMPLES, len(test)))
rows = []
for run in ["synthetic_generated", "synthetic_labeled"]:
    if run not in mae.index:
        continue
    for seed in mae.loc[run].dropna().index:
        d = np.abs(predictions(run, seed) - test["tg_celsius"].values) - zs_abs     # paired, per polymer
        boot = d[resamples].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"student": run, "seed": seed, "student MAE - teacher MAE": d.mean(), "CI 2.5 %": lo, "CI 97.5 %": hi,
                     "CI excludes 0": bool(lo > 0 or hi < 0)})
if rows:
    save(pd.DataFrame(rows).round(2), "student_vs_teacher",
         "Stage-1 (synthetic-only) students against the zero-shot teacher on the same test polymers (paired bootstrap, 1,000 resamples)")

# ---------------------------------------------------------------- 4. noise floor
multi = (test["n_points"] >= 2).values
sd = test.loc[multi, "tg_std"]
floor = {"test polymers with 2+ measurements": int(multi.sum()), "median experimental SD": sd.median(), "mean experimental SD": sd.mean(),
         "expected |difference| of two measurements, from the median SD (1.128 x SD)": 2 / np.sqrt(np.pi) * sd.median(),
         "expected |difference| of two measurements, from the mean SD": 2 / np.sqrt(np.pi) * sd.mean()}
rows = [{"quantity": k, "value": v} for k, v in floor.items()]
rows += [{"quantity": f"MAE on these polymers: {label}", "value": e[multi].mean()} for label, e in main_models.items()]
rows += [{"quantity": f"MAE on the other {int((~multi).sum())} test polymers: {label}", "value": e[~multi].mean()} for label, e in main_models.items()]
save(pd.DataFrame(rows).round(2), "noise_floor", "Experimental noise floor and model error on the test polymers with repeated measurements (°C)")

# ---------------------------------------------------------------- 5. stratified errors


def stratified(groups, name, title):
    table = pd.DataFrame({label: pd.Series(e).groupby(groups, observed=True).mean() for label, e in main_models.items()})
    table.insert(0, "n", pd.Series(groups).value_counts())
    table.loc["all"] = [len(test)] + [e.mean() for e in main_models.values()]
    table.index = table.index.astype(str)
    save(table.round(1).rename_axis(name).reset_index(), f"mae_by_{name}", title)


stratified(test["reliability"].values, "reliability", "Test MAE (°C) by reliability flag of the dataset (black = one data point)")
stratified(pd.cut(test["tg_celsius"], config.TG_BINS, right=False).values, "tg_bin", "Test MAE (°C) by band of the measured Tg")
similarity = pd.read_csv(os.path.join(config.RESULTS_DIR, "test_similarity.csv"))["max_tanimoto_to_train"]
stratified(pd.cut(similarity, config.SIMILARITY_BINS, right=False).values, "similarity_bin",
           "Test MAE (°C) by maximum Tanimoto similarity of the test polymer to the real training set")
stratified(np.where(test["tg_std"].fillna(0) > 30, "SD > 30 C", "other"), "high_sd",
           "Sensitivity: test polymers whose repeated measurements disagree by SD > 30 °C, against all others")

# ---------------------------------------------------------------- 6. patience ablation (patience_ablation.py)
ABLATION = os.path.join(config.RESULTS_DIR, "patience_ablation.jsonl")


def stopped_at(history, patience):
    """The entry with the best validation MAE when training stops after `patience` epochs without improvement."""
    best, since_best = history[0], 0
    for h in history[1:]:
        if h["val_mae"] < best["val_mae"]:
            best, since_best = h, 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    return best


if os.path.exists(ABLATION):
    with open(ABLATION, encoding="utf-8") as f:
        runs = [json.loads(line) for line in f]
    rows = [{"n_real": r["n"], "arm": r["arm"], "seed": r["seed"], "patience": p, "best_epoch": stopped_at(r["history"], p)["epoch"],
             "mae": stopped_at(r["history"], p)["test_mae"]} for r in runs for p in (config.PATIENCE, 5, 10)]   # the runs used 10
    ablation = pd.DataFrame(rows)
    rows = []
    for (n, p), part in ablation.groupby(["n_real", "patience"]):
        wide = part.pivot(index="seed", columns="arm", values="mae").dropna()
        row = {"n_real": n, "patience": p, "seeds": len(wide),
               "best epoch, real only": part.loc[part["arm"] == "real", "best_epoch"].mean()}
        for arm in ["real", "generated", "labeled"]:
            row[f"MAE {arm}"], row[f"SD {arm}"] = wide[arm].mean(), wide[arm].std()
        for arm in ["generated", "labeled"]:
            d = wide[arm] - wide["real"]
            row[f"dMAE {arm}"], row[f"dSD {arm}"] = d.mean(), d.std()
            row[f"same sign {arm}"] = f"{int((np.sign(d) == np.sign(d.mean())).sum())}/{len(d)}"
            row[f"paired t p {arm}"] = stats.ttest_rel(wide[arm], wide["real"]).pvalue
        rows.append(row)
    save(pd.DataFrame(rows).round(3), "patience_ablation",
         "Patience ablation: test MAE (°C) of the real-only and two-stage runs when early stopping waits 3, 5 or 10 epochs; "
         "dMAE = two-stage minus real only, seed-matched")

# ---------------------------------------------------------------- 7. training time and software
gpu_hours = results.loc[results["model"] == "molformer", "train_seconds"].sum() / 3600
print(f"\nTotal MoLFormer training time: {gpu_hours:.1f} GPU-hours")
with open(os.path.join(TABLES_DIR, "all_tables.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(markdown))
import platform
import rdkit, sklearn, torch, transformers
run_info.update({"software": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                              "rdkit": rdkit.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"}})
run_info.update({"training": {"model": config.MOLFORMER_NAME, "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE,
                              "max_tokens": config.MAX_SMILES_TOKENS, "early_stopping": "validation MAE, best weights kept",
                              "patience": config.PATIENCE, "max_epochs_real": config.MAX_EPOCHS_REAL,
                              "max_epochs_stage1": config.MAX_EPOCHS_STAGE1, "seeds": config.SEEDS,
                              "subset_sizes": config.SUBSET_SIZES, "runs_completed": int(len(results)),
                              "gpu_hours_molformer": round(float(gpu_hours), 2),
                              "cpu_hours_random_forest": round(float(results.loc[results["model"] == "random_forest", "train_seconds"].sum() / 3600), 2)}})
