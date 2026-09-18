"""
Figures (step 6, matplotlib only).

- Bar chart of MAE for all experiments: mean over the training seeds with the standard
  deviation as error bars (MolFormer and random forest side by side, plus zero-shot).
- Predicted vs. true Tg scatter plots for experiments 1, 2, 3 (MolFormer, seed 42) and the zero-shot predictions.
- Histograms of the Tg distributions of the real, generated, and labeled datasets.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)
summary = pd.read_csv(os.path.join(config.RESULTS_DIR, "results_summary.csv"))
results = pd.read_csv(config.RESULTS_CSV)

COLORS = {"molformer": "#2b6cb0", "random_forest": "#a0aec0", "zero_shot": "#dd6b20"}

# ---------------------------------------------------------------- 1. MAE bar chart (mean and SD over seeds)
molformer = summary[summary["model"] == "molformer"].sort_values("experiment")
forest = summary[summary["model"] == "random_forest"].sort_values("experiment")
labels = [f"{r.experiment}: {r.training_data}" for r in molformer.itertuples()]
zero_shot_mae = results[results["experiment"] == 0]["mae"].values[0]
n_seeds = int(molformer["n_seeds"].iloc[0])

x = np.arange(len(labels))
width = 0.38
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width / 2, molformer["mae_mean"], width, yerr=molformer["mae_std"], capsize=3,
       label=f"MolFormer (mean of {n_seeds} seeds)", color=COLORS["molformer"])
ax.bar(x + width / 2, forest["mae_mean"], width, yerr=forest["mae_std"], capsize=3,
       label=f"Random forest (mean of {n_seeds} seeds)", color=COLORS["random_forest"])
ax.axhline(zero_shot_mae, color=COLORS["zero_shot"], linestyle="--", label=f"Zero-shot {config.LLM_MODEL}")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("MAE on real test set (°C)")
ax.set_title("Tg prediction error by training data (error bars: SD over seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "mae_bar_chart.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. predicted vs true scatter plots (seed 42 runs)
panels = [
    (f"exp1_molformer_seed{config.SEED}.csv", "Exp 1: real (MolFormer)"),
    (f"exp2_molformer_seed{config.SEED}.csv", "Exp 2: generated (MolFormer)"),
    (f"exp3_molformer_seed{config.SEED}.csv", "Exp 3: labeled (MolFormer)"),
    (None, f"Zero-shot {config.LLM_MODEL}"),
]
fig, axes = plt.subplots(2, 2, figsize=(9, 9))
for ax, (filename, title) in zip(axes.flat, panels):
    if filename is None:
        df = pd.read_csv(config.ZERO_SHOT_CSV).dropna(subset=["tg_pred"])
        color = COLORS["zero_shot"]
    else:
        df = pd.read_csv(os.path.join(config.PREDICTIONS_DIR, filename))
        color = COLORS["molformer"]
    mae = (df["tg_true"] - df["tg_pred"]).abs().mean()
    ax.scatter(df["tg_true"], df["tg_pred"], s=8, alpha=0.4, color=color)
    limits = [config.TG_MIN_CELSIUS, config.TG_MAX_CELSIUS]
    ax.plot(limits, limits, color="black", linewidth=0.8)   # the perfect-prediction line
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("True Tg (°C)")
    ax.set_ylabel("Predicted Tg (°C)")
    ax.set_title(f"{title}\nMAE = {mae:.1f} °C, n = {len(df)}", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "predicted_vs_true.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. Tg histograms
datasets = [
    ("Real (train)", pd.read_csv(config.TRAIN_REAL_CSV), "#4a5568"),
    ("Generated", pd.read_csv(config.GENERATED_CLEAN_CSV), COLORS["molformer"]),
    ("Labeled PI1M", pd.read_csv(config.LABELED_CLEAN_CSV), COLORS["zero_shot"]),
]
bins = np.arange(config.TG_MIN_CELSIUS, config.TG_MAX_CELSIUS + 1, 20)
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, df, color in datasets:
    # density=True so datasets of different sizes are comparable
    ax.hist(df["tg_celsius"], bins=bins, density=True, alpha=0.45, color=color, label=f"{name} (n={len(df)})")
ax.set_xlabel("Tg (°C)")
ax.set_ylabel("Density")
ax.set_title("Tg distributions of the three training datasets")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "tg_histograms.png"), dpi=200)
plt.close(fig)

print(f"Saved 3 figures to {config.FIGURES_DIR}/")
