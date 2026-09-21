"""
Figures (matplotlib only), written to results/figures/.

- learning_curve.png       test MAE against the number of real polymers (log axis): real only, two-stage
                           generated, two-stage labeled (mean and SD over seeds); band = zero-shot teacher
- controls_n500.png        the n = 500 configurations: real only, two-stage with LLM labels, with
                           self-training labels, with the generated set, and the two shuffled-label controls
- predicted_vs_true.png    real only (full), generated only, labeled only (seed 42), zero-shot teacher
- tg_histograms.png        Tg distributions of the real training set and the two synthetic sets
Colour follows the data source everywhere: grey = real, blue = generated, orange = labeled,
green = the teacher itself, violet = self-training. Shuffled-label controls are hatched.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

os.makedirs(config.FIGURES_DIR, exist_ok=True)
results = pd.read_csv(config.RESULTS_CSV)
with open(os.path.join(config.RESULTS_DIR, "zero_shot_metrics.json"), encoding="utf-8") as f:
    zero_shot = json.load(f)
n_full = len(pd.read_csv(config.TRAIN_REAL_CSV))
n_seeds = results["seed"].nunique()

COLORS = {"real": "#4a5568", "generated": "#2a78d6", "labeled": "#eb6834", "teacher": "#1baf7a", "selftrain": "#4a3aa7"}
INK, MUTED = "#1a1a19", "#6b6b66"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True})


def mae_of(run):
    values = results.loc[results["run"] == run, "mae"]
    return values.mean(), values.std()


# ---------------------------------------------------------------- 1. learning curve
sizes = config.SUBSET_SIZES + [n_full]
names = [f"n{n}" for n in config.SUBSET_SIZES] + ["nfull"]
fig, ax = plt.subplots(figsize=(7, 4.4))
lo, hi = zero_shot["mae_ci95"]
ax.axhspan(lo, hi, color=COLORS["teacher"], alpha=0.18, linewidth=0)
ax.axhline(zero_shot["mae"], color=COLORS["teacher"], linewidth=1.5, linestyle="--")
ax.text(sizes[-1], hi + 0.3, f"zero-shot teacher, {zero_shot['mae']:.1f} °C (95 % CI)", color=INK, ha="right", va="bottom", fontsize=9)
for key, label, prefix, marker in [("real", "real only", "real", "o"), ("generated", "two-stage: generated → real", "twostage_generated", "s"),
                                   ("labeled", "two-stage: labeled → real", "twostage_labeled", "^")]:
    stats = [mae_of(f"{prefix}_{n}") for n in names]
    mean, sd = np.array([s[0] for s in stats]), np.array([s[1] for s in stats])
    ax.errorbar(sizes, mean, yerr=sd, color=COLORS[key], marker=marker, markersize=7, linewidth=2, capsize=3,
                markeredgecolor="white", markeredgewidth=1, label=label)
ax.set_xscale("log")
ax.set_xticks(sizes)
ax.set_xticklabels([f"{n:,}" for n in sizes])
ax.minorticks_off()
ax.set_xlabel("Number of real training polymers")
ax.set_ylabel("MAE on the real test set (°C)")
ax.set_title(f"Learning curve (mean ± SD over {n_seeds} seeds)", loc="left", color=INK)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "learning_curve.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. controls at n = 500
n = f"n{config.CONTROL_SIZE}"
bars = [(f"real_{n}", "real only", "real", False), (f"twostage_labeled_{n}", "LLM-labeled\n→ real", "labeled", False),
        (f"twostage_selftrain_{n}", "self-trained\nlabels → real", "selftrain", False),
        (f"twostage_shuffled_labeled_{n}", "labeled,\nTg shuffled → real", "labeled", True),
        (f"twostage_generated_{n}", "generated\n→ real", "generated", False),
        (f"twostage_shuffled_generated_{n}", "generated,\nTg shuffled → real", "generated", True)]
fig, ax = plt.subplots(figsize=(8, 4.4))
for x, (run, label, key, shuffled) in enumerate(bars):
    mean, sd = mae_of(run)
    ax.bar(x, mean, width=0.62, yerr=sd, capsize=3, color=COLORS[key], alpha=0.45 if shuffled else 1.0,
           hatch="///" if shuffled else None, edgecolor="white", linewidth=1.5, error_kw={"ecolor": INK, "linewidth": 1})
    ax.text(x, mean + sd + 0.6, f"{mean:.1f}", ha="center", va="bottom", color=INK, fontsize=9)
ax.axhline(zero_shot["mae"], color=COLORS["teacher"], linewidth=1.5, linestyle="--")
ax.text(-0.45, 84, f"dashed line: zero-shot teacher, {zero_shot['mae']:.1f} °C", color=INK, ha="left", va="top", fontsize=9)
ax.set_ylim(0, 86)
ax.set_xticks(range(len(bars)))
ax.set_xticklabels([b[1] for b in bars], color=INK, fontsize=9)
ax.set_ylabel("MAE on the real test set (°C)")
ax.set_title(f"Controls with {config.CONTROL_SIZE} real polymers, two-stage protocol (mean ± SD over {n_seeds} seeds)", loc="left", color=INK)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "controls_n500.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. predicted against measured Tg (seed 42)
panels = [("real_nfull", "(a) real only, full training set", "real"), ("synthetic_generated", "(b) generated only", "generated"),
          ("synthetic_labeled", "(c) labeled only", "labeled"), (None, f"(d) zero-shot {config.LLM_MODEL}", "teacher")]
fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
limits = [config.TG_MIN_CELSIUS, config.TG_MAX_CELSIUS]
for ax, (run, title, key) in zip(axes.flat, panels):
    path = config.ZERO_SHOT_CSV if run is None else os.path.join(config.PREDICTIONS_DIR, f"{run}_seed{config.SEED}.csv")
    df = pd.read_csv(path)
    mae = (df["tg_true"] - df["tg_pred"]).abs().mean()
    ax.scatter(df["tg_true"], df["tg_pred"], s=7, alpha=0.4, color=COLORS[key], linewidths=0)
    ax.plot(limits, limits, color=INK, linewidth=0.8)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal")
    ax.set_xlabel("Measured Tg (°C)")
    ax.set_ylabel("Predicted Tg (°C)")
    ax.set_title(f"{title}\nMAE = {mae:.1f} °C, n = {len(df):,}", fontsize=10, loc="left", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "predicted_vs_true.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. Tg distributions (small multiples, shared axis)
datasets = [("real training set", pd.read_csv(config.TRAIN_REAL_CSV), "real"), ("generated set", pd.read_csv(config.GENERATED_CLEAN_CSV), "generated"),
            ("labeled set", pd.read_csv(config.LABELED_CLEAN_CSV), "labeled")]
bins = np.arange(config.TG_MIN_CELSIUS, config.TG_MAX_CELSIUS + 1, 20)
fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True, sharey=True)
for ax, (name, df, key) in zip(axes, datasets):
    ax.hist(df["tg_celsius"], bins=bins, density=True, color=COLORS[key], edgecolor="white", linewidth=0.4)
    ax.axvline(df["tg_celsius"].mean(), color=INK, linewidth=1)
    ax.set_title(f"{name}: n = {len(df):,}, mean {df['tg_celsius'].mean():.0f} °C, SD {df['tg_celsius'].std():.0f} °C",
                 loc="left", fontsize=10, color=INK)
    ax.set_ylabel("Density (per °C)")
axes[-1].set_xlabel("Tg (°C); vertical line = mean")
fig.tight_layout()
fig.savefig(os.path.join(config.FIGURES_DIR, "tg_histograms.png"), dpi=200)
plt.close(fig)

print(f"Saved 4 figures to {config.FIGURES_DIR}/")
