"""
The figures of the paper and the Supplementary Information, drawn from results/ and data/ only.
Writes PDF (vector) and PNG copies to results/figures/:
  fig_tg_distributions        Figure 2   Tg distributions of the real training set and the two synthetic sets
  fig_learning_curve          Figure 3   test MAE against the number of real polymers, with the patience-10 runs
  fig_controls                Figure 4   the controls at 500 real polymers
  fig_predicted_vs_true       Figure 5   predicted against measured Tg (seed 42) and the zero-shot teacher
  fig_si_repetition_rounding  Figure S1  saturation of the generation route; rounding of the labels
  fig_si_similarity           Figure S2  test MAE against the similarity to the training set
  fig_si_zero_shot_error      Figure S3  signed zero-shot error against the measured Tg
(Figure 1, the pipeline overview, is drawn in TikZ inside paper/sections/methods.tex.)
Run with  python report/make_figures.py  (from any directory) or through pipeline/run.py.

Terminology is kept identical to the tables and the text: "real only", "generated set",
"labeled set", "two-stage", "zero-shot teacher".

COLOUR. One hue per data source in every figure, never reused for anything else:
  real / baseline  #1A2233  a near-black ink, because the real-only run is the reference the other
                            configurations are judged against, not a fifth competing series;
  generated set    #2F6BE8  blue
  labeled set      #E02D4E  crimson
  self-training    #B5179E  magenta
  zero-shot teacher #00876C green; wherever it appears as a reference LEVEL (Figures 3, 4 and S2) it
                            is dashed, because it is a level and not a trained model. In Figure 5d and
                            Figure S3 its own predictions are the data, so they are a scatter.
The four chromatic slots were checked with the all-pairs colour-vision validator against the panel
colour used here (#F7F9FC): worst pair 8.6 under deuteranopia and 16.6 for normal vision, both above
the floors. The worst tritanopia separation is 5.0 (green against magenta), which is why identity is
never left to hue alone: every series also carries a marker shape, a dash pattern, a hatch, a direct
label or its own panel. Do not substitute a "nicer" hue without re-running that check -- the obvious
violet for self-training collides with the blue of the generated set (deuteranopia 0.2, i.e. the same
colour), which is why self-training is magenta and not violet.

No figure carries its own title: the LaTeX caption names it, and a title inside the image would
print twice. Panel letters (a), (b), ... stay, because the captions refer to them.

Every figure is authored at the width it is PRINTED at: its figsize is f * TEXT_WIDTH (TEXT_WIDTH is
the \\textwidth of the manuscript) and it is included with width=f\\textwidth for the same f, so LaTeX
never rescales it and a font size means the same thing in every figure. If you change a figsize here,
change the matching \\includegraphics width in the .tex, and the other way round.
"""
import json
import os
import sys

# config.py and common/ live in pipeline/, one folder up and over; anchored to this file, not to the working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import config  # noqa: E402
from common.cleaning import canonical_smiles, polymer_id  # noqa: E402

OUT = config.FIGURES_DIR

TEXT_WIDTH = 6.38        # inches; A4 with 2.4 cm margins, as set in main.tex
C = {"real": "#1A2233", "generated": "#2F6BE8", "labeled": "#E02D4E", "selftrain": "#B5179E", "teacher": "#00876C"}
# Panel tints for the distribution figure: the same hues at very low saturation.
TINT = {"real": "#EEF1FA", "generated": "#EBF2FE", "labeled": "#FDEEF1"}
INK, MUTED, GRID, PANEL = "#1A2233", "#6B7A90", "#DFE7F2", "#F7F9FC"
TG = r"$T_\mathrm{g}$"          # same symbol as in the text

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    # a white page with a very light panel, so the plotting area reads as a surface of its own
    "axes.facecolor": PANEL, "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150, "savefig.dpi": 300,
})


def save(fig, name):
    # NO bbox_inches="tight": a tight crop makes the saved canvas narrower than figsize, and LaTeX then
    # scales the file up to the requested width by a different factor per figure (measured 1.01x to
    # 1.16x), so the nominal 8.5 pt would print between 8.6 and 9.9 pt. Constrained layout already fits
    # the decorations inside the canvas, so the file is saved at exactly figsize and LaTeX scales 1:1.
    # CreationDate=None omits matplotlib's wall-clock stamp, so re-running this script leaves the
    # committed PDFs byte-identical instead of churning seven binaries on every run
    fig.savefig(os.path.join(OUT, name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)
    print("wrote", name)


def mae_of(results, run):
    """Mean and SD over seeds of the test MAE of one run."""
    values = results.loc[results["run"] == run, "mae"]
    return values.mean(), values.std(ddof=1)


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=6)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = pd.read_csv(config.RESULTS_CSV)
    with open(config.ZERO_SHOT_METRICS_JSON, encoding="utf-8") as f:
        zero_shot = json.load(f)
    zs_pred = pd.read_csv(config.ZERO_SHOT_CSV)
    train = pd.read_csv(config.TRAIN_REAL_CSV)
    generated = pd.read_csv(config.GENERATED_CLEAN_CSV)
    labeled = pd.read_csv(config.LABELED_CLEAN_CSV)
    ablation = pd.read_csv(os.path.join(config.TABLES_DIR, "patience_ablation.csv"))
    n_full = len(train)

    # ---------------------------------------------------------------- Figure 2: Tg distributions (three tinted panels)
    # The panels share the x axis but not the y axis: the labeled set's spikes are three times the
    # height of anything in the other two, and a shared scale would flatten them.
    sets = [(train, "real", "Real training set"), (generated, "generated", "Generated set"),
            (labeled, "labeled", "Labeled set (PI1M)")]
    bins = np.arange(-150, 501, 10)
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 4.6), sharex=True, layout="constrained")
    for ax, (df, key, label) in zip(axes, sets):
        tg = df["tg_celsius"]
        ax.set_facecolor(TINT[key])
        # a thin white edge separates the 10 C bins, so the histogram reads as a row of bars
        # rather than one filled silhouette (each bar is only about 6 pt wide at this size)
        ax.hist(tg, bins=bins, density=True, color=C[key], edgecolor="white", linewidth=0.4)
        ax.axvline(tg.mean(), color=INK, linewidth=1.1, linestyle=(0, (4, 2.5)), zorder=4)
        ax.text(0.012, 0.88, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(0.988, 0.88, f"n = {len(tg):,}   ·   mean {tg.mean():.0f} °C", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color=MUTED)
        ax.set_ylabel("Density (per °C)")
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="white", linewidth=0.8)
        ax.set_ylim(0, 1.42 * np.histogram(tg, bins=bins, density=True)[0].max())
    axes[-1].set_xlabel(f"Glass transition temperature {TG}  (°C);  dashed line = mean")
    axes[-1].set_xlim(-150, 500)
    fig.get_layout_engine().set(hspace=0.04, h_pad=0.02)
    save(fig, "fig_tg_distributions")

    # ---------------------------------------------------------------- Figure 3: learning curve
    sizes = config.SUBSET_SIZES + [n_full]
    names = [f"n{n}" for n in config.SUBSET_SIZES] + ["nfull"]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.9), layout="constrained")
    lo, hi = zero_shot["mae_ci95"]
    ax.axhspan(lo, hi, color=C["teacher"], alpha=0.13, linewidth=0, zorder=1)
    ax.axhline(zero_shot["mae"], color=C["teacher"], linestyle=(0, (5, 2.5)), linewidth=1.8, zorder=2)
    ax.text(0.985, 0.955, f"Zero-shot teacher    {zero_shot['mae']:.1f} °C   (95 % CI)", transform=ax.transAxes,
            color=C["teacher"], ha="right", va="top", fontsize=8)
    # The three series are shifted slightly sideways so that their error bars do not hide each other.
    series = [("real", "Real only", "real", "o", 0.965), ("generated", "Generated → real", "twostage_generated", "s", 1.0),
              ("labeled", "Labeled → real", "twostage_labeled", "^", 1.035)]
    handles = []
    for key, label, prefix, marker, shift in series:
        stats = [mae_of(results, f"{prefix}_{n}") for n in names]
        mean, sd = np.array([m for m, _ in stats]), np.array([d for _, d in stats])
        ax.errorbar(np.array(sizes) * shift, mean, yerr=sd, color=C[key], marker=marker, markersize=5.5,
                    linewidth=1.9, elinewidth=1.0, capsize=2.5, capthick=1.0, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=4)
        # a clean handle: the errorbar container would draw its whisker into the legend as a dash
        handles.append(Line2D([], [], color=C[key], marker=marker, markersize=5.5, linewidth=1.9,
                              markeredgecolor="white", markeredgewidth=0.7, label=label))
    # The patience ablation is a self-contained triple (baseline AND both two-stage arms at patience 10),
    # so that the vertical gaps a reader measures inside it compare like with like. Open markers, dashed.
    patient = ablation[ablation["patience"] == 10].sort_values("n_real")
    for key, column, marker, shift in [("real", "MAE real", "o", 0.965), ("generated", "MAE generated", "s", 1.0),
                                       ("labeled", "MAE labeled", "^", 1.035)]:
        ax.plot(patient["n_real"] * shift, patient[column], color=C[key], marker=marker, markersize=5.5,
                markerfacecolor="white", markeredgewidth=1.4, linewidth=1.0, linestyle=(0, (2, 2)), zorder=3)
    handles.append(Line2D([], [], color=C["real"], marker="o", markersize=5.5, markerfacecolor="white",
                          markeredgewidth=1.4, linestyle="none", label="Same three, patience 10"))
    ax.legend(handles=handles, loc="lower left", frameon=True, facecolor="white", edgecolor=GRID,
              framealpha=1.0, handlelength=2.4, borderpad=0.7, labelspacing=0.5)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] * 0.80, sizes[-1] * 1.16)
    ax.set_xlabel("Number of real training polymers")
    ax.set_ylabel(f"MAE on the real test set  (°C)")
    ax.set_ylim(22.0, 41.4)
    ax.set_yticks(np.arange(22.5, 40.1, 2.5))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_learning_curve")

    # ---------------------------------------------------------------- Figure 4: controls at n = 500
    # Vertical bars from zero, grouped by which synthetic set the first stage used, with the two
    # reference levels drawn across the panel. The axis reaches the top of the longest whisker.
    n = f"n{config.CONTROL_SIZE}"
    BARS = [(f"real_{n}", "Real only", "real", False),
            (f"twostage_labeled_{n}", "LLM labels", "labeled", False),
            (f"twostage_selftrain_{n}", "Self-training", "selftrain", False),
            (f"twostage_shuffled_labeled_{n}", "Labels shuffled", "labeled", True),
            (f"twostage_generated_{n}", "LLM labels", "generated", False),
            (f"twostage_shuffled_generated_{n}", "Labels shuffled", "generated", True)]
    positions = [0.0, 1.35, 2.35, 3.35, 4.70, 5.70]     # the gaps separate the three families
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.9), layout="constrained")
    real_mean, _ = mae_of(results, f"real_{n}")
    lo, hi = zero_shot["mae_ci95"]
    ax.axhspan(lo, hi, color=C["teacher"], alpha=0.13, linewidth=0, zorder=1)
    ax.axhline(zero_shot["mae"], color=C["teacher"], linestyle=(0, (5, 2.5)), linewidth=1.8, zorder=2)
    ax.axhline(real_mean, color=INK, linestyle=(0, (1.5, 2)), linewidth=1.3, zorder=2)
    top = 0
    for (run, label, key, shuffled), x in zip(BARS, positions):
        mean, sd = mae_of(results, run)
        ax.bar(x, mean, width=0.86, color=C[key], alpha=0.30 if shuffled else 1.0,
               hatch="////" if shuffled else None, edgecolor=C[key] if shuffled else "none",
               linewidth=0.9 if shuffled else 0, zorder=3)
        ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=5)
        # the bbox keeps the two reference lines from striking through a value label
        ax.text(x, mean + sd + 1.6, f"{mean:.1f}", ha="center", va="bottom", fontsize=8, color=INK, zorder=6,
                bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 1.4})
        top = max(top, mean + sd)
    ax.set_xticks(positions)
    ax.set_xticklabels([b[1] for b in BARS], fontsize=8, color=MUTED)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(-0.72, positions[-1] + 0.72)
    ax.set_ylim(0, top + 8)
    ax.set_ylabel(f"MAE on the real test set  (°C)")
    ax.grid(axis="x", visible=False)
    legend = [Patch(facecolor=C["real"], label="No first stage (real only)"),
              Patch(facecolor=C["labeled"], label="First stage: labeled set"),
              Patch(facecolor=C["selftrain"], label="First stage: self-training"),
              Patch(facecolor=C["generated"], label="First stage: generated set"),
              Patch(facecolor="#c8d2e0", alpha=0.45, hatch="////", edgecolor=MUTED, label=f"{TG} values shuffled"),
              Line2D([], [], color=C["teacher"], linestyle=(0, (5, 2.5)), linewidth=1.8,
                     label=f"Zero-shot {config.LLM_MODEL.replace('claude-sonnet-5', 'Claude Sonnet 5')}   ({zero_shot['mae']:.1f} °C)"),
              Line2D([], [], color=INK, linestyle=(0, (1.5, 2)), linewidth=1.3, label=f"Real data alone   ({real_mean:.1f} °C)")]
    ax.legend(handles=legend, loc="upper left", frameon=True, facecolor="white", edgecolor=GRID,
              framealpha=1.0, handlelength=1.9, borderpad=0.7, labelspacing=0.45)
    # Which synthetic set each group's first stage used, written under the axis with a thin rule.
    for lo_i, hi_i, text in [(1, 3, "First stage on the labeled set"), (4, 5, "Generated set")]:
        left, right = positions[lo_i] - 0.43, positions[hi_i] + 0.43
        ax.plot([left, right], [-0.115, -0.115], transform=ax.get_xaxis_transform(), color=GRID,
                linewidth=1.0, clip_on=False, zorder=2)
        ax.text((left + right) / 2, -0.145, text, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.5, color=MUTED, clip_on=False)
    # rect is (left, bottom, WIDTH, HEIGHT): hold back the bottom 12 % of the canvas for the group
    # rules written below the axes, which constrained layout cannot reserve room for by itself.
    fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))
    save(fig, "fig_controls")

    # ---------------------------------------------------------------- Figure 5: predicted against measured (seed 42)
    panels = [("real_nfull", "Real only, full training set", "real"),
              ("synthetic_generated", "Generated set only", "generated"),
              ("synthetic_labeled", "Labeled set only", "labeled"),
              (None, f"Zero-shot {config.LLM_MODEL.replace('claude-sonnet-5', 'Claude Sonnet 5')}", "teacher")]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 6.1), layout="constrained")
    axes = list(axes.flat)
    for k, (ax, (run, title, key)) in enumerate(zip(axes, panels)):
        df = zs_pred if run is None else pd.read_csv(os.path.join(config.PREDICTIONS_DIR, f"{run}_seed{config.SEED}.csv"))
        err = df["tg_pred"] - df["tg_true"]
        ax.plot([-150, 500], [-150, 500], color=MUTED, linewidth=0.8, zorder=1)
        ax.scatter(df["tg_true"], df["tg_pred"], s=5, alpha=0.42, color=C[key], linewidths=0, rasterized=True, zorder=2)
        panel_label(ax, f"({'abcd'[k]})   {title}")
        ax.text(0.035, 0.965, f"MAE = {err.abs().mean():.1f} °C     n = {len(df):,}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=MUTED)
        ax.set_xlim(-150, 500)
        ax.set_ylim(-150, 500)
        ax.set_xticks([-100, 0, 100, 200, 300, 400, 500])
        ax.set_yticks([-100, 0, 100, 200, 300, 400, 500])
        ax.set_aspect("equal")
        ax.set_xlabel(f"Measured {TG}  (°C)")
        ax.set_ylabel(f"Predicted {TG}  (°C)")
    fig.get_layout_engine().set(hspace=0.06, wspace=0.06)
    save(fig, "fig_predicted_vs_true")

    # ---------------------------------------------------------------- Figure S1: saturation of the generation route, and label rounding
    gen_raw = pd.read_csv(config.GENERATED_RAW_CSV)
    valid = gen_raw.copy()
    valid["canon"] = valid["smiles"].map(canonical_smiles)
    valid = valid.dropna(subset=["canon"])
    valid = valid[valid["canon"].str.count(r"\*") == 2]
    valid = valid[pd.to_numeric(valid["tg_celsius"], errors="coerce").between(config.TG_MIN_CELSIUS, config.TG_MAX_CELSIUS)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9), layout="constrained")
    # (a) distinct polymers obtained against requests spent, against the 50-per-request ideal
    request_number = valid["request_id"].str.extract(r"(\d+)$")[0].astype(int)
    first_seen = valid.assign(request=request_number, polymer=valid["canon"].map(polymer_id)).groupby("polymer")["request"].min()
    requests = np.arange(config.GENERATION_MAX_REQUESTS + 1)
    distinct = np.concatenate([[0], np.cumsum(np.bincount(first_seen.values, minlength=config.GENERATION_MAX_REQUESTS))])
    ax1.plot(requests, config.PAIRS_PER_REQUEST * requests, color=MUTED, linewidth=1.0, linestyle=(0, (4, 2)))
    ax1.text(112, 9950, "no repetition\n(50 per request)", color=MUTED, fontsize=7, ha="right", va="top", linespacing=1.3)
    ax1.axvline(202, color="white", linewidth=1.4)                   # the boundary between the two rounds
    ax1.plot(requests, distinct, color=C["generated"], linewidth=2.0)
    ax1.plot([config.GENERATION_MAX_REQUESTS], [distinct[-1]], marker="o", markersize=4.5, color=C["generated"],
             markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax1.annotate(f"{distinct[-1]:,} distinct polymers\nfrom {config.GENERATION_MAX_REQUESTS} requests", xy=(396, distinct[-1] + 120),
                 xytext=(372, 10150), textcoords="data", color=C["generated"], fontsize=7.5, ha="right", va="top", linespacing=1.3,
                 arrowprops={"arrowstyle": "-", "color": C["generated"], "linewidth": 0.6})
    for lo_, hi_ in [(0, 202), (202, config.GENERATION_MAX_REQUESTS)]:
        ax1.text((lo_ + hi_) / 2, distinct[lo_] + (distinct[hi_] - distinct[lo_]) / 2 - 700,
                 f"{(distinct[hi_] - distinct[lo_]) / (hi_ - lo_):.0f} new\nper request", color=INK, fontsize=7, ha="center", va="top", linespacing=1.3)
    panel_label(ax1, "(a)   The teacher runs out of new polymers")
    ax1.set_xlabel("Generation requests sent")
    ax1.set_ylabel("Distinct polymers obtained")
    ax1.set_xlim(0, config.GENERATION_MAX_REQUESTS)
    ax1.set_ylim(0, 10500)
    ax1.set_yticks([0, 2500, 5000, 7500, 10000])
    ax1.set_yticklabels(["0", "2,500", "5,000", "7,500", "10,000"])
    ax1.grid(axis="x", visible=False)
    # (b) the rounding of the labels
    vc = labeled["tg_celsius"].value_counts().sort_index()
    ax2.bar(vc.index, vc.values, width=4.0, color=C["labeled"], linewidth=0)
    for v, tx, ty in [(-20, -128, 1480), (45, 135, 1560)]:
        ax2.annotate(f"{v:g} °C: {vc[v]:,} polymers", xy=(v, vc[v] + 30), xytext=(tx, ty), textcoords="data",
                     fontsize=7, color=INK, ha="left", va="top",
                     arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkB": 2})
    panel_label(ax2, "(b)   The teacher's labels are rounded")
    ax2.set_xlabel(f"{TG} assigned by the LLM (°C)")
    ax2.set_ylabel("Number of polymers")
    ax2.set_xlim(-140, 340)
    ax2.set_ylim(0, 1600)
    ax2.grid(axis="x", visible=False)
    fig.get_layout_engine().set(w_pad=0.08)
    save(fig, "fig_si_repetition_rounding")

    # ---------------------------------------------------------------- Figure S2: error against similarity to the training set
    by_sim = pd.read_csv(os.path.join(config.TABLES_DIR, "mae_by_similarity_bin.csv"))
    by_sim = by_sim[by_sim["similarity_bin"] != "all"]
    fig, ax = plt.subplots(figsize=(0.85 * TEXT_WIDTH, 3.4), layout="constrained")
    x = np.arange(len(by_sim))
    series = [("real only, full", "Real only, full", "real", "o"), ("random forest, real full", "Random forest, real, full", "real", "D"),
              ("generated only", "Generated only", "generated", "s"), ("labeled only", "Labeled only", "labeled", "^"),
              ("zero-shot teacher", "Zero-shot teacher", "teacher", "v")]
    for col, label, key, marker in series:
        forest = col.startswith("random forest")     # same data source as the real-only student, so the same colour, dashed
        ax.plot(x, by_sim[col], color=C[key], marker=marker, markersize=4.5,
                linewidth=1.2 if forest else (1.9 if key == "teacher" else 1.6),
                linestyle=(0, (2, 2)) if forest else ((0, (5, 2.5)) if key == "teacher" else "-"),
                alpha=0.75 if forest else 1.0, markeredgecolor="white", markeredgewidth=0.6, label=label)
    ax.set_xticks(x)
    bands = [str(b).replace("[", "").replace(")", "").replace(", ", " to ") for b in by_sim["similarity_bin"]]
    bands[-1] = bands[-1].replace(" to 1.0", " to 1.00")
    ax.set_xticklabels([f"{b}\nn = {int(n):,}" for b, n in zip(bands, by_sim["n"])], fontsize=7)
    ax.set_xlabel("Maximum Tanimoto similarity of the test polymer to the real training set")
    ax.set_ylabel("MAE (°C)")
    ax.set_xlim(-0.3, len(by_sim) - 0.7)
    ax.legend(loc="upper right", ncol=2, handlelength=2.4, columnspacing=1.2, frameon=True,
              facecolor="white", edgecolor=GRID, framealpha=1.0, borderpad=0.6)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_si_similarity")

    # ---------------------------------------------------------------- Figure S3: zero-shot signed error
    fig, ax = plt.subplots(figsize=(0.70 * TEXT_WIDTH, 3.1), layout="constrained")
    err = zs_pred["tg_pred"] - zs_pred["tg_true"]
    ax.scatter(zs_pred["tg_true"], err, s=6, alpha=0.45, color=C["teacher"], linewidths=0, rasterized=True, label="Test polymers")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    edges = np.arange(-150, 501, 50)
    mids, meds = [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        part = err[(zs_pred["tg_true"] >= lo_) & (zs_pred["tg_true"] < hi_)]
        if len(part) >= 10:
            mids.append((lo_ + hi_) / 2)
            meds.append(part.median())
    # the running median is a summary of the teacher's own points, not a second data source, so it is
    # drawn in the neutral ink; a series hue here would read as "the labeled set"
    ax.plot(mids, meds, color=INK, linewidth=1.8, marker="o", markersize=3.5, markeredgecolor="white",
            markeredgewidth=0.6, label="Median error per 50 °C band")
    ax.set_xlabel(f"Measured {TG} (°C)")
    ax.set_ylabel(f"Predicted minus measured {TG} (°C)")
    ax.set_xlim(-150, 500)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID, framealpha=1.0, borderpad=0.6)
    save(fig, "fig_si_zero_shot_error")
    print("figures done")


if __name__ == "__main__":
    main()
