"""
Generate every LaTeX table of the paper and the SI, and the inline-number macros (numbers.tex), from
results/ and data/, so that no number in the manuscript is typed by hand. Written to
results/tables/latex/, from where paper/main.tex and paper/supplementary.tex read them.
Run with  python report/make_tables.py  (from any directory) or through pipeline/run.py.

Sources: results/results.csv and results/tables/*.csv (written by analyze_results.py),
results/run_info.json, results/memorization.csv, results/zero_shot_metrics.json, results/api_cost.csv,
results/test_similarity.csv, results/patience_ablation.jsonl, and the CSVs in data/.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# config.py and common/ live in pipeline/, one folder up and over; anchored to this file, not to the working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
import config  # noqa: E402
from common.cleaning import canonical_smiles, polymer_id  # noqa: E402

OUT = config.LATEX_TABLES_DIR
TABLES = config.TABLES_DIR


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", os.path.join(OUT, name))


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def signed(x, digits=1):
    """A number with an explicit sign; a value that rounds to zero is printed without one (macro text)."""
    text = f"{x:+.{digits}f}"
    return text[1:] if float(text) == 0 else text


def snum(x, digits=1):
    """A signed number for a table cell, typeset by siunitx (proper minus sign, explicit plus)."""
    return "\\num{" + signed(x, digits) + "}"


def num(n):
    return f"{int(n):,}"


def pm(mean, sd, digits=1):
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def pvalue(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def tabular(spec, header, rows, note=None):
    text = "\\begin{tabular}{" + spec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
    if note:
        text += note + "\n"
    return text + "\\end{tabular}\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    # ------------------------------------------------------------ load
    info = json.load(open(config.RUN_INFO_JSON, encoding="utf-8"))
    zs = json.load(open(config.ZERO_SHOT_METRICS_JSON, encoding="utf-8"))
    results = pd.read_csv(config.RESULTS_CSV)
    summary = pd.read_csv(config.RESULTS_SUMMARY_CSV).set_index("run")
    comparisons = pd.read_csv(os.path.join(TABLES, "seed_matched_comparisons.csv")).set_index("comparison (first minus second)")
    student = pd.read_csv(os.path.join(TABLES, "student_vs_teacher.csv"))
    noise = pd.read_csv(os.path.join(TABLES, "noise_floor.csv")).set_index("quantity")["value"]
    by_tg = pd.read_csv(os.path.join(TABLES, "mae_by_tg_bin.csv")).set_index("tg_bin")
    by_sim = pd.read_csv(os.path.join(TABLES, "mae_by_similarity_bin.csv")).set_index("similarity_bin")
    by_rel = pd.read_csv(os.path.join(TABLES, "mae_by_reliability.csv")).set_index("reliability")
    inference = pd.read_csv(os.path.join(TABLES, "inference_noise.csv"))
    memo = pd.read_csv(config.MEMORIZATION_CSV)
    cost = pd.read_csv(config.API_COST_CSV).set_index("task")
    similarity = pd.read_csv(config.TEST_SIMILARITY_CSV)["max_tanimoto_to_train"]
    real_all = pd.read_csv(config.REAL_CLEAN_CSV)
    train = pd.read_csv(config.TRAIN_REAL_CSV)
    val = pd.read_csv(config.VAL_REAL_CSV)
    test = pd.read_csv(config.TEST_REAL_CSV)
    generated = pd.read_csv(config.GENERATED_CLEAN_CSV)
    labeled = pd.read_csv(config.LABELED_CLEAN_CSV)
    gen_raw = pd.read_csv(config.GENERATED_RAW_CSV)
    zero_shot = pd.read_csv(config.ZERO_SHOT_CSV)
    n_seeds = int(summary["n_seeds"].iloc[0])
    n_full = len(train)

    SIZES = [("TwoFifty", "n250", "250"), ("FiveHundred", "n500", "500"), ("Thousand", "n1000", "1{,}000"), ("Full", "nfull", num(n_full))]
    FAMILIES = [("Real", "real", "Real only"), ("TsGen", "twostage_generated", "Two-stage: generated $\\rightarrow$ real"),
                ("TsLab", "twostage_labeled", "Two-stage: labeled $\\rightarrow$ real"),
                ("CatGen", "concat_generated", "Concatenation: real + generated"), ("CatLab", "concat_labeled", "Concatenation: real + labeled")]


    def mean_sd(run, key="mae"):
        return summary.loc[run, f"{key}_mean"], summary.loc[run, f"{key}_std"]


    def cell(run, key="mae", digits=1):
        if run not in summary.index:
            return "--"
        return pm(*mean_sd(run, key), digits)


    # ---------------------------------------------------------------- Table: learning curve and mixing protocols (MoLFormer)
    ablation = pd.read_csv(os.path.join(TABLES, "patience_ablation.csv"))
    patient = ablation[ablation["patience"] == 10].set_index("n_real")
    COMPARISON = {"twostage_generated": "two-stage generated vs real only, {}", "twostage_labeled": "two-stage labeled vs real only, {}",
                  "concat_generated": "concatenation generated vs real only, {}", "concat_labeled": "concatenation labeled vs real only, {}"}


    def gain_cell(key):
        """Seed-matched difference to real only: mean, SD, and the seeds on which it has the sign of the mean."""
        if key not in comparisons.index:
            return "--"
        c = comparisons.loc[key]
        return f"{snum(c['dMAE mean'])} $\\pm$ {fmt(c['SD'])} ({c['same sign']})"


    def row(label, size, protocol):
        """
        One size of the learning curve: the real-only baseline and each synthetic set beside it, so that the
        three numbers a reader compares sit on one line. The seed-matched differences and their tests are in
        SI Table~S14. Nothing is set in bold: the two synthetic sets cannot be separated at any size
        (Section 3.3), and a bold cell in every row would assert an ordering that no comparison supports.
        """
        return (f"\\quad {label} & {cell(f'real_{size}')} & {cell(f'{protocol}_generated_{size}')}"
                f" & {cell(f'{protocol}_labeled_{size}')} \\\\")


    def band(title):
        return ("\\addlinespace\n" if rows else "") + f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"


    rows = []
    rows.append(band("Two-stage: synthetic first, then real (main protocol, patience 3)"))
    for _, size, shown in SIZES:
        rows.append(row(shown, size, "twostage"))
    # The conservative comparison: the same two-stage models against a real-only baseline that trains
    # about four times longer (patience 10; separate runs, the patience ablation). It follows the main
    # block directly because it is the estimate the text leads with.
    rows.append(band("Two-stage, against a real-only baseline trained to convergence (patience 10; separate runs)"))
    for n_real, r in patient.iterrows():
        rows.append(f"\\quad {num(n_real)} & {pm(r['MAE real'], r['SD real'])} & {pm(r['MAE generated'], r['SD generated'])} & "
                    f"{pm(r['MAE labeled'], r['SD labeled'])} \\\\")
    rows.append(band("Concatenation: real + synthetic in one training set"))
    for _, size, shown in SIZES:
        if f"concat_generated_{size}" in summary.index:
            rows.append(row(shown, size, "concat"))
    # the stage-1 models are the same learning curve at zero real polymers, so they belong in the grid
    rows.append("\\addlinespace\\midrule")
    # indented like every other size row, so the 0 lines up with the 250/500/1,000/5,297 above it
    rows.append(f"\\quad 0 (synthetic data only) & -- & {cell('synthetic_generated')} & {cell('synthetic_labeled')} \\\\")
    rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{Teacher, zero-shot: {fmt(zs['mae'])} (95\\,\\% CI {fmt(zs['mae_ci95'][0])} to {fmt(zs['mae_ci95'][1])})}} \\\\")
    header = (" & \\multicolumn{3}{c}{First stage before the real polymers} \\\\\n"
              "\\cmidrule(lr){2-4}\n"
              "Real polymers & None (real only) & Generated set & Labeled set")
    write("results_main.tex", tabular("@{}llll@{}", header, rows))

    # ---------------------------------------------------------------- Table: run matrix (Methods)
    # Which configuration was trained at which number of real polymers, per seed (the A-D scheme of
    # train_molformer.py). A cell holds a bullet when the run exists in results/runs/ for every seed.
    MATRIX_SIZES = [("0", None), ("250", "n250"), ("500", "n500"), ("1{,}000", "n1000"), (num(n_full), "nfull")]
    MATRIX = [("Real only (MoLFormer)", None, "real_{}"),
              ("Two-stage: generated set, then real", "synthetic_generated", "twostage_generated_{}"),
              ("Two-stage: labeled set, then real", "synthetic_labeled", "twostage_labeled_{}"),
              ("Concatenation: real + generated", None, "concat_generated_{}"),
              ("Concatenation: real + labeled", None, "concat_labeled_{}"),
              ("Control: self-training labels, then real", "synthetic_selftrain", "twostage_selftrain_{}"),
              ("Control: generated set with shuffled \\Tg, then real", "synthetic_shuffled_generated", "twostage_shuffled_generated_{}"),
              ("Control: labeled set with shuffled \\Tg, then real", "synthetic_shuffled_labeled", "twostage_shuffled_labeled_{}"),
              ("Random forest, real only", None, "rf_real_{}"),
              ("Random forest, real + generated", "rf_synthetic_generated", "rf_concat_generated_{}"),
              ("Random forest, real + labeled", "rf_synthetic_labeled", "rf_concat_labeled_{}")]
    rows = []
    for label, stage1, pattern in MATRIX:
        cells = []
        for _, size in MATRIX_SIZES:
            run = stage1 if size is None else pattern.format(size)
            cells.append("$\\bullet$" if run is not None and run in summary.index and summary.loc[run, "n_seeds"] == n_seeds else "--")
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write("run_matrix.tex", tabular("@{}l" + "c" * len(MATRIX_SIZES) + "@{}",
          " & \\multicolumn{" + str(len(MATRIX_SIZES)) + "}{c}{Real training polymers} \\\\\n\\cmidrule(lr){2-" + str(len(MATRIX_SIZES) + 1) + "}\n"
          "Configuration & " + " & ".join(shown for shown, _ in MATRIX_SIZES), rows))

    # ---------------------------------------------------------------- Table: controls at n = 500
    CONTROLS = [("real_n500", "None (real only)", "--", None),
                ("twostage_labeled_n500", "LLM labels", "synthetic_labeled", "two-stage labeled vs real only, n500"),
                ("twostage_selftrain_n500", "Self-training labels", "synthetic_selftrain", "self-training vs real only, n500"),
                ("twostage_shuffled_labeled_n500", "LLM labels, shuffled", "synthetic_shuffled_labeled", "shuffled labeled vs real only, n500"),
                ("twostage_generated_n500", "LLM labels", "synthetic_generated", "two-stage generated vs real only, n500"),
                ("twostage_shuffled_generated_n500", "LLM labels, shuffled", "synthetic_shuffled_generated", "shuffled generated vs real only, n500")]
    rows = []
    for i, (run, label, stage1, key) in enumerate(CONTROLS):
        if i == 1:
            rows.append("\\addlinespace\n\\multicolumn{5}{@{}l}{\\emph{First stage on the labeled set (PI1M structures)}} \\\\")
        if i == 4:
            rows.append("\\addlinespace\n\\multicolumn{5}{@{}l}{\\emph{First stage on the generated set}} \\\\")
        s1 = "--" if stage1 == "--" else cell(stage1)
        indent = "" if key is None else "\\quad "
        rows.append(f"{indent}{label} & {s1} & {cell(run)} & {cell(run, 'rmse')} & {cell(run, 'r2', 3)} \\\\")
    write("controls.tex", tabular("@{}lllll@{}",
          " & & \\multicolumn{3}{c}{After the 500 real polymers} \\\\\n\\cmidrule(lr){3-5}\n"
          "First stage & Stage-1 model alone & MAE & RMSE & $R^2$", rows))

    # ---------------------------------------------------------------- Table: seed-matched differences (main text)
    PAIRS = [("TsGenTwoFifty", "two-stage generated vs real only, n250", "Generated $\\rightarrow$ real vs.\\ real only, 250"),
             ("TsLabTwoFifty", "two-stage labeled vs real only, n250", "Labeled $\\rightarrow$ real vs.\\ real only, 250"),
             ("TsGenFiveHundred", "two-stage generated vs real only, n500", "Generated $\\rightarrow$ real vs.\\ real only, 500"),
             ("TsLabFiveHundred", "two-stage labeled vs real only, n500", "Labeled $\\rightarrow$ real vs.\\ real only, 500"),
             ("TsGenThousand", "two-stage generated vs real only, n1000", "Generated $\\rightarrow$ real vs.\\ real only, 1{,}000"),
             ("TsLabThousand", "two-stage labeled vs real only, n1000", "Labeled $\\rightarrow$ real vs.\\ real only, 1{,}000"),
             ("TsGenFull", "two-stage generated vs real only, nfull", f"Generated $\\rightarrow$ real vs.\\ real only, {num(n_full)}"),
             ("TsLabFull", "two-stage labeled vs real only, nfull", f"Labeled $\\rightarrow$ real vs.\\ real only, {num(n_full)}"),
             ("CatGenFiveHundred", "concatenation generated vs real only, n500", "Real + generated vs.\\ real only, 500"),
             ("CatLabFiveHundred", "concatenation labeled vs real only, n500", "Real + labeled vs.\\ real only, 500"),
             ("CatGenFull", "concatenation generated vs real only, nfull", f"Real + generated vs.\\ real only, {num(n_full)}"),
             ("CatLabFull", "concatenation labeled vs real only, nfull", f"Real + labeled vs.\\ real only, {num(n_full)}"),
             ("TsVsCatGenFiveHundred", "two-stage vs concatenation, generated, n500", "Two-stage vs.\\ concatenation, generated, 500"),
             ("TsVsCatLabFiveHundred", "two-stage vs concatenation, labeled, n500", "Two-stage vs.\\ concatenation, labeled, 500"),
             ("TsVsCatGenFull", "two-stage vs concatenation, generated, nfull", f"Two-stage vs.\\ concatenation, generated, {num(n_full)}"),
             ("TsVsCatLabFull", "two-stage vs concatenation, labeled, nfull", f"Two-stage vs.\\ concatenation, labeled, {num(n_full)}"),
             ("SelfVsLab", "self-training vs LLM-labeled (two-stage, n500)", "Self-training vs.\\ LLM labels, 500"),
             ("SelfVsReal", "self-training vs real only, n500", "Self-training $\\rightarrow$ real vs.\\ real only, 500"),
             ("ShufGen", "shuffled vs unshuffled generated (two-stage, n500)", "Shuffled vs.\\ original labels, generated, 500"),
             ("ShufLab", "shuffled vs unshuffled labeled (two-stage, n500)", "Shuffled vs.\\ original labels, labeled, 500"),
             ("ShufGenVsReal", "shuffled generated vs real only, n500", "Shuffled generated $\\rightarrow$ real vs.\\ real only, 500"),
             ("ShufLabVsReal", "shuffled labeled vs real only, n500", "Shuffled labeled $\\rightarrow$ real vs.\\ real only, 500"),
             ("LabVsGen", "labeled only vs generated only (stage 1)", "Labeled only vs.\\ generated only"),
             ("RfCatGenFiveHundred", "random forest: real + generated vs real only, n500", "Random forest: real + generated vs.\\ real, 500"),
             ("RfCatLabFiveHundred", "random forest: real + labeled vs real only, n500", "Random forest: real + labeled vs.\\ real, 500"),
             ("RfCatGenFull", "random forest: real + generated vs real only, nfull", f"Random forest: real + generated vs.\\ real, {num(n_full)}"),
             ("RfCatLabFull", "random forest: real + labeled vs real only, nfull", f"Random forest: real + labeled vs.\\ real, {num(n_full)}")]
    GROUP_HEADINGS = {"TsGenTwoFifty": "Synthetic first stage against real data alone", "CatGenFiveHundred": "Concatenation against real data alone",
                      "TsVsCatGenFiveHundred": "Two-stage against concatenation", "SelfVsLab": "Controls with 500 real polymers",
                      "LabVsGen": "Synthetic data alone", "RfCatGenFiveHundred": "Random forest"}
    rows = []
    for tag, key, label in PAIRS:
        if tag in GROUP_HEADINGS:
            rows.append(("\\addlinespace\n" if rows else "") + f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{GROUP_HEADINGS[tag]}}}}} \\\\")
        c = comparisons.loc[key]
        rows.append(f"\\quad {label} & {snum(c['dMAE mean'])} $\\pm$ {fmt(c['SD'])} & [{snum(c['min'])}, {snum(c['max'])}] & "
                    f"{c['same sign']} & {pvalue(c['paired t p'])} & {pvalue(c['Wilcoxon p'])} \\\\")
    write("paired_differences_full.tex", tabular("@{}llllll@{}",
          "Comparison (first minus second), $n_\\mathrm{real}$ & $\\Delta$MAE (\\unit{\\celsius}) & Range & Same sign & $p$ (paired $t$) & $p$ (Wilcoxon)", rows))

    # ---------------------------------------------------------------- Tables for the SI: every run, four metrics; per-seed MAE; best epochs
    SIZE_ROWS = [(size, shown) for _, size, shown in SIZES]
    ORDER = [("MoLFormer, no real data", [("synthetic_generated", "Generated set"), ("synthetic_labeled", "Labeled set")]),
             ("MoLFormer, real data alone", [(f"real_{s}", shown) for s, shown in SIZE_ROWS]),
             ("Two-stage: generated $\\rightarrow$ real", [(f"twostage_generated_{s}", shown) for s, shown in SIZE_ROWS]),
             ("Two-stage: labeled $\\rightarrow$ real", [(f"twostage_labeled_{s}", shown) for s, shown in SIZE_ROWS]),
             ("Concatenation: real + synthetic", [("concat_generated_n500", "Generated, 500"), ("concat_labeled_n500", "Labeled, 500"),
                                                  ("concat_generated_nfull", f"Generated, {num(n_full)}"), ("concat_labeled_nfull", f"Labeled, {num(n_full)}")]),
             ("Controls, stage 1 alone", [("synthetic_selftrain", "Self-training labels"), ("synthetic_shuffled_generated", "Generated, shuffled"),
                                          ("synthetic_shuffled_labeled", "Labeled, shuffled")]),
             ("Controls, after 500 real polymers", [("twostage_selftrain_n500", "Self-training labels"), ("twostage_shuffled_generated_n500", "Generated, shuffled"),
                                                    ("twostage_shuffled_labeled_n500", "Labeled, shuffled")]),
             ("Random forest", [("rf_real_n500", "Real only, 500"), ("rf_real_nfull", f"Real only, {num(n_full)}"),
                                ("rf_synthetic_generated", "Generated set"), ("rf_synthetic_labeled", "Labeled set"),
                                ("rf_concat_generated_n500", "Real + generated, 500"), ("rf_concat_labeled_n500", "Real + labeled, 500"),
                                ("rf_concat_generated_nfull", f"Real + generated, {num(n_full)}"), ("rf_concat_labeled_nfull", f"Real + labeled, {num(n_full)}")])]

    rows = []
    for group, runs in ORDER:
        rows.append(("\\addlinespace\n" if rows else "") + f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{group}}}}} \\\\")
        for run, label in runs:
            rows.append(f"\\quad {label} & {cell(run)} & {cell(run, 'rmse')} & {cell(run, 'r2', 3)} & {cell(run, 'spearman', 3)} \\\\")
    rows.append("\\midrule")
    rows.append(f"Zero-shot teacher & {fmt(zs['mae'])} & {fmt(zs['rmse'])} & {fmt(zs['r2'], 3)} & {fmt(zs['spearman'], 3)} \\\\")
    write("results_full.tex", tabular("@{}lllll@{}", "Training data & MAE & RMSE & $R^2$ & $\\rho$", rows))

    mae = results[results["model"] == "molformer"].pivot(index="run", columns="seed", values="mae")
    epochs = results[results["model"] == "molformer"].pivot(index="run", columns="seed", values="best_epoch")
    seeds = list(mae.columns)
    rows, erows = [], []
    for group, runs in ORDER[:-1]:
        rows.append(("\\addlinespace\n" if rows else "") + f"\\multicolumn{{{len(seeds) + 2}}}{{@{{}}l}}{{\\emph{{{group}}}}} \\\\")
        erows.append(("\\addlinespace\n" if erows else "") + f"\\multicolumn{{{len(seeds) + 2}}}{{@{{}}l}}{{\\emph{{{group}}}}} \\\\")
        for run, label in runs:
            rows.append(f"\\quad {label} & " + " & ".join(fmt(mae.loc[run, s]) for s in seeds) + f" & {cell(run)} \\\\")
            erows.append(f"\\quad {label} & " + " & ".join(str(int(epochs.loc[run, s])) for s in seeds) + f" & {fmt(epochs.loc[run].mean())} \\\\")
    header = "Training data & " + " & ".join(f"seed {s}" for s in seeds)
    write("per_seed_molformer.tex", tabular("@{}l" + "r" * len(seeds) + "l@{}", header + " & Mean $\\pm$ SD", rows))
    write("best_epochs.tex", tabular("@{}l" + "r" * len(seeds) + "r@{}", header + " & Mean", erows))

    # ---------------------------------------------------------------- cleaning counts
    g, l = info["generated_cleaning_counts"], info["labeled_cleaning_counts"]
    STEPS = [("0_raw", "Raw pairs from the responses"), ("1_numeric_tg", "1. Numeric \\Tg"),
             ("2_valid_smiles", "2. RDKit parses the SMILES"), ("3_two_stars", "3. Exactly two \\texttt{*}"),
             ("4_tg_in_range", "4. \\Tg{} in $[-150, 500]$\\,\\unit{\\celsius}"),
             ("5a_unique_exact_smiles", "5. Duplicates merged, exact SMILES"), ("5b_unique_either_key", "\\quad and by the ring key"),
             ("6a_not_in_test", "6. Test polymers removed"), ("6b_not_in_validation", "\\quad and validation polymers")]
    rows, prev_g, prev_l = [], None, None
    for key, label in STEPS:
        if key in ("5a_unique_exact_smiles", "6a_not_in_test"):
            rows.append("\\addlinespace")
        rg = "" if prev_g is None else f"({prev_g - g[key]:,})"
        rl = "" if prev_l is None else f"({prev_l - l[key]:,})"
        rows.append(f"{label} & {num(g[key])} & {rg} & {num(l[key])} & {rl} \\\\")
        prev_g, prev_l = g[key], l[key]
    write("cleaning_counts.tex", tabular("@{}lrlrl@{}",
          " & \\multicolumn{2}{c}{Generated} & \\multicolumn{2}{c}{Labeled (PI1M)} \\\\\n\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\nStep & Rows kept & (removed) & Rows kept & (removed)", rows))

    # ---------------------------------------------------------------- dataset statistics
    rows = []
    for name, df in [("Real, all", real_all), ("\\quad training split", train), ("\\quad validation split", val),
                     ("\\quad test split", test), ("Generated", generated), ("Labeled (PI1M)", labeled)]:
        if name.startswith("Generated"):
            rows.append("\\addlinespace")
        tg = df["tg_celsius"]
        rows.append(f"{name} & {num(len(tg))} & {fmt(tg.mean())} & {fmt(tg.std())} & \\num{{{fmt(tg.min(), 0)}}} & \\num{{{fmt(tg.max(), 0)}}} \\\\")
    write("dataset_stats.tex", tabular("@{}lrrrrr@{}", "Dataset & $n$ & Mean & SD & Min & Max", rows))

    # ---------------------------------------------------------------- memorization
    rows = []
    for split in ["train", "validation", "test"]:
        for kind, label in [("exact", "exact SMILES"), ("ring", "ring key only"), ("either", "either key")]:
            r = memo[(memo["synthetic_set"] == "generated") & (memo["real_split"] == split) & (memo["match"] == kind)].iloc[0]
            rows.append(f"{split} & {label} & {num(r['n'])} & {r['percent_of_unique_synthetic']:.1f} & {fmt(r['teacher_mae'])} & "
                        f"{fmt(r['teacher_median_abs_error'])} & {fmt(r['teacher_spearman'], 2)} & {snum(r['teacher_mean_signed_error'])} \\\\")
    write("memorization.tex", tabular("@{}llrrrrrr@{}",
          "Real split & Match & $n$ & \\% of unique & MAE & Median AE & $\\rho$ & Mean signed error", rows))

    # ---------------------------------------------------------------- API usage and cost
    rows = []
    for task, label, n_clean in [("generation", "Generation (50 pairs per request)", len(generated)), ("labeling of PI1M", "Labeling of PI1M polymers", len(labeled)),
                                 ("zero-shot on the test set", "Zero-shot test-set predictions", None)]:
        c = cost.loc[task]
        per = f"{1000 * c['cost_usd'] / n_clean:.2f}" if n_clean else "--"
        rows.append(f"{label} & {num(c['requests'])} & {num(c['input_tokens'])} & {num(c['output_tokens'])} & {c['cost_usd']:.2f} & {per} \\\\")
    t = cost.loc["total"]
    rows.append(f"\\midrule\nTotal & {num(t['requests'])} & {num(t['input_tokens'])} & {num(t['output_tokens'])} & {t['cost_usd']:.2f} & -- \\\\")
    write("cost_full.tex", tabular("@{}lrrrrr@{}", "Task & Requests & Input tokens & Output tokens & Cost (USD) & USD / 1{,}000 clean rows", rows))
    # ---------------------------------------------------------------- zero-shot by band, top labels
    rows = []
    err = zero_shot["tg_pred"] - zero_shot["tg_true"]
    for lo, hi in zip(config.TG_BINS[:-1], config.TG_BINS[1:]):
        part = (zero_shot["tg_true"] >= lo) & (zero_shot["tg_true"] < hi)
        band = f"$[{lo}, {hi})$" if hi < 500 else f"$\\geq {lo}$"
        rows.append(f"{band} & {num(part.sum())} & {fmt(err[part].abs().mean())} & {snum(err[part].mean())} \\\\")
    write("zero_shot_bands.tex", tabular("@{}lrrr@{}", "Measured \\Tg{} band (\\unit{\\celsius}) & $n$ & MAE (\\unit{\\celsius}) & Mean signed error (\\unit{\\celsius})", rows))
    vc = labeled["tg_celsius"].value_counts().head(8)
    write("labeled_top_values.tex", tabular("@{}rrr@{}", "\\Tg{} value (\\unit{\\celsius}) & Count & Share (\\%)",
          [f"{k:g} & {num(v)} & {100 * v / len(labeled):.1f} \\\\" for k, v in vc.items()]))

    # ---------------------------------------------------------------- stratified errors, noise floor, student vs teacher, inference noise
    MAIN = [("zero-shot teacher", "Teacher"), ("real only, full", "Real, full"), ("real only, n500", "Real, 500"),
            ("generated only", "Generated"), ("labeled only", "Labeled"), ("two-stage generated, n500", "Gen.\\,$\\rightarrow$\\,500"),
            ("two-stage labeled, n500", "Lab.\\,$\\rightarrow$\\,500"), ("two-stage generated, full", "Gen.\\,$\\rightarrow$\\,full"),
            ("two-stage labeled, full", "Lab.\\,$\\rightarrow$\\,full"), ("random forest, real full", "RF, real full")]
    header = "Band & $n$ & " + " & ".join(short for _, short in MAIN)


    def band_label(band, last):
        """Render a pandas interval; the final bin has an open upper edge that is not a real value."""
        if band == "all":
            return "all"
        low = str(band).strip("[)").split(", ")[0]
        if band == last:
            return f"$\\geq {low}$"
        return str(band).replace("[", "$[").replace(")", ")$")


    def strata(table, name, label):
        rows = []
        last = [b for b in table.index if b != "all"][-1]
        for band, r in table.iterrows():
            shown = band_label(band, last)
            rows.append(f"{shown} & {num(r['n'])} & " + " & ".join(fmt(r[col]) for col, _ in MAIN) + " \\\\")
        write(name, tabular("@{}lr" + "r" * len(MAIN) + "@{}", header.replace("Band", label), rows))


    strata(by_tg, "mae_by_tg_bin.tex", "Measured \\Tg{} (\\unit{\\celsius})")
    strata(by_sim, "mae_by_similarity.tex", "Max.\\ Tanimoto to training set")

    rows = []
    for label, key in [("Test polymers with two or more measurements", "test polymers with 2+ measurements"),
                       ("Median experimental SD (\\unit{\\celsius})", "median experimental SD"), ("Mean experimental SD (\\unit{\\celsius})", "mean experimental SD"),
                       ("Expected $|$difference$|$ of two measurements, median-based", "expected |difference| of two measurements, from the median SD (1.128 x SD)"),
                       ("Expected $|$difference$|$ of two measurements, mean-based", "expected |difference| of two measurements, from the mean SD")]:
        v = noise[key]
        rows.append(f"{label} & {num(v) if v == int(v) else fmt(v)} \\\\")
    rows.append("\\midrule")
    n_other = next(int(re.search(r"other (\d+)", i).group(1)) for i in noise.index if "other" in i)
    for col, short in MAIN:
        rows.append(f"MAE of {short} on these polymers / on the other {num(n_other)} & "
                    f"{fmt(noise[f'MAE on these polymers: {col}'])} / {fmt(noise[[i for i in noise.index if i.startswith('MAE on the other') and i.endswith(col)][0]])} \\\\")
    write("noise_floor.tex", tabular("@{}lr@{}", "Quantity & Value", rows))

    rows = []
    for r in student.itertuples():
        rows.append(f"{'Generated' if 'generated' in r.student else 'Labeled'} & {r.seed} & {snum(r[3])} & [{snum(r[4])}, {snum(r[5])}] & {'yes' if r[6] else 'no'} \\\\")
    write("student_vs_teacher.tex", tabular("@{}llrll@{}", "Stage-1 student & Seed & Student minus teacher MAE & 95\\,\\% CI & CI excludes 0", rows))

    rows = []
    for r in inference.itertuples():
        rows.append(f"{'Generated' if 'generated' in r.run else 'Labeled'} & {r.seed} & {fmt(r.mae_mean, 2)} & {fmt(r.mae_sd, 2)} & "
                    f"{fmt(r.mae_min, 2)} to {fmt(r.mae_max, 2)} & {fmt(r.per_polymer_prediction_sd_mean)} \\\\")
    write("inference_noise.tex", tabular("@{}llrrlr@{}", "Stage-1 model & Seed & MAE, mean of 10 passes & SD & Range & Per-polymer SD", rows))

    # ---------------------------------------------------------------- hyperparameters (from config.py)
    write("hyperparameters.tex", tabular("@{}ll@{}", "Setting & Value", [
        f"Random seed for the data split and samples & {config.SEED} \\\\",
        f"Training seeds & {', '.join(str(s) for s in config.SEEDS)} \\\\",
        f"Test fraction / validation fraction of the rest & {config.TEST_FRACTION} / {config.VAL_FRACTION} \\\\",
        f"LLM & \\texttt{{{config.LLM_MODEL}}} \\\\",
        "Extended thinking & disabled \\\\",
        "Sampling parameters & none set (not accepted by the model) \\\\",
        f"Pairs requested per generation request & {config.PAIRS_PER_REQUEST} \\\\",
        f"Generation requests per round / hard cap & {config.GENERATION_ROUND_REQUESTS} / {config.GENERATION_MAX_REQUESTS} \\\\",
        f"Generation target (clean pairs) & {num(config.GENERATION_TARGET)} \\\\",
        f"\\texttt{{max\\_tokens}} generation / labeling / labeling retry & {config.GENERATION_MAX_TOKENS} / {config.LABEL_MAX_TOKENS} / {config.LABEL_RETRY_MAX_TOKENS} \\\\",
        f"PI1M polymers sampled & {num(config.LABEL_SAMPLE_SIZE)} \\\\",
        f"Plausible \\Tg{{}} range & $[{config.TG_MIN_CELSIUS}, {config.TG_MAX_CELSIUS}]$\\,\\unit{{\\celsius}} \\\\",
        f"Real subsets (nested, per seed) & {', '.join(num(s) for s in config.SUBSET_SIZES)}, full \\\\",
        f"MoLFormer checkpoint & \\texttt{{{config.MOLFORMER_NAME}}} \\\\",
        f"Learning rate / batch size & {config.LEARNING_RATE:g} / {config.BATCH_SIZE} \\\\",
        "Optimizer / loss & AdamW / MSE on standardized \\Tg{} \\\\",
        f"Early stopping & validation MAE after every epoch, patience {config.PATIENCE}, best weights kept \\\\",
        f"Maximum epochs, real data / stage 1 & {config.MAX_EPOCHS_REAL} / {config.MAX_EPOCHS_STAGE1} \\\\",
        f"Maximum SMILES tokens & {config.MAX_SMILES_TOKENS} \\\\",
        f"Random forest trees & {config.RF_TREES} \\\\",
        f"Morgan fingerprint bits / radius & {config.FINGERPRINT_BITS} / {config.FINGERPRINT_RADIUS} \\\\",
        f"Bootstrap resamples & {num(config.BOOTSTRAP_RESAMPLES)} \\\\",
    ]))

    # ---------------------------------------------------------------- inline numbers (LaTeX macros)
    rc = info["real_cleaning_counts"]
    rel = info["real_reliability_counts_raw"]
    multi = test["n_points"] >= 2
    macros = {
        "nSeeds": str(n_seeds), "nSeedsWord": {3: "three", 5: "five"}.get(n_seeds, str(n_seeds)),
        "nRawReal": num(rc["0_raw"]), "nUniqueReal": num(rc["5b_unique_either_key"]), "nRealMerged": num(rc["5a_unique_exact_smiles"] - rc["5b_unique_either_key"]),
        "nRealColumns": str(info["real_dataset"]["columns_in_file"]),
        "nTrainReal": num(len(train)), "nValReal": num(len(val)), "nTestReal": num(len(test)),
        "nNoRingKey": num(info["real_ring_key_not_computable"]), "pctNoRingKey": fmt(100 * info["real_ring_key_not_computable"] / len(real_all)),
        "nRelBlack": num(rel["black"]), "nRelGold": num(rel["gold"]), "nRelYellow": num(rel["yellow"]), "nRelRed": num(rel["red"]),
        "nRealMultiPoint": num((real_all["n_points"] >= 2).sum()), "nTestMultiPoint": num(multi.sum()), "nTrainMultiPoint": num((train["n_points"] >= 2).sum()),
        "nRealGrea": num(info["real_source_counts_raw"]["GREA"]),
        "simMedian": fmt(similarity.median(), 2), "simQOne": fmt(similarity.quantile(0.25), 2), "simQThree": fmt(similarity.quantile(0.75), 2),
        "pctSimHigh": fmt(100 * (similarity >= 0.85).mean(), 0), "pctSimLow": fmt(100 * (similarity < 0.55).mean(), 0),
        "nSimIdentical": num(info["test_similarity_to_train"]["n_identical_fingerprint"]),
        "pctSimIdentical": fmt(100 * info["test_similarity_to_train"]["n_identical_fingerprint"] / len(test), 0),
        "nGenRequests": num(info["generation_requests_sent"]), "nGenRaw": num(g["0_raw"]), "genPairsPerRequest": fmt(gen_raw.groupby("request_id").size().mean()),
        "nGenValidPairs": num(g["4_tg_in_range"]), "nGenUniqueExact": num(g["5a_unique_exact_smiles"]), "nGenUniqueEither": num(g["5b_unique_either_key"]),
        "nGenRingDuplicates": num(g["5a_unique_exact_smiles"] - g["5b_unique_either_key"]),
        "nGenDuplicatesRemoved": num(g["4_tg_in_range"] - g["5b_unique_either_key"]),
        "pctGenDuplicates": fmt(100 * (g["4_tg_in_range"] - g["5b_unique_either_key"]) / g["4_tg_in_range"], 0),
        "pctGenInvalid": fmt(100 * (g["1_numeric_tg"] - g["2_valid_smiles"]) / g["1_numeric_tg"]),
        "pctGenNotTwoStars": fmt(100 * (g["2_valid_smiles"] - g["3_two_stars"]) / g["2_valid_smiles"]),
        "nGenTestRemoved": num(g["removed_test_exact"] + g["removed_test_ring"]), "nGenTestExact": num(g["removed_test_exact"]), "nGenTestRing": num(g["removed_test_ring"]),
        "nGenValRemoved": num(g["removed_validation_exact"] + g["removed_validation_ring"]), "nGenClean": num(len(generated)),
        "nGenMultiFragment": num(generated["smiles"].str.contains(".", regex=False).sum()),
        "nLabRaw": num(l["0_raw"]), "nLabInvalid": num(l["1_numeric_tg"] - l["2_valid_smiles"]), "nLabRingDuplicates": num(l["5a_unique_exact_smiles"] - l["5b_unique_either_key"]),
        "nLabTestRemoved": num(l["removed_test_exact"] + l["removed_test_ring"]), "nLabValRemoved": num(l["removed_validation_exact"] + l["removed_validation_ring"]),
        "nLabClean": num(len(labeled)),
        "meanTgTrain": fmt(train["tg_celsius"].mean()), "sdTgTrain": fmt(train["tg_celsius"].std()),
        "meanTgGen": fmt(generated["tg_celsius"].mean()), "sdTgGen": fmt(generated["tg_celsius"].std()),
        "meanTgLab": fmt(labeled["tg_celsius"].mean()), "sdTgLab": fmt(labeled["tg_celsius"].std()),
        "minTgGen": fmt(generated["tg_celsius"].min(), 0), "maxTgGen": fmt(generated["tg_celsius"].max(), 0), "maxTgLab": fmt(labeled["tg_celsius"].max(), 0),
        "nDistinctLab": str(labeled["tg_celsius"].nunique()), "pctLabMultipleFive": fmt(100 * (labeled["tg_celsius"] % 5 == 0).mean()),
        "pctLabFortyFive": fmt(100 * (labeled["tg_celsius"] == 45).mean()), "pctLabMinusTwenty": fmt(100 * (labeled["tg_celsius"] == -20).mean()),
        "nLabAboveTwoFifty": num((labeled["tg_celsius"] > 250).sum()), "pctLabAboveTwoFifty": fmt(100 * (labeled["tg_celsius"] > 250).mean(), 0),
        "pctTrainAboveTwoFifty": fmt(100 * (train["tg_celsius"] > 250).mean(), 0),
        "genSmilesLength": fmt(generated["smiles"].str.len().mean(), 0), "trainSmilesLength": fmt(train["smiles"].str.len().mean(), 0),
        "zsMae": fmt(zs["mae"]), "zsCiLo": fmt(zs["mae_ci95"][0]), "zsCiHi": fmt(zs["mae_ci95"][1]), "zsRmse": fmt(zs["rmse"]),
        "zsRTwo": fmt(zs["r2"], 3), "zsSpearman": fmt(zs["spearman"], 3), "zsBias": signed(zs["mean_signed_error"]),
        "pctZsMultipleFive": fmt(zs["percent_multiples_of_5"]), "nDistinctZs": str(zs["distinct_values"]),
        "nZsTruncated": str(info["zero_shot_requests_repeated_with_higher_limit"]),
        "totalCostUSD": f"{t['cost_usd']:.2f}", "totalRequests": num(t["requests"]), "nRequestsFullPrice": str(int(t["requests_full_price"])),
        "genCostUSD": f"{cost.loc['generation', 'cost_usd']:.2f}", "labCostUSD": f"{cost.loc['labeling of PI1M', 'cost_usd']:.2f}",
        "zsCostUSD": f"{cost.loc['zero-shot on the test set', 'cost_usd']:.2f}",
        "genCostPerThousand": f"{1000 * cost.loc['generation', 'cost_usd'] / len(generated):.2f}",
        "labCostPerThousand": f"{1000 * cost.loc['labeling of PI1M', 'cost_usd'] / len(labeled):.2f}",
        "totalInputTokens": f"{t['input_tokens'] / 1e6:.1f}", "totalOutputTokens": f"{t['output_tokens'] / 1e6:.1f}",
        "gpuHours": fmt(info["training"]["gpu_hours_molformer"], 0), "nRuns": num(len(results)), "nRunsMolformer": num((results["model"] == "molformer").sum()),
        "nTestAboveThreeHundred": num(by_tg.loc["[300, 501)", "n"]),
        "noiseMaeSd": fmt(inference["mae_sd"].mean(), 2), "noisePolymerSd": fmt(inference["per_polymer_prediction_sd_mean"].mean()),
        "nfMedianSd": fmt(noise["median experimental SD"]), "nfMeanSd": fmt(noise["mean experimental SD"]),
        "nfExpectedMedian": fmt(noise["expected |difference| of two measurements, from the median SD (1.128 x SD)"]),
        "nfExpectedMean": fmt(noise["expected |difference| of two measurements, from the mean SD"]),
        "nRealSdAboveThirty": num((real_all["tg_std"].fillna(0) > 30).sum()), "nTestSdAboveThirty": num((test["tg_std"].fillna(0) > 30).sum()),
    }
    # Repetition in the generated set: the valid pairs (cleaning rules 1 to 4) before duplicates are merged
    pairs = gen_raw.assign(tg=pd.to_numeric(gen_raw["tg_celsius"], errors="coerce"), smiles=gen_raw["smiles"].map(canonical_smiles)).dropna(subset=["tg", "smiles"])
    pairs = pairs[(pairs["smiles"].str.count(r"\*") == 2) & (pairs["tg"] >= config.TG_MIN_CELSIUS) & (pairs["tg"] <= config.TG_MAX_CELSIUS)]
    assert len(pairs) == g["4_tg_in_range"]
    by_smiles = pairs.groupby("smiles")["tg"]
    macros["nGenTopRepeat"] = str(by_smiles.size().max())
    macros["genRepeatSd"] = fmt(by_smiles.std()[by_smiles.size() >= 3].median())     # polymers written at least three times
    macros["nGenRepeatedThrice"] = num((by_smiles.size() >= 3).sum())
    # New distinct polymers (either key) per request in the two rounds, requests 0 to 201 and 202 to 399
    pairs = pairs.assign(polymer=pairs["smiles"].map(polymer_id), request=pairs["request_id"].str.extract(r"(\d+)$")[0].astype(int))
    first_seen = pairs.groupby("polymer")["request"].min()
    macros["genYieldRoundOne"] = fmt((first_seen < 202).sum() / 202, 0)
    macros["genYieldRoundTwo"] = fmt((first_seen >= 202).sum() / (info["generation_requests_sent"] - 202), 0)
    # Cost of the two synthetic sets without the zero-shot query; share of the LLM-label gain that self-training gives
    macros["synCostUSD"] = f"{cost.loc['generation', 'cost_usd'] + cost.loc['labeling of PI1M', 'cost_usd']:.2f}"
    macros["pctSelfShare"] = fmt(100 * comparisons.loc["self-training vs real only, n500", "dMAE mean"]
                                 / comparisons.loc["two-stage labeled vs real only, n500", "dMAE mean"], 0)
    # Epochs after which the real-only runs stopped (the best epoch plus the patience)
    for size_tag, size, _ in SIZES:
        macros[f"epochsRunReal{size_tag}"] = fmt(results.loc[results["run"] == f"real_{size}", "epochs_run"].mean())
    # Patience ablation (pipeline/patience_ablation.py): table for the SI and macros P<patience><size>...
    with open(config.PATIENCE_ABLATION_JSONL, encoding="utf-8") as f:
        ablation_runs = [json.loads(line) for line in f]
    macros["nRunsPatience"] = str(len(ablation_runs)); macros["gpuHoursPatience"] = fmt(sum(r["seconds"] for r in ablation_runs) / 3600)
    rows = []
    for _, r in ablation.iterrows():
        if r["patience"] == ablation["patience"].min() and rows:
            rows.append("\\addlinespace")
        rows.append(f"{num(r['n_real'])} & {int(r['patience'])} & {fmt(r['best epoch, real only'])} & {pm(r['MAE real'], r['SD real'])} & "
                    f"{pm(r['MAE generated'], r['SD generated'])} & {pm(r['MAE labeled'], r['SD labeled'])} & "
                    f"{snum(r['dMAE generated'])} $\\pm$ {fmt(r['dSD generated'])} ({r['same sign generated']}; {pvalue(r['paired t p generated'])}) & "
                    f"{snum(r['dMAE labeled'])} $\\pm$ {fmt(r['dSD labeled'])} ({r['same sign labeled']}; {pvalue(r['paired t p labeled'])}) \\\\")
        size_tag = {250: "TwoFifty", 500: "FiveHundred"}[int(r["n_real"])]
        tag = {3: "Three", 5: "Five", 10: "Ten"}[int(r["patience"])]
        macros[f"pat{tag}Real{size_tag}"] = fmt(r["MAE real"]); macros[f"pat{tag}Epoch{size_tag}"] = fmt(r["best epoch, real only"], 0)
        for arm, a in [("generated", "Gen"), ("labeled", "Lab")]:
            macros[f"pat{tag}{a}{size_tag}"] = fmt(r[f"MAE {arm}"]); macros[f"dPat{tag}{a}{size_tag}"] = signed(r[f"dMAE {arm}"])
            macros[f"dPat{tag}{a}{size_tag}Sd"] = fmt(r[f"dSD {arm}"]); macros[f"dPat{tag}{a}{size_tag}Same"] = str(r[f"same sign {arm}"]).split("/")[0]
            macros[f"dPat{tag}{a}{size_tag}Pt"] = pvalue(r[f"paired t p {arm}"])
    # Share of the main-protocol (patience 3) gain that disappears against the patience-10 baseline
    for n_real, size_tag in [(250, "TwoFifty"), (500, "FiveHundred")]:
        for arm, a in [("generated", "Gen"), ("labeled", "Lab")]:
            d3 = ablation.loc[(ablation["n_real"] == n_real) & (ablation["patience"] == 3), f"dMAE {arm}"].iloc[0]
            d10 = ablation.loc[(ablation["n_real"] == n_real) & (ablation["patience"] == 10), f"dMAE {arm}"].iloc[0]
            macros[f"pctPatLost{a}{size_tag}"] = fmt(100 * (1 - d10 / d3), 0)
    write("patience_ablation.tex", tabular("@{}rrrlllll@{}",
          "$n_\\mathrm{real}$ & Patience & Best epoch & Real only & Gen.\\,$\\rightarrow$\\,real & Lab.\\,$\\rightarrow$\\,real & "
          "$\\Delta$MAE generated & $\\Delta$MAE labeled", rows))
    # Memorization of the generated set
    for split, tag in [("train", "Train"), ("validation", "Val"), ("test", "Test")]:
        for kind, k in [("exact", "Exact"), ("ring", "Ring"), ("either", "")]:
            r = memo[(memo["synthetic_set"] == "generated") & (memo["real_split"] == split) & (memo["match"] == kind)].iloc[0]
            macros[f"nGenIn{tag}{k}"] = num(r["n"])
            if kind == "either":
                macros[f"memMae{tag}"] = fmt(r["teacher_mae"]); macros[f"memMedian{tag}"] = fmt(r["teacher_median_abs_error"])
                macros[f"memSpearman{tag}"] = fmt(r["teacher_spearman"], 2); macros[f"memBias{tag}"] = signed(r["teacher_mean_signed_error"])
    macros["pctGenInTrain"] = fmt(100 * int(macros["nGenInTrain"].replace(",", "")) / len(generated))
    macros["nGenAnyReal"] = num(info["generated_matching_any_real_polymer"])
    macros["pctGenAnyReal"] = fmt(100 * info["generated_matching_any_real_polymer"] / info["generated_unique_before_removal"])
    macros["nLabInTrain"] = num(memo[(memo["synthetic_set"] == "labeled") & (memo["real_split"] == "train") & (memo["match"] == "either")]["n"].iloc[0])
    macros["nLabAnyReal"] = num(info["labeled_matching_any_real_polymer"])
    # Per-configuration metrics
    for tag, prefix, _ in FAMILIES:
        for size_tag, size, _ in SIZES:
            run = f"{prefix}_{size}"
            if run in summary.index:
                for key, name, digits in [("mae", "", 1), ("r2", "RTwo", 3)]:
                    mu, sd = mean_sd(run, key)
                    macros[f"mf{tag}{size_tag}{name}"] = fmt(mu, digits); macros[f"mf{tag}{size_tag}{name}Sd"] = fmt(sd, digits)
    for tag, run in [("SynGen", "synthetic_generated"), ("SynLab", "synthetic_labeled"), ("SelfTrain", "twostage_selftrain_n500"),
                     ("ShufGen", "twostage_shuffled_generated_n500"), ("ShufLab", "twostage_shuffled_labeled_n500"), ("SynSelfTrain", "synthetic_selftrain"),
                     ("SynShufGen", "synthetic_shuffled_generated"), ("SynShufLab", "synthetic_shuffled_labeled")]:
        for key, name, digits in [("mae", "", 1), ("r2", "RTwo", 3), ("spearman", "Spearman", 3)]:
            mu, sd = mean_sd(run, key)
            macros[f"mf{tag}{name}"] = fmt(mu, digits); macros[f"mf{tag}{name}Sd"] = fmt(sd, digits)
    for tag, run in [("RealFull", "rf_real_nfull"), ("RealFiveHundred", "rf_real_n500"), ("SynGen", "rf_synthetic_generated"), ("SynLab", "rf_synthetic_labeled"),
                     ("CatGenFull", "rf_concat_generated_nfull"), ("CatLabFull", "rf_concat_labeled_nfull"),
                     ("CatGenFiveHundred", "rf_concat_generated_n500"), ("CatLabFiveHundred", "rf_concat_labeled_n500")]:
        mu, sd = mean_sd(run)
        macros[f"rf{tag}"] = fmt(mu); macros[f"rf{tag}Sd"] = fmt(sd)
    # Seed-matched differences
    for tag, key, _ in PAIRS:
        c = comparisons.loc[key]
        macros[f"d{tag}"] = signed(c["dMAE mean"]); macros[f"d{tag}Sd"] = fmt(c["SD"]); macros[f"d{tag}Min"] = signed(c["min"]); macros[f"d{tag}Max"] = signed(c["max"])
        macros[f"d{tag}Same"] = str(c["same sign"]).split("/")[0]; macros[f"d{tag}Pt"] = pvalue(c["paired t p"]); macros[f"d{tag}Pw"] = pvalue(c["Wilcoxon p"])
    # Student against teacher
    for tag, run in [("Gen", "synthetic_generated"), ("Lab", "synthetic_labeled")]:
        part = student[student["student"] == run]
        macros[f"dZs{tag}"] = signed(part["student MAE - teacher MAE"].mean()); macros[f"dZs{tag}Min"] = signed(part["student MAE - teacher MAE"].min())
        macros[f"dZs{tag}Max"] = signed(part["student MAE - teacher MAE"].max()); macros[f"dZs{tag}Excl"] = str(int(part["CI excludes 0"].sum()))
    # Best epochs
    for tag, prefix, _ in FAMILIES[:3]:
        for size_tag, size, _ in SIZES:
            macros[f"epochs{tag}{size_tag}"] = fmt(epochs.loc[f"{prefix}_{size}"].mean())
    macros["epochsSynGen"] = fmt(epochs.loc["synthetic_generated"].mean()); macros["epochsSynLab"] = fmt(epochs.loc["synthetic_labeled"].mean())
    macros["maxEpochsRun"] = str(int(results.loc[results["model"] == "molformer", "epochs_run"].max()))
    # The panels of the predicted-against-measured figure show one seed (config.SEED); the caption gives both values
    for tag, run in [("RealFull", "real_nfull"), ("SynGen", "synthetic_generated"), ("SynLab", "synthetic_labeled")]:
        macros[f"mf{tag}SeedOne"] = fmt(mae.loc[run, config.SEED])
    macros["figureSeed"] = str(config.SEED)
    # Stratified numbers used in the text
    for col, tag in [("zero-shot teacher", "Teacher"), ("real only, full", "RealFull"), ("real only, n500", "RealFiveHundred"), ("generated only", "SynGen"),
                     ("labeled only", "SynLab"), ("random forest, real full", "Rf"), ("two-stage generated, full", "TsGenFull"), ("two-stage labeled, full", "TsLabFull")]:
        macros[f"high{tag}"] = fmt(by_tg.loc["[300, 501)", col])
        macros[f"simLow{tag}"] = fmt(by_sim.loc["[0.0, 0.4)", col]); macros[f"simMid{tag}"] = fmt(by_sim.loc["[0.4, 0.55)", col]); macros[f"simHigh{tag}"] = fmt(by_sim.loc["[0.85, 1.0)", col])
        macros[f"nf{tag}"] = fmt(noise[f"MAE on these polymers: {col}"])
        macros[f"nfOther{tag}"] = fmt(noise[[i for i in noise.index if i.startswith("MAE on the other") and i.endswith(col)][0]])
        macros[f"relGold{tag}"] = fmt(by_rel.loc["gold", col]); macros[f"relYellow{tag}"] = fmt(by_rel.loc["yellow", col]); macros[f"relBlack{tag}"] = fmt(by_rel.loc["black", col])
    macros["nSimLow"] = num(by_sim.loc["[0.0, 0.4)", "n"]); macros["nSimHigh"] = num(by_sim.loc["[0.85, 1.0)", "n"])
    macros["nRelGold"] = num(rel["gold"]); macros["nTestGold"] = num(by_rel.loc["gold", "n"]); macros["nTestYellow"] = num(by_rel.loc["yellow", "n"])
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    write("numbers.tex", "% Numbers used inline in the text, generated by make_tables.py\n" + "\n".join(lines) + "\n")
    print(len(macros), "macros")


if __name__ == "__main__":
    main()
