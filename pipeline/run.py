"""
Run the study, or part of it, in dependency order.

    python pipeline/run.py                  every step whose outputs are missing or out of date
    python pipeline/run.py analyze figures  only these steps (and nothing else)
    python pipeline/run.py --list           the steps, their scripts, and whether each is up to date
    python pipeline/run.py --dry-run        show what would run, run nothing
    python pipeline/run.py --force analyze  run even if up to date

A step is up to date when every output exists and is newer than every input. The training steps
(train_molformer, train_random_forest, inference_noise, patience_ablation) are the exception: they
run only when an output is MISSING, never because an input file is merely newer, so that regenerating
a data file with identical content cannot trigger a 29-GPU-hour rerun. The training scripts are
themselves resumable and skip every finished run (common/resume.py). To retrain after a real change
of the data, delete results/runs/ and results/predictions/ first -- and, because they are derived
from the same weights, results/patience_ablation.jsonl and results/tables/inference_noise.csv too,
or those two will keep describing the previous training runs.

The "api" step (the three Claude scripts) is never run unless named explicitly: every raw response is
in data/claude_responses/, and the scripts only send requests whose id has no saved response.
"""
import argparse
import os
import subprocess
import sys
import time

import config

PIPELINE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(config.PROJECT_DIR, "report")
TESTS = os.path.join(config.PROJECT_DIR, "tests")


def p(*parts):
    return os.path.join(*parts)


# name, scripts (run in this order), inputs, outputs (one representative file is enough), only_if_missing
STEPS = [
    ("download", [p(PIPELINE, "download_data.py")],
     [], [config.REAL_TG_CSV, config.PI1M_CSV], True),
    ("prepare", [p(PIPELINE, "prepare_real_data.py")],
     [config.REAL_TG_CSV], [config.REAL_CLEAN_CSV, config.TRAIN_REAL_CSV, config.VAL_REAL_CSV, config.TEST_REAL_CSV], False),
    ("api", [p(PIPELINE, "generate_with_claude.py"), p(PIPELINE, "label_pi1m_with_claude.py"), p(PIPELINE, "zero_shot_with_claude.py")],
     [config.TEST_REAL_CSV], [config.GENERATED_RAW_CSV, config.LABELED_RAW_CSV, config.ZERO_SHOT_CSV], True),
    ("clean", [p(PIPELINE, "clean_synthetic_data.py")],
     [config.GENERATED_RAW_CSV, config.LABELED_RAW_CSV, config.TEST_REAL_CSV, config.VAL_REAL_CSV],
     [config.GENERATED_CLEAN_CSV, config.LABELED_CLEAN_CSV], False),
    ("describe", [p(PIPELINE, "describe_split.py")],
     [config.TRAIN_REAL_CSV, config.TEST_REAL_CSV], [config.TEST_SIMILARITY_CSV], False),
    ("memorization", [p(PIPELINE, "memorization.py")],
     [config.GENERATED_RAW_CSV, config.LABELED_RAW_CSV, config.TRAIN_REAL_CSV, config.VAL_REAL_CSV, config.TEST_REAL_CSV],
     [config.MEMORIZATION_CSV], False),
    ("score_zero_shot", [p(PIPELINE, "score_zero_shot.py")],
     [config.ZERO_SHOT_CSV], [config.ZERO_SHOT_METRICS_JSON, config.API_COST_CSV], False),
    ("train_molformer", [p(PIPELINE, "train_molformer.py")],
     [config.GENERATED_CLEAN_CSV, config.LABELED_CLEAN_CSV, config.TRAIN_REAL_CSV],
     [p(config.RUNS_DIR, "twostage_shuffled_labeled_n500_seed46.json")], True),
    ("train_random_forest", [p(PIPELINE, "train_random_forest.py")],
     [config.GENERATED_CLEAN_CSV, config.LABELED_CLEAN_CSV, config.TRAIN_REAL_CSV],
     [p(config.RUNS_DIR, "rf_concat_labeled_n500_seed46.json")], True),
    # the next two need results/stage1_weights/ (written by train_molformer, not in the repository)
    ("inference_noise", [p(PIPELINE, "inference_noise.py")],
     [p(config.RUNS_DIR, "twostage_shuffled_labeled_n500_seed46.json")], [p(config.TABLES_DIR, "inference_noise.csv")], True),
    ("patience_ablation", [p(PIPELINE, "patience_ablation.py")],
     [p(config.RUNS_DIR, "twostage_shuffled_labeled_n500_seed46.json")], [config.PATIENCE_ABLATION_JSONL], True),
    ("analyze", [p(PIPELINE, "analyze_results.py")],
     [p(config.RUNS_DIR, "twostage_shuffled_labeled_n500_seed46.json"), p(config.RUNS_DIR, "rf_concat_labeled_n500_seed46.json"),
      config.PATIENCE_ABLATION_JSONL, config.TEST_SIMILARITY_CSV, config.ZERO_SHOT_CSV],
     [config.RESULTS_CSV, p(config.TABLES_DIR, "summary.csv"), p(config.TABLES_DIR, "patience_ablation.csv"),
      p(config.TABLES_DIR, "mae_by_similarity_bin.csv")], False),
    # both report scripts also read the datasets themselves (dataset_stats, the Tg histograms, the
    # repetition panel), so those files belong in the inputs or a re-clean would leave them stale
    ("tables", [p(REPORT, "make_tables.py")],
     [config.RESULTS_CSV, config.MEMORIZATION_CSV, config.ZERO_SHOT_METRICS_JSON, config.API_COST_CSV, config.TEST_SIMILARITY_CSV,
      p(config.TABLES_DIR, "inference_noise.csv"), config.PATIENCE_ABLATION_JSONL,
      config.TRAIN_REAL_CSV, config.GENERATED_CLEAN_CSV, config.LABELED_CLEAN_CSV, config.GENERATED_RAW_CSV],
     [p(config.LATEX_TABLES_DIR, "numbers.tex")], False),
    ("figures", [p(REPORT, "make_figures.py")],
     [config.RESULTS_CSV, config.ZERO_SHOT_METRICS_JSON, p(config.TABLES_DIR, "patience_ablation.csv"),
      p(config.TABLES_DIR, "mae_by_similarity_bin.csv"),
      config.TRAIN_REAL_CSV, config.GENERATED_CLEAN_CSV, config.LABELED_CLEAN_CSV, config.GENERATED_RAW_CSV],
     [p(config.FIGURES_DIR, "fig_learning_curve.pdf")], False),
]
DEFAULT = [name for name, *_ in STEPS if name != "api"]


def status(inputs, outputs, only_if_missing):
    """'up to date', 'missing outputs', 'inputs newer' or 'missing inputs'."""
    if any(not os.path.exists(f) for f in inputs):
        return "missing inputs"
    if any(not os.path.exists(f) for f in outputs):
        return "missing outputs"
    if not only_if_missing and inputs and max(os.path.getmtime(f) for f in inputs) > min(os.path.getmtime(f) for f in outputs):
        return "inputs newer"
    return "up to date"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", help=f"steps to run (default: all but api): {', '.join(n for n, *_ in STEPS)}")
    parser.add_argument("--list", action="store_true", help="show the steps and their state")
    parser.add_argument("--dry-run", action="store_true", help="show what would run")
    parser.add_argument("--force", action="store_true", help="run the named steps even when up to date")
    args = parser.parse_args()

    steps = {name: (scripts, inputs, outputs, only_missing) for name, scripts, inputs, outputs, only_missing in STEPS}
    for name in args.steps:
        if name not in steps:
            sys.exit(f"unknown step {name!r}; choose from: {', '.join(steps)}")
    wanted = args.steps or DEFAULT

    if args.list:
        for name, (scripts, inputs, outputs, only_missing) in steps.items():
            print(f"{name:<20} {status(inputs, outputs, only_missing):<16} "
                  + ", ".join(os.path.relpath(s, config.PROJECT_DIR) for s in scripts))
        return

    for name in wanted:
        scripts, inputs, outputs, only_missing = steps[name]
        state = status(inputs, outputs, only_missing)
        if state == "missing inputs":
            # In a real run an earlier step creates these before we get here. A dry run executes
            # nothing, so they are legitimately absent: report and carry on, or the preview would
            # stop at the first step whose inputs a clone does not ship (e.g. prepare needs data/raw/).
            if args.dry_run:
                print(f"{name:<20} inputs not there yet; an earlier step would create them")
                continue
            sys.exit(f"{name}: an input is missing; run the earlier steps first (python pipeline/run.py --list)")
        if state == "up to date" and not args.force:
            print(f"{name:<20} up to date, skipped")
            continue
        for script in scripts:
            rel = os.path.relpath(script, config.PROJECT_DIR)
            print(f"{name:<20} {state}: python {rel}", flush=True)
            if args.dry_run:
                continue
            start = time.time()
            subprocess.run([sys.executable, script], check=True)
            print(f"{name:<20} done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
