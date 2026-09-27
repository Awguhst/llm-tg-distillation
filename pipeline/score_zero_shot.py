"""
Score the zero-shot predictions of the teacher on the real test set.

MAE with a bootstrap confidence interval (1,000 resamples of the test polymers), RMSE, R2,
Spearman rho, the rounding of the answers, the error by band of the measured Tg, and the
API usage and cost of all three tasks. Writes results/zero_shot_metrics.json,
zero_shot_by_tg_bin.csv and api_cost.csv.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

import config
from common import claude_batches, run_info


def main():
    zs = pd.read_csv(config.ZERO_SHOT_CSV)
    assert zs["tg_pred"].notna().all(), "some test polymers have no zero-shot prediction"
    error = zs["tg_pred"] - zs["tg_true"]

    rng = np.random.RandomState(config.SEED)
    absolute = error.abs().values
    boot = [absolute[rng.randint(0, len(absolute), len(absolute))].mean() for _ in range(config.BOOTSTRAP_RESAMPLES)]
    metrics = {
        "n": len(zs), "mae": float(absolute.mean()),
        "mae_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "rmse": float(np.sqrt(mean_squared_error(zs["tg_true"], zs["tg_pred"]))),
        "r2": float(r2_score(zs["tg_true"], zs["tg_pred"])),
        "spearman": float(spearmanr(zs["tg_true"], zs["tg_pred"]).statistic),
        "mean_signed_error": float(error.mean()),
        "percent_multiples_of_5": float(100 * (zs["tg_pred"] % 5 == 0).mean()),
        "percent_multiples_of_10": float(100 * (zs["tg_pred"] % 10 == 0).mean()),
        "distinct_values": int(zs["tg_pred"].nunique()),
    }
    print(json.dumps(metrics, indent=2))

    bands = pd.cut(zs["tg_true"], config.TG_BINS, right=False)
    by_bin = pd.DataFrame({"band": bands, "abs_error": absolute, "error": error}).groupby("band", observed=True).agg(
        n=("error", "size"), mae=("abs_error", "mean"), mean_signed_error=("error", "mean")).round(1)
    print("\nError by band of the measured Tg (C):\n" + by_bin.to_string())
    by_bin.to_csv(os.path.join(config.RESULTS_DIR, "zero_shot_by_tg_bin.csv"))

    # API usage and cost of every task; every saved line was paid for, also the cut-off answers.
    rows = []
    for task, path in [("generation", config.GENERATED_RESPONSES), ("labeling of PI1M", config.LABELED_RESPONSES),
                       ("zero-shot on the test set", config.ZERO_SHOT_RESPONSES)]:
        records = claude_batches.load_all_records(path)
        rows.append({"task": task, "requests": len(records), "requests_full_price": sum(r["batch_id"] is None for r in records),
                     "input_tokens": sum(r["input_tokens"] for r in records), "output_tokens": sum(r["output_tokens"] for r in records),
                     "cost_usd": round(sum(r["cost_usd"] for r in records), 4), "dates": "/".join(sorted({r["date"] for r in records}))})
    cost = pd.DataFrame(rows)
    cost.loc[len(cost)] = ["total", cost["requests"].sum(), cost["requests_full_price"].sum(), cost["input_tokens"].sum(),
                           cost["output_tokens"].sum(), round(cost["cost_usd"].sum(), 4), ""]
    print("\n" + cost.to_string(index=False))
    cost.to_csv(config.API_COST_CSV, index=False)

    with open(config.ZERO_SHOT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    run_info.update({"zero_shot_metrics": metrics, "api_cost": cost.to_dict(orient="records"),
                     "total_api_cost_usd": float(cost["cost_usd"].iloc[-1])})


if __name__ == "__main__":
    main()
