"""
How much do MoLFormer's predictions vary between forward passes of the same weights?

The checkpoint's configuration has deterministic_eval = False: the linear-attention feature map
draws new random projections at every forward pass, in eval mode too. We left this as it is. Here
the saved stage-1 models (results/stage1_weights/, written by train_molformer.py) are predicted ten
times on the test set, and the spread of the per-polymer predictions and of the MAE is reported.
Writes results/tables/inference_noise.csv.
"""
import os

import numpy as np
import pandas as pd
import torch

import config
from common import molformer, run_info


def main():
    # results/stage1_weights/ is not in the repository (about 180 MB per model), so say why rather
    # than let torch.load raise a bare FileNotFoundError on a fresh clone.
    if not os.path.isdir(config.WEIGHTS_DIR) or not os.listdir(config.WEIGHTS_DIR):
        raise SystemExit(f"{config.WEIGHTS_DIR} is empty or missing. The stage-1 weights are not in the "
                         f"repository; run 'python pipeline/train_molformer.py' to produce them. "
                         f"This script's own output, results/tables/inference_noise.csv, is committed.")
    test = pd.read_csv(config.TEST_REAL_CSV)
    rows = []
    for name in ["synthetic_generated", "synthetic_labeled"]:
        for seed in config.SEEDS:
            saved = torch.load(os.path.join(config.WEIGHTS_DIR, f"{name}_seed{seed}.pt"), map_location="cpu")
            model = molformer.load_model()
            model.load_state_dict(saved["state_dict"])
            passes = np.array([molformer.predict(model, test["smiles"], saved["mean"], saved["std"]) for _ in range(10)])
            maes = np.abs(passes - test["tg_celsius"].values).mean(axis=1)
            rows.append({"run": name, "seed": seed, "passes": 10, "mae_mean": maes.mean(), "mae_sd": maes.std(ddof=1),
                         "mae_min": maes.min(), "mae_max": maes.max(),
                         "per_polymer_prediction_sd_mean": passes.std(axis=0, ddof=1).mean(),
                         "per_polymer_prediction_sd_max": passes.std(axis=0, ddof=1).max()})
            print(rows[-1], flush=True)
            del model
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows).round(3)
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    table.to_csv(os.path.join(config.TABLES_DIR, "inference_noise.csv"), index=False)
    run_info.update({"inference_noise": {"note": "deterministic_eval = False, left unchanged; ten forward passes of each stage-1 model on the test set",
                                         "mae_sd_between_passes_mean": float(table["mae_sd"].mean()),
                                         "per_polymer_prediction_sd_mean": float(table["per_polymer_prediction_sd_mean"].mean())}})


if __name__ == "__main__":
    main()
