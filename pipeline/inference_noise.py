"""
How much do MoLFormer's predictions vary between forward passes of the same weights?

The checkpoint's configuration has deterministic_eval = False: the linear-attention feature map
draws new random projections at every forward pass, in eval mode too. We left this as it is. Here the saved stage-1 models are predicted ten times on the test set, and the
spread of the per-polymer predictions and of the MAE is reported. Writes results/tables/inference_noise.csv.
"""
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from common import run_info

test = pd.read_csv(config.TEST_REAL_CSV)
tokenizer = AutoTokenizer.from_pretrained(config.MOLFORMER_NAME, trust_remote_code=True)
rows = []
for name in ["synthetic_generated", "synthetic_labeled"]:
    for seed in config.SEEDS:
        saved = torch.load(os.path.join(config.WEIGHTS_DIR, f"{name}_seed{seed}.pt"), map_location="cpu")
        model = AutoModelForSequenceClassification.from_pretrained(config.MOLFORMER_NAME, num_labels=1, trust_remote_code=True)
        model.load_state_dict(saved["state_dict"])
        model = model.to("cuda").eval()
        passes = []
        with torch.no_grad():
            for _ in range(10):
                out = []
                for start in range(0, len(test), config.EVAL_BATCH_SIZE):
                    batch = tokenizer(list(test["smiles"][start:start + config.EVAL_BATCH_SIZE]), padding=True, truncation=True,
                                      max_length=config.MAX_SMILES_TOKENS, return_tensors="pt").to("cuda")
                    out.append(model(**batch).logits.squeeze(-1).cpu())
                passes.append(torch.cat(out).numpy() * saved["std"] + saved["mean"])
        passes = np.array(passes)
        maes = np.abs(passes - test["tg_celsius"].values).mean(axis=1)
        rows.append({"run": name, "seed": seed, "passes": 10, "mae_mean": maes.mean(), "mae_sd": maes.std(ddof=1),
                     "mae_min": maes.min(), "mae_max": maes.max(),
                     "per_polymer_prediction_sd_mean": passes.std(axis=0, ddof=1).mean(),
                     "per_polymer_prediction_sd_max": passes.std(axis=0, ddof=1).max()})
        print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache()
table = pd.DataFrame(rows).round(3)
table.to_csv(os.path.join(config.RESULTS_DIR, "tables", "inference_noise.csv"), index=False)
run_info.update({"inference_noise": {"note": "deterministic_eval = False, left unchanged; ten forward passes of each stage-1 model on the test set",
                                     "mae_sd_between_passes_mean": float(table["mae_sd"].mean()),
                                     "per_polymer_prediction_sd_mean": float(table["per_polymer_prediction_sd_mean"].mean())}})
