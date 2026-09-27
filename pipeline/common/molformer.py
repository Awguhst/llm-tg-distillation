"""
MoLFormer fine-tuning, shared by train_molformer.py, patience_ablation.py and the smoke test.

Every run fine-tunes ibm/MoLFormer-XL-both-10pct with fixed settings (AdamW, learning rate 3e-5,
batch size 16, 128 tokens) and EARLY STOPPING ON THE VALIDATION MAE: the validation set is scored
after every epoch, the best weights are kept, and training stops after `patience` epochs without
improvement or at `max_epochs`. Targets are standardized with the mean and SD of the run's own
training data. A stage-1 model (trained on a synthetic set) can be loaded as the starting point of
stage 2; its output layer is rescaled so that it predicts exactly the same temperatures under the new
standardization.

The order in which random numbers are drawn is part of the published protocol: set_seed, then
load_model (the fresh regression head), then one permutation per epoch inside fit. Do not reorder.
"""
import random

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = None


def get_tokenizer():
    """The MoLFormer tokenizer, loaded once. It treats * as a token of its own."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.MOLFORMER_NAME, trust_remote_code=True)
    return _tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(smiles_list):
    return get_tokenizer()(list(smiles_list), padding=True, truncation=True,
                           max_length=config.MAX_SMILES_TOKENS, return_tensors="pt")


def predict(model, smiles_list, mean, std):
    """Predicted Tg in Celsius for a list of SMILES."""
    model.eval()
    smiles_list = list(smiles_list)
    out = []
    with torch.no_grad():
        for start in range(0, len(smiles_list), config.EVAL_BATCH_SIZE):
            batch = encode(smiles_list[start:start + config.EVAL_BATCH_SIZE]).to(DEVICE)
            out.append(model(**batch).logits.squeeze(-1).float().cpu())
    return torch.cat(out).numpy() * std + mean


def load_model(stage1_weights=None, new_mean=None, new_std=None):
    """
    The pretrained checkpoint with a fresh regression head, or a stage-1 model. A stage-1 model was
    trained on targets standardized with its own mean and SD; its output layer is rescaled so that
    it predicts the same temperatures under the standardization of stage 2 (new_mean, new_std).
    """
    model = AutoModelForSequenceClassification.from_pretrained(config.MOLFORMER_NAME, num_labels=1, trust_remote_code=True)
    if stage1_weights is not None:
        saved = torch.load(stage1_weights, map_location="cpu")
        model.load_state_dict(saved["state_dict"])
        with torch.no_grad():
            layer = model.classifier.out_proj
            layer.weight.mul_(saved["std"] / new_std)
            layer.bias.copy_((layer.bias * saved["std"] + saved["mean"] - new_mean) / new_std)
    return model.to(DEVICE)


def fit(model, train, val, mean, std, max_epochs, patience, test=None, verbose=True):
    """
    Fine-tune with early stopping on the validation MAE. Returns the history (one entry per epoch:
    epoch, train_mse_standardized, val_mae) and leaves the best weights in the model.

    With `test` given, the test MAE is recorded after every epoch as well (key test_mae). The test
    pass draws random attention features, so the random state is saved before it and restored
    afterwards: training is not disturbed and the history is the same with or without it. The epoch
    is always chosen on the validation MAE; the test MAE is never used for a choice.
    """
    y = torch.tensor(((train["tg_celsius"] - mean) / std).values, dtype=torch.float32)
    smiles = list(train["smiles"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = torch.nn.MSELoss()
    history, best_mae, best_state, since_best = [], float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = np.random.permutation(len(smiles))   # new shuffle every epoch
        total_loss = 0.0
        for start in range(0, len(order), config.BATCH_SIZE):
            idx = order[start:start + config.BATCH_SIZE]
            batch = encode([smiles[i] for i in idx]).to(DEVICE)
            loss = loss_function(model(**batch).logits.squeeze(-1), y[idx].to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        val_mae = float(np.abs(predict(model, val["smiles"], mean, std) - val["tg_celsius"].values).mean())
        entry = {"epoch": epoch, "train_mse_standardized": total_loss / len(smiles), "val_mae": val_mae}
        if test is not None:
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state() if DEVICE == "cuda" else None
            entry["test_mae"] = float(np.abs(predict(model, test["smiles"], mean, std) - test["tg_celsius"].values).mean())
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state)
        history.append(entry)
        if verbose:
            print(f"    epoch {epoch:>2}  train MSE = {entry['train_mse_standardized']:.4f}  validation MAE = {val_mae:.2f}", flush=True)
        if val_mae < best_mae:
            best_mae, since_best = val_mae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return history


def nested_subsets(train_real, seed):
    """
    The real training subsets of one seed: the first n rows of one seeded shuffle, so that the
    subsets are nested (250 < 500 < 1000 < full). Keys are "250", "500", "1000" and "full".
    """
    order = np.random.RandomState(seed).permutation(len(train_real))
    subsets = {str(n): train_real.iloc[order[:n]] for n in config.SUBSET_SIZES}
    subsets["full"] = train_real
    return subsets


def compute_metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "spearman": float(spearmanr(y_true, y_pred).statistic)}


def training_settings(patience, max_epochs):
    """The settings that determine a run's result, for the resume check (common/resume.py)."""
    return {"model": config.MOLFORMER_NAME, "learning_rate": config.LEARNING_RATE, "batch_size": config.BATCH_SIZE,
            "max_tokens": config.MAX_SMILES_TOKENS, "patience": patience, "max_epochs": max_epochs}
