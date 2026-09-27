"""
Resume safety for the training scripts.

Every training script skips a run whose output file already exists. That is only safe when the
run was made with the same settings, so every record carries a short hash of its TRAINING
hyperparameters. On resume the stored hash must equal the hash of the current settings; otherwise
the script stops with a clear message instead of silently mixing two protocols.

What the hash covers is exactly the dicts below: for MoLFormer the checkpoint, learning rate,
batch size, token limit, patience and epoch cap; for the random forest the number of trees and the
fingerprint. It deliberately does NOT cover the data or the split -- the seed, which synthetic set a
run started from, the cleaning rules, the subset sizes. Change any of those and the finished runs
still look current, so delete results/runs/ and results/predictions/ yourself (see pipeline/run.py).

The records of the published run were written before the hash existed. A record without a hash
is taken to have been produced with the PUBLISHED settings below, which are the defaults of
config.py and of the scripts, so nothing needs to be rerun.
"""
import hashlib
import json

# The settings of the published run. Kept here as literal values, not read from config.py, so
# that an edited config.py cannot make an old record look like a new one.
PUBLISHED_MOLFORMER = {"model": "ibm/MoLFormer-XL-both-10pct", "learning_rate": 3e-05, "batch_size": 16,
                       "max_tokens": 128, "patience": 3}
PUBLISHED_MAX_EPOCHS = {"real": 30, "stage1": 10}          # runs containing real data / stage 1 on synthetic data only
PUBLISHED_ABLATION = dict(PUBLISHED_MOLFORMER, patience=10, max_epochs=40)
PUBLISHED_RANDOM_FOREST = {"trees": 500, "fingerprint_bits": 2048, "fingerprint_radius": 2}


def settings_hash(settings):
    """Twelve hex characters of the SHA-256 of the settings, serialized with sorted keys."""
    text = json.dumps(settings, sort_keys=True, default=float)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def check_record(record, current, published, name):
    """
    A record of a finished run exists. Return True when it may be skipped, i.e. its settings hash
    (or, for a record without one, the published settings) equals the current one. Otherwise stop.
    """
    current_hash, published_hash = settings_hash(current), settings_hash(published)
    stored = record.get("settings_hash", published_hash)
    if stored == current_hash:
        return True
    raise SystemExit(
        f"{name}: a finished run exists with settings hash {stored}, but the current settings hash to "
        f"{current_hash} ({current}). Delete or move that output before rerunning with new settings.")
