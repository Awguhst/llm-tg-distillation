"""
The reproducibility log results/run_info.json. Every script adds its own entries.
"""
import json
import os

import config


def update(new_entries):
    """Merge a dict into results/run_info.json (created if missing)."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    info = {}
    if os.path.exists(config.RUN_INFO_JSON):
        with open(config.RUN_INFO_JSON, encoding="utf-8") as f:
            info = json.load(f)
    info.update(new_entries)
    with open(config.RUN_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
