"""
All settings for the pipeline live in this one file.
Every script imports it. Change things here, not in the scripts.
"""
import os

# ---------------------------------------------------------------- folders
# This file lives in <project>/pipeline/, so the project folder is one level up.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_DIR, ".env")   # holds ANTHROPIC_API_KEY=..., see .env.example
# data/ and results/ are relative paths: run every script from the project folder,
# for example  python pipeline/prepare_real_data.py
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")          # original downloaded files
CLAUDE_RESPONSES_DIR = os.path.join(DATA_DIR, "claude_responses")  # every raw Claude response, never deleted
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

# ---------------------------------------------------------------- input files
REAL_TG_CSV = os.path.join(RAW_DIR, "Tg_SMILES_class_pid_polyinfo_median.csv")
PI1M_CSV = os.path.join(RAW_DIR, "PI1M.csv")
# PI1M lives in the GitHub repo of Ruimin Ma. The script tries "master" then "main".
PI1M_URLS = [
    "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/master/PI1M.csv",
    "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/main/PI1M.csv",
]

# ---------------------------------------------------------------- files produced by the pipeline
TRAIN_REAL_CSV = os.path.join(DATA_DIR, "train_real.csv")
TEST_REAL_CSV = os.path.join(DATA_DIR, "test_real.csv")
PI1M_SAMPLE_CSV = os.path.join(DATA_DIR, "pi1m_sample.csv")
GENERATED_RAW_CSV = os.path.join(DATA_DIR, "generated_raw.csv")   # parsed pairs before cleaning
LABELED_RAW_CSV = os.path.join(DATA_DIR, "labeled_raw.csv")       # parsed pairs before cleaning
GENERATED_CLEAN_CSV = os.path.join(DATA_DIR, "generated_clean.csv")
LABELED_CLEAN_CSV = os.path.join(DATA_DIR, "labeled_clean.csv")
ZERO_SHOT_CSV = os.path.join(RESULTS_DIR, "zero_shot_predictions.csv")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
RUN_INFO_JSON = os.path.join(RESULTS_DIR, "run_info.json")

# raw Claude responses, one JSON line per request (see common/claude_batches.py)
GENERATED_RESPONSES = os.path.join(CLAUDE_RESPONSES_DIR, "generated.jsonl")
LABELED_RESPONSES = os.path.join(CLAUDE_RESPONSES_DIR, "labeled.jsonl")
ZERO_SHOT_RESPONSES = os.path.join(CLAUDE_RESPONSES_DIR, "zero_shot.jsonl")

# ---------------------------------------------------------------- general
SEED = 42            # data split, PI1M sample, generation sampling: fixed once
TEST_FRACTION = 0.2
# Training is repeated with these seeds (model initialisation, shuffling, the 500-polymer
# subset, the random forest). The first one equals SEED so the original run is reused.
SEEDS = [42, 43, 44, 45, 46]

# ---------------------------------------------------------------- Claude API
LLM_MODEL = "claude-sonnet-5"
# Sonnet 5 does not accept temperature/top_p/top_k (the API returns 400),
# so no sampling settings appear anywhere. Diversity in generation comes from the prompt.
# Sonnet 5 turns on "adaptive thinking" by default; we switch it off explicitly.
THINKING = {"type": "disabled"}

# PILOT = True runs only a small sample of requests so a bug cannot cost much.
# False is the full run of the paper; switch to True first when trying the scripts with a new key.
PILOT = False
PILOT_GENERATION_REQUESTS = 2    # ~100 raw pairs
PILOT_LABEL_REQUESTS = 100

# Hard stop: an API script refuses to submit a batch whose estimate would push
# the total spend (all scripts, all batches so far) above this number.
BUDGET_LIMIT_USD = 18.0

# The Message Batches API costs half price but can sit in a queue for hours.
# False sends the requests one by one at full price (fine for the pilot, too
# expensive for the full run).
USE_BATCH_API = True

# Normal Sonnet 5 prices per million tokens; batch requests pay half.
PRICE_INPUT_PER_MTOK = 2.0
PRICE_OUTPUT_PER_MTOK = 10.0
BATCH_DISCOUNT = 0.5

# --- 3a: fully generated dataset
PAIRS_PER_REQUEST = 50
GENERATION_ROUND_REQUESTS = 200      # requests per round (~10,000 raw pairs)
GENERATION_MAX_REQUESTS = 400        # hard cap over all rounds
GENERATION_TARGET = 10000            # stop when this many clean pairs exist
GENERATION_MAX_TOKENS = 8000         # 50 pairs of JSON is ~3,000 tokens; leave headroom
GENERATION_EXPECTED_OUTPUT_TOKENS = 3000   # used only for the cost estimate

# Each request names one class and one "focus" hint, rotating through both lists.
POLYMER_CLASSES = [
    "polyimides", "polyesters", "polyamides", "polysiloxanes", "polyvinyls",
    "polyacrylates and polymethacrylates", "polycarbonates", "polyethers",
    "polyurethanes", "polysulfones", "polyolefins", "polystyrenes and styrene copolymers",
    "polyketones", "polyanhydrides", "polyphosphazenes", "epoxy and phenolic polymers",
    "polyoxazolines and polyoxazoles", "conjugated polymers (polythiophenes, polyanilines, etc.)",
    "fluoropolymers", "polydienes and rubbers",
]
FOCUS_HINTS = [
    "Cover a wide range of Tg values, from very low to very high.",
    "Prefer polymers with aromatic or rigid backbones.",
    "Prefer polymers with long flexible aliphatic segments.",
    "Prefer polymers with bulky side groups.",
    "Prefer polymers containing heteroatoms in the backbone (N, O, S, Si, P).",
    "Prefer less common or specialty polymers rather than textbook examples.",
    "Prefer polymers with polar or hydrogen-bonding side groups.",
    "Prefer polymers reported in the polymer-science literature after the year 2000.",
    "Prefer halogenated polymers (F, Cl, Br).",
    "Prefer polymers with cyclic units (rings) in the backbone.",
]

# --- 3b / 3c: labeling
LABEL_SAMPLE_SIZE = 10000
LABEL_MAX_TOKENS = 200
# The answer echoes the SMILES before the number, so a very long SMILES can use up all
# LABEL_MAX_TOKENS. The zero-shot script asks such cut-off answers again with this limit.
LABEL_RETRY_MAX_TOKENS = 600
LABEL_EXPECTED_OUTPUT_TOKENS = 50    # used only for the cost estimate

# ---------------------------------------------------------------- cleaning
TG_MIN_CELSIUS = -150
TG_MAX_CELSIUS = 500
# Cleaning rule 3 counts * characters. Setting this to True also drops SMILES made of more
# than one fragment (a "." in the canonical SMILES), which the star count cannot catch.
# False reproduces the published run (36 such rows stayed in the generated set).
REQUIRE_SINGLE_FRAGMENT = False

# ---------------------------------------------------------------- training
SMALL_REAL_SUBSET = 500
MOLFORMER_NAME = "ibm/MoLFormer-XL-both-10pct"
EPOCHS = 5
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
MAX_SMILES_TOKENS = 128   # SMILES longer than this are truncated by the tokenizer
RF_TREES = 500
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2
