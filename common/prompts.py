"""
The two prompts sent to Claude, and the parsing of its JSON answers.

- GENERATION_PROMPT (generate_with_claude.py): ask for ~50 polymer / Tg pairs of one polymer class.
- LABEL_PROMPT (label_pi1m_with_claude.py and zero_shot_with_claude.py): ask for the Tg of one given polymer.
"""
import json
import re

import pandas as pd

# ---------------------------------------------------------------- generation (3a)

GENERATION_PROMPT = """You are an expert polymer chemist. List {n} distinct, real {polymer_class} whose glass transition temperature (Tg) is known from the literature. {hint} {variant}

For each polymer give:
- "smiles": the repeat unit as a SMILES string with exactly two "*" atoms marking the connection points to the neighbouring repeat units (for example polyethylene is "*CC*" and polystyrene is "*CC(*)c1ccccc1"),
- "tg_celsius": its glass transition temperature in degrees Celsius, as a number.

Answer with only a JSON array and no other text:
[{{"smiles": "...", "tg_celsius": number}}, ...]"""

# After every polymer class has been combined with every focus hint once, the next
# pass adds one of these sentences so the model does not simply repeat the same list.
GENERATION_VARIANTS = [
    "",
    "Avoid the most well-known textbook examples; choose polymers a specialist would know.",
    "Choose polymers that are structurally unusual or rarely listed in tables.",
]


def parse_generation_response(text):
    """Return a list of {"smiles", "tg_celsius"} dicts, or None if the JSON is invalid."""
    # Models sometimes wrap JSON in ```json ... ``` fences; take the outermost [ ... ].
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append({"smiles": item.get("smiles"), "tg_celsius": item.get("tg_celsius")})
    return rows


# ---------------------------------------------------------------- labeling (3b and 3c)

# Kept short because it is sent about 11,000 times. The model echoes the SMILES so
# each answer can be matched to its input even if results come back out of order.
LABEL_PROMPT = (
    "Predict the glass transition temperature (Tg) of the polymer whose repeat unit is "
    "this SMILES (the two * atoms are the connection points to the neighbouring units): {smiles}\n"
    "Answer with only this JSON and no other text: {{\"smiles\": \"{smiles}\", \"tg_celsius\": number}}"
)


def parse_label_response(text):
    """
    Find the first {...} in the response and parse it. Returns a dict or None.
    SMILES with stereo bonds contain a backslash, which the model often echoes
    unescaped; that breaks the JSON parser, so we fall back to reading only the
    number after "tg_celsius" with a regular expression.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r'"tg_celsius"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if match is None:
        return None
    return {"smiles": None, "tg_celsius": float(match.group(1))}


def label_responses_to_dataframe(saved, input_smiles):
    """
    Turn saved labeling responses into a DataFrame with one row per request:
    custom_id, smiles (the SMILES we sent), echoed_smiles, tg_celsius (NaN if unparsable).
    input_smiles maps custom_id -> the SMILES that was in the prompt.
    """
    rows = []
    for custom_id, record in saved.items():
        parsed = parse_label_response(record["text"]) or {}
        rows.append({
            "custom_id": custom_id,
            "smiles": input_smiles[custom_id],
            "echoed_smiles": parsed.get("smiles"),
            "tg_celsius": parsed.get("tg_celsius"),
        })
    return pd.DataFrame(rows, columns=["custom_id", "smiles", "echoed_smiles", "tg_celsius"])
