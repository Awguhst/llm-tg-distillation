"""
The LLM-labeled dataset (step 3b).

Sample 10,000 polymers from PI1M (seed 42) and ask Claude, one polymer per request,
to predict the Tg. Raw responses go to data/claude_responses/labeled.jsonl; the parsed
table goes to data/labeled_raw.csv (cleaning happens in clean_synthetic_data.py).
"""
import os

import pandas as pd

import config
from common import claude_batches, prompts, run_info

# ---------------------------------------------------------------- 1. fixed random sample of PI1M
if os.path.exists(config.PI1M_SAMPLE_CSV):
    sample = pd.read_csv(config.PI1M_SAMPLE_CSV)
else:
    pi1m = pd.read_csv(config.PI1M_CSV)
    smiles = pi1m["SMILES"].drop_duplicates()
    sample = smiles.sample(n=config.LABEL_SAMPLE_SIZE, random_state=config.SEED).reset_index(drop=True)
    sample = pd.DataFrame({"custom_id": [f"lab_{i:05d}" for i in range(len(sample))], "smiles": sample})
    sample.to_csv(config.PI1M_SAMPLE_CSV, index=False)
print(f"PI1M sample: {len(sample)} polymers ({config.PI1M_SAMPLE_CSV})")

if config.PILOT:
    sample = sample.head(config.PILOT_LABEL_REQUESTS)
    print(f"PILOT mode: using only the first {len(sample)} polymers")

input_smiles = dict(zip(sample["custom_id"], sample["smiles"]))

# ---------------------------------------------------------------- 2. send what is not yet saved
client = claude_batches.get_client()
saved = claude_batches.load_saved_responses(config.LABELED_RESPONSES)
todo = sample[~sample["custom_id"].isin(saved.keys())]
print(f"Already saved: {len(saved)}, still to send: {len(todo)}")

if len(todo) > 0:
    requests = [
        claude_batches.make_request(row.custom_id, prompts.LABEL_PROMPT.format(smiles=row.smiles), config.LABEL_MAX_TOKENS)
        for row in todo.itertuples()
    ]
    claude_batches.estimate_and_confirm(client, requests, config.LABEL_EXPECTED_OUTPUT_TOKENS)
    claude_batches.run_requests(client, requests, config.LABELED_RESPONSES)
    saved = claude_batches.load_saved_responses(config.LABELED_RESPONSES)

# ---------------------------------------------------------------- 3. parse into a table
# Only keep responses for polymers in the current sample (matters in PILOT mode).
saved = {k: v for k, v in saved.items() if k in input_smiles}
labeled = prompts.label_responses_to_dataframe(saved, input_smiles)
labeled.to_csv(config.LABELED_RAW_CSV, index=False)

n_missing = labeled["tg_celsius"].isna().sum()
n_echo_mismatch = (labeled["echoed_smiles"] != labeled["smiles"]).sum()
print(f"\nParsed {len(labeled)} responses -> {config.LABELED_RAW_CSV}")
print(f"  responses without a usable Tg: {n_missing}")
print(f"  responses whose echoed SMILES differs from the input: {n_echo_mismatch} "
      f"(we always keep the input SMILES, so this is only informative)")
print(labeled.head(5).to_string())

for record in list(saved.values())[:3]:
    print(f"\n--- example response {record['custom_id']} ---")
    print(record["text"][:300])

run_info.update({
    "labeling_model": config.LLM_MODEL,
    "labeling_thinking": config.THINKING,
    "labeling_prompt_template": prompts.LABEL_PROMPT,
    "labeling_requests_sent": len(saved),
    "labeling_responses_without_tg": int(n_missing),
    "labeling_dates": sorted({r["date"] for r in saved.values()}),
    "labeling_cost_usd": round(sum(r["cost_usd"] for r in saved.values()), 4),
})
