"""
Zero-shot baseline (step 3c).

Ask Claude to predict Tg for every polymer in test_real.csv with the same prompt as
label_pi1m_with_claude.py. This tells us how accurate the LLM labels are, and gives a
baseline row in the results table. Predictions go to results/zero_shot_predictions.csv.
"""
import pandas as pd

import config
from common import claude_batches, prompts, run_info

test = pd.read_csv(config.TEST_REAL_CSV)
test["custom_id"] = [f"zs_{i:05d}" for i in range(len(test))]
if config.PILOT:
    test = test.head(config.PILOT_LABEL_REQUESTS)
    print(f"PILOT mode: using only the first {len(test)} test polymers")
input_smiles = dict(zip(test["custom_id"], test["smiles"]))

client = claude_batches.get_client()
saved = claude_batches.load_saved_responses(config.ZERO_SHOT_RESPONSES)
todo = test[~test["custom_id"].isin(saved.keys())]
print(f"Test polymers: {len(test)}, already saved: {len(saved)}, still to send: {len(todo)}")

if len(todo) > 0:
    requests = [
        claude_batches.make_request(row.custom_id, prompts.LABEL_PROMPT.format(smiles=row.smiles), config.LABEL_MAX_TOKENS)
        for row in todo.itertuples()
    ]
    claude_batches.estimate_and_confirm(client, requests, config.LABEL_EXPECTED_OUTPUT_TOKENS)
    claude_batches.run_requests(client, requests, config.ZERO_SHOT_RESPONSES)
    saved = claude_batches.load_saved_responses(config.ZERO_SHOT_RESPONSES)

# Answers that were cut off by the output limit (stop_reason "max_tokens") have no Tg.
# Ask them again, one by one, with a higher limit. The cut-off response stays in the JSONL
# file; load_saved_responses keeps the later line of a custom_id, so the new answer is used.
# The output_tokens check stops us from asking again if even the higher limit was not enough.
truncated = [k for k, v in saved.items() if k in input_smiles and v["stop_reason"] == "max_tokens"
             and v["output_tokens"] < config.LABEL_RETRY_MAX_TOKENS]
if truncated:
    print(f"\n{len(truncated)} answers were cut off at {config.LABEL_MAX_TOKENS} tokens; asking again with {config.LABEL_RETRY_MAX_TOKENS}.")
    requests = [
        claude_batches.make_request(k, prompts.LABEL_PROMPT.format(smiles=input_smiles[k]), config.LABEL_RETRY_MAX_TOKENS)
        for k in truncated
    ]
    claude_batches.estimate_and_confirm(client, requests, config.LABEL_EXPECTED_OUTPUT_TOKENS, batch=False)
    claude_batches.run_one_by_one(client, requests, config.ZERO_SHOT_RESPONSES, None)
    saved = claude_batches.load_saved_responses(config.ZERO_SHOT_RESPONSES)

saved = {k: v for k, v in saved.items() if k in input_smiles}
predictions = prompts.label_responses_to_dataframe(saved, input_smiles)
predictions["tg_celsius"] = pd.to_numeric(predictions["tg_celsius"], errors="coerce")
predictions = predictions.rename(columns={"tg_celsius": "tg_pred"})

# Attach the true value and save one row per test polymer (tg_pred is NaN if the answer was unusable).
predictions = test[["custom_id", "smiles", "tg_celsius"]].rename(columns={"tg_celsius": "tg_true"}).merge(
    predictions[["custom_id", "tg_pred"]], on="custom_id", how="left")
predictions.to_csv(config.ZERO_SHOT_CSV, index=False)

n_missing = predictions["tg_pred"].isna().sum()
print(f"\nSaved {len(predictions)} rows -> {config.ZERO_SHOT_CSV} ({n_missing} without a usable prediction)")
usable = predictions.dropna(subset=["tg_pred"])
print(f"Quick MAE on the {len(usable)} usable predictions: {(usable['tg_true'] - usable['tg_pred']).abs().mean():.1f} C")
print(usable.head(5).to_string())

# Requests, dates and cost count every saved line, also the cut-off answers that were asked again.
all_records = claude_batches.load_all_records(config.ZERO_SHOT_RESPONSES)
run_info.update({
    "zero_shot_requests_sent": len(all_records),
    "zero_shot_requests_repeated_with_higher_limit": len(all_records) - len(saved),
    "zero_shot_responses_without_tg": int(n_missing),
    "zero_shot_dates": sorted({r["date"] for r in all_records}),
    "zero_shot_cost_usd": round(sum(r["cost_usd"] for r in all_records), 4),
})
