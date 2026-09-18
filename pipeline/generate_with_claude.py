"""
The fully generated dataset (step 3a).

Ask Claude for polymers together with their Tg, about 50 pairs per request, as JSON.
Work in rounds: send a round of requests, apply the cleaning rules from common/cleaning.py,
count the clean pairs, and send another round if we are still below the target.
Stop at the target or at the hard cap on the number of requests.

Every raw response is saved in data/claude_responses/generated.jsonl. Requests that
already have a saved response are never sent again. The final "keep exactly 10,000"
sampling happens in clean_synthetic_data.py; this script saves all parsed pairs to
data/generated_raw.csv.
"""
import pandas as pd

import config
from common import claude_batches, prompts, run_info
from common.cleaning import clean_dataset, print_counts


def request_settings(i):
    """Which polymer class, focus hint, and variant sentence request number i uses."""
    n_classes = len(config.POLYMER_CLASSES)
    n_hints = len(config.FOCUS_HINTS)
    polymer_class = config.POLYMER_CLASSES[i % n_classes]
    hint = config.FOCUS_HINTS[(i // n_classes) % n_hints]
    variant = prompts.GENERATION_VARIANTS[(i // (n_classes * n_hints)) % len(prompts.GENERATION_VARIANTS)]
    return polymer_class, hint, variant


def build_request(i):
    custom_id = f"gen_{i:04d}"
    polymer_class, hint, variant = request_settings(i)
    prompt = prompts.GENERATION_PROMPT.format(
        n=config.PAIRS_PER_REQUEST, polymer_class=polymer_class, hint=hint, variant=variant)
    request = claude_batches.make_request(custom_id, prompt, config.GENERATION_MAX_TOKENS)
    metadata = {"polymer_class": polymer_class, "hint": hint, "variant": variant}
    return request, metadata


def parsed_pairs(saved):
    """All pairs from all saved responses as a DataFrame, plus the number of unparsable responses."""
    rows = []
    n_invalid = 0
    for custom_id, record in saved.items():
        items = prompts.parse_generation_response(record["text"])
        if items is None:
            n_invalid += 1
            continue
        for item in items:
            item["request_id"] = custom_id
            item["polymer_class"] = record["polymer_class"]
            rows.append(item)
    columns = ["request_id", "polymer_class", "smiles", "tg_celsius"]
    return pd.DataFrame(rows, columns=columns), n_invalid


# ---------------------------------------------------------------- main loop

client = claude_batches.get_client()
test_smiles = set(pd.read_csv(config.TEST_REAL_CSV)["smiles"])
round_size = config.PILOT_GENERATION_REQUESTS if config.PILOT else config.GENERATION_ROUND_REQUESTS
max_requests = config.PILOT_GENERATION_REQUESTS if config.PILOT else config.GENERATION_MAX_REQUESTS
stop_reason = None

while True:
    saved = claude_batches.load_saved_responses(config.GENERATED_RESPONSES)
    raw, n_invalid = parsed_pairs(saved)
    clean, counts = clean_dataset(raw, test_smiles) if len(raw) > 0 else (raw, {})
    print(f"\nSaved responses: {len(saved)} ({n_invalid} with invalid JSON) -> "
          f"{len(raw)} raw pairs -> {len(clean)} clean unique pairs (target {config.GENERATION_TARGET})")

    if len(clean) >= config.GENERATION_TARGET:
        stop_reason = "target reached"
        break

    # Request ids that still need a response, in order, limited to the cap.
    todo = [i for i in range(max_requests) if f"gen_{i:04d}" not in saved]
    if not todo:
        stop_reason = "pilot finished" if config.PILOT else f"hard cap of {max_requests} requests reached"
        break
    todo = todo[:round_size]

    requests, metadata = [], {}
    for i in todo:
        request, meta = build_request(i)
        requests.append(request)
        metadata[request["custom_id"]] = meta

    claude_batches.estimate_and_confirm(client, requests, config.GENERATION_EXPECTED_OUTPUT_TOKENS)
    claude_batches.run_requests(client, requests, config.GENERATED_RESPONSES, metadata)

# ---------------------------------------------------------------- summary

if counts:
    print_counts(counts, "generated (all rounds so far)")
raw.to_csv(config.GENERATED_RAW_CSV, index=False)
print(f"\nStopped because: {stop_reason}")
print(f"Saved {len(raw)} raw pairs -> {config.GENERATED_RAW_CSV}")
if len(clean) < config.GENERATION_TARGET and not config.PILOT:
    print(f"Only {len(clean)} clean pairs. The model may be repeating polymers. "
          f"Raise GENERATION_MAX_REQUESTS in config.py only if the budget allows it.")

# Show a few example responses so they can be checked by eye.
for record in list(saved.values())[:2]:
    print(f"\n--- example response {record['custom_id']} ({record['polymer_class']}) ---")
    print(record["text"][:600])

run_info.update({
    "generation_model": config.LLM_MODEL,
    "generation_thinking": config.THINKING,
    "generation_prompt_template": prompts.GENERATION_PROMPT,
    "generation_variants": prompts.GENERATION_VARIANTS,
    "generation_requests_sent": len(saved),
    "generation_invalid_json_responses": n_invalid,
    "generation_raw_pairs": int(len(raw)),
    "generation_clean_pairs_before_sampling": int(len(clean)),
    "generation_stop_reason": stop_reason,
    "generation_dates": sorted({r["date"] for r in saved.values()}),
    "generation_cost_usd": round(sum(r["cost_usd"] for r in saved.values()), 4),
})
