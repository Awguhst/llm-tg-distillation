"""
Sending requests to Claude and saving the raw responses.
Used by generate_with_claude.py, label_pi1m_with_claude.py and zero_shot_with_claude.py.

Two ways to send (config.USE_BATCH_API):
- Message Batches API: half price, but the batch can wait in a queue for hours.
- One request at a time: full price, results arrive immediately.

Rules that keep the budget safe:
- One request = one line in a JSONL file under data/claude_responses/. A request whose
  custom_id already has a line is never sent again. (One exception: zero_shot_with_claude.py
  asks again, once, for answers that were cut off by the output limit; both lines are kept.)
- A batch id is written to a ".pending" file before we wait for it, so a crash
  while waiting can resume the same batch instead of paying for a new one.
- Before sending, we print the request count and an estimated cost and wait
  for the user to type "yes". Passing "--yes" on the command line skips the prompt.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

import config


def get_client():
    """
    The SDK reads ANTHROPIC_API_KEY from the environment. load_dotenv() first copies the
    variables from the .env file in the project folder into the environment (an already
    set environment variable wins). The key is never written in code.
    """
    load_dotenv(config.ENV_FILE)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in .env (see .env.example) or in your environment.")
    # max_retries: the SDK waits and retries on rate limits (HTTP 429) and server errors.
    return anthropic.Anthropic(max_retries=5)


def make_request(custom_id, prompt, max_tokens):
    """Build one request. Thinking is disabled; no sampling parameters (see config.py)."""
    params = MessageCreateParamsNonStreaming(
        model=config.LLM_MODEL,
        max_tokens=max_tokens,
        thinking=config.THINKING,
        messages=[{"role": "user", "content": prompt}],
    )
    return Request(custom_id=custom_id, params=params)


# ---------------------------------------------------------------- saved responses

def load_saved_responses(jsonl_path):
    """Return {custom_id: record} for every response already on disk."""
    saved = {}
    if not os.path.exists(jsonl_path):
        return saved
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            saved[record["custom_id"]] = record
    return saved


def load_all_records(jsonl_path):
    """
    Every saved line, including an earlier cut-off answer that was asked again later.
    Used for counting requests and cost, because every request was paid for.
    """
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_responses(jsonl_path, records):
    """Append records to the JSONL file (one JSON object per line)."""
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def make_record(custom_id, message, batch_id, extra_fields):
    """One saved line: ids, date, token usage, cost, and the raw text of the answer."""
    text = "".join(block.text for block in message.content if block.type == "text")
    usage = message.usage
    record = {
        "custom_id": custom_id,
        "batch_id": batch_id,            # None when sent one by one
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "model": config.LLM_MODEL,
        "stop_reason": message.stop_reason,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost_usd(usage.input_tokens, usage.output_tokens, batch=batch_id is not None),
        "text": text,
    }
    if extra_fields and custom_id in extra_fields:
        record.update(extra_fields[custom_id])
    return record


def total_cost_so_far():
    """Sum the cost of every saved response across all three response files."""
    total = 0.0
    for path in [config.GENERATED_RESPONSES, config.LABELED_RESPONSES, config.ZERO_SHOT_RESPONSES]:
        for record in load_all_records(path):
            total += record["cost_usd"]
    return total


# ---------------------------------------------------------------- cost estimate + confirmation

def cost_usd(input_tokens, output_tokens, batch):
    cost = (input_tokens * config.PRICE_INPUT_PER_MTOK
            + output_tokens * config.PRICE_OUTPUT_PER_MTOK) / 1_000_000
    return cost * config.BATCH_DISCOUNT if batch else cost


def estimate_and_confirm(client, requests, expected_output_tokens, batch=None):
    """
    Print request count and cost estimate, then require "yes" (or --yes) to continue.
    batch says how the requests will be sent; None means "as config.USE_BATCH_API says".
    """
    batch = config.USE_BATCH_API if batch is None else batch
    # Count the input tokens of the first request; all requests in a script have
    # nearly the same length, so one count is a good estimate for all of them.
    first = requests[0]["params"]
    count = client.messages.count_tokens(model=config.LLM_MODEL, messages=first["messages"])
    n = len(requests)
    est_input = count.input_tokens * n
    est_output = expected_output_tokens * n
    estimate = cost_usd(est_input, est_output, batch=batch)
    spent = total_cost_so_far()

    how = "Batches API, half price" if batch else "one by one, full price"
    print(f"\nRequests to send: {n} ({how})")
    print(f"Input tokens per request (counted): {count.input_tokens}")
    print(f"Estimated tokens: {est_input:,} input + {est_output:,} output")
    print(f"Estimated cost: ${estimate:.3f}")
    print(f"Spent so far (all saved responses): ${spent:.3f}")
    print(f"Budget limit: ${config.BUDGET_LIMIT_USD:.2f}")

    if spent + estimate > config.BUDGET_LIMIT_USD:
        sys.exit("This would exceed BUDGET_LIMIT_USD in config.py. Not sending.")

    if "--yes" in sys.argv:
        print("(--yes given on the command line, skipping the prompt)")
        return
    answer = input("Type yes to send: ")
    if answer.strip().lower() != "yes":
        sys.exit("Aborted by user. Nothing was sent.")


# ---------------------------------------------------------------- sending

def run_requests(client, requests, jsonl_path, extra_fields=None):
    """Send the requests the way config.USE_BATCH_API says, save the responses, and report cost."""
    if config.USE_BATCH_API:
        records, n_failed = run_batch(client, requests, jsonl_path, extra_fields)
    else:
        records, n_failed = run_one_by_one(client, requests, jsonl_path, extra_fields)

    total_in = sum(r["input_tokens"] for r in records)
    total_out = sum(r["output_tokens"] for r in records)
    print(f"\nDone: {len(records)} succeeded, {n_failed} failed "
          f"(failed ones will be resent next time the script runs).")
    print(f"Actual tokens: {total_in:,} input + {total_out:,} output")
    print(f"Actual cost: ${sum(r['cost_usd'] for r in records):.3f}")
    print(f"Total spent so far: ${total_cost_so_far():.3f}")
    return records


def run_one_by_one(client, requests, jsonl_path, extra_fields):
    """Send each request directly and save its response right away."""
    records = []
    n_failed = 0
    for i, request in enumerate(requests, start=1):
        try:
            message = client.messages.create(**request["params"])
        except anthropic.APIError as error:
            n_failed += 1
            print(f"  {request['custom_id']}: {error}")
            continue
        record = make_record(request["custom_id"], message, None, extra_fields)
        append_responses(jsonl_path, [record])   # saved immediately, so a crash loses nothing
        records.append(record)
        if i % 25 == 0 or i == len(requests):
            print(f"  {i}/{len(requests)} done", flush=True)
    return records, n_failed


def run_batch(client, requests, jsonl_path, extra_fields):
    """Submit one Message Batch, wait until it has ended, and save every successful response."""
    pending_path = jsonl_path + ".pending"

    # Resume a batch that was submitted earlier but never collected (crash while waiting).
    if os.path.exists(pending_path):
        batch_id = open(pending_path).read().strip()
        print(f"Found pending batch {batch_id}; resuming instead of submitting a new one.")
    else:
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        os.makedirs(os.path.dirname(pending_path), exist_ok=True)
        with open(pending_path, "w") as f:
            f.write(batch_id)
        print(f"Submitted batch {batch_id}")

    # Poll until the batch has ended. Anthropic promises this within 24 hours.
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  status={batch.processing_status} processing={counts.processing} "
              f"succeeded={counts.succeeded} errored={counts.errored}", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(30)

    # Collect results. They arrive in any order, so we key everything by custom_id.
    records = []
    n_failed = 0
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            n_failed += 1
            print(f"  {result.custom_id}: {result.result.type}")
            continue
        records.append(make_record(result.custom_id, result.result.message, batch_id, extra_fields))

    append_responses(jsonl_path, records)
    os.remove(pending_path)   # only after results are safely on disk
    return records, n_failed
