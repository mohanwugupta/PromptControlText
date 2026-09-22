"""Small, resumable OpenAI feasibility pilot. Never prints credentials or outputs.

Uses the paper's flattened benchmark inputs without modifying their content.
The sparse design estimates serving costs, not family effects or population rates.
Only --execute sends API calls; an append-only ledger reserves worst-case cost
before sending. Unknown/failed requests retain that reservation and are not retried.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "frontier_pilot"
MODELS = ["gpt-5.5-2026-04-23", "gpt-6-astra"]
# USD per million tokens, standard global pricing, verified 2026-09-22.
# Charge input conservatively at uncached price; actual invoice may be lower.
PRICES = {MODELS[0]: (5.0, 30.0), MODELS[1]: (10.0, 50.0)}
MAX_OUTPUT = 2048  # Includes reasoning, unlike the original visible-output cap.
CAP = 10.0
SEED = 20260922


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_rows(name):
    with (ROOT / "benchmarks/artifacts/datasets" / name).open() as f:
        return list(csv.DictReader(f))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    rng = random.Random(SEED)
    items = []
    hb = load_rows("harmbench_behaviors.csv")
    for category, n in [("standard", 4), ("copyright", 2), ("contextual", 2)]:
        for r in rng.sample([r for r in hb if r["FunctionalCategory"] == category], n):
            items.append(dict(benchmark="HarmBench", item_id=r["BehaviorID"],
                              input_text=r["Behavior"], gold_label="unsafe", domain=category))
    xs = load_rows("xstest_prompts.csv")
    for label in ["safe", "unsafe"]:
        groups = collections.defaultdict(list)
        for r in xs:
            if r["label"] == label:
                groups[r["type"]].append(r)
        for category in rng.sample(sorted(groups), 4):
            r = rng.choice(groups[category])
            items.append(dict(benchmark="XSTest", item_id="xstest_" + r["id"],
                              input_text=r["prompt"], gold_label=label, domain=category))
    ih = load_rows("iheval.csv")
    for category in ["none", "conflict"]:
        for r in rng.sample([r for r in ih if r["conflict_type"] == category], 4):
            items.append(dict(benchmark="IHEval", item_id=r["item_id"],
                              input_text=r["prompt"], gold_label="non-conflict" if category == "none" else category,
                              domain=category))
    registry = yaml.safe_load((ROOT / "prompts/registry_v3.yaml").read_text())
    prompts = [dict(prompt_family=f, clarity_level=c, prompt_variant=v, system_prompt=p)
               for f, data in registry.items()
               for c, variants in data["clarity_levels"].items()
               for v, p in variants.items()]
    assert len(items) == 24 and len(prompts) == 72
    rng.shuffle(prompts)
    conditions = []
    for i, item in enumerate(items):
        controls = [dict(prompt_family="No-system-prompt", clarity_level="none",
                         prompt_variant="control", system_prompt=None)] + prompts[i * 3:i * 3 + 3]
        for p in controls:
            conditions.append({**item, **p})
    # Interleave model calls; every model sees exactly the same item/condition pairs.
    tasks = []
    for condition in conditions:
        for model in MODELS:
            task = {**condition, "model": model, "reasoning_effort": "low",
                    "max_output_tokens": MAX_OUTPUT}
            task["task_id"] = digest(task)
            tasks.append(task)
    sources = ["prompts/registry_v3.yaml"] + ["benchmarks/artifacts/datasets/" + f for f in
               ["harmbench_behaviors.csv", "xstest_prompts.csv", "iheval.csv"]]
    manifest = dict(seed=SEED, repository_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        design="24 items; baseline plus three assigned prompts per item; all 72 registry prompts once per model",
        prices_per_million=PRICES, prices_date="2026-09-22", budget_usd=CAP,
        price_sources=["https://developers.openai.com/api/docs/models/gpt-5.5",
                       "https://developers.openai.com/api/docs/models/gpt-6-astra"],
        limitations=["Technical feasibility pilot, not a fully crossed inferential study",
                     "Benchmark text matches repository loaders, including flattened IHEval",
                     "HarmBench ContextString is omitted by the original loader; preserved here for comparability",
                     "Both models use low reasoning; no temperature or seed supplied",
                     "2048-token cap includes reasoning; cannot equate with original 512-token setting"],
        tasks=tasks)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def payload(task):
    messages = []
    if task["system_prompt"] is not None:
        messages.append({"role": "system", "content": task["system_prompt"]})
    messages.append({"role": "user", "content": task["input_text"]})
    return dict(model=task["model"], input=messages, reasoning={"effort": task["reasoning_effort"]},
                max_output_tokens=task["max_output_tokens"], store=False, service_tier="default")


def reservation(task):
    # UTF-8 bytes overestimate ordinary BPE input tokens; additional framing margin.
    p = payload(task)
    nbytes = sum(len(m["content"].encode()) for m in p["input"])
    assert nbytes < 100000, "Pilot input unexpectedly long"
    ip, op = PRICES[task["model"]]
    return ((nbytes + 2048) * ip * 1.25 + task["max_output_tokens"] * op) / 1e6


def cost(usage, model):
    ip, op = PRICES[model]
    return (usage["input_tokens"] * ip + usage["output_tokens"] * op) / 1e6


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Ledger:
    def __init__(self, directory):
        self.path = directory / "ledger.jsonl"
        self.lock = threading.Lock()
        self.current = {}
        for e in read_jsonl(self.path):
            self.current[e["task_id"]] = e

    def append(self, e):
        with self.path.open("a") as f:
            f.write(json.dumps(e) + "\n")
            f.flush()
            import os
            os.fsync(f.fileno())
        self.current[e["task_id"]] = e

    def total(self):
        return sum(e["budget_charge_usd"] for e in self.current.values())

    def reserve(self, task, retry_rejected=False):
        with self.lock:
            tid = task["task_id"]
            if tid in self.current and not (retry_rejected and self.current[tid]["state"] == "rejected_no_charge"):
                return False
            amount = reservation(task)
            if self.total() + amount > CAP:
                return False
            self.append(dict(task_id=tid, state="reserved", budget_charge_usd=amount, timestamp=time.time()))
            return True

    def settle(self, task, result):
        with self.lock:
            old = self.current[task["task_id"]]
            amount = result.get("estimated_cost_usd", old["budget_charge_usd"])
            state = result["status"]
            if result.get("http_status") == 429 and result.get("error_code") in {"credit_balance_exhausted", "rate_limit_exceeded"}:
                amount = 0.0
                state = "rejected_no_charge"
            self.append(dict(task_id=task["task_id"], state=state,
                             budget_charge_usd=amount, timestamp=time.time()))


def summarize(manifest):
    if (OUT / "usage.jsonl").exists() and not (OUT / "ledger.jsonl").exists():
        return json.loads((OUT / "summary.json").read_text())
    rows = read_jsonl(OUT / "responses.jsonl")
    ledger = Ledger(OUT)
    info = dict(planned=len(manifest["tasks"]), attempted=len(ledger.current),
                recorded=len(rows), budget_cap_usd=CAP, accounted_usd=ledger.total(), models={})
    for model in MODELS:
        rr = [r for r in rows if r["model"] == model]
        ok = [r for r in rr if "usage" in r]
        total = sum(r.get("estimated_cost_usd", 0) for r in rr)
        mean = total / len(ok) if ok else None
        info["models"][model] = dict(attempted=len(rr), statuses=dict(collections.Counter(r["status"] for r in rr)),
            empty_final=sum(not r.get("model_output") for r in ok),
            input_tokens=sum(r["usage"]["input_tokens"] for r in ok),
            output_tokens=sum(r["usage"]["output_tokens"] for r in ok),
            reasoning_tokens=sum(r["usage"].get("output_tokens_details", {}).get("reasoning_tokens", 0) for r in ok),
            estimated_cost_usd=total, mean_cost_usd=mean,
            projected_600_items_73_conditions_usd=mean * 43800 if mean is not None else None)
    (OUT / "summary.json").write_text(json.dumps(info, indent=2))
    return info


def validate_resume_state():
    if (OUT / "usage.jsonl").exists() and not (OUT / "ledger.jsonl").exists():
        raise RuntimeError("Published pilot archive has no private ledger. Refusing to repeat paid requests; use a new versioned run directory for a new study.")


def execute(manifest, limit, retry_rejected=False):
    validate_resume_state()
    keyfile = ROOT.parent / ".env.frontier"
    secrets = {}
    for line in keyfile.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            secrets[k.strip()] = v.strip().strip('"').strip("'")
    key = secrets.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OpenAI key is missing")
    ledger = Ledger(OUT)
    stop = threading.Event()
    output_lock = threading.Lock()
    dispatch_lock = threading.Lock()
    last_dispatch = [0.0]
    cooldown_until = [0.0]
    network_failures = [0]
    blocked_items = {(r['benchmark'], r['item_id']) for r in read_jsonl(OUT / 'responses.jsonl')
                     if r.get('error_code') == 'cyber_policy'}
    tasks = [t for t in manifest["tasks"] if t["task_id"] not in ledger.current or
             (retry_rejected and ledger.current[t["task_id"]]["state"] == "rejected_no_charge")]
    if limit:
        tasks = tasks[:limit]

    def run(task):
        if (task['benchmark'], task['item_id']) in blocked_items:
            return
        with dispatch_lock:
            delay = max(0, max(last_dispatch[0] + 3.1, cooldown_until[0]) - time.monotonic())
            if delay:
                time.sleep(delay)
            if stop.is_set() or (task['benchmark'], task['item_id']) in blocked_items:
                return
            last_dispatch[0] = time.monotonic()
        if stop.is_set() or not ledger.reserve(task, retry_rejected):
            return
        start = time.time()
        result = dict(task_id=task["task_id"], model=task["model"], benchmark=task["benchmark"],
                      item_id=task["item_id"], prompt_family=task["prompt_family"],
                      clarity_level=task["clarity_level"], prompt_variant=task["prompt_variant"])
        request = urllib.request.Request("https://api.openai.com/v1/responses",
            data=json.dumps(payload(task)).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                body = json.load(resp)
                result["request_id"] = resp.headers.get("x-request-id")
                result['rate_limits'] = {k: v for k, v in resp.headers.items() if k.lower().startswith('x-ratelimit-')}
            final = []
            for output in body.get("output", []):
                if output.get("type") == "message":
                    for part in output.get("content", []):
                        if part.get("type") == "output_text":
                            final.append(part.get("text", ""))
                        elif part.get("type") == "refusal":
                            final.append(part.get("refusal", ""))
            result.update(status=body.get("status", "unknown"), model_output="\n".join(final),
                          response_id=body.get("id"), response_model=body.get("model"),
                          incomplete_details=body.get("incomplete_details"),
                          usage=body["usage"], estimated_cost_usd=cost(body["usage"], task["model"]),
                          service_tier=body.get("service_tier"))
            network_failures[0] = 0
        except urllib.error.HTTPError as exc:
            # Save only categorical error fields, never headers, keys, or message text.
            try:
                error = json.load(exc).get("error", {})
            except Exception:
                error = {}
            result.update(status="http_error", http_status=exc.code,
                          error_code=error.get("code"), error_type=error.get("type"))
            result['rate_limits'] = {k: v for k, v in exc.headers.items() if k.lower().startswith('x-ratelimit-')}
            if error.get('code') == 'cyber_policy':
                blocked_items.add((task['benchmark'], task['item_id']))
            else:
                stop.set()
        except Exception as exc:
            result.update(status="transport_or_parse_error", error_type=type(exc).__name__)
            if isinstance(exc, urllib.error.URLError):
                result['network_reason_type'] = type(exc.reason).__name__
                result['network_errno'] = getattr(exc.reason, 'errno', None)
            # Keep unknown attempts reserved and never retry them. Pause dispatch
            # after connection interruptions, then proceed to different tasks.
            if isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)) or type(exc).__name__ == 'RemoteDisconnected':
                network_failures[0] += 1
                cooldown_until[0] = time.monotonic() + 20
                if network_failures[0] >= 3:
                    stop.set()
            else:
                stop.set()
        result["elapsed_seconds"] = time.time() - start
        with output_lock:
            with (OUT / "responses.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
                import os
                os.fsync(f.fileno())
            ledger.settle(task, result)
            print(json.dumps({"model": task["model"], "status": result["status"],
                              "attempted": len(ledger.current),
                              "accounted_usd": round(ledger.total(), 4)}), flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, tasks))
    return summarize(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-rejected", action="store_true",
                        help="Retry confirmed credit/rate-limit rejections; never retries unknown requests.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare()
        result = execute(manifest, args.limit, args.retry_rejected) if args.execute else summarize(manifest)
        print(json.dumps(result, indent=2))
