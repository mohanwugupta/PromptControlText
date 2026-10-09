"""Frozen main study: bounded Batch submissions with durable cost reservations.

Default is an offline dry run. --step polls and submits once; --watch-minutes
runs bounded polling. Never resubmits a request after ambiguous submission.
"""
import argparse
import json
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path

from frontier.batch_api import ApiError, Client, result_row, terminal
from frontier.prepare import ROOT, digest, validate_manifest
from frontier.providers import KEYS, request_spec
from frontier.run import cases, reservation, run_lock, validate_config
from scripts.credentials import credential_environment


def validate_main(config, manifest, pilot):
    validate_config(pilot)
    validate_manifest(manifest)
    if (config.get("schema_version") != 2 or config.get("stage") != "main"
            or config.get("transport") != "batch" or config.get("batch_discount") != .5):
        raise ValueError("Unreviewed main transport/configuration")
    if config.get("manifest_sha256") != manifest["manifest_sha256"] or config.get("pilot_config_sha256") != digest(pilot):
        raise ValueError("Main configuration must reference the reviewed frozen inputs")
    for name in ("models", "max_output_tokens", "input_token_reserve", "timeout_seconds"):
        if config[name] != pilot[name]:
            raise ValueError("Main generation settings must match the pilot: " + name)
    fixed = {"study_ceiling_usd": 250, "main_generation_ceiling_usd": 160,
             "main_generation_allocation_usd": 160, "judge_allocation_usd": 30,
             "setup_and_contingency_allocation_usd": 60, "prior_pilot_accounted_usd": 9.402844,
             "main_runpod_reserve_usd": 5, "initial_batch_size": 5, "batch_size": 200,
             "poll_seconds": 60, "canonical_judge": "meta-llama/Llama-3.1-8B-Instruct"}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("Unreviewed budget, judge, or scheduling configuration")
    return config


def main_cases(manifest, config):
    for case in cases(manifest, config, "main"):
        _, body = request_spec(case["model"], case["messages"], config["max_output_tokens"])
        # Batch scheduling supplies the service tier. Generation settings match.
        if case["model"]["provider"] == "openai":
            body.pop("service_tier")
        case["request_body_sha256"] = digest(body)
        case["request_id"] = digest([manifest["manifest_sha256"], digest(config), "main-batch",
                                      case["item"]["item_id"], case["condition"]["condition_id"],
                                      case["model"]["provider"], body])
        case["body"] = body
        yield case


def batch_rows(provider, requests):
    if provider == "openai":
        return [{"custom_id": c["request_id"], "method": "POST", "url": "/v1/responses", "body": c["body"]}
                for c in requests]
    if provider == "anthropic":
        return [{"custom_id": c["request_id"], "params": c["body"]} for c in requests]
    return [{"request": c["body"], "metadata": {"key": c["request_id"]}} for c in requests]


def reserve_cost(case, config):
    return reservation(case["model"], config) * config["batch_discount"]


def cost_result(case, result, config):
    tokens = [result.get("input_tokens"), result.get("output_tokens")]
    if result.get("documented_unbilled"):
        return 0.0, "documented_batch_rejection_unbilled"
    if any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in tokens):
        return reserve_cost(case, config), "reserved_unknown"
    m = case["model"]
    return ((tokens[0] * m["input_reserve_usd_per_million"] + tokens[1] * m["output_usd_per_million"])
            / 1e6 * config["batch_discount"], "usage_at_conservative_batch_rates")


def reserved_terminal_block(case, result):
    """A Gemini prompt block is final even when its charge remains unknown.

    Do not invent zero output usage or release its full reservation. Other
    missing-usage outcomes and provider errors still require review.
    """
    return (case["model"]["provider"] == "google"
            and result.get("outcome") == "blocked" and result.get("text") == ""
            and (result.get("prompt_feedback") or {}).get("blockReason")
            in {"SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT"}
            and result.get("finish_reason") is None
            and result.get("output_tokens") is None
            and result.get("reasoning_tokens") is None
            and isinstance(result.get("input_tokens"), int)
            and not isinstance(result.get("input_tokens"), bool)
            and result["input_tokens"] >= 0)


class MainLedger:
    def __init__(self, path, manifest, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.config = path, config
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS batches (
              batch_key TEXT PRIMARY KEY, provider TEXT NOT NULL, state TEXT NOT NULL,
              remote_id TEXT, file_id TEXT, rows_json TEXT NOT NULL,
              status_json TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
              request_id TEXT PRIMARY KEY, batch_key TEXT NOT NULL, provider TEXT NOT NULL,
              state TEXT NOT NULL, reserve REAL NOT NULL, cost REAL NOT NULL,
              case_json TEXT NOT NULL, result_json TEXT);
        """)
        expected = {"manifest_sha256": manifest["manifest_sha256"], "config_sha256": digest(config)}
        current = dict(self.db.execute("SELECT key,value FROM metadata"))
        if current and current != expected:
            raise ValueError("Ledger belongs to another experiment/configuration")
        if not current:
            with self.db:
                self.db.executemany("INSERT INTO metadata VALUES (?,?)", expected.items())

    def total(self):
        return self.db.execute("SELECT coalesce(sum(cost),0) FROM requests").fetchone()[0]

    def prepare(self, provider, requests):
        """Reserve the entire batch atomically before upload or inference submission."""
        if not requests or any(c["model"]["provider"] != provider for c in requests):
            raise ValueError("Empty/mixed batch")
        if len(requests) > self.config["batch_size"]:
            raise ValueError("Oversized batch")
        key = provider + "-" + digest([c["request_id"] for c in requests])[:32]
        amount = sum(reserve_cost(c, self.config) for c in requests)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.total() + amount > self.config["main_generation_ceiling_usd"] + 1e-9:
                return None
            now = time.time()
            self.db.execute("INSERT INTO batches VALUES (?,?, 'submitting',NULL,NULL,?,NULL,NULL,?,?)",
                            (key, provider, json.dumps(batch_rows(provider, requests)), now, now))
            for c in requests:
                amount = reserve_cost(c, self.config)
                self.db.execute("INSERT INTO requests VALUES (?,?,?,'reserved',?,?,?,NULL)",
                                (c["request_id"], key, provider, amount, amount, json.dumps(c)))
        return key

    def update_batch(self, key, **fields):
        if not set(fields) <= {"state", "remote_id", "file_id", "status_json", "error"}:
            raise ValueError("Invalid update")
        fields["updated"] = time.time()
        with self.db:
            self.db.execute("UPDATE batches SET " + ",".join(k + "=?" for k in fields) + " WHERE batch_key=?",
                            tuple(fields.values()) + (key,))

    def ingest(self, batch, data):
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        expected = {r["request_id"]: dict(r) for r in self.db.execute(
            "SELECT * FROM requests WHERE batch_key=?", (batch["batch_key"],))}
        parsed = [result_row(batch["provider"], row) for row in rows]
        keys = [key for key, _ in parsed]
        if len(keys) != len(set(keys)) or set(keys) != set(expected):
            raise ValueError("Incomplete, duplicate, missing-identity, or foreign Batch results")
        updates, states = [], []
        for key, result in parsed:
            record = expected[key]
            case = json.loads(record["case_json"])
            cost, basis = cost_result(case, result, self.config)
            result["cost_basis"] = basis
            state = "done" if result["outcome"] in ("text", "blocked", "truncated") else "needs_review"
            if ((basis == "reserved_unknown" and not reserved_terminal_block(case, result))
                    or not math.isfinite(cost) or cost > record["reserve"] + 1e-9
                    or (isinstance(result.get("input_tokens"), int)
                        and result["input_tokens"] > self.config["input_token_reserve"])
                    or (isinstance(result.get("output_tokens"), int)
                        and result["output_tokens"] > self.config["max_output_tokens"])):
                state = "needs_review"
            updates.append((state, cost, json.dumps(result), key))
            states.append(state)
        with self.db:
            self.db.executemany("UPDATE requests SET state=?,cost=?,result_json=? WHERE request_id=?", updates)
            self.db.execute("UPDATE batches SET state=?,updated=? WHERE batch_key=?",
                            ("done" if set(states) == {"done"} else "needs_review", time.time(), batch["batch_key"]))

    def reconcile_terminal_blocks(self):
        """Resolve only reviewed terminal blocks; preserve costs and results exactly."""
        resolved, affected_batches = [], set()
        with self.db:
            for row in self.db.execute("SELECT * FROM requests WHERE state='needs_review'").fetchall():
                case, result = json.loads(row["case_json"]), json.loads(row["result_json"] or "null")
                if (result is None or not reserved_terminal_block(case, result)
                        or result.get("cost_basis") != "reserved_unknown"
                        or result["input_tokens"] > self.config["input_token_reserve"]
                        or row["cost"] != row["reserve"]):
                    continue
                self.db.execute("UPDATE requests SET state='done' WHERE request_id=?", (row["request_id"],))
                resolved.append(row["request_id"])
                affected_batches.add(row["batch_key"])
            # Batch-level parsing/identity errors are never cleared by this review.
            for batch in self.db.execute("SELECT * FROM batches WHERE state='needs_review' AND error IS NULL").fetchall():
                if batch["batch_key"] not in affected_batches:
                    continue
                states = {r[0] for r in self.db.execute("SELECT state FROM requests WHERE batch_key=?", (batch["batch_key"],))}
                if states == {"done"}:
                    self.db.execute("UPDATE batches SET state='done',updated=? WHERE batch_key=?", (time.time(), batch["batch_key"]))
        return resolved

    def summary(self, expected=10950):
        rows = self.db.execute("SELECT provider,state,result_json FROM requests").fetchall()
        done = Counter(r["provider"] for r in rows if r["state"] == "done")
        outcomes = Counter((json.loads(r["result_json"]) if r["result_json"] else {}).get("outcome", "pending") for r in rows)
        batches = [dict(r) for r in self.db.execute(
            "SELECT batch_key,provider,state,remote_id,created,updated,error FROM batches ORDER BY created")]
        cost = self.total()
        return {"expected_requests": expected, "submitted_or_reserved": len(rows), "complete": sum(done.values()) == expected,
                "done_by_provider": dict(done), "outcomes": dict(outcomes), "main_accounted_usd": round(cost, 6),
                "prior_pilot_accounted_usd": self.config["prior_pilot_accounted_usd"],
                "main_runpod_reserved_usd": self.config["main_runpod_reserve_usd"],
                "study_accounted_including_reserved_usd": round(cost + self.config["prior_pilot_accounted_usd"]
                                                                 + self.config["main_runpod_reserve_usd"], 6),
                "main_generation_cap_usd": self.config["main_generation_ceiling_usd"],
                "judge_allocation_usd": self.config["judge_allocation_usd"], "batches": batches}

    def checkpoint(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        backup = directory / "checkpoint.sqlite.tmp"
        destination = sqlite3.connect(backup)
        try:
            self.db.backup(destination)
        finally:
            destination.close()
        backup.replace(directory / "checkpoint.sqlite")
        temp = directory / "results.jsonl.tmp"
        with temp.open("w") as out:
            for row in self.db.execute("SELECT * FROM requests ORDER BY batch_key,request_id"):
                record = dict(row)
                record["case"] = json.loads(record.pop("case_json"))
                raw = record.pop("result_json")
                record["result"] = json.loads(raw) if raw else None
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp.replace(directory / "results.jsonl")
        temp = directory / "status.json.tmp"
        temp.write_text(json.dumps(self.summary(), indent=2) + "\n")
        temp.replace(directory / "status.json")


def step(ledger, requests, client, artifacts, submit=True):
    """Read-only recovery of existing batches always precedes new submissions."""
    artifacts.mkdir(parents=True, exist_ok=True)
    for batch in list(ledger.db.execute("SELECT * FROM batches WHERE state='active' ORDER BY created")):
        batch = dict(batch)
        try:
            status = client.poll(batch["provider"], batch["remote_id"])
            ledger.update_batch(batch["batch_key"], status_json=json.dumps(status), error=None)
            if terminal(batch["provider"], status):
                data = client.results(batch["provider"], batch["remote_id"], status)
                (artifacts / (batch["batch_key"] + ".jsonl")).write_bytes(data)
                ledger.ingest(batch, data)
        except ApiError as error:
            ledger.update_batch(batch["batch_key"], error=str(error))
            # A failed GET can be retried safely; no new paid work in this step.
            return "poll_error"
        except (ValueError, KeyError, TypeError) as error:
            ledger.update_batch(batch["batch_key"], state="needs_review", error=type(error).__name__)
    batches = list(ledger.db.execute("SELECT * FROM batches"))
    if any(b["state"] in ("submitting", "needs_review") for b in batches):
        return "needs_review"
    states = {r["request_id"]: r["state"] for r in ledger.db.execute("SELECT request_id,state FROM requests")}
    if len(states) == len(requests) and set(states.values()) == {"done"}:
        return "complete"
    if not submit:
        return "polled"
    # All three initial five-case batches must finish before bulk submission.
    initial_complete = all(any(b["provider"] == p and b["state"] == "done" for b in batches) for p in KEYS)
    active = {b["provider"] for b in batches if b["state"] == "active"}
    size = ledger.config["batch_size"] if initial_complete else ledger.config["initial_batch_size"]
    budget_stopped = False
    for provider in KEYS:
        if provider in active or (not initial_complete and any(b["provider"] == provider for b in batches)):
            continue
        pending = [c for c in requests if c["model"]["provider"] == provider and c["request_id"] not in states]
        if not pending:
            continue
        # Smaller final batches may use the remainder without increasing the cap.
        count = min(size, len(pending))
        while count and ledger.total() + sum(reserve_cost(c, ledger.config) for c in pending[:count]) > ledger.config["main_generation_ceiling_usd"] + 1e-9:
            count -= 1
        if not count:
            budget_stopped = True
            continue
        selected = pending[:count]
        key = ledger.prepare(provider, selected)
        if key is None:
            budget_stopped = True
            continue
        rows = batch_rows(provider, selected)
        file_id = None
        try:
            if provider == "openai":
                file_id = client.upload_openai(rows, key)
                ledger.update_batch(key, file_id=file_id)
            result = client.submit(provider, rows, key, selected[0]["model"]["model"], file_id)
            remote_id = result.get("name") if provider == "google" else result.get("id")
            if not remote_id:
                raise ValueError("Submission lacks a batch ID")
            ledger.update_batch(key, state="active", remote_id=remote_id, status_json=json.dumps(result))
            print(json.dumps({"event": "submitted", "provider": provider, "count": count,
                              "batch_key": key, "remote_id": remote_id,
                              "main_accounted_usd": round(ledger.total(), 6)}), flush=True)
        except (ApiError, ValueError, KeyError, TypeError) as error:
            ledger.update_batch(key, error=str(error))
            return "needs_review"
    if budget_stopped and not ledger.db.execute("SELECT 1 FROM batches WHERE state='active'").fetchone():
        return "budget_stopped"
    return "running"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/frontier-main.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/frontier/manifest.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".local/frontier/main-20261008/main.sqlite")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/frontier/main-20261008")
    parser.add_argument("--step", action="store_true", help="Poll and submit bounded paid batches once")
    parser.add_argument("--poll-only", action="store_true", help="Read/recover existing batches, no new generation")
    parser.add_argument("--watch-minutes", type=int, default=0, help="Run steps for at most 120 minutes")
    args = parser.parse_args()
    if not 0 <= args.watch_minutes <= 120:
        raise SystemExit("Worker session must be bounded to 120 minutes")
    manifest = json.loads(args.manifest.read_text())
    config = validate_main(json.loads(args.config.read_text()), manifest,
                           json.loads((ROOT / "configs/frontier-pilot.json").read_text()))
    requests = list(main_cases(manifest, config))
    if not (args.step or args.watch_minutes or args.poll_only):
        print(json.dumps({"status": "dry_run", "requests": len(requests),
                          "unique_request_ids": len({c["request_id"] for c in requests}),
                          "by_provider": dict(Counter(c["model"]["provider"] for c in requests)),
                          "manifest_sha256": manifest["manifest_sha256"], "config_sha256": digest(config),
                          "first_wave_max_reservation_usd": sum(reservation(m, config) for m in config["models"]) * .5 * config["initial_batch_size"],
                          "main_generation_cap_usd": 160, "study_ceiling_usd": 250}, indent=2))
        return
    environment = credential_environment(ROOT / ".env")
    if any(not environment.get(key) for key in KEYS.values()):
        raise SystemExit("Missing provider credential")
    with run_lock(args.ledger.with_suffix(".lock")):
        ledger = MainLedger(args.ledger, manifest, config)
        client = Client(environment, config["timeout_seconds"])
        deadline = time.monotonic() + args.watch_minutes * 60
        try:
            while True:
                status = step(ledger, requests, client, args.output / "provider-results", not args.poll_only)
                ledger.checkpoint(args.output)
                summary = ledger.summary()
                print(json.dumps({"status": status, **{k: v for k, v in summary.items() if k != "batches"}}), flush=True)
                if status in ("complete", "needs_review", "budget_stopped") or time.monotonic() >= deadline:
                    break
                time.sleep(min(config["poll_seconds"], max(0, deadline - time.monotonic())))
                if time.monotonic() >= deadline:
                    break
        finally:
            ledger.checkpoint(args.output)
        if status in ("needs_review", "budget_stopped", "poll_error"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
