"""Dry-run the frozen design, or execute its capped pilot with a durable ledger.

The first version deliberately permits paid PILOT calls only. Main generation
requires a reviewed cost/completeness report and a subsequent main-run config.
"""

import argparse
import fcntl
import json
import math
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from frontier.prepare import ROOT, digest, render_messages, validate_manifest
from frontier.providers import KEYS, MODEL_IDS, generate, request_spec
from scripts.credentials import credential_environment


def validate_config(config):
    if config["schema_version"] != 1 or config["stage"] != "pilot":
        raise ValueError("This runner supports the reviewed pilot configuration only")
    if len(config["models"]) != 3 or {m["provider"] for m in config["models"]} != set(KEYS):
        raise ValueError("Expected one model per provider")
    if not 0 < config["pilot_ceiling_usd"] <= 10 or config["study_ceiling_usd"] != 250:
        raise ValueError("Pilot must remain within its $10 cap and the $250 study ceiling")
    infrastructure = config.get("runpod_reserve_usd", 0)
    total_cap = config.get("total_pilot_ceiling_usd", 10)
    if (not math.isfinite(infrastructure) or infrastructure < 0 or total_cap != 10
            or config["pilot_ceiling_usd"] + infrastructure > total_cap):
        raise ValueError("API cap plus infrastructure reserve must fit the $10 total cap")
    if not 1 <= config["max_output_tokens"] <= 4096 or config["input_token_reserve"] != 32768:
        raise ValueError("Unreviewed token limits")
    for model in config["models"]:
        if MODEL_IDS.get(model["provider"]) != model["model"] or model["effort"] != "low":
            raise ValueError("Unreviewed model or reasoning configuration")
        for key in ("input_usd_per_million", "input_reserve_usd_per_million", "output_usd_per_million"):
            if not math.isfinite(model[key]) or model[key] <= 0:
                raise ValueError("Invalid price")
        if model["input_reserve_usd_per_million"] < model["input_usd_per_million"]:
            raise ValueError("Input reserve must cover standard pricing")
    return config


def cases(manifest, config, split):
    conditions = [c for c in manifest["conditions"]
                  if split == "main" or c["condition_id"] in manifest["pilot_condition_ids"]]
    pairs = [(item, condition) for item in manifest[split] for condition in conditions]
    pairs.sort(key=lambda p: digest([manifest["seed"], split, p[0]["item_id"], p[1]["condition_id"]]))
    for item, condition in pairs:
        messages = render_messages(item, condition)
        for model in config["models"]:
            url, body = request_spec(model, messages, config["max_output_tokens"])
            if len(json.dumps(body, ensure_ascii=False).encode()) + 1024 > config["input_token_reserve"]:
                raise ValueError("Input exceeds conservative token reserve; review before running")
            request_id = digest([manifest["manifest_sha256"], digest(config), split,
                                 item["item_id"], condition["condition_id"], url, body])
            yield {"request_id": request_id, "split": split, "item": item,
                   "condition": condition, "model": model, "messages": messages,
                   "request_body_sha256": digest(body)}


def reservation(model, config):
    return (config["input_token_reserve"] * model["input_reserve_usd_per_million"]
            + config["max_output_tokens"] * model["output_usd_per_million"]) / 1e6


def accounted_cost(result, model, config):
    """Do not release the reservation if usage is absent or the outcome uncertain."""
    reserve = reservation(model, config)
    tokens = [result.get("input_tokens"), result.get("output_tokens")]
    if any(not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in tokens):
        return reserve, "reserved_unknown"
    amount = (tokens[0] * model["input_reserve_usd_per_million"]
              + tokens[1] * model["output_usd_per_million"]) / 1e6
    return amount, "usage_at_conservative_rates"


class Ledger:
    def __init__(self, path, manifest, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY, state TEXT NOT NULL, cost REAL NOT NULL,
            case_json TEXT NOT NULL, result_json TEXT, started REAL NOT NULL, ended REAL)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS attempt_history (
            request_id TEXT NOT NULL, attempt_number INTEGER NOT NULL, record_json TEXT NOT NULL,
            cost REAL NOT NULL, PRIMARY KEY (request_id, attempt_number))""")
        expected = {"manifest_sha256": manifest["manifest_sha256"], "config_sha256": digest(config)}
        existing = dict(self.db.execute("SELECT key,value FROM metadata"))
        if existing and existing != expected:
            raise ValueError("Ledger belongs to a different manifest or model configuration")
        if not existing:
            self.db.executemany("INSERT INTO metadata VALUES (?,?)", expected.items())
            self.db.commit()

    def total(self):
        return self.db.execute("""SELECT coalesce(sum(cost),0) FROM (
            SELECT cost FROM requests UNION ALL SELECT cost FROM attempt_history)""").fetchone()[0]

    def retry_funded_anthropic(self, request_id, reason):
        """Reopen one known credit rejection after an explicit funding update."""
        return self._retry_setup_error(request_id, reason, "credit")

    def retry_workspace_anthropic(self, request_id, reason):
        """Reopen one missing-workspace rejection after a routing/key update."""
        return self._retry_setup_error(request_id, reason, "workspace")

    def retry_google_quota(self, request_id, reason):
        """Reopen one known zero-free-tier rejection after an explicit update."""
        return self._retry_setup_error(request_id, reason, "quota")

    def _retry_setup_error(self, request_id, reason, error_kind):
        """Archive one verified account-setup rejection after an explicit update.

        The failed attempt and its full reservation remain in the ledger. This
        makes the unchanged request eligible for one new attempt, not a retry loop.
        """
        if not reason.strip():
            raise ValueError("An explicit account-update reason is required")
        with self.db:
            row = self.db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                raise ValueError("No current failed attempt to reconcile")
            key, state, cost, case_json, result_json, started, ended = row
            case, result = json.loads(case_json), json.loads(result_json or "{}")
            message = (result.get("error_diagnostic") or {}).get("message", "").lower()
            if error_kind == "quota":
                provider, status, label = "google", 429, "Google quota"
                matches = ((result.get("error_diagnostic") or {}).get("status") == "RESOURCE_EXHAUSTED"
                           and "generate_content_free_tier" in message and "limit: 0," in message)
            elif error_kind in ("credit", "workspace"):
                provider, status, label = "anthropic", 400, f"Anthropic {error_kind}"
                matches = ("credit balance is too low" in message if error_kind == "credit" else
                           "this api key is not scoped to a workspace" in message
                           and "anthropic-workspace-id" in message
                           and (result.get("error_diagnostic") or {}).get("type") == "invalid_request_error")
            else:
                raise ValueError("Unrecognized account-update kind")
            if not (state == "needs_review" and case["model"]["provider"] == provider
                    and result.get("outcome") == "http_error" and result.get("http_status") == status
                    and matches and ended is not None):
                raise ValueError(f"Only a known {label} rejection can be retried here")
            number = self.db.execute("SELECT coalesce(max(attempt_number),0)+1 FROM attempt_history WHERE request_id=?",
                                     (key,)).fetchone()[0]
            record = {"request_id": key, "attempt_number": number, "state": state,
                      "accounted_usd": cost, "case": case, "result": result,
                      "started_unix": started, "ended_unix": ended,
                      "reconciled_unix": time.time(), "retry_reason": reason}
            self.db.execute("INSERT INTO attempt_history VALUES (?,?,?,?)",
                            (key, number, json.dumps(record), cost))
            self.db.execute("DELETE FROM requests WHERE request_id=?", (key,))
        return record

    def states(self):
        return dict(self.db.execute("SELECT request_id,state FROM requests"))

    def classify_anthropic_refusal(self, request_id):
        """Correct a recorded provider refusal; never resend or change its cost."""
        with self.db:
            row = self.db.execute("SELECT state,cost,case_json,result_json FROM requests WHERE request_id=?",
                                  (request_id,)).fetchone()
            if row is None:
                raise ValueError("No recorded response to classify")
            state, cost, case_json, result_json = row
            case, result = json.loads(case_json), json.loads(result_json or "{}")
            if not (state == "needs_review" and case["model"]["provider"] == "anthropic"
                    and result.get("outcome") == "empty" and not result.get("text")
                    and result.get("finish_reason") == "refusal" and result.get("provider_refusal") is True
                    and result.get("cost_basis") == "usage_at_conservative_rates"
                    and result.get("output_tokens") == 0):
                raise ValueError("Only a recorded empty Anthropic provider refusal can be classified here")
            previous = dict(result)
            result.update(outcome="blocked", block_origin="provider_stop_reason")
            result["normalization_correction"] = {
                "previous_outcome": "empty", "previous_state": state, "corrected_unix": time.time(),
                "reason": "Anthropic stop_reason=refusal is a terminal provider block, not an unexplained empty response",
                "source": "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback"}
            self.db.execute("UPDATE requests SET state='done',result_json=? WHERE request_id=?",
                            (json.dumps(result), request_id))
        return {"request_id": request_id, "accounted_usd_unchanged": cost,
                "previous_result": previous, "corrected_result": result, "api_request_repeated": False}

    def reserve(self, case, amount, cap):
        with self.db:
            if self.total() + amount > cap:
                return False
            self.db.execute("INSERT INTO requests VALUES (?, 'reserved', ?, ?, NULL, ?, NULL)",
                            (case["request_id"], amount, json.dumps(case), time.time()))
        return True

    def finish(self, case, result, config):
        cost, basis = accounted_cost(result, case["model"], config)
        result["cost_basis"] = basis
        state = "done" if result["outcome"] in ("text", "blocked", "truncated") else "needs_review"
        known_http_block = (result["outcome"] == "blocked" and result.get("block_origin") == "provider_http"
                            and (result.get("error_diagnostic") or {}).get("code") == "cyber_policy")
        if (basis == "reserved_unknown" and not known_http_block) or cost > reservation(case["model"], config):
            state = "needs_review"
        with self.db:
            self.db.execute("UPDATE requests SET state=?, cost=?, result_json=?, ended=? WHERE request_id=?",
                            (state, cost, json.dumps(result), time.time(), case["request_id"]))
        return state

    def export(self, path):
        """Export all states, including errors and unresolved reservations."""
        records = []
        prior = {}
        for key, data in self.db.execute("SELECT request_id,record_json FROM attempt_history ORDER BY request_id,attempt_number"):
            prior.setdefault(key, []).append(json.loads(data))
        for request_id, state, cost, case, result, started, ended in self.db.execute(
                "SELECT * FROM requests ORDER BY started,request_id"):
            records.append({"request_id": request_id, "state": state, "accounted_usd": cost,
                            "started_unix": started, "ended_unix": ended,
                            "case": json.loads(case), "result": json.loads(result) if result else None,
                            "prior_attempts": prior.pop(request_id, [])})
        # A crash between reconciliation and the next reservation must still
        # export every previous attempt and dollar held against the cap.
        for key, attempts in prior.items():
            records.append({"request_id": key, "state": "retry_ready", "accounted_usd": 0,
                            "started_unix": None, "ended_unix": None, "case": attempts[-1]["case"],
                            "result": None, "prior_attempts": attempts})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        temporary.replace(path)
        return records


@contextmanager
def run_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another worker already owns this ledger") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def execute(requests, ledger, config, environment, call=generate):
    states = ledger.states()
    selected_ids = {case["request_id"] for case in requests}
    if any(s != "done" and key in selected_ids for key, s in states.items()):
        return "needs_review: unresolved attempt; retain reservation and reconcile before retrying"
    for case in requests:
        if case["request_id"] in states:
            continue
        reserve = reservation(case["model"], config)
        if not ledger.reserve(case, reserve, config["pilot_ceiling_usd"]):
            return "budget_stopped"
        result = call(case["model"], case["messages"], config, environment)
        state = ledger.finish(case, result, config)
        print(json.dumps({"request_id": case["request_id"], "provider": case["model"]["provider"],
                          "outcome": result["outcome"], "accounted_usd": round(ledger.total(), 6)}), flush=True)
        if state != "done":
            return "needs_review: provider error, missing usage, or unexpected billing"
    return "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/frontier/manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/frontier-pilot.json")
    parser.add_argument("--split", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--providers", nargs="+", choices=tuple(KEYS), default=list(KEYS),
                        help="Run a provider subset using the same shared ledger and budget")
    parser.add_argument("--execute", action="store_true", help="Make paid PILOT calls; otherwise dry run")
    parser.add_argument("--ledger", type=Path, default=ROOT / ".local/frontier/pilot.sqlite")
    parser.add_argument("--export", type=Path, default=ROOT / ".local/frontier/pilot.jsonl")
    args = parser.parse_args()
    manifest = validate_manifest(json.loads(args.manifest.read_text()))
    config = validate_config(json.loads(args.config.read_text()))
    requests = [case for case in cases(manifest, config, args.split)
                if case["model"]["provider"] in args.providers]
    if not args.execute:
        print(json.dumps({"status": "dry_run", "split": args.split, "requests": len(requests),
                          "unique_request_ids": len({r["request_id"] for r in requests}),
                          "manifest_sha256": manifest["manifest_sha256"], "config_sha256": digest(config),
                          "providers": dict(Counter(r["model"]["provider"] for r in requests)),
                          "max_token_reservations_usd": round(sum(reservation(r["model"], config) for r in requests), 4),
                          "note": "Token-limit bound, not expected cost. No API calls or credential access."}))
        return
    if args.split != "pilot":
        raise SystemExit("Main generation is gated on pilot cost/completeness review; no calls made")
    environment = credential_environment(ROOT / ".env")
    if any(not environment.get(KEYS[provider]) for provider in args.providers):
        raise SystemExit("Missing provider credentials; use scripts/credentials.py status")
    with run_lock(args.ledger.with_suffix(".lock")):
        ledger = Ledger(args.ledger, manifest, config)
        try:
            status = execute(requests, ledger, config, environment)
        finally:
            ledger.export(args.export)
        print(json.dumps({"status": status, "states": dict(Counter(ledger.states().values())),
                          "accounted_usd": round(ledger.total(), 6), "target_requests": len(requests)}))
        if status != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
