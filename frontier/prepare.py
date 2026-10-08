"""Freeze context-complete benchmark items and the existing 73 prompt conditions.

No model requests. Rebuilding an existing manifest must produce identical bytes.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from prompts.registry import iter_prompt_triples, load_registry, validate_registry_schema

ROOT = Path(__file__).resolve().parents[1]
IHEVAL_REV = "726a62924c3050045954df94347d53fe2bd1090d"
SEED = "frontier-2026-10-07-v1"
MAIN_QUOTAS = {"xstest_safe": 20, "xstest_unsafe": 10, "harmbench": 10,
               "iheval_aligned": 5, "iheval_conflict": 5}
PILOT_QUOTAS = {key: 2 for key in MAIN_QUOTAS}
PILOT_CONDITIONS = ["No-system-prompt/none/v1", "Answer-first/explicit/v1",
                    "Refuse-first/explicit/v1", "Source-isolation-first/explicit/v1",
                    "Hierarchy-first/explicit/v1"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def source_record(path, url=None):
    record = {"sha256": file_digest(path)}
    if url:
        record["url"] = url
    else:
        record["path"] = str(path.relative_to(ROOT))
    return record


def restore_iheval(cache, download=False):
    """Use official ready-input files, preserving the historical row population."""
    recovered, sources = {}, []
    for task in ("user-prompt-hijack", "system-prompt-extract"):
        for setting, defense in (("aligned", "default"), ("conflict", "weak_defense")):
            relative = f"benchmark/safety/{task}/{setting}/{defense}/input_data.json"
            path = cache / IHEVAL_REV / relative
            url = f"https://raw.githubusercontent.com/ytyz1307zzh/IHEval/{IHEVAL_REV}/{relative}"
            if not path.exists():
                if not download:
                    raise ValueError("Missing pinned IHEval source; run prepare with --download")
                path.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(url, timeout=60) as response:
                    data = response.read()
                json.loads(data)  # Do not cache a non-JSON error response.
                path.write_bytes(data)
            sources.append(source_record(path, url))
            for row in json.loads(path.read_text()):
                item_id = "safety_" + setting + "_" + row["id"].replace("-", "_")
                if item_id in recovered:
                    raise ValueError("Duplicate IHEval ID: " + item_id)
                if not row["system"] or row["answer"]["system_prompt"] != row["system"]:
                    raise ValueError("IHEval task instructions and reference disagree")
                recovered[item_id] = {
                    "item_id": item_id, "benchmark": "iheval", "stratum": "iheval_" + setting,
                    "domain": task, "group_id": "iheval/" + row["id"].rsplit("-", 1)[0],
                    "base_system": row["system"], "user_text": row["instruction"],
                    "reference": row["answer"], "gold_label": None,
                    "metadata": {"setting": setting, "defense": defense, "upstream_id": row["id"]},
                }
    return recovered, sources


def load_population(cache, download=False):
    directory = ROOT / "benchmarks/artifacts/datasets"
    sources, items = [], []
    for name in ("xstest_prompts.csv", "harmbench_behaviors.csv", "iheval.csv"):
        sources.append(source_record(directory / name))
    for row in read_csv(directory / "xstest_prompts.csv"):
        if row["label"] not in ("safe", "unsafe"):
            raise ValueError("Unexpected XSTest label")
        item_id = "xstest_" + row["id"]
        items.append({"item_id": item_id, "benchmark": "xstest",
                      "stratum": "xstest_" + row["label"], "domain": row["type"],
                      "group_id": item_id, "base_system": "", "user_text": row["prompt"],
                      "gold_label": row["label"], "reference": None,
                      "metadata": {key: row[key] for key in ("focus", "note")}})
    for row in read_csv(directory / "harmbench_behaviors.csv"):
        context, behavior = row["ContextString"], row["Behavior"]
        # Matches HarmBench's official DirectRequest method, including delimiter.
        user_text = context + "\n\n---\n\n" + behavior if context else behavior
        item_id = "harmbench_" + row["BehaviorID"]
        items.append({"item_id": item_id, "benchmark": "harmbench", "stratum": "harmbench",
                      "domain": row["FunctionalCategory"], "group_id": item_id,
                      "base_system": "", "user_text": user_text, "gold_label": None,
                      "reference": {"behavior": behavior, "context": context},
                      "metadata": {"behavior_id": row["BehaviorID"], "tags": row["Tags"],
                                   "semantic_category": row["SemanticCategory"]}})
    recovered, upstream_sources = restore_iheval(cache, download)
    historical = read_csv(directory / "iheval.csv")
    if len(historical) != len(recovered) or {r["item_id"] for r in historical} != set(recovered):
        raise ValueError("Pinned IHEval population differs from historical snapshot")
    for row in historical:
        item = recovered[row["item_id"]]
        setting = "aligned" if row["conflict_type"] == "none" else "conflict"
        if (item["user_text"].replace("\r\n", "\n") != row["prompt"].replace("\r\n", "\n")
                or item["stratum"] != "iheval_" + setting):
            raise ValueError("Pinned IHEval prompt/setting mismatch: " + row["item_id"])
        item["metadata"]["historical_crlf_normalized"] = item["user_text"] != row["prompt"]
    items.extend(recovered.values())
    sources.extend(upstream_sources)
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("Duplicate benchmark item IDs")
    return items, sources


def sample(items, quotas, seed, excluded_groups=(), excluded_texts=()):
    """Hash-ranked stratified sample, excluding repeated parents or exact queries.

    Sampling is uniform by item before these exclusions; no subgroup quotas or
    model outputs enter selection. Stratum order is part of the frozen protocol.
    """
    selected, groups, texts = [], set(excluded_groups), set(excluded_texts)
    for stratum, count in quotas.items():
        candidates = sorted((i for i in items if i["stratum"] == stratum),
                            key=lambda i: digest([seed, stratum, i["item_id"]]))
        chosen = []
        for item in candidates:
            text_key = digest(item["user_text"])
            if item["group_id"] in groups or text_key in texts:
                continue
            chosen.append(item)
            groups.add(item["group_id"])
            texts.add(text_key)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError("Insufficient independent items for " + stratum)
        selected.extend(chosen)
    return selected


def render_messages(item, condition):
    # Task system instructions stay fixed, before the experimental intervention.
    system = "\n\n".join(p for p in (item["base_system"], condition["text"]) if p)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": item["user_text"]})
    return messages


def prepare(cache, download=False):
    population, sources = load_population(cache, download)
    conditions = []
    for name in ("registry_control.yaml", "registry_v3.yaml"):
        path = ROOT / "prompts" / name
        registry = load_registry(str(path))
        validate_registry_schema(registry)
        sources.append(source_record(path))
        for family, clarity, variant, text in iter_prompt_triples(registry):
            conditions.append({"condition_id": f"{family}/{clarity or 'none'}/{variant}",
                               "family": family, "clarity": clarity, "variant": variant,
                               "text": text})
    if len(conditions) != 73 or len({c["condition_id"] for c in conditions}) != 73:
        raise ValueError("Expected exactly 73 unique prompt conditions")
    main = sample(population, MAIN_QUOTAS, SEED + "/main")
    pilot = sample(population, PILOT_QUOTAS, SEED + "/pilot",
                   [i["group_id"] for i in main], [digest(i["user_text"]) for i in main])
    manifest = {
        "schema_version": 1, "experiment": "frontier-extension-v1", "seed": SEED,
        "design": "exploratory; within-item; one model per provider",
        "sample_method": "SHA256 rank by seed/stratum/item; skip duplicate parents and exact user texts",
        "main_quotas": MAIN_QUOTAS, "pilot_quotas": PILOT_QUOTAS,
        "iheval_composition": "task system then intervention, separated by two newlines",
        "control_semantics": "no added intervention; IHEval retains its required task system",
        "judge_model": "meta-llama/Llama-3.1-8B-Instruct",
        "population_counts": dict(sorted(Counter(i["stratum"] for i in population).items())),
        "sources": sources, "conditions": conditions, "main": main, "pilot": pilot,
        "pilot_condition_ids": PILOT_CONDITIONS,
    }
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def validate_manifest(manifest):
    content = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest(content) != manifest["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    if len(manifest["conditions"]) != 73:
        raise ValueError("Expected 73 conditions")
    groups, texts = set(), set()
    for split, quotas in (("main", MAIN_QUOTAS), ("pilot", PILOT_QUOTAS)):
        if Counter(i["stratum"] for i in manifest[split]) != Counter(quotas):
            raise ValueError("Sample quota mismatch")
        for item in manifest[split]:
            text_key = digest(item["user_text"])
            if item["group_id"] in groups or text_key in texts:
                raise ValueError("Sample/pilot overlap or duplicate query")
            groups.add(item["group_id"])
            texts.add(text_key)
            if not item["user_text"]:
                raise ValueError("Empty user input")
            if item["benchmark"] == "iheval" and not item["base_system"]:
                raise ValueError("Missing IHEval task system")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch pinned public IHEval inputs")
    parser.add_argument("--cache", type=Path, default=ROOT / ".local/iheval")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/frontier/manifest.json")
    args = parser.parse_args()
    manifest = validate_manifest(prepare(args.cache, args.download))
    data = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != data:
        raise SystemExit("Refusing to replace a different frozen manifest; choose a new --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data)
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "main_items": 50,
                      "pilot_items": 10, "conditions": 73, "main_requests_three_models": 10950,
                      "pilot_requests_three_models": 150}))


if __name__ == "__main__":
    main()
