"""Portable dataset discovery, source selection, and schema validation."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from .io import read_jsonl

BENCHMARKS = ("JEEBench", "MMLU-Pro", "GPQA", "PhysicsQA", "PHYSICS",
              "UGPhysics", "PhyBench", "TheoremQA", "SciBench", "HiPhO")
ANSWER_TYPES = {"mcq", "mcq(multiple)", "boolean", "open-ended"}
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "Eval/benchmark/uniphysics.jsonl"


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def benchmark_name(identifier: str) -> str:
    prefix = re.split(r"\d", identifier, maxsplit=1)[0].rstrip("_")
    return next((name for name in BENCHMARKS if normalize(name) == normalize(prefix)), prefix)


def load_dataset(path: Path, benchmark="all", test_mode="all", limit=None) -> list[dict]:
    records = read_jsonl(path)
    seen = set()
    for number, record in enumerate(records, 1):
        for field in ("id", "problem", "answer_type"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Record {number}: missing or empty {field}")
        if "answer" not in record or not isinstance(record["answer"], str):
            raise ValueError(f"Record {number}: answer must be a string (empty allowed for removed data)")
        if record["id"] in seen:
            raise ValueError(f"Duplicate dataset ID: {record['id']}")
        seen.add(record["id"])
        if record["answer_type"].lower().strip() not in ANSWER_TYPES:
            raise ValueError(f"Record {number}: unsupported answer_type")
    requested = {normalize(s.strip()) for s in benchmark.split(",") if s.strip()}
    available = {normalize(benchmark_name(r["id"])) for r in records}
    if not requested or requested - available - {normalize(n) for n in BENCHMARKS} - {"all"}:
        raise ValueError(f"Unknown benchmark selection: {benchmark}")
    if "all" not in requested:
        records = [r for r in records if normalize(benchmark_name(r["id"])) in requested]
    if test_mode == "mini":
        counts = Counter()
        selected = []
        for record in records:
            name = benchmark_name(record["id"])
            if counts[name] < 500:
                selected.append(record)
            counts[name] += 1
        records = selected
    elif test_mode != "all":
        raise ValueError("test_mode must be all or mini")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        records = records[:limit]
    if not records:
        raise ValueError("Dataset selection is empty")
    return records


def dataset_summary(records: list[dict]) -> dict:
    return {
        "records": len(records),
        "benchmarks": dict(sorted(Counter(benchmark_name(r["id"]) for r in records).items())),
        "answer_types": dict(Counter(r["answer_type"] for r in records)),
        "question_types": dict(Counter(r.get("question_type", "Unknown") for r in records)),
        "domains": dict(Counter(r.get("domain", "Unknown") for r in records)),
        "empty_answers": sum(not r["answer"].strip() for r in records),
    }
