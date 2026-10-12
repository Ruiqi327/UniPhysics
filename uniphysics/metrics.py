"""Accuracy and coverage, including failed and unjudgeable records."""
from collections import defaultdict
from pathlib import Path
from .data import benchmark_name
from .io import atomic_text, read_jsonl, write_json


def decision(value):
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    return None


def counter(records):
    decisions = [decision(r.get("judge")) for r in records]
    total = len(records)
    judged = sum(d is not None for d in decisions)
    correct = sum(d is True for d in decisions)
    return {
        "total": total, "judged": judged, "correct": correct,
        "unresolved": total - judged,
        "api_errors": sum(r.get("judge_status") == "api_error" for r in records),
        "generation_errors": sum(r.get("judge_status") == "generation_error" for r in records),
        "missing_gold": sum(r.get("judge_status") == "missing_gold" for r in records),
        "accuracy": correct / judged if judged else None,
        "coverage": judged / total if total else None,
        "accuracy_over_all": correct / total if total else None,
    }


def calculate_metrics(records):
    result = {"overall": counter(records)}
    for dimension in ("benchmark", "domain", "question_type", "answer_type"):
        groups = defaultdict(list)
        for record in records:
            name = benchmark_name(record["id"]) if dimension == "benchmark" else record.get(dimension, "Unknown")
            groups[name or "Unknown"].append(record)
        result[dimension] = {name: counter(rows) for name, rows in sorted(groups.items())}
    scores = [c["accuracy"] for c in result["benchmark"].values() if c["accuracy"] is not None]
    result["macro_accuracy"] = sum(scores) / len(scores) if scores else None
    result["macro_benchmarks_judged"] = len(scores)
    result["micro_accuracy"] = result["overall"]["accuracy"]
    return result


def run_metric(path):
    path = Path(path)
    records = read_jsonl(path)
    if not records:
        raise ValueError("No judge results to summarize")
    result = calculate_metrics(records)
    write_json(path.with_name("metrics.json"), result)
    def pct(value):
        return "N/A" if value is None else f"{100 * value:.2f}%"
    lines = ["UniPhysics evaluation", "",
             f"Micro accuracy (correct / judged): {pct(result['micro_accuracy'])}",
             f"Macro accuracy (mean over judged sources): {pct(result['macro_accuracy'])}",
             f"Coverage (judged / total): {pct(result['overall']['coverage'])}",
             f"Conservative accuracy (correct / total): {pct(result['overall']['accuracy_over_all'])}",
             "Unresolved records are excluded from accuracy; always report coverage."]
    for dimension in ("benchmark", "domain", "question_type", "answer_type"):
        lines.extend(["", f"By {dimension}"])
        for name, counts in result[dimension].items():
            lines.append(f"{name}: {counts['correct']}/{counts['judged']} = {pct(counts['accuracy'])}; "
                         f"coverage {pct(counts['coverage'])}; unresolved {counts['unresolved']}")
    atomic_text(path.with_name("metric.txt"), "\n".join(lines) + "\n")
    return result
