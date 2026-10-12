"""Generation and hybrid judging orchestration, independent of CLI parsing."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from .api import APIClient, APIConfig
from .data import load_dataset
from .extract import extract_answer, extract_judge
from .io import fingerprint, read_jsonl, write_json, write_jsonl, write_record
from .metrics import run_metric
from .prompts import JUDGE_PROMPT, SYSTEM_PROMPT
from .rules import RULE_ANSWER_TYPES, rule_decision

ERROR_RESPONSES = {"API_ERROR", "LOCAL_ERROR"}


def code_fingerprint():
    return fingerprint({p.name: p.read_text(encoding="utf-8")
                        for p in sorted(Path(__file__).parent.glob("*.py"))})


def run_directory(output_root: Path, project: str) -> Path:
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", project):
        raise ValueError("project_name must contain only letters, numbers, dots, underscores and hyphens")
    directory = output_root / project
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def run_lock(directory: Path):
    import os
    lock = directory / ".run.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError("Run is locked. If its process has stopped, remove .run.lock and resume.") from None
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def cached_rows(path: Path):
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except ValueError:
        # An interrupted final JSONL write must never be accepted as a valid cache.
        return []


def generate(args, config, api_key=None, base_url=None):
    records = load_dataset(args.uniphysics, args.benchmark, args.test_mode, args.limit)
    prompts = [SYSTEM_PROMPT(r["answer_type"]) + "\n" + r["problem"] for r in records]
    directory = run_directory(args.output_dir, args.project_name)
    parameters = asdict(config)
    if "model_path" in parameters:
        # A local model path can identify a person or private directory.
        model = parameters["model_path"]
        if Path(model).exists() or Path(model).is_absolute():
            parameters["model_path"] = Path(model).name
    manifest = {"backend": args.command, "configuration": parameters,
                "dataset_sha256": fingerprint(records), "selected_records": len(records),
                "benchmark": args.benchmark, "test_mode": args.test_mode,
                "endpoint_sha256": fingerprint(base_url or "default"),
                "model_identity_sha256": fingerprint(asdict(config)),
                "code_sha256": code_fingerprint()}
    path = directory / "response.jsonl"
    metadata = directory / "run.json"
    with run_lock(directory):
        if args.resume:
            if not metadata.exists() or json.loads(metadata.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Cannot resume: data, code or generation configuration changed (or run.json is absent)")
        elif path.exists() or metadata.exists():
            raise ValueError("Run already exists; use --resume with identical settings or a new project_name")
        write_json(metadata, manifest)
        write_jsonl(directory / "data.jsonl", records)
        valid_ids = {r["id"] for r in records}
        completed = {r["id"]: r for r in cached_rows(path)
                     if args.resume and r.get("id") in valid_ids and
                     isinstance(r.get("response"), str) and r["response"].strip() and
                     r["response"] not in ERROR_RESPONSES and r.get("run_sha256") == fingerprint(manifest)}
        remaining = [i for i, record in enumerate(records) if record["id"] not in completed]
        write_jsonl(path, completed.values())
        client = None
        try:
            if remaining:
                if args.command == "api":
                    client = APIClient(api_key, base_url, config)
                    iterator = client.iter_results([prompts[i] for i in remaining])
                else:
                    from .local import call_local_iter
                    iterator = call_local_iter([prompts[i] for i in remaining], config)
                with path.open("a", encoding="utf-8", newline="\n") as stream:
                    for index, response in iterator:
                        record = dict(records[remaining[index]])
                        if args.command == "api":
                            record.update(api_attempts=response.attempts, api_retry_errors=list(response.errors))
                            response = response.response
                        extracted, match = extract_answer(response)
                        record.update(response=response, extracted_answer=extracted, extract_match_type=match,
                                      model=parameters.get("model", parameters.get("model_path")),
                                      generation_status="error" if response in ERROR_RESPONSES or not response.strip() else "ok",
                                      run_sha256=fingerprint(manifest))
                        write_record(stream, record)
                        completed[record["id"]] = record
        finally:
            if client is not None:
                client.close()
        write_jsonl(path, [completed[r["id"]] for r in records])
    failed = sum(r.get("generation_status") == "error" for r in completed.values())
    print(f"Wrote {len(completed)} responses to {path}; failures: {failed}")
    return failed


def judge_prompt(record):
    candidate = record.get("extracted_answer")
    if candidate is None:
        candidate, _ = extract_answer(str(record.get("response", "")))
    if isinstance(candidate, (list, tuple)):
        candidate = candidate[0] if candidate else ""
    return (JUDGE_PROMPT(record["answer_type"]) + "\nQuestion: " + record["problem"]
            + "\nModel's answer: " + str(candidate) + "\nGold answer: " + record["answer"])


def judge_stage(records, indices, path, config, api_key, base_url, resume):
    prompts = {i: judge_prompt(records[i]) for i in indices}
    implementation_hash = code_fingerprint()
    identities = {i: fingerprint({"id": records[i]["id"], "prompt": prompts[i],
                                  "configuration": config.public_dict(), "endpoint": base_url,
                                  "code": implementation_hash}) for i in indices}
    cached = {r.get("id"): r for r in cached_rows(path)} if resume else {}
    results, rows, remaining = {}, {}, []
    for index in indices:
        previous = cached.get(records[index]["id"], {})
        if previous.get("request_sha256") == identities[index] and extract_judge(str(previous.get("response", ""))) is not None:
            results[index] = previous["response"]
            rows[index] = previous
        else:
            remaining.append(index)
    write_jsonl(path, rows.values())
    if remaining:
        if not api_key:
            raise ValueError("Set UNIPHYSICS_API_KEY or OPENAI_API_KEY for open-ended judging")
        client = APIClient(api_key, base_url, config)
        try:
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                for position, result in client.iter_results([prompts[i] for i in remaining]):
                    index = remaining[position]
                    results[index] = result.response
                    write_record(stream, {"index": index, "id": records[index]["id"],
                                          "request_sha256": identities[index], "model": config.model,
                                          "response": result.response, "api_attempts": result.attempts,
                                          "api_retry_errors": list(result.errors)})
        finally:
            client.close()
    return results


def judge(args, config: APIConfig, api_key=None, base_url=None):
    directory = run_directory(args.output_dir, args.project_name)
    records = read_jsonl(directory / "response.jsonl")
    if not records or len({r.get("id") for r in records}) != len(records):
        raise ValueError("Responses must be nonempty and have unique IDs")
    if (directory / "run.json").exists():
        expected = json.loads((directory / "run.json").read_text(encoding="utf-8"))["selected_records"]
        if len(records) != expected:
            raise ValueError("Generation is incomplete; resume generation before judging")
    final, api_indices = {}, []
    for index, record in enumerate(records):
        if not record.get("answer", "").strip():
            final[index] = (None, "missing_gold", "skipped")
        elif record.get("response") in ERROR_RESPONSES or record.get("generation_status") == "error":
            final[index] = (None, "generation_error", "skipped")
        elif not str(record.get("response", "")).strip() and not str(record.get("extracted_answer", "")).strip():
            final[index] = (False, "ok", "rule")
        elif record.get("answer_type", "").lower().strip() in RULE_ANSWER_TYPES:
            candidate = record.get("extracted_answer")
            if candidate is None:
                candidate, _ = extract_answer(record.get("response", ""))
            final[index] = (rule_decision(record["answer_type"], candidate, record["answer"]), "ok", "rule")
        else:
            # Validate type before any billable request.
            judge_prompt(record)
            api_indices.append(index)
    if api_indices and (not api_key or not all((args.model_1, args.model_2, args.model_3))):
        raise ValueError("Open-ended judging requires an API key and --model_1, --model_2, --model_3")
    with run_lock(directory):
        if (directory / "judge.jsonl").exists() and not args.resume:
            raise ValueError("Judge results already exist; use --resume to reuse matching stages and retry failures")
        stages = []
        for number, model in enumerate((args.model_1, args.model_2), 1):
            stages.append(judge_stage(records, api_indices, directory / f"judge_model_{number}.jsonl",
                                      replace(config, model=model or "unused"), api_key, base_url, args.resume))
        disagreements = []
        for index in api_indices:
            first, second = extract_judge(stages[0][index]), extract_judge(stages[1][index])
            if first is not None and first == second:
                final[index] = (first == "true", "ok", "api")
            else:
                disagreements.append(index)
        third = judge_stage(records, disagreements, directory / "judge_model_3.jsonl",
                            replace(config, model=args.model_3 or "unused"), api_key, base_url, args.resume)
        for index in disagreements:
            verdict = extract_judge(third[index])
            final[index] = (verdict == "true" if verdict is not None else None,
                            "ok" if verdict is not None else "api_error", "api")
        output = []
        for index, record in enumerate(records):
            verdict, status, strategy = final[index]
            output.append({**record, "judge": verdict, "judge_status": status, "judge_strategy": strategy,
                           "model1_judge": stages[0].get(index, ""), "model2_judge": stages[1].get(index, ""),
                           "model3_judge": third.get(index, "")})
        write_jsonl(directory / "judge.jsonl", output)
        write_json(directory / "judge_config.json", {"models": [args.model_1, args.model_2, args.model_3],
                   "configuration": config.public_dict(), "endpoint_sha256": fingerprint(base_url),
                   "response_sha256": fingerprint(records), "code_sha256": code_fingerprint()})
        metrics = run_metric(directory / "judge.jsonl")
    print(f"Wrote {len(records)} judgments to {directory / 'judge.jsonl'}; "
          f"unresolved: {metrics['overall']['unresolved']}")
    return metrics["overall"]["api_errors"] + metrics["overall"]["generation_errors"]
