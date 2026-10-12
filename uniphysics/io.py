"""Validated JSONL and atomic metadata writes shared by all workflows."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{Path(path).name}:{number}: invalid JSON ({error.msg})") from None
            if not isinstance(record, dict):
                raise ValueError(f"{Path(path).name}:{number}: expected a JSON object")
            records.append(record)
    return records


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records) -> None:
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".uniphysics-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_record(stream, record: dict) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()
