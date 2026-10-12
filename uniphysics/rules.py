"""Deterministic scoring for closed-form answers."""
import re
import unicodedata
from typing import Any
from .extract import extract_answer

RULE_ANSWER_TYPES = frozenset({"mcq", "mcq(multiple)", "boolean"})
LATEX_TEXT_COMMAND = re.compile(
    r"\\(?:text|textrm|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}",
    flags=re.IGNORECASE,
)
ANSWER_PREFIX = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?"
    r"(?:answer|options?|choices?)(?:\s+(?:is|are))?\s*[:=\-]?\s*",
    flags=re.IGNORECASE,
)


def _answer_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    text = "" if value is None else str(value)
    extracted, _ = extract_answer(text)
    return extracted.strip()


def _strip_latex_wrappers(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _answer_text(value))
    previous = None
    while previous != text:
        previous = text
        text = LATEX_TEXT_COMMAND.sub(r"\1", text)
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = text.replace(r"\[", "").replace(r"\]", "")
    return text.strip().strip("$").strip()


def _single_choice_letter(value: Any, *, gold: bool = False) -> str | None:
    text = _strip_latex_wrappers(value)
    if gold:
        match = re.match(r"^\s*([A-J])\s*[.)：:]", text, flags=re.IGNORECASE)
        return match.group(1).upper() if match else _single_choice_letter(text)

    text = ANSWER_PREFIX.sub("", text)
    pattern = (
        r"^\s*(?:(?:option|choice)\s*)?"
        r"[({\[]?\s*([A-J])\s*[)}\]]?"
        r"(?:\s*[.：:]\s*.*)?\s*$"
    )
    match = re.fullmatch(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).upper() if match else None


def _multiple_choice_letters(value: Any) -> frozenset[str] | None:
    text = ANSWER_PREFIX.sub("", _strip_latex_wrappers(value))
    text = re.sub(
        r"\s+(?:(?:is|are)\s+)?correct\s*[.!]?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:option|choice)s?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[{}()\[\]]", "", text).strip()

    if re.fullmatch(r"[A-J]{1,10}", text, flags=re.IGNORECASE):
        return frozenset(text.upper())

    parts = re.split(r"\s*(?:[,;/&+]|\band\b)\s*", text, flags=re.IGNORECASE)
    if not parts or any(not re.fullmatch(r"[A-J]", part, re.IGNORECASE) for part in parts):
        return None
    return frozenset(part.upper() for part in parts)


def _canonical_choice_text(value: Any, *, remove_gold_letter: bool = False) -> str:
    text = _strip_latex_wrappers(value)
    if remove_gold_letter:
        text = re.sub(r"^\s*[A-J]\s*[.)：:]\s*", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"\\(?:,|;|:|!|quad|qquad|~)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", "", text).lower()
    text = text.translate(str.maketrans("", "", "${}[]()"))
    return text.rstrip(".;,。；，")


def _boolean_values(value: Any) -> tuple[bool, ...] | None:
    text = ANSWER_PREFIX.sub("", _strip_latex_wrappers(value)).lower()
    text = re.sub(r"[{}()\[\]]", "", text).strip().rstrip(".。")
    parts = re.split(r"\s*(?:[,;/&+]|\band\b)\s*", text)
    mapping = {
        "true": True,
        "yes": True,
        "correct": True,
        "false": False,
        "no": False,
        "incorrect": False,
    }
    if not parts or any(part not in mapping for part in parts):
        return None
    return tuple(mapping[part] for part in parts)


def rule_decision(answer_type: Any, candidate: Any, gold: Any) -> bool:
    """Judge a closed-form answer using the same strictness as JUDGE_PROMPT."""
    normalized_type = str(answer_type).strip().lower()

    if normalized_type == "mcq":
        gold_letter = _single_choice_letter(gold, gold=True)
        candidate_letter = _single_choice_letter(candidate)
        if candidate_letter is not None:
            return gold_letter is not None and candidate_letter == gold_letter

        candidate_text = _canonical_choice_text(candidate)
        gold_text = _canonical_choice_text(gold, remove_gold_letter=True)
        return bool(candidate_text) and candidate_text == gold_text

    if normalized_type == "mcq(multiple)":
        candidate_letters = _multiple_choice_letters(candidate)
        gold_letters = _multiple_choice_letters(gold)
        return candidate_letters is not None and candidate_letters == gold_letters

    if normalized_type == "boolean":
        candidate_values = _boolean_values(candidate)
        gold_values = _boolean_values(gold)
        return candidate_values is not None and candidate_values == gold_values

    raise ValueError(f"Rule judging does not support answer type: {answer_type!r}")


