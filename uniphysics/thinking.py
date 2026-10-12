from __future__ import annotations

import argparse
import re


_QWEN_VERSION = re.compile(r"qwen\s*(\d+)", re.IGNORECASE)
_TRAILING_THINK_SWITCH = re.compile(
    r"\s*/(?:no_)?think\s*$",
    re.IGNORECASE,
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def supports_qwen_thinking(model: str) -> bool:
    match = _QWEN_VERSION.search(model)
    return match is not None and int(match.group(1)) >= 3


def add_qwen_thinking_switch(
    prompt: str,
    model: str,
    enable_thinking: bool,
) -> str:
    if not supports_qwen_thinking(model):
        return prompt

    prompt_without_switch = _TRAILING_THINK_SWITCH.sub("", prompt.rstrip())
    switch = "/think" if enable_thinking else "/no_think"
    return f"{prompt_without_switch}\n{switch}"
