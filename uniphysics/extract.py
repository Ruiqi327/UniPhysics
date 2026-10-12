import re


BOXED_OPEN_PATTERN = re.compile(
    r"(?<![A-Za-z])\\?boxed[ \t]*\{",
    flags=re.IGNORECASE,
)
FINAL_ANSWER_PATTERN = re.compile(
    r"\bfinal[ \t]+answer\b"
    r"(?:[ \t]+is)?[ \t]*(?:[:：=-][ \t]*)?([^\r\n]+)",
    flags=re.IGNORECASE,
)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _extract_last_boxed(response: str) -> str | None:
    """Extract the last complete boxed value while preserving nested LaTeX."""
    matches = list(BOXED_OPEN_PATTERN.finditer(response))
    if not matches:
        return None

    opening_brace = matches[-1].end() - 1
    depth = 0
    for index in range(opening_brace, len(response)):
        character = response[index]
        if character == "{" and not _is_escaped(response, index):
            depth += 1
        elif character == "}" and not _is_escaped(response, index):
            depth -= 1
            if depth == 0:
                return response[opening_brace + 1 : index].strip()

    return None


def extract_answer(response: str) -> tuple[str, str]:
    """
    Supported formats (matches the LAST occurrence from the end):
    1. boxed{answer}
    2. <answer>answer</answer>
    3. Final answer: ...
    4. **** answer

    If all extraction attempts fail, returns the last 300 characters.
    """

    # 1. Extract the last \boxed{} with balanced nested braces.
    boxed_answer = _extract_last_boxed(response)
    if boxed_answer is not None:
        return boxed_answer, "boxed"

    # 2. Try to extract from <answer></answer> (find LAST match)
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_tag_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip(), "xml"

    # 3. Extract the last "Final answer" line without splitting decimals.
    matches = FINAL_ANSWER_PATTERN.findall(response)
    if matches:
        answer = re.sub(r"\s*\*{4}\s*$", "", matches[-1]).strip()
        return answer, "final"

    # 4. Try to extract from "****" format (find LAST match)
    star_pattern = r'\*\*\*\*\s*(.+?)(?:\n|$)'
    matches = re.findall(star_pattern, response)
    if matches:
        return matches[-1].strip(), "star"

    # 5. If all extraction attempts fail, return last 300 characters
    if len(response) > 300:
        return response[-300:], "char"
    return response, "raw"


def extract_judge(response: str) -> str | None:
    matches = re.findall(
        r"<judge\s*>(.*?)</judge\s*>",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        return None

    value = matches[-1].strip().lower()
    if value not in {"true", "false"}:
        return None

    return value
