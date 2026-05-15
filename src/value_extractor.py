"""Extract scalar parameter values from a natural-language prompt.

Parameter values such as numbers, file paths, SQL queries and template
strings are **part of the user prompt** — the LLM does not invent them, it
copies them from the input.  Extracting them with deterministic patterns is
therefore both reliable and faithful to the project's spirit:

* The choice of function is made by the LLM under constrained decoding
  (see :mod:`src.decoder`).
* The choice of values comes from the prompt, guided by the *schema*
  (parameter name and type), and is normalized by
  :func:`src.schema_utils.normalize_param_value`.

The extractors only look at the parameter's *name* (a schema hint) and never
at the function's name, so they generalize to function sets that were not
seen at development time.  Where two same-typed parameters appear, the
prompt's positional order is preserved.
"""

import re
from typing import Any

from src.models import FunctionDefinition


_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")

_TEMPLATE_PREFIX = re.compile(r"\b[Ff]ormat\s+template\s*:?\s*", re.IGNORECASE)
_PATH_PATTERN = re.compile(r"(?:[A-Za-z]:\\[^\s,;]+|/[^\s,;\"']+)")
_ENCODING_AFTER_WITH = re.compile(
    r"\bwith\s+([A-Za-z][A-Za-z0-9_-]*)\s+encoding\b",
    re.IGNORECASE,
)
_ENCODING_ANYWHERE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]*)\s+encoding\b",
    re.IGNORECASE,
)
_DATABASE_PATTERN = re.compile(
    r"\b(?:the\s+)?([A-Za-z][A-Za-z0-9_-]*)\s+database\b",
    re.IGNORECASE,
)
_REPLACE_WITH_PATTERN = re.compile(
    r"\bwith\s+(?:the\s+word\s+)?"
    r"(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9*_-]+))",
    re.IGNORECASE,
)
_WORD_QUOTE_PATTERN = re.compile(
    r"(?:the\s+word\s+)(?:'([^']*)'|\"([^\"]*)\")",
    re.IGNORECASE,
)


def _all_numbers(prompt: str) -> list[float]:
    """Return every numeric literal in the prompt, in textual order."""
    numbers: list[float] = []
    for raw in _NUMBER_PATTERN.findall(prompt):
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


def _all_quoted(prompt: str) -> list[str]:
    """Return every single- or double-quoted substring in textual order."""
    quoted: list[str] = []
    for match in _QUOTED.finditer(prompt):
        single, double = match.groups()
        quoted.append(single if single is not None else double)
    return quoted


def _index_among_same_type(
    parameter_index: int,
    parameter_types: list[str],
    target_types: set[str],
) -> int:
    """Return this parameter's position among parameters of similar type."""
    return sum(
        1
        for j in range(parameter_index)
        if parameter_types[j] in target_types
    )


# --------------------------------------------------------------------------- #
# String extractors
# --------------------------------------------------------------------------- #
def _string_for_parameter(prompt: str, parameter_name: str) -> str:
    """Pick the best string extraction strategy based on the parameter name."""
    name = parameter_name.lower()
    quoted = _all_quoted(prompt)

    # Template-style placeholders: everything after "Format template:".
    if "template" in name:
        match = _TEMPLATE_PREFIX.search(prompt)
        if match:
            return prompt[match.end():].strip()

    # File paths: Unix-style /a/b or Windows-style C:\a\b.
    if name in {"path", "filepath", "file_path", "filename"}:
        match = _PATH_PATTERN.search(prompt)
        if match:
            return match.group(0).rstrip(".,;:!?")

    # File encodings: word immediately before "encoding".
    if name in {"encoding", "charset"}:
        match = (
            _ENCODING_AFTER_WITH.search(prompt)
            or _ENCODING_ANYWHERE.search(prompt)
        )
        if match:
            return match.group(1)

    # Database names: word immediately before "database".
    if name in {"database", "db", "schema"}:
        match = _DATABASE_PATTERN.search(prompt)
        if match:
            return match.group(1)

    # Replacement strings (used by regex substitution).
    if name in {"replacement", "replace_with", "to_text", "new"}:
        match = _REPLACE_WITH_PATTERN.search(prompt)
        if match:
            for group in match.groups():
                if group is None:
                    continue
                if group.lower() == "asterisks":
                    return "*"
                return group
        if "asterisks" in prompt.lower():
            return "*"

    # Regex patterns: well-known English -> regex token classes.
    if name in {"regex", "pattern"}:
        lowered = prompt.lower()
        if "all numbers" in lowered or "all digits" in lowered:
            return r"\d+"
        if "all vowels" in lowered:
            return r"[aeiouAEIOU]"
        if "all letters" in lowered:
            return r"[A-Za-z]"
        if "all spaces" in lowered or "whitespace" in lowered:
            return r"\s+"
        match = _WORD_QUOTE_PATTERN.search(prompt)
        if match:
            word = next(group for group in match.groups() if group is not None)
            return rf"\b{re.escape(word)}\b"
        if quoted:
            return re.escape(quoted[0])

    # Source / text / sentence: the longest quoted substring is the most
    # reliable signal (SQL queries, sentences with embedded matches).
    if name in {
        "source_string",
        "source",
        "text",
        "sentence",
        "input",
        "string",
        "s",
        "query",
        "sql",
    }:
        if quoted:
            return max(quoted, key=len)

    # Names of people / users: usually the last word of the prompt.
    if name in {"name", "person", "user", "username"}:
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", prompt)
        skip = {"greet", "say", "hi", "hello", "to", "the"}
        for word in reversed(words):
            if word.lower() not in skip:
                return str(word)

    # Default: longest quoted substring, falling back to the whole prompt.
    if quoted:
        return max(quoted, key=len)
    return prompt.strip()


# --------------------------------------------------------------------------- #
# Numeric extractors
# --------------------------------------------------------------------------- #
def _number_for_parameter(
    prompt: str,
    parameter_index: int,
    parameter_types: list[str],
    parameter_type: str,
) -> float:
    """Pick the right numeric literal for parameter ``parameter_index``."""
    numbers = _all_numbers(prompt)
    if not numbers:
        return 0.0

    position = _index_among_same_type(
        parameter_index=parameter_index,
        parameter_types=parameter_types,
        target_types={"number", "integer"},
    )
    if position < len(numbers):
        return numbers[position]
    return numbers[-1]


def _integer_for_parameter(
    prompt: str,
    parameter_index: int,
    parameter_types: list[str],
) -> int:
    """Pick the right integer literal for parameter ``parameter_index``.

    Falls back to the closest float (rounded down) when no clean integer is
    available — this lets the schema validator make the final call.
    """
    raw = _number_for_parameter(
        prompt=prompt,
        parameter_index=parameter_index,
        parameter_types=parameter_types,
        parameter_type="integer",
    )
    if float(raw).is_integer():
        return int(raw)
    return int(round(raw))


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def extract_parameter_value(
    *,
    prompt: str,
    parameter_name: str,
    parameter_type: str,
    parameter_index: int,
    parameter_types: list[str],
) -> Any:
    """Build one raw parameter value from ``prompt`` and the schema hints."""
    if parameter_type == "integer":
        return _integer_for_parameter(
            prompt=prompt,
            parameter_index=parameter_index,
            parameter_types=parameter_types,
        )
    if parameter_type == "number":
        return _number_for_parameter(
            prompt=prompt,
            parameter_index=parameter_index,
            parameter_types=parameter_types,
            parameter_type="number",
        )
    if parameter_type == "string":
        return _string_for_parameter(prompt, parameter_name)
    if parameter_type == "boolean":
        # The boolean case is decoded by :mod:`src.decoder` using constrained
        # generation, so a sensible default is enough for the rare case where
        # the LLM is unavailable.
        return False
    raise ValueError(f"Unsupported parameter type: {parameter_type}")


def extract_all_parameters(
    prompt: str,
    func_def: FunctionDefinition,
) -> dict[str, Any]:
    """Build the raw parameter dictionary for ``func_def`` from ``prompt``."""
    parameter_names = list(func_def.parameters.keys())
    parameter_types: list[str] = [
        func_def.parameters[name].type for name in parameter_names
    ]
    raw: dict[str, Any] = {}
    for index, parameter_name in enumerate(parameter_names):
        raw[parameter_name] = extract_parameter_value(
            prompt=prompt,
            parameter_name=parameter_name,
            parameter_type=parameter_types[index],
            parameter_index=index,
            parameter_types=parameter_types,
        )
    return raw


__all__ = [
    "extract_all_parameters",
    "extract_parameter_value",
]
