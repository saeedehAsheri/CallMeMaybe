"""JSON input/output helpers for the project.

Every error path raises :class:`InputFileError` (a subclass of
:class:`ProjectError`) so the entry point can produce a single, clear error
message even when something goes wrong inside ``json`` or ``pydantic``.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.errors import InputFileError
from src.models import FunctionCall, FunctionCallResult, FunctionDefinition


def load_json_file(path: Path) -> Any:
    """Load a JSON file and raise :class:`InputFileError` on any failure."""
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except FileNotFoundError as exc:
        raise InputFileError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputFileError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise InputFileError(f"Could not read {path}: {exc}") from exc


def write_json_file(path: Path, results: list[FunctionCallResult]) -> None:
    """Write schema-compliant results as valid JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump() for result in results]
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)
            file_handle.write("\n")
    except OSError as exc:
        raise InputFileError(
            f"Could not write output file {path}: {exc}"
        ) from exc


def load_func_def(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions from JSON."""
    data = load_json_file(path)
    if not isinstance(data, list):
        raise InputFileError(
            "Function definitions file must contain a JSON array."
        )
    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as exc:
        raise InputFileError(
            f"Invalid function definitions schema: {exc}"
        ) from exc


def load_func_call(path: Path) -> list[FunctionCall]:
    """Load and validate prompt items from JSON."""
    data = load_json_file(path)
    if not isinstance(data, list):
        raise InputFileError("Prompt input file must contain a JSON array.")
    try:
        return [FunctionCall.model_validate(item) for item in data]
    except ValidationError as exc:
        raise InputFileError(f"Invalid prompt input schema: {exc}") from exc


__all__ = [
    "load_func_call",
    "load_func_def",
    "load_json_file",
    "write_json_file",
]
