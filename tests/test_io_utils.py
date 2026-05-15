"""Tests for the JSON I/O helpers."""

import json
from pathlib import Path

import pytest

from src.errors import InputFileError
from src.io_utils import (
    load_func_call,
    load_func_def,
    load_json_file,
    write_json_file,
)
from src.models import FunctionCallResult


def test_load_json_file_returns_data(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text('{"key": 42}', encoding="utf-8")
    assert load_json_file(target) == {"key": 42}


def test_load_json_file_missing(tmp_path: Path) -> None:
    with pytest.raises(InputFileError):
        load_json_file(tmp_path / "missing.json")


def test_load_json_file_invalid(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputFileError):
        load_json_file(target)


def test_load_func_def_must_be_array(tmp_path: Path) -> None:
    target = tmp_path / "defs.json"
    target.write_text('{"oops": true}', encoding="utf-8")
    with pytest.raises(InputFileError):
        load_func_def(target)


def test_load_func_def_accepts_valid_schema(tmp_path: Path) -> None:
    payload = [
        {
            "name": "fn_x",
            "description": "x",
            "parameters": {"a": {"type": "number"}},
            "returns": {"type": "number"},
        }
    ]
    target = tmp_path / "defs.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    funcs = load_func_def(target)
    assert len(funcs) == 1
    assert funcs[0].name == "fn_x"


def test_load_func_call_must_be_array(tmp_path: Path) -> None:
    target = tmp_path / "calls.json"
    target.write_text('"oops"', encoding="utf-8")
    with pytest.raises(InputFileError):
        load_func_call(target)


def test_load_func_call_rejects_empty_prompt(tmp_path: Path) -> None:
    target = tmp_path / "calls.json"
    target.write_text(json.dumps([{"prompt": ""}]), encoding="utf-8")
    with pytest.raises(InputFileError):
        load_func_call(target)


def test_write_json_file_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.json"
    results = [
        FunctionCallResult(
            prompt="prompt",
            name="fn_x",
            parameters={"a": 1, "b": 2.0, "c": "x"},
        )
    ]
    write_json_file(target, results)

    with target.open("r", encoding="utf-8") as file_handle:
        written = json.load(file_handle)
    assert written == [
        {
            "prompt": "prompt",
            "name": "fn_x",
            "parameters": {"a": 1, "b": 2.0, "c": "x"},
        }
    ]
