"""Tests for the Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models import FunctionCall, FunctionCallResult, FunctionDefinition


def test_function_definition_accepts_all_scalar_types() -> None:
    payload = {
        "name": "fn_all",
        "description": "demonstrates every scalar type",
        "parameters": {
            "s": {"type": "string"},
            "n": {"type": "number"},
            "i": {"type": "integer"},
            "b": {"type": "boolean"},
        },
        "returns": {"type": "string"},
    }
    func = FunctionDefinition.model_validate(payload)
    assert {name: spec.type for name, spec in func.parameters.items()} == {
        "s": "string",
        "n": "number",
        "i": "integer",
        "b": "boolean",
    }


def test_function_definition_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(
            {
                "name": "fn_bad",
                "description": "x",
                "parameters": {"a": {"type": "array"}},
                "returns": {"type": "string"},
            }
        )


def test_function_definition_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(
            {
                "name": "fn_x",
                "description": "x",
                "parameters": {},
                "returns": {"type": "string"},
                "extra": True,
            }
        )


def test_function_call_rejects_blank_prompt() -> None:
    with pytest.raises(ValidationError):
        FunctionCall.model_validate({"prompt": "   "})


def test_function_call_result_accepts_mixed_value_types() -> None:
    result = FunctionCallResult(
        prompt="p",
        name="fn_x",
        parameters={"a": 1, "b": "two", "c": True, "d": 3.14},
    )
    assert result.parameters["a"] == 1
    assert result.parameters["c"] is True
