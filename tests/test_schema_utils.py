"""Tests for the schema validation helpers."""

import pytest

from src.models import FunctionDefinition
from src.schema_utils import (
    get_function_by_name,
    normalize_param_value,
    validate_params_against_schema,
)


def _build_function() -> FunctionDefinition:
    """Build a small function definition for tests."""
    func_def: FunctionDefinition = FunctionDefinition.model_validate(
        {
            "name": "fn_test",
            "description": "test function",
            "parameters": {
                "a": {"type": "number"},
                "n": {"type": "integer"},
                "label": {"type": "string"},
                "active": {"type": "boolean"},
            },
            "returns": {"type": "string"},
        }
    )
    return func_def


def test_normalize_integer_keeps_int() -> None:
    assert normalize_param_value("integer", 5) == 5


def test_normalize_integer_from_clean_float() -> None:
    assert normalize_param_value("integer", 5.0) == 5


def test_normalize_integer_rejects_non_integer_float() -> None:
    with pytest.raises(ValueError):
        normalize_param_value("integer", 5.3)


def test_normalize_integer_from_string() -> None:
    assert normalize_param_value("integer", "42") == 42


def test_normalize_integer_rejects_boolean() -> None:
    with pytest.raises(ValueError):
        normalize_param_value("integer", True)


def test_normalize_number_returns_float() -> None:
    result = normalize_param_value("number", 5)
    assert isinstance(result, float)
    assert result == 5.0


def test_normalize_string_casts_to_str() -> None:
    assert normalize_param_value("string", 42) == "42"


def test_normalize_boolean_recognizes_yes() -> None:
    assert normalize_param_value("boolean", "yes") is True


def test_normalize_boolean_recognizes_no() -> None:
    assert normalize_param_value("boolean", "no") is False


def test_get_function_by_name_returns_matching() -> None:
    func = _build_function()
    assert get_function_by_name([func], "fn_test") is func


def test_get_function_by_name_returns_none_when_missing() -> None:
    func = _build_function()
    assert get_function_by_name([func], "fn_other") is None


def test_validate_params_normalizes_each_value() -> None:
    func = _build_function()
    result = validate_params_against_schema(
        func,
        {"a": "1.5", "n": "3", "label": 7, "active": "yes"},
    )
    assert result == {"a": 1.5, "n": 3, "label": "7", "active": True}
    assert isinstance(result["a"], float)
    assert isinstance(result["n"], int)


def test_validate_params_reports_missing_keys() -> None:
    func = _build_function()
    with pytest.raises(ValueError):
        validate_params_against_schema(func, {"a": 1.0})


def test_validate_params_reports_extra_keys() -> None:
    func = _build_function()
    with pytest.raises(ValueError):
        validate_params_against_schema(
            func,
            {
                "a": 1.0,
                "n": 1,
                "label": "x",
                "active": True,
                "extra": 1,
            },
        )
