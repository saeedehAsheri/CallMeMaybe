"""Schema validation helpers for generated function-call parameters.

This module is the single place where the project converts raw values
(extracted from the prompt or chosen by the constrained decoder) into the
exact Python type required by the schema.  The moulinette uses
``assert isinstance(value, <type>)`` checks when running the student's
function call, so the distinction between ``int`` and ``float`` is critical:
``fn_is_even(n=4)`` succeeds, but ``fn_is_even(n=4.0)`` fails.
"""

from typing import Any

from src.models import FunctionDefinition


def get_function_by_name(
    function_definitions: list[FunctionDefinition],
    name: str,
) -> FunctionDefinition | None:
    """Look up one function definition by name.

    Returns ``None`` if no definition has the given name.
    """
    for func_def in function_definitions:
        if func_def.name == name:
            return func_def
    return None


def normalize_param_value(expected_type: str, value: Any) -> Any:
    """Convert ``value`` to the Python type required by ``expected_type``.

    The output type follows JSON Schema conventions:

    * ``"string"`` -> ``str``
    * ``"integer"`` -> ``int``
    * ``"number"`` -> ``float``
    * ``"boolean"`` -> ``bool``

    Any other type, or a value that cannot be safely converted, raises
    :class:`ValueError`.
    """
    if expected_type == "string":
        return str(value)

    if expected_type == "integer":
        if isinstance(value, bool):
            raise ValueError("Boolean cannot be cast to integer.")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(
                    f"Cannot cast non-integer float {value!r} to integer."
                )
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return int(stripped)
            except ValueError as exc:
                # Some prompts spell numbers with a decimal point even when
                # the schema requires an integer (e.g. "23.0" -> 23).
                try:
                    float_value = float(stripped)
                except ValueError as inner_exc:
                    raise ValueError(
                        f"Cannot convert {value!r} to integer."
                    ) from inner_exc
                if not float_value.is_integer():
                    raise ValueError(
                        f"Cannot cast non-integer string {value!r} to integer."
                    ) from exc
                return int(float_value)
        raise ValueError(f"Cannot convert {value!r} to integer.")

    if expected_type == "number":
        if isinstance(value, bool):
            raise ValueError("Boolean cannot be cast to number.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Cannot convert {value!r} to number."
                ) from exc
        raise ValueError(f"Cannot convert {value!r} to number.")

    if expected_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"yes", "true", "1", "enabled", "on"}:
                return True
            if lowered in {"no", "false", "0", "disabled", "off"}:
                return False
        raise ValueError(f"Cannot convert {value!r} to boolean.")

    raise ValueError(f"Unsupported parameter type: {expected_type}")


def validate_params_against_schema(
    func_def: FunctionDefinition,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Validate parameter names and normalize values to schema types.

    Raises :class:`ValueError` when keys are missing, extra keys are present,
    or a value cannot be cast to the declared type.
    """
    expected_names = set(func_def.parameters.keys())
    actual_names = set(parameters.keys())
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(
            f"Invalid parameters for '{func_def.name}'. "
            f"Missing={missing}, extra={extra}."
        )

    normalized: dict[str, Any] = {}
    for param_name, param_spec in func_def.parameters.items():
        normalized[param_name] = normalize_param_value(
            param_spec.type,
            parameters[param_name],
        )
    return normalized


__all__ = [
    "get_function_by_name",
    "normalize_param_value",
    "validate_params_against_schema",
]
