"""Pydantic models used by the function-calling pipeline.

All input and output schemas are described here.  Pydantic enforces the schema
so that the rest of the pipeline never has to repeat type checks.

Notes
-----
The ``JsonScalarType`` literal includes ``"integer"`` because the moulinette
emits this type for parameters declared as ``int`` in the Python function
signatures (see ``moulinette/extract_functions_infos.py``).  Without this
entry the pydantic validation would fail on the private exercise set as soon
as a function such as ``fn_is_even(n: int)`` is encountered.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


JsonScalarType = Literal["string", "number", "integer", "boolean"]


class ParamSpec(BaseModel):
    """Description of one function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: JsonScalarType


class ReturnSpec(BaseModel):
    """Description of a function return type."""

    model_config = ConfigDict(extra="forbid")

    type: JsonScalarType


class FunctionDefinition(BaseModel):
    """Function definition loaded from ``functions_definition.json``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParamSpec]
    returns: ReturnSpec

    @field_validator("name", "description")
    @classmethod
    def _no_blank_text(cls, value: str) -> str:
        """Reject empty or whitespace-only text fields."""
        if not value.strip():
            raise ValueError("Text fields cannot be empty.")
        return value

    @field_validator("parameters")
    @classmethod
    def _no_blank_parameter_names(
        cls,
        value: dict[str, ParamSpec],
    ) -> dict[str, ParamSpec]:
        """Reject empty parameter names while allowing zero-arg functions."""
        for param_name in value:
            if not param_name.strip():
                raise ValueError("Parameter names cannot be empty.")
        return value


class FunctionCall(BaseModel):
    """Prompt item loaded from ``function_calling_tests.json``."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def _no_blank_prompt(cls, value: str) -> str:
        """Reject blank prompts."""
        if not value.strip():
            raise ValueError("Prompt cannot be empty.")
        return value


class FunctionCallResult(BaseModel):
    """Final schema-compliant output item written to disk."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


__all__ = [
    "FunctionCall",
    "FunctionCallResult",
    "FunctionDefinition",
    "JsonScalarType",
    "ParamSpec",
    "ReturnSpec",
]
