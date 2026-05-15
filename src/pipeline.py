"""High-level processing pipeline.

The pipeline glues the LLM-driven constrained decoder (:mod:`src.decoder`)
together with prompt-driven value extraction (:mod:`src.value_extractor`)
and schema validation (:mod:`src.schema_utils`).

For each natural-language prompt:

1. The LLM picks one function name under token-level constraint.
2. Each parameter value is built from the prompt (with constrained
   generation only for booleans).
3. The full parameter dictionary is validated against the schema, which
   casts every value to the exact Python type required by the moulinette.
4. The result is wrapped in a :class:`FunctionCallResult` pydantic model so
   no invalid object can ever reach :func:`src.io_utils.write_json_file`.
"""

from typing import Any

from src.decoder import choose_boolean, choose_function_name
from src.display import print_result
from src.errors import DecodingError
from src.llm_interface import LLMInterface
from src.models import FunctionCall, FunctionCallResult, FunctionDefinition
from src.schema_utils import (
    get_function_by_name,
    validate_params_against_schema,
)
from src.value_extractor import extract_all_parameters


def _resolve_parameters(
    *,
    llm: LLMInterface,
    vocab: list[str],
    user_prompt: str,
    func_def: FunctionDefinition,
    verbose: bool,
) -> dict[str, Any]:
    """Build the parameter dictionary for ``func_def`` from ``user_prompt``."""
    raw_parameters: dict[str, Any] = extract_all_parameters(
        prompt=user_prompt,
        func_def=func_def,
    )
    for parameter_name, parameter_spec in func_def.parameters.items():
        if parameter_spec.type == "boolean":
            raw_parameters[parameter_name] = choose_boolean(
                llm=llm,
                vocab=vocab,
                user_prompt=user_prompt,
                parameter_name=parameter_name,
                verbose=verbose,
            )
    return validate_params_against_schema(func_def, raw_parameters)


def process_single_prompt(
    *,
    llm: LLMInterface,
    vocab: list[str],
    prompt_item: FunctionCall,
    function_definitions: list[FunctionDefinition],
    verbose: bool = False,
) -> FunctionCallResult:
    """Convert one prompt into one structured function call."""
    selected_name = choose_function_name(
        llm=llm,
        vocab=vocab,
        user_prompt=prompt_item.prompt,
        function_definitions=function_definitions,
        verbose=verbose,
    )
    selected_function = get_function_by_name(
        function_definitions,
        selected_name,
    )
    if selected_function is None:
        raise DecodingError(
            "Constrained decoder selected an unknown function: "
            f"{selected_name!r}."
        )

    parameters = _resolve_parameters(
        llm=llm,
        vocab=vocab,
        user_prompt=prompt_item.prompt,
        func_def=selected_function,
        verbose=verbose,
    )
    return FunctionCallResult(
        prompt=prompt_item.prompt,
        name=selected_function.name,
        parameters=parameters,
    )


def process_all_prompts(
    *,
    llm: LLMInterface,
    vocab: list[str],
    prompt_items: list[FunctionCall],
    function_definitions: list[FunctionDefinition],
    verbose: bool = False,
) -> list[FunctionCallResult]:
    """Convert every prompt into a structured function call."""
    results: list[FunctionCallResult] = []
    total = len(prompt_items)
    for index, item in enumerate(prompt_items, start=1):
        result = process_single_prompt(
            llm=llm,
            vocab=vocab,
            prompt_item=item,
            function_definitions=function_definitions,
            verbose=verbose,
        )
        print_result(index, total, item.prompt, result.name, result.parameters)
        results.append(result)
    return results


__all__ = ["process_all_prompts", "process_single_prompt"]
