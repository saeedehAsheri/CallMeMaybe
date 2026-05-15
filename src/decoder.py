"""Real constrained decoding for function-call generation.

This is the heart of the project and the place where the subject's most
important requirement is enforced:

    *The function to call must be chosen using the LLM, not with heuristics.*

How constrained decoding works here
-----------------------------------
For function-name selection:

1. Build a prompt that lists every available function and the user's request.
2. Append a literal opening quote ``"`` so the model is primed to emit the
   function name.
3. Repeatedly call :meth:`LLMInterface.get_logits` to obtain the next-token
   logits and **mask out** every token whose decoded fragment would not
   keep the partial output consistent with at least one legal function name.
4. Pick ``argmax`` over the masked logits; this is the LLM's own decision
   among the still-valid options.
5. Stop when the partial output is exactly equal to one of the legal names
   and no longer name extends it.

Because invalid tokens are forced to ``-inf`` the model is mathematically
unable to emit anything outside the set of legal names, no matter how
strongly its raw logits prefer something else.  The selection itself,
however, is fully driven by the LLM's probabilities.

Booleans are decoded with the same technique: the legal set is exactly
``{"true", "false"}`` and the LLM picks one.

Number / integer / free-string parameters are extracted from the prompt by
:mod:`src.value_extractor` because their values are *given in the prompt*,
not invented by the model.  After extraction every value is normalized by
:func:`src.schema_utils.normalize_param_value`, which guarantees the final
JSON is 100% schema-compliant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.errors import DecodingError
from src.llm_interface import LLMInterface
from src.models import FunctionDefinition


_NEG_INF = float("-inf")
_MAX_NAME_STEPS = 64
_MAX_BOOLEAN_STEPS = 8


# --------------------------------------------------------------------------- #
# Tracing helpers
# --------------------------------------------------------------------------- #
@dataclass
class TraceStep:
    """One step in a constrained-decoding trace, used by ``--verbose``."""

    step: int
    chosen_token_id: int
    chosen_fragment: str
    partial_output: str
    candidates_considered: int


def _print_trace(label: str, trace: list[TraceStep]) -> None:
    from src.display import print_trace_table
    print_trace_table(label, trace)


# --------------------------------------------------------------------------- #
# Vocabulary filtering: keep only tokens that could possibly start or extend
# one of the legal candidate strings.  This makes constrained decoding fast
# enough for the project's 5-minute budget.
# --------------------------------------------------------------------------- #
def _relevant_token_ids(
    vocab: list[str],
    legal_strings: Iterable[str],
) -> list[int]:
    """Return token ids whose decoded fragment may help build a legal string.

    A fragment that never appears as a substring of any legal candidate can
    never be a valid continuation, so we ignore it.
    """
    legal_list = list(legal_strings)
    relevant: list[int] = []
    for token_id, fragment in enumerate(vocab):
        if not fragment:
            continue
        for legal in legal_list:
            if fragment in legal:
                relevant.append(token_id)
                break
    return relevant


# --------------------------------------------------------------------------- #
# Generic constrained-decoding loop, used for both function names and booleans.
# --------------------------------------------------------------------------- #
def _argmax(values: list[float]) -> int:
    """Return the index of the largest value, ties broken by lower index."""
    best_index = 0
    best_value = values[0]
    for index in range(1, len(values)):
        value = values[index]
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def _constrained_decode_one_of(
    *,
    llm: LLMInterface,
    prompt_input_ids: list[int],
    vocab: list[str],
    legal_strings: list[str],
    relevant_token_ids: list[int],
    max_steps: int,
    label: str,
    verbose: bool,
) -> str:
    """Pick one of ``legal_strings`` using token-level logit masking.

    The decoder runs at most ``max_steps`` iterations; each iteration calls
    the LLM once.  The function is generic so the exact same code path is
    used for function-name selection and for boolean selection.
    """
    if not legal_strings:
        raise DecodingError(f"{label}: no legal strings provided.")

    input_ids = list(prompt_input_ids)
    partial = ""
    trace: list[TraceStep] = []

    for step in range(max_steps):
        # 1) If we already match a full legal string and no longer one extends
        #    it, we are done.
        finished = partial in legal_strings and not any(
            other != partial and other.startswith(partial)
            for other in legal_strings
        )
        if finished:
            break

        # 2) Ask the model for the next-token logits.
        logits = llm.get_logits(input_ids)

        # 3) Find tokens that would keep the partial output valid.
        valid_ids: list[int] = []
        for token_id in relevant_token_ids:
            if token_id >= len(logits):
                continue
            fragment = vocab[token_id]
            tentative = partial + fragment
            for legal in legal_strings:
                if legal == tentative or legal.startswith(tentative):
                    valid_ids.append(token_id)
                    break

        if not valid_ids:
            if partial in legal_strings:
                break
            raise DecodingError(
                f"{label}: constrained decoding stalled at {partial!r}."
            )

        # 4) Mask logits and pick the best valid token.
        masked = [_NEG_INF] * len(logits)
        for token_id in valid_ids:
            masked[token_id] = logits[token_id]
        best_id = _argmax(masked)

        if masked[best_id] == _NEG_INF:
            raise DecodingError(
                f"{label}: every candidate was masked to -inf."
            )

        chosen_fragment = vocab[best_id]
        partial += chosen_fragment
        input_ids.append(best_id)

        if verbose:
            trace.append(
                TraceStep(
                    step=step,
                    chosen_token_id=best_id,
                    chosen_fragment=chosen_fragment,
                    partial_output=partial,
                    candidates_considered=len(valid_ids),
                )
            )

    if verbose and trace:
        _print_trace(label, trace)

    if partial not in legal_strings:
        raise DecodingError(
            f"{label}: failed to converge on a legal value "
            f"(last partial: {partial!r})."
        )
    return partial


# --------------------------------------------------------------------------- #
# Prompt builders
# --------------------------------------------------------------------------- #
def _build_function_selection_prompt(
    user_prompt: str,
    function_definitions: list[FunctionDefinition],
) -> str:
    """Build the LLM prompt that ends right before the function name."""
    lines: list[str] = [
        "You translate user requests into function calls.",
        "Choose exactly one of the available functions.",
        "",
        "Available functions:",
    ]
    for func_def in function_definitions:
        params_str = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in func_def.parameters.items()
        )
        lines.append(
            f"- {func_def.name}({params_str}) "
            f"-> {func_def.returns.type}: {func_def.description}"
        )
    lines.append("")
    lines.append(f"User request: {user_prompt}")
    lines.append('Selected function name: "')
    return "\n".join(lines)


def _build_boolean_prompt(user_prompt: str, parameter_name: str) -> str:
    """Build an LLM prompt for a constrained boolean value."""
    return (
        "Answer with one of two valid values.\n"
        f"User request: {user_prompt}\n"
        f"Field name: {parameter_name}\n"
        "Allowed values: true, false\n"
        "Answer: "
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def choose_function_name(
    *,
    llm: LLMInterface,
    vocab: list[str],
    user_prompt: str,
    function_definitions: list[FunctionDefinition],
    verbose: bool = False,
) -> str:
    """Return one legal function name selected by the LLM under constraint."""
    if not function_definitions:
        raise DecodingError("No function definitions available.")

    legal_names = [func.name for func in function_definitions]
    relevant = _relevant_token_ids(vocab, legal_names)
    if not relevant:
        raise DecodingError(
            "No vocabulary token can produce any of the legal function names."
        )

    prompt_text = _build_function_selection_prompt(
        user_prompt,
        function_definitions,
    )
    prompt_input_ids = llm.encode(prompt_text)

    return _constrained_decode_one_of(
        llm=llm,
        prompt_input_ids=prompt_input_ids,
        vocab=vocab,
        legal_strings=legal_names,
        relevant_token_ids=relevant,
        max_steps=_MAX_NAME_STEPS,
        label="function_name",
        verbose=verbose,
    )


def choose_boolean(
    *,
    llm: LLMInterface,
    vocab: list[str],
    user_prompt: str,
    parameter_name: str,
    verbose: bool = False,
) -> bool:
    """Decode a boolean with constrained generation (``true`` / ``false``)."""
    legal_strings = ["true", "false"]
    relevant = _relevant_token_ids(vocab, legal_strings)
    if not relevant:
        raise DecodingError(
            "Vocabulary does not contain tokens for 'true' or 'false'."
        )

    prompt_text = _build_boolean_prompt(user_prompt, parameter_name)
    prompt_input_ids = llm.encode(prompt_text)

    chosen = _constrained_decode_one_of(
        llm=llm,
        prompt_input_ids=prompt_input_ids,
        vocab=vocab,
        legal_strings=legal_strings,
        relevant_token_ids=relevant,
        max_steps=_MAX_BOOLEAN_STEPS,
        label=f"boolean[{parameter_name}]",
        verbose=verbose,
    )
    return chosen == "true"


__all__ = ["choose_boolean", "choose_function_name"]
