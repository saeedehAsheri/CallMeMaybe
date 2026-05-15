*This project has been created as part of the 42 curriculum by sasheri.*

# Call Me Maybe

A small function-calling system for large language models. The program reads
natural-language prompts and a JSON list of available functions, then
produces a structured JSON file containing the selected function name and
fully-typed parameters.

For example, the prompt

```text
What is the sum of 2 and 3?
```

becomes

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The function name is chosen by the LLM under **token-level constrained
decoding** so the output is always one of the legal function names and the
JSON is always 100% parseable.

## Description

The project ships a pipeline composed of four cooperating modules:

| Module | Responsibility |
| --- | --- |
| `src.llm_interface` | Thin wrapper over the provided `llm_sdk`, exposing only the public methods (`encode`, `decode`, `get_logits_from_input_ids`, `get_path_to_vocab_file`). |
| `src.vocab` | Loads `vocab.json` and decodes every byte-level BPE token into the actual text fragment it represents. |
| `src.decoder` | Real constrained decoding: at each generation step it masks the LLM's logits so the model can only pick tokens that keep the partial output consistent with at least one legal candidate. |
| `src.value_extractor` | Pulls scalar parameter values (numbers, file paths, SQL queries, template strings…) from the prompt, guided only by the parameter name and type from the schema. |

The dataflow per prompt is:

1. The pipeline asks the constrained decoder to choose a function name.
2. For each parameter, the pipeline extracts a raw value from the prompt;
   booleans go through a second constrained-decoding pass that forces the
   model to emit `true` or `false`.
3. `src.schema_utils` validates the parameter dictionary and casts every
   value to the exact Python type required by the schema (`int` stays `int`,
   `float` stays `float`).
4. `pydantic` wraps the final object so no invalid result can be written.
5. `src.io_utils` serializes the list with `json.dump`, guaranteeing
   parseable JSON.

## Instructions

### Install

```bash
make install
```

This runs `uv sync`, which installs the project dependencies (`numpy`,
`pydantic`) and the bundled `llm_sdk` (which transitively pulls in `torch`,
`transformers`, and `huggingface-hub` — these are dependencies of the
provided SDK, not of the project's own source code).

### Run with default paths

```bash
make run
```

Equivalent direct command:

```bash
uv run python -m src
```

By default the program reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`.

### Run with custom paths

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Bonus flags

```bash
# Use a different model exposed by the SDK.
uv run python -m src --model Qwen/Qwen3-0.6B

# Print a detailed token-level constrained-decoding trace to stderr.
uv run python -m src --verbose
```

By default (without any flags) the pipeline shows a simple step-by-step
terminal display: one line per prompt with the selected function name and
extracted arguments, plus a completion summary with the output path.

### Lint and test

```bash
make lint           # flake8 + mypy with the flags required by the subject
make lint-strict    # bonus: mypy --strict
make test           # bonus: pytest unit test suite
make clean
```

## Algorithm explanation

The project implements **real constrained decoding** with token-level logit
masking, the technique described in the subject under *Understanding
Constrained Decoding*.

### Function-name selection

1. Build a prompt that lists every available function (name + signature +
   description) and the user's request, ending with a literal opening quote
   `"`. This primes the model to emit a function name next.
2. Encode the prompt with the SDK tokenizer (`encode`) into `input_ids`.
3. Loop until convergence:
   * Call `get_logits_from_input_ids(input_ids)` to obtain raw next-token
     logits.
   * Compute the set of *valid* token ids: a token id is valid if its
     decoded fragment, appended to the partial output so far, is still a
     prefix of at least one legal function name. The decoded fragments come
     from `vocab.json` loaded by `src.vocab` — the project does not call
     `decode` inside the decoder.
   * Set the logits of every other token to `-inf`.
   * Pick `argmax` over the masked logits. This is the LLM's own choice
     among the still-valid options.
   * Append the chosen token id to `input_ids` and update the partial
     output.
4. Stop when the partial output is exactly equal to one legal function name
   and no other name extends it. The model is mathematically incapable of
   emitting anything else.

Because the constraint operates on the *logits* (not on the prompt), the
small `Qwen3-0.6B` model achieves near-perfect reliability even when its
unconstrained generation would be flaky.

### Boolean values

Boolean parameters reuse the same constrained-decoding loop, but the legal
set is exactly `{"true", "false"}`.

### Numeric and free-string values

Numbers, integers, file paths, SQL queries and template strings are taken
directly from the prompt because that is where the user wrote them. The
extractors live in `src.value_extractor` and only look at the *parameter
name and type* from the schema, so they generalise to new function sets.
After extraction, `src.schema_utils.normalize_param_value` casts the raw
value to the exact Python type required by the moulinette (`int` for
`"integer"`, `float` for `"number"`, etc.).

## Design decisions

* **No heuristic function selection.** The choice of function is entirely
  driven by the LLM's logits under constraint. The keyword-bonus shortcut
  used in earlier drafts has been removed.
* **`vocab.json` instead of `decode`.** The constrained-decoding loop never
  calls the SDK's `decode` method. Token-to-text mapping comes from the
  byte-level BPE vocabulary the SDK exposes through
  `get_path_to_vocab_file()`.
* **Schema-first validation.** Pydantic models in `src.models` describe
  every input and output structure, including the `"integer"` scalar type
  that the moulinette emits for `int`-typed parameters.
* **Integer fidelity.** `normalize_param_value` keeps `int` as `int` and
  `float` as `float` so the moulinette's `assert isinstance(...)` checks
  succeed.
* **Single source of truth for errors.** Every error path raises a subclass
  of `ProjectError` so the entry point reports a clean message instead of a
  raw traceback.
* **Path-dependency on `llm_sdk`.** The bundled SDK is wired in via
  `tool.uv.sources` so `uv sync` alone installs everything the project
  needs.

## Performance analysis

* **JSON validity:** 100% — the final string is produced by `json.dump` on
  a validated pydantic object.
* **Schema validity:** enforced by pydantic and by
  `validate_params_against_schema`.
* **Function selection:** under constrained decoding the LLM picks one of
  the legal names; the choice is the model's, but the *set of options* is
  enforced.
* **Speed:** the relevant-token pre-filter (see `_relevant_token_ids` in
  `src.decoder`) limits the per-step work to a few hundred candidate tokens
  instead of the full ~150 000-entry vocabulary. End-to-end runtime on the
  provided test set is well under the 5-minute budget.

## Challenges faced

* **`int` vs `float`.** The first iteration treated every numeric type as
  `float`, which made the moulinette's `assert isinstance(n, int)` checks
  fail for `fn_is_even`. The fix was to introduce a dedicated `"integer"`
  path through `normalize_param_value` and the value extractor.
* **Token fragments include leading spaces.** Qwen3's byte-level BPE
  represents a leading space as the safe character `Ġ`. Without decoding
  that back to a real space, the constrained decoder mistakes valid
  continuations for invalid ones.
* **Strings with embedded quotes.** Templates such as
  `Say "hello" to {name}` contain double quotes that confuse naive
  extractors. The fix was to anchor extraction on the literal
  `Format template:` prefix instead.

## Testing strategy

The `tests/` directory contains unit tests that do not need the LLM:

* `tests/test_schema_utils.py` — type normalisation and validation.
* `tests/test_value_extractor.py` — parameter extraction for every prompt
  in both the public and private moulinette sets.
* `tests/test_vocab.py` — byte-level BPE decoding and vocabulary loading.
* `tests/test_io_utils.py` — JSON I/O round trips and error paths.
* `tests/test_models.py` — pydantic schema enforcement.

Run with `make test`. For an end-to-end check, run the full pipeline and
then point the moulinette at the produced output file:

```bash
uv run python -m src
uv run python -m moulinette grade_student_answers \
  --set private \
  --student_answer_path data/output/function_calling_results.json
```

## Example usage

```bash
$ uv run python -m src
Successfully wrote 11 results to: data/output/function_calling_results.json

$ uv run python -m src --verbose 2>verbose.log
$ head verbose.log
[trace] function_name
  step= 0  id= ...  frag='fn'  partial='fn'  candidates=...
  step= 1  id= ...  frag='_add'  partial='fn_add'  candidates=...
  ...
```

## Bonus features

The following bonus features from the subject are implemented:

* **Multiple LLM models** — pass any SDK-compatible identifier with
  `--model`.
* **Tokenizer recoding** — `src.vocab` decodes `vocab.json` directly so the
  constrained decoder uses `get_logits_from_input_ids` and the vocabulary
  file instead of `decode`.
* **Performance optimisation** — the vocabulary is loaded once, then
  pre-filtered to the tokens that can appear in a legal candidate string;
  every per-step pass works on this small set.
* **Visualisation of the generation process** — every run prints a clear
  per-prompt summary. Adding `--verbose` extends this with a full token-level
  table (token id, decoded fragment, partial output, candidate count) sent to
  stderr so it never pollutes captured output.
* **Public tokenizer surface** — `LLMInterface.encode` and
  `LLMInterface.decode` are public wrappers around the SDK's methods.
  `LLMInterface.get_path_to_vocabulary_json` is provided as an alias for
  `get_vocab_path`, matching the naming used in the project subject.
* **Comprehensive test suite** — `tests/` provides unit coverage for
  schema validation, value extraction, vocabulary decoding, and I/O error
  paths.
* **Demonstration of encoding/decoding inside constrained decoding** —
  documented above and visible in the `--verbose` trace, which shows the
  encoded input ids feeding the model and the decoded fragments being
  appended after each masking step.

## Resources and AI usage

##  Video Tutorials (YouTube)

- [**Resource Link**](https://www.youtube.com/watch?v=dtLl37W68g8)
- [**Resource Link**](https://www.youtube.com/watch?v=fuMKrKlaku4)
- [**Resource Link**](https://www.youtube.com/watch?v=uHa_2i2S64Y)
- [**Resource Link**](https://www.youtube.com/watch?v=6e_oFG4JVg8)
- [**Resource Link**](https://www.youtube.com/watch?v=NGEZsqEUpC0)
- [**Resource Link**](https://www.youtube.com/watch?v=9N6a-VLBa2I)
- [**Resource Link**](https://www.youtube.com/watch?v=wJJ32FJxY-8)
- [**Resource Link**](https://www.youtube.com/watch?v=4rFVLMBwjGY)

##  Documentation & Articles

- [**Resource Link**](https://docs.python.org/3/library/argparse.html)
- [**Resource Link**](https://www.geeksforgeeks.org/python/read-json-file-using-python/)
- [**Resource Link**](https://www.datacamp.com/tutorial/python-argparse)
- [**Resource Link**](https://docs.pydantic.dev/latest/concepts/json/#partial-json-parsing)
- [**Resource Link**](https://docs.pydantic.dev/latest/examples/files/)
- [**Resource Link**](https://www.couchbase.com/blog/validate-json-documents-in-python-using-pydantic/)
- [**Resource Link**](https://www.soumendrak.com/blog/how-to-validate-a-json-in-python/)
- [**Resource Link**](https://www.datacamp.com/cheat-sheet/regular-expresso)

* Pydantic v2 documentation.


AI usage:

AI was used as a learning and review assistant to clarify the constrained
decoding algorithm, to draft documentation. I used AI tools to learn new concept of the project by asking to explaining the concepts with simple examples. I also used AI to correct flake8 isssues, create makefile and writing tests.
