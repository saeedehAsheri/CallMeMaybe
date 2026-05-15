"""Tests for prompt-driven value extraction."""

from src.models import FunctionDefinition
from src.value_extractor import extract_all_parameters, extract_parameter_value


def _func(parameters: dict[str, str]) -> FunctionDefinition:
    """Build a small function definition for tests."""
    func_def: FunctionDefinition = FunctionDefinition.model_validate(
        {
            "name": "fn_x",
            "description": "x",
            "parameters": {
                name: {"type": type_name}
                for name, type_name in parameters.items()
            },
            "returns": {"type": "string"},
        }
    )
    return func_def


# --------------------------------------------------------------------------- #
# Numeric extraction
# --------------------------------------------------------------------------- #
def test_number_first_position() -> None:
    value = extract_parameter_value(
        prompt="What is the sum of 2 and 3?",
        parameter_name="a",
        parameter_type="number",
        parameter_index=0,
        parameter_types=["number", "number"],
    )
    assert value == 2.0


def test_number_second_position() -> None:
    value = extract_parameter_value(
        prompt="What is the sum of 2 and 3?",
        parameter_name="b",
        parameter_type="number",
        parameter_index=1,
        parameter_types=["number", "number"],
    )
    assert value == 3.0


def test_integer_extracted_cleanly() -> None:
    value = extract_parameter_value(
        prompt="Is 4 an even number?",
        parameter_name="n",
        parameter_type="integer",
        parameter_index=0,
        parameter_types=["integer"],
    )
    assert value == 4
    assert isinstance(value, int)


def test_compound_interest_three_numbers() -> None:
    func = _func(
        {"principal": "number", "rate": "number", "years": "integer"}
    )
    prompt = (
        "Calculate compound interest on 1234567.89 "
        "at 0.0375 rate for 23 years"
    )
    result = extract_all_parameters(prompt, func)
    assert result["principal"] == 1234567.89
    assert result["rate"] == 0.0375
    assert result["years"] == 23


# --------------------------------------------------------------------------- #
# String extraction
# --------------------------------------------------------------------------- #
def test_greet_extracts_last_word() -> None:
    value = extract_parameter_value(
        prompt="Greet shrek",
        parameter_name="name",
        parameter_type="string",
        parameter_index=0,
        parameter_types=["string"],
    )
    assert value == "shrek"


def test_reverse_extracts_quoted_string() -> None:
    value = extract_parameter_value(
        prompt="Reverse the string 'hello'",
        parameter_name="s",
        parameter_type="string",
        parameter_index=0,
        parameter_types=["string"],
    )
    assert value == "hello"


def test_sql_query_extraction() -> None:
    func = _func({"query": "string", "database": "string"})
    prompt = (
        "Execute SQL query 'SELECT * FROM users' "
        "on the production database"
    )
    result = extract_all_parameters(prompt, func)
    assert result["query"] == "SELECT * FROM users"
    assert result["database"] == "production"


def test_read_file_path_and_encoding() -> None:
    func = _func({"path": "string", "encoding": "string"})
    prompt = "Read the file at /home/user/data.json with utf-8 encoding"
    result = extract_all_parameters(prompt, func)
    assert result["path"] == "/home/user/data.json"
    assert result["encoding"] == "utf-8"


def test_windows_path_extraction() -> None:
    func = _func({"path": "string", "encoding": "string"})
    prompt = "Read C:\\Users\\john\\config.ini with latin-1 encoding"
    result = extract_all_parameters(prompt, func)
    assert result["path"] == "C:\\Users\\john\\config.ini"
    assert result["encoding"] == "latin-1"


def test_template_preserves_quotes() -> None:
    func = _func({"template": "string"})
    prompt = 'Format template: Say "hello" to {name}'
    result = extract_all_parameters(prompt, func)
    assert result["template"] == 'Say "hello" to {name}'


def test_substitute_numbers_regex() -> None:
    func = _func(
        {"source_string": "string", "regex": "string", "replacement": "string"}
    )
    prompt = (
        "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    )
    result = extract_all_parameters(prompt, func)
    assert result["source_string"] == "Hello 34 I'm 233 years old"
    assert result["regex"] == r"\d+"
    assert result["replacement"] == "NUMBERS"


def test_substitute_vowels_regex_with_asterisks() -> None:
    func = _func(
        {"source_string": "string", "regex": "string", "replacement": "string"}
    )
    prompt = "Replace all vowels in 'Programming is fun' with asterisks"
    result = extract_all_parameters(prompt, func)
    assert result["source_string"] == "Programming is fun"
    assert result["regex"] == r"[aeiouAEIOU]"
    assert result["replacement"] == "*"


def test_substitute_word_with_word_boundary() -> None:
    func = _func(
        {"source_string": "string", "regex": "string", "replacement": "string"}
    )
    prompt = (
        "Substitute the word 'cat' with 'dog' "
        "in 'The cat sat on the mat with another cat'"
    )
    result = extract_all_parameters(prompt, func)
    assert result["source_string"] == "The cat sat on the mat with another cat"
    assert result["regex"] == r"\bcat\b"
    assert result["replacement"] == "dog"
