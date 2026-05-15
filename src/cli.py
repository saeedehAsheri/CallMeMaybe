"""Command-line argument parsing."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Read and parse command-line arguments.

    The flag layout matches the subject exactly; the optional ``--model`` and
    ``--verbose`` flags are bonus features.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts into structured function "
            "calls using constrained decoding."
        )
    )
    parser.add_argument(
        "--functions_definition",
        help="Path to the JSON file containing function definitions.",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        help="Path to the JSON file containing input prompts.",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        help="Path to the JSON output file.",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional model identifier passed to the SDK. Defaults to the "
            "SDK default (Qwen/Qwen3-0.6B)."
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        help="Print a step-by-step constrained-decoding trace to stderr.",
        action="store_true",
    )
    return parser.parse_args()


__all__ = ["parse_args"]
