"""Simple terminal display helpers.

The project should run after a plain ``uv sync`` with only the allowed project
runtime dependencies.  For that reason this module uses only the Python
standard library instead of external display packages.
"""

import sys
from typing import Any


class _Console:
    """Tiny compatibility wrapper used by ``__main__`` for error messages."""

    def __init__(self, *, stderr: bool = False) -> None:
        self._stream = sys.stderr if stderr else sys.stdout

    def print(self, message: str = "") -> None:
        """Print ``message`` to the configured stream.

        A few old rich-style tags are stripped so existing call sites can keep
        readable messages without depending on ``rich``.
        """
        clean = (
            message.replace("[bold red]", "")
            .replace("[/bold red]", "")
            .replace("[bold green]", "")
            .replace("[/bold green]", "")
            .replace("[dim]", "")
            .replace("[/dim]", "")
            .replace("[cyan]", "")
            .replace("[/cyan]", "")
        )
        print(clean, file=self._stream)


console = _Console()
err_console = _Console(stderr=True)


def print_header(input_path: str, output_path: str) -> None:
    """Print a small startup message."""
    print("\nCall Me Maybe - LLM Function-Calling Pipeline")
    print(f"input : {input_path}")
    print(f"output: {output_path}\n")


def print_summary(count: int, output_path: str) -> None:
    """Print the completion summary."""
    noun = "result" if count == 1 else "results"
    print("-" * 60)
    print(f"Done: wrote {count} {noun} to {output_path}\n")


def print_result(
    index: int,
    total: int,
    prompt: str,
    func_name: str,
    parameters: dict[str, Any],
) -> None:
    """Print one processed prompt result."""
    params_str = ", ".join(
        f"{key}={value!r}" for key, value in parameters.items()
    )
    suffix = f" ({params_str})" if params_str else ""
    print(f"[{index}/{total}] {prompt}")
    print(f"      -> {func_name}{suffix}\n")


def print_trace_table(label: str, trace: list[Any]) -> None:
    """Print a compact token-level trace to stderr."""
    print(f"\n[trace] {label}", file=sys.stderr)
    print(
        "step | token id | fragment | partial output | candidates",
        file=sys.stderr,
    )
    print(
        "-----+----------+----------+----------------+-----------",
        file=sys.stderr,
    )
    for step in trace:
        print(
            f"{step.step:>4} | {step.chosen_token_id:>8} | "
            f"{step.chosen_fragment!r:<8} | "
            f"{step.partial_output!r:<14} | {step.candidates_considered:>10}",
            file=sys.stderr,
        )


__all__ = [
    "console",
    "err_console",
    "print_header",
    "print_result",
    "print_summary",
    "print_trace_table",
]
