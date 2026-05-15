"""Command-line entry point for the Call Me Maybe project.

The entry point intentionally keeps a flat structure so that the moulinette
``uv run python -m src ...`` invocation can grade the project without any
extra setup.

Failure modes are all caught and reported with a non-zero exit code; the
program never prints a raw traceback to the user.
"""

import sys

from src.cli import parse_args
from src.display import console, print_header, print_summary
from src.errors import ProjectError
from src.io_utils import load_func_call, load_func_def, write_json_file
from src.llm_interface import LLMInterface
from src.pipeline import process_all_prompts
from src.vocab import load_vocab


def main() -> None:
    """Run the application from command-line arguments."""
    args = parse_args()

    print_header(str(args.input), str(args.output))

    try:
        function_definitions = load_func_def(args.functions_definition)
        prompt_items = load_func_call(args.input)

        llm = LLMInterface(model_name=args.model)
        vocab = load_vocab(llm.get_vocab_path())

        results = process_all_prompts(
            llm=llm,
            vocab=vocab,
            prompt_items=prompt_items,
            function_definitions=function_definitions,
            verbose=args.verbose,
        )
        write_json_file(args.output, results)
        print_summary(len(results), str(args.output))
    except ProjectError as exc:
        console.print(f"\n  [bold red]❌  Error:[/bold red]  {exc}\n")
        sys.exit(1)
    except Exception as exc:  # defensive guard required by the subject
        console.print(
            f"\n  [bold red]❌  Unexpected error:[/bold red]  "
            f"{exc}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
