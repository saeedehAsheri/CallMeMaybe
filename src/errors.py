"""Project-specific exceptions.

Every error raised by the project inherits from :class:`ProjectError` so the
main entry point can catch the whole project family with a single ``except``
clause and exit with a clear message.
"""


class ProjectError(Exception):
    """Base exception for project errors."""


class InputFileError(ProjectError):
    """Raised when an input or output file is missing or invalid."""


class ModelError(ProjectError):
    """Raised when the provided LLM SDK cannot be used correctly."""


class DecodingError(ProjectError):
    """Raised when a valid structured output cannot be produced."""


class VocabError(ProjectError):
    """Raised when the tokenizer vocabulary cannot be loaded or decoded."""


__all__ = [
    "ProjectError",
    "InputFileError",
    "ModelError",
    "DecodingError",
    "VocabError",
]
