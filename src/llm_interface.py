"""Thin, safe wrapper around the provided ``llm_sdk`` package.

The wrapper exposes only the documented public methods of ``Small_LLM_Model``:

* :meth:`encode` — tokenizer encoding (used to feed the prompt to the model)
* :meth:`decode` — optional decoder (used for verbose tracing only)
* :meth:`get_logits` — wraps ``get_logits_from_input_ids``
* :meth:`get_vocab_path` — wraps ``get_path_to_vocab_file``

The decoder in :mod:`src.decoder` performs constrained decoding only with
``get_logits`` and the ``vocab.json`` loaded from :meth:`get_vocab_path`; it
never accesses ``decode`` for control flow.  This is the bonus pattern
recommended by the project subject (use ``get_logits_from_input_ids`` and
``get_path_to_vocabulary_json``).
"""

from pathlib import Path
from typing import Any

from src.errors import ModelError


class LLMInterface:
    """Public, safe interface around the provided SDK.

    Parameters
    ----------
    model_name:
        Optional model identifier.  When ``None`` (default) the SDK falls
        back to ``Qwen/Qwen3-0.6B``.  Any model exposing the same SDK API
        also works; this is the *Support for multiple LLM models* bonus.
    """

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
        except ImportError as exc:  # pragma: no cover - depends on SDK setup
            raise ModelError(
                "Could not import Small_LLM_Model from llm_sdk. "
                "Run 'uv sync' and ensure the llm_sdk package is present at "
                "the project root."
            ) from exc

        try:
            if model_name is None:
                self.model: Any = Small_LLM_Model()
            else:
                self.model = Small_LLM_Model(model_name=model_name)
        except Exception as exc:  # pragma: no cover
            raise ModelError(f"Failed to initialize the model: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Tokenizer methods (public SDK surface)
    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> list[int]:
        """Encode ``text`` and return token ids as a flat ``list[int]``."""
        try:
            encoded = self.model.encode(text)
            return [int(token_id) for token_id in encoded[0].tolist()]
        except Exception as exc:
            raise ModelError(f"Encoding failed: {exc}") from exc

    def decode(self, token_ids: list[int]) -> str:
        """Decode ``token_ids`` back to text using the SDK tokenizer.

        Used only for verbose tracing.  The constrained decoder relies on
        :func:`src.vocab.load_vocab` instead, which is the recommended bonus
        path.
        """
        try:
            return str(self.model.decode(token_ids))
        except Exception as exc:
            raise ModelError(f"Decoding failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Model methods (public SDK surface)
    # ------------------------------------------------------------------ #
    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for ``input_ids``."""
        try:
            raw = self.model.get_logits_from_input_ids(input_ids)
            return [float(value) for value in raw]
        except Exception as exc:
            raise ModelError(f"Failed to obtain logits: {exc}") from exc

    def get_vocab_path(self) -> Path:
        """Return the path to the tokenizer's ``vocab.json`` file.

        Wraps ``get_path_to_vocab_file`` (the SDK method); also accessible as
        ``get_path_to_vocabulary_json`` for compatibility with subject wording.
        """
        try:
            return Path(self.model.get_path_to_vocab_file())
        except Exception as exc:
            raise ModelError(
                f"Failed to locate the vocabulary file: {exc}"
            ) from exc

    def get_path_to_vocabulary_json(self) -> Path:
        """Alias for :meth:`get_vocab_path` — matches the subject's naming.

        The subject refers to this SDK method as
        ``get_path_to_vocabulary_json``. The SDK exposes it as
        ``get_path_to_vocab_file``. Both return the same vocab path.
        """
        return self.get_vocab_path()


__all__ = ["LLMInterface"]
