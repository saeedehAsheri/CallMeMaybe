"""Vocabulary helper.

The constrained decoder needs a mapping ``token_id -> string fragment`` for
every token in the tokenizer's vocabulary.  The SDK exposes
``get_path_to_vocab_file()`` which points to the BPE ``vocab.json`` shipped
with the Qwen3 tokenizer; this module reads that file and converts each
byte-level BPE token into the actual UTF-8 string fragment it represents.

Why the byte decoder?
---------------------
Qwen3 (like GPT-2) uses byte-level BPE: every byte of UTF-8 input is mapped
to a "safe" Unicode character before BPE merging happens.  The ``vocab.json``
keys therefore look like ``Ġhello`` (where ``Ġ`` is the safe character for
byte ``0x20``, i.e. a leading space) instead of ``" hello"``.  The
``_bytes_to_unicode`` function below reproduces the OpenAI byte mapping so
that we can invert the encoding and recover the original UTF-8 text.

This module is also a bonus deliverable: it removes the need to call
``decode`` on the SDK during constrained decoding because we already know the
exact string each token will produce.
"""

import json
from functools import lru_cache
from pathlib import Path

from src.errors import VocabError


def _bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 byte -> safe-Unicode mapping used by Qwen3.

    This is the inverse of the byte decoder embedded in the tokenizer.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


@lru_cache(maxsize=1)
def _unicode_to_bytes() -> dict[str, int]:
    """Inverse of :func:`_bytes_to_unicode`, cached."""
    return {v: k for k, v in _bytes_to_unicode().items()}


def decode_token_repr(token_repr: str) -> str:
    """Decode a single BPE-token string from ``vocab.json`` to actual text.

    Example: ``"Ġhello"`` becomes ``" hello"``.

    Unknown characters fall back to the raw token string so that one corrupt
    entry does not crash the whole vocabulary load.
    """
    byte_decoder = _unicode_to_bytes()
    try:
        byte_sequence = bytes(
            byte_decoder[character] for character in token_repr
        )
    except KeyError:
        return token_repr
    return byte_sequence.decode("utf-8", errors="replace")


def load_vocab(vocab_path: Path) -> list[str]:
    """Load ``vocab.json`` and return ``[token_id] -> decoded text``.

    The list is indexed by token id so that the constrained decoder can do
    ``vocab[token_id]`` without an extra lookup.  Token ids that have no
    entry in ``vocab.json`` (for example added special tokens) get an empty
    string.
    """
    try:
        with vocab_path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except FileNotFoundError as exc:
        raise VocabError(f"Vocabulary file not found: {vocab_path}") from exc
    except json.JSONDecodeError as exc:
        raise VocabError(
            f"Vocabulary file is not valid JSON: {vocab_path}"
        ) from exc

    if not isinstance(data, dict) or not data:
        raise VocabError("vocab.json must contain a non-empty JSON object.")

    max_id = 0
    for token_id in data.values():
        try:
            integer_id = int(token_id)
        except (TypeError, ValueError) as exc:
            raise VocabError(
                f"Vocabulary contains non-integer token id {token_id!r}."
            ) from exc
        if integer_id > max_id:
            max_id = integer_id

    decoded: list[str] = [""] * (max_id + 1)
    for token_repr, token_id in data.items():
        decoded[int(token_id)] = decode_token_repr(token_repr)
    return decoded


__all__ = ["decode_token_repr", "load_vocab"]
