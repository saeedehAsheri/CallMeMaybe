"""Tests for the BPE vocabulary helper."""

import json
from pathlib import Path

import pytest

from src.errors import VocabError
from src.vocab import decode_token_repr, load_vocab


def test_decode_token_repr_handles_leading_space() -> None:
    # GPT-2 / Qwen byte-level BPE uses 'Ġ' (U+0120) for a leading space.
    assert decode_token_repr("Ġhello") == " hello"


def test_decode_token_repr_passes_ascii_through() -> None:
    assert decode_token_repr("hello") == "hello"


def test_load_vocab_indexes_by_id(tmp_path: Path) -> None:
    vocab_data = {"hello": 5, "Ġworld": 7}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_data), encoding="utf-8")

    loaded = load_vocab(vocab_path)
    assert loaded[5] == "hello"
    assert loaded[7] == " world"
    # Holes between ids are filled with empty strings.
    assert loaded[0] == ""


def test_load_vocab_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VocabError):
        load_vocab(tmp_path / "does_not_exist.json")


def test_load_vocab_invalid_json(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text("not json", encoding="utf-8")
    with pytest.raises(VocabError):
        load_vocab(vocab_path)


def test_load_vocab_rejects_empty(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text("{}", encoding="utf-8")
    with pytest.raises(VocabError):
        load_vocab(vocab_path)
