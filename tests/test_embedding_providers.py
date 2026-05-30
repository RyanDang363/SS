from __future__ import annotations

import sys

import pytest

from video_rag.index.embedding_providers import (
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_provider,
)


def test_mock_is_deterministic():
    p = MockEmbeddingProvider(dim=8)
    a = p.embed_texts(["hello", "world"])
    b = p.embed_texts(["hello", "world"])
    assert a == b
    # Distinct texts give distinct vectors.
    assert a[0] != a[1]


def test_mock_configurable_dim():
    assert all(len(v) == 4 for v in MockEmbeddingProvider(dim=4).embed_texts(["x", "y"]))
    assert all(len(v) == 16 for v in MockEmbeddingProvider(dim=16).embed_texts(["x"]))


def test_mock_rejects_nonpositive_dim():
    with pytest.raises(ValueError):
        MockEmbeddingProvider(dim=0)


def test_mock_vectors_are_floats_in_range():
    vec = MockEmbeddingProvider(dim=8).embed_texts(["sample"])[0]
    assert all(isinstance(x, float) for x in vec)
    assert all(-1.0 <= x < 1.0 for x in vec)


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("not-a-real-provider")


def test_get_provider_mock_and_openai():
    assert isinstance(get_provider("mock"), MockEmbeddingProvider)
    p = get_provider("openai", model="text-embedding-3-large")
    assert isinstance(p, OpenAIEmbeddingProvider)
    assert p.model == "text-embedding-3-large"


def test_openai_provider_clear_error_without_dependency(monkeypatch):
    """When ``openai`` cannot be imported, the provider raises a clear error
    rather than an ImportError — and no network call is attempted."""
    # Force the lazy ``import openai`` to fail.
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(RuntimeError, match="embed"):
        OpenAIEmbeddingProvider().embed_texts(["hi"])


def test_openai_empty_input_short_circuits():
    # No texts -> returns [] without importing openai or needing a key.
    assert OpenAIEmbeddingProvider().embed_texts([]) == []
