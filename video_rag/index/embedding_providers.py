"""Embedding provider adapters for Stage 11.

The stage code (``embed_chunks``) is decoupled from any specific embedding
backend via the :class:`EmbeddingProvider` protocol. Each provider takes a list
of texts and returns one vector (``list[float]``) per text, in order.

This module ships a deterministic, offline ``MockEmbeddingProvider`` for tests
and CLI smoke checks. The real ``OpenAIEmbeddingProvider`` lazy-imports the
``openai`` package so that importing this module — and running the mock-provider
tests — never requires the OpenAI dependency or an API key.
"""

from __future__ import annotations

import hashlib
import os
import struct
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Adapter interface for an embedding backend."""

    name: str
    model: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbeddingProvider:
    """Deterministic, offline provider for tests and smoke runs.

    Produces a fixed-dimension vector derived from a hash of each text, so the
    same text always maps to the same vector and no network/API key is needed.
    """

    name = "mock"
    model = "mock-embedding"

    def __init__(self, dim: int = 8):
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vector: list[float] = []
        for i in range(self.dim):
            digest = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
            # Map the first 8 bytes to a stable float in [-1.0, 1.0).
            (value,) = struct.unpack("<Q", digest[:8])
            vector.append((value / 2**64) * 2.0 - 1.0)
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbeddingProvider:
    """Real provider backed by the OpenAI embeddings API.

    The ``openai`` package is imported lazily so the base install does not need
    it; install with ``pip install -e .[embed]``. Reads ``OPENAI_API_KEY`` from
    the environment, loading a project ``.env`` file first when ``python-dotenv``
    is installed. The default model ``text-embedding-3-small`` returns 1536-dim
    vectors.
    """

    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "install the 'embed' extra to use the openai provider: "
                "pip install -e .[embed]"
            ) from e

        try:
            from dotenv import load_dotenv
        except ImportError as e:
            raise RuntimeError(
                "install the 'embed' extra to load OPENAI_API_KEY from .env: "
                "pip install -e .[embed]"
            ) from e

        load_dotenv()

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set; cannot use the openai provider"
            )

        client = OpenAI(api_key=api_key)
        try:
            response = client.embeddings.create(model=self.model, input=texts)
        except Exception as e:
            raise RuntimeError(f"openai embedding failed: {e}") from e

        # The API preserves input order in ``data``; sort defensively by index.
        items = sorted(response.data, key=lambda d: d.index)
        return [list(item.embedding) for item in items]


PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "mock": MockEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
}


def get_provider(name: str, **kwargs) -> EmbeddingProvider:
    """Resolve a provider by registry name. Raises ``ValueError`` if unknown.

    Extra keyword arguments are forwarded to the provider constructor (e.g.
    ``model=`` for openai, ``dim=`` for mock). Unsupported kwargs are dropped so
    a single call site can pass ``model`` to any provider.
    """
    cls = PROVIDERS.get(name)
    if cls is None:
        known = ", ".join(sorted(PROVIDERS)) or "(none)"
        raise ValueError(f"unknown provider {name!r} (known: {known})")
    if name == "openai":
        model = kwargs.get("model")
        return cls(model=model) if model else cls()
    return cls()
