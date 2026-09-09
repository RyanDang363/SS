from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_rag.index.embedding_providers import MockEmbeddingProvider
from video_rag.index.store_vectors import store_vectors
from video_rag.io_utils import write_json, write_jsonl
from video_rag.schemas import Chunk, EmbeddingRecord, RetrievalResult, VectorStoreManifest
from video_rag.search.retrieve import main, retrieve


VIDEO_ID = "lecture_001"
VARIANT = "transcript_ocr_vlm"
CHUNK_SECONDS = 30


def _provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dim=8)


def _chunk(index: int, text: str) -> Chunk:
    start = float(index * CHUNK_SECONDS)
    return Chunk(
        chunk_id=f"{VIDEO_ID}_chunk_{index:04d}",
        video_id=VIDEO_ID,
        chunk_index=index,
        start_time=start,
        end_time=start + CHUNK_SECONDS,
        transcript_text=f"Transcript {index}",
        ocr_text=f"OCR {index}",
        vlm_caption=f"Caption {index}",
        frame_paths=[f"data/frames/{VIDEO_ID}/frame_{index:06d}.jpg"],
        chunk_seconds=float(CHUNK_SECONDS),
        chunking_strategy="fixed",
        combined_text_all=text,
    )


def _embedding(chunk: Chunk, text: str) -> EmbeddingRecord:
    vector = _provider().embed_texts([text])[0]
    return EmbeddingRecord(
        chunk_id=chunk.chunk_id,
        video_id=chunk.video_id,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        embedding_model="mock-embedding",
        embedding_provider="mock",
        embedding_variant=VARIANT,
        vector=vector,
        vector_dim=len(vector),
    )


def _write_index(data_dir: Path) -> list[Chunk]:
    target = _chunk(0, "alpha target answer")
    distractor = _chunk(1, "unrelated beta content")
    chunks = [target, distractor]
    write_jsonl(
        data_dir / "chunks" / f"{VIDEO_ID}_{CHUNK_SECONDS}s_enriched.jsonl",
        chunks,
    )
    write_jsonl(
        data_dir / "embeddings" / f"{VIDEO_ID}_{CHUNK_SECONDS}s_{VARIANT}.jsonl",
        [_embedding(target, "alpha target answer"), _embedding(distractor, "unrelated beta content")],
    )
    store_vectors(
        VIDEO_ID,
        data_dir=data_dir,
        chunk_seconds=CHUNK_SECONDS,
        variant=VARIANT,
    )
    return chunks


def test_retrieve_returns_nearest_enriched_chunk(tmp_path: Path):
    data_dir = tmp_path / "data"
    chunks = _write_index(data_dir)

    results = retrieve(
        VIDEO_ID,
        "alpha target answer",
        data_dir=data_dir,
        provider=_provider(),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0] == RetrievalResult(
        chunk_id=chunks[0].chunk_id,
        video_id=VIDEO_ID,
        score=results[0].score,
        start_time=0.0,
        end_time=30.0,
        transcript_text="Transcript 0",
        ocr_text="OCR 0",
        vlm_caption="Caption 0",
        combined_text="alpha target answer",
        frame_paths=[f"data/frames/{VIDEO_ID}/frame_000000.jpg"],
    )


def test_retrieve_requires_existing_index(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing vector index"):
        retrieve(VIDEO_ID, "question", data_dir=tmp_path / "data", provider=_provider())


def test_retrieve_validates_manifest_variant(tmp_path: Path):
    data_dir = tmp_path / "data"
    chunks = _write_index(data_dir)
    index_dir = data_dir / "indexes" / f"{VIDEO_ID}_{CHUNK_SECONDS}s_{VARIANT}"
    write_json(
        index_dir / "vector_store_manifest.json",
        VectorStoreManifest(
            video_id=VIDEO_ID,
            chunk_seconds=CHUNK_SECONDS,
            embedding_variant="transcript_only",
            backend="chroma",
            index_path=index_dir.as_posix(),
            num_vectors=len(chunks),
            vector_dim=8,
        ),
    )

    with pytest.raises(ValueError, match="embedding_variant"):
        retrieve(VIDEO_ID, "alpha", data_dir=data_dir, provider=_provider())


def test_retrieve_can_use_stored_document_without_chunk_file(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_index(data_dir)
    (data_dir / "chunks" / f"{VIDEO_ID}_{CHUNK_SECONDS}s_enriched.jsonl").unlink()

    results = retrieve(
        VIDEO_ID,
        "alpha target answer",
        data_dir=data_dir,
        provider=_provider(),
        top_k=1,
    )

    assert results[0].combined_text == "alpha target answer"
    assert results[0].transcript_text is None
    assert results[0].frame_paths == []


def test_retrieve_cli_success(tmp_path: Path, capsys):
    data_dir = tmp_path / "data"
    _write_index(data_dir)

    rc = main(
        [
            "--video-id",
            VIDEO_ID,
            "--question",
            "alpha target answer",
            "--data-dir",
            str(data_dir),
            "--provider",
            "mock",
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload[0]["chunk_id"] == f"{VIDEO_ID}_chunk_0000"


def test_retrieve_cli_failure(tmp_path: Path, capsys):
    rc = main(
        [
            "--video-id",
            VIDEO_ID,
            "--question",
            "missing",
            "--data-dir",
            str(tmp_path / "data"),
            "--provider",
            "mock",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL  FileNotFoundError:" in captured.err
