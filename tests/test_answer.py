from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_rag.index.embedding_providers import MockEmbeddingProvider
from video_rag.index.store_vectors import store_vectors
from video_rag.io_utils import write_jsonl
from video_rag.schemas import AnswerResult, Chunk, EmbeddingRecord, RetrievalResult
from video_rag.search.answer import (
    MockAnswerProvider,
    answer_from_results,
    answer_video,
    format_context,
    main,
)


VIDEO_ID = "lecture_001"
VARIANT = "transcript_ocr_vlm"
CHUNK_SECONDS = 30


def _retrieval_result(**overrides) -> RetrievalResult:
    base = {
        "chunk_id": "lecture_001_chunk_0000",
        "video_id": VIDEO_ID,
        "score": 0.12,
        "start_time": 210.0,
        "end_time": 240.0,
        "transcript_text": "The speaker explains Bayes theorem.",
        "ocr_text": "P(A|B) = P(B|A)P(A)/P(B)",
        "vlm_caption": "A slide shows Bayes theorem.",
        "combined_text": "Bayes theorem evidence.",
        "frame_paths": ["data/frames/lecture_001/frame_000210.jpg"],
    }
    base.update(overrides)
    return RetrievalResult(**base)


def _embedding_provider() -> MockEmbeddingProvider:
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
    vector = _embedding_provider().embed_texts([text])[0]
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


def _write_index(data_dir: Path) -> None:
    target = _chunk(0, "alpha target answer")
    distractor = _chunk(1, "unrelated beta content")
    write_jsonl(
        data_dir / "chunks" / f"{VIDEO_ID}_{CHUNK_SECONDS}s_enriched.jsonl",
        [target, distractor],
    )
    write_jsonl(
        data_dir / "embeddings" / f"{VIDEO_ID}_{CHUNK_SECONDS}s_{VARIANT}.jsonl",
        [
            _embedding(target, "alpha target answer"),
            _embedding(distractor, "unrelated beta content"),
        ],
    )
    store_vectors(
        VIDEO_ID,
        data_dir=data_dir,
        chunk_seconds=CHUNK_SECONDS,
        variant=VARIANT,
    )


def test_format_context_labels_modalities_and_timestamps():
    context = format_context([_retrieval_result()])

    assert "chunk_id=lecture_001_chunk_0000" in context
    assert "time=03:30-04:00" in context
    assert "Transcript: The speaker explains Bayes theorem." in context
    assert "On-screen text: P(A|B)" in context
    assert "Visual caption: A slide shows Bayes theorem." in context


def test_answer_from_results_returns_grounded_payload():
    evidence = [_retrieval_result()]

    result = answer_from_results(
        VIDEO_ID,
        "What formula was shown?",
        evidence,
        provider=MockAnswerProvider(),
    )

    assert isinstance(result, AnswerResult)
    assert result.video_id == VIDEO_ID
    assert result.provider == "mock"
    assert result.model == "mock-answer"
    assert result.citations[0].chunk_id == "lecture_001_chunk_0000"
    assert result.citations[0].start_time == 210.0
    assert result.retrieval_results == evidence


def test_answer_from_empty_results_does_not_call_provider():
    class _BoomProvider:
        name = "boom"
        model = "boom-model"

        def answer(self, question: str, context: str) -> str:
            raise AssertionError("provider should not be called")

    result = answer_from_results(
        VIDEO_ID,
        "Unknown?",
        [],
        provider=_BoomProvider(),
    )

    assert "not found" in result.answer
    assert result.citations == []


def test_answer_video_retrieves_then_answers(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_index(data_dir)

    result = answer_video(
        VIDEO_ID,
        "alpha target answer",
        data_dir=data_dir,
        retrieval_provider="mock",
        answer_provider=MockAnswerProvider(),
        top_k=1,
    )

    assert result.video_id == VIDEO_ID
    assert len(result.retrieval_results) == 1
    assert result.citations[0].chunk_id == f"{VIDEO_ID}_chunk_0000"


def test_answer_from_results_rejects_empty_question():
    with pytest.raises(ValueError, match="question"):
        answer_from_results(VIDEO_ID, " ", [], provider=MockAnswerProvider())


def test_answer_cli_success(tmp_path: Path, capsys):
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
            "--retrieval-provider",
            "mock",
            "--answer-provider",
            "mock",
            "--top-k",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["video_id"] == VIDEO_ID
    assert payload["citations"][0]["chunk_id"] == f"{VIDEO_ID}_chunk_0000"


def test_answer_cli_failure(tmp_path: Path, capsys):
    rc = main(
        [
            "--video-id",
            VIDEO_ID,
            "--question",
            "missing",
            "--data-dir",
            str(tmp_path / "data"),
            "--retrieval-provider",
            "mock",
            "--answer-provider",
            "mock",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL  FileNotFoundError:" in captured.err
