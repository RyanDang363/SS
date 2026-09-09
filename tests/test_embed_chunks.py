from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from video_rag.index.embed_chunks import embed_chunks, main
from video_rag.index.embedding_providers import MockEmbeddingProvider
from video_rag.io_utils import read_jsonl
from video_rag.schemas import EmbeddingRecord

VIDEO_ID = "lecture_001"


def _enriched(index: int, **overrides) -> dict:
    """A Stage 10 enriched-chunk dict with all four search-text variants.

    Includes an extra field to confirm the reader tolerates unknown keys.
    """
    start = float(index * 30)
    base = {
        "chunk_id": f"{VIDEO_ID}_chunk_{index:04d}",
        "video_id": VIDEO_ID,
        "chunk_index": index,
        "start_time": start,
        "end_time": start + 30.0,
        "combined_text_transcript_only": f"transcript text {index}",
        "combined_text_transcript_ocr": f"transcript ocr text {index}",
        "combined_text_transcript_vlm": f"transcript vlm text {index}",
        "combined_text_all": f"transcript ocr vlm text {index}",
        "transcript_text": f"raw transcript {index}",  # extra, must be ignored
    }
    base.update(overrides)
    return base


def _write_enriched(
    data_dir: Path, records: list[dict], chunk_seconds: int = 30
) -> Path:
    path = data_dir / "chunks" / f"{VIDEO_ID}_{chunk_seconds}s_enriched.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


# 1. Happy path with the mock provider.
def test_happy_path_with_mock(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0), _enriched(1)])

    records = embed_chunks(
        VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider(dim=8)
    )

    assert len(records) == 2
    output = data_dir / "embeddings" / f"{VIDEO_ID}_30s_transcript_ocr_vlm.jsonl"
    assert output.is_file()
    on_disk = list(read_jsonl(output, EmbeddingRecord))
    assert on_disk == records
    assert [r.chunk_id for r in records] == [
        f"{VIDEO_ID}_chunk_0000",
        f"{VIDEO_ID}_chunk_0001",
    ]
    assert all(r.embedding_provider == "mock" for r in records)
    assert all(r.embedding_variant == "transcript_ocr_vlm" for r in records)


# 2. Correct variant-to-field mapping.
@pytest.mark.parametrize(
    "variant,field",
    [
        ("transcript_only", "combined_text_transcript_only"),
        ("transcript_ocr", "combined_text_transcript_ocr"),
        ("transcript_vlm", "combined_text_transcript_vlm"),
        ("transcript_ocr_vlm", "combined_text_all"),
    ],
)
def test_variant_field_mapping(tmp_path: Path, variant, field):
    data_dir = tmp_path / "data"
    rec = _enriched(0)
    _write_enriched(data_dir, [rec])
    provider = MockEmbeddingProvider(dim=8)

    records = embed_chunks(
        VIDEO_ID, data_dir=data_dir, variant=variant, provider=provider
    )

    # The embedded vector must come from this variant's source field.
    expected = provider.embed_texts([rec[field]])[0]
    assert records[0].vector == expected


# 3. Unknown variant fails clearly.
def test_unknown_variant_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0)])
    with pytest.raises(ValueError, match="unknown variant"):
        embed_chunks(
            VIDEO_ID, data_dir=data_dir, variant="nope", provider=MockEmbeddingProvider()
        )


# 4. Missing enriched file fails clearly.
def test_missing_enriched_file_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    with pytest.raises(FileNotFoundError) as exc_info:
        embed_chunks(VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider())
    assert "enriched chunks" in str(exc_info.value)
    assert "Run Stage 10" in str(exc_info.value)


# 5. Missing selected variant field fails clearly.
def test_missing_variant_field_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    rec = _enriched(0)
    del rec["combined_text_all"]  # drop the field for transcript_ocr_vlm
    _write_enriched(data_dir, [rec])

    with pytest.raises(ValueError) as exc_info:
        embed_chunks(
            VIDEO_ID,
            data_dir=data_dir,
            variant="transcript_ocr_vlm",
            provider=MockEmbeddingProvider(),
        )
    assert "combined_text_all" in str(exc_info.value)


# 6. Empty / whitespace selected text is skipped with a warning.
def test_empty_text_is_skipped_with_warning(tmp_path: Path, caplog):
    data_dir = tmp_path / "data"
    _write_enriched(
        data_dir,
        [
            _enriched(0),
            _enriched(1, combined_text_all="   "),  # whitespace-only -> skip
            _enriched(2, combined_text_all=""),  # empty -> skip
        ],
    )

    with caplog.at_level(logging.WARNING):
        records = embed_chunks(
            VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider(dim=8)
        )

    assert [r.chunk_id for r in records] == [f"{VIDEO_ID}_chunk_0000"]
    assert "skipping chunk" in caplog.text
    assert f"{VIDEO_ID}_chunk_0001" in caplog.text
    assert f"{VIDEO_ID}_chunk_0002" in caplog.text


# 7 & 8. Vector dimension is recorded and equals len(vector).
def test_vector_dim_recorded_and_consistent(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0), _enriched(1)])

    records = embed_chunks(
        VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider(dim=8)
    )

    assert all(r.vector_dim == 8 for r in records)
    assert all(r.vector_dim == len(r.vector) for r in records)


# 9. Output JSONL contains valid EmbeddingRecords in input order.
def test_output_jsonl_round_trips_in_order(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0), _enriched(1), _enriched(2)])

    records = embed_chunks(
        VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider(dim=8)
    )
    output = data_dir / "embeddings" / f"{VIDEO_ID}_30s_transcript_ocr_vlm.jsonl"

    on_disk = list(read_jsonl(output, EmbeddingRecord))
    assert on_disk == records
    assert [r.chunk_id for r in on_disk] == [
        f"{VIDEO_ID}_chunk_0000",
        f"{VIDEO_ID}_chunk_0001",
        f"{VIDEO_ID}_chunk_0002",
    ]


# Empty enriched file fails clearly.
def test_empty_enriched_file_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [])
    with pytest.raises(ValueError, match="no enriched chunks"):
        embed_chunks(VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider())


# 10. Existing output without overwrite fails.
def test_existing_output_without_overwrite_fails(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0)])
    embed_chunks(VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider())

    with pytest.raises(FileExistsError):
        embed_chunks(VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider())


# 11. Overwrite replaces only the target embedding file.
def test_overwrite_replaces_only_target_file(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0)])
    embeddings_dir = data_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)
    target = embeddings_dir / f"{VIDEO_ID}_30s_transcript_ocr_vlm.jsonl"
    target.write_text('{"existing": true}\n', encoding="utf-8")
    # A different-variant embeddings file must survive.
    sibling = embeddings_dir / f"{VIDEO_ID}_30s_transcript_only.jsonl"
    sibling.write_text("KEEP ME\n", encoding="utf-8")

    records = embed_chunks(
        VIDEO_ID,
        data_dir=data_dir,
        provider=MockEmbeddingProvider(dim=8),
        overwrite=True,
    )

    assert sibling.read_text(encoding="utf-8") == "KEEP ME\n"
    assert list(read_jsonl(target, EmbeddingRecord)) == records


# Stored metadata carries the fields needed for timestamped evidence.
def test_metadata_preserved(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(7)])

    records = embed_chunks(
        VIDEO_ID, data_dir=data_dir, provider=MockEmbeddingProvider(dim=8)
    )

    r = records[0]
    assert r.chunk_id == f"{VIDEO_ID}_chunk_0007"
    assert r.video_id == VIDEO_ID
    assert r.start_time == 210.0
    assert r.end_time == 240.0


# 12. CLI success with the mock provider.
def test_cli_success_with_mock(tmp_path: Path, capsys):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0), _enriched(1)])

    rc = main(
        [
            "--video-id",
            VIDEO_ID,
            "--data-dir",
            str(data_dir),
            "--chunk-seconds",
            "30",
            "--variant",
            "transcript_ocr_vlm",
            "--provider",
            "mock",
        ]
    )

    captured = capsys.readouterr()
    output = data_dir / "embeddings" / f"{VIDEO_ID}_30s_transcript_ocr_vlm.jsonl"
    assert rc == 0
    assert "Wrote embeddings:" in captured.out
    assert output.as_posix() in captured.out
    assert output.is_file()


# 13. CLI failure with an unknown provider (argparse rejects the choice).
def test_cli_unknown_provider_exits_nonzero(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0)])
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--video-id",
                VIDEO_ID,
                "--data-dir",
                str(data_dir),
                "--variant",
                "transcript_ocr_vlm",
                "--provider",
                "not-a-real-provider",
            ]
        )
    assert exc.value.code != 0


def test_cli_missing_provider_exits_nonzero(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        main(["--video-id", VIDEO_ID, "--variant", "transcript_ocr_vlm"])
    assert exc.value.code != 0


def test_cli_failure_on_missing_enriched_file(tmp_path: Path, capsys):
    data_dir = tmp_path / "data"
    rc = main(
        [
            "--video-id",
            VIDEO_ID,
            "--data-dir",
            str(data_dir),
            "--variant",
            "transcript_ocr_vlm",
            "--provider",
            "mock",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL" in captured.err
    assert "FileNotFoundError" in captured.err


# 14. The OpenAI dependency is lazily imported: a full mock-provider run works
#     even when importing ``openai`` is impossible.
def test_mock_run_does_not_import_openai(tmp_path: Path, monkeypatch):
    # Poison the lazy ``import openai`` so any attempt would raise ImportError.
    monkeypatch.setitem(sys.modules, "openai", None)
    data_dir = tmp_path / "data"
    _write_enriched(data_dir, [_enriched(0)])

    records = embed_chunks(
        VIDEO_ID, data_dir=data_dir, provider="mock"
    )
    assert len(records) == 1
    assert records[0].embedding_provider == "mock"
