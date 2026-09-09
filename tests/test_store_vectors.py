from __future__ import annotations

from pathlib import Path

import pytest

from video_rag.index.store_vectors import main, store_vectors
from video_rag.io_utils import read_json, write_jsonl
from video_rag.schemas import Chunk, EmbeddingRecord, VectorStoreManifest
from video_rag.validate import main as validate_main


VIDEO_ID = "lecture_001"
VARIANT = "transcript_ocr_vlm"
CHUNK_SECONDS = 30
VECTOR_DIM = 4


def _index_name() -> str:
    return f"{VIDEO_ID}_{CHUNK_SECONDS}s_{VARIANT}"


def _embedding(
    chunk_index: int,
    *,
    video_id: str = VIDEO_ID,
    variant: str = VARIANT,
    vector: list[float] | None = None,
    vector_dim: int | None = VECTOR_DIM,
    embedding_model: str = "text-embedding-3-small",
) -> EmbeddingRecord:
    start = float(chunk_index * CHUNK_SECONDS)
    end = start + CHUNK_SECONDS
    if vector is None:
        vector = [0.01 * (chunk_index + 1 + i) for i in range(VECTOR_DIM)]
    return EmbeddingRecord(
        chunk_id=f"{video_id}_chunk_{chunk_index:04d}",
        video_id=video_id,
        start_time=start,
        end_time=end,
        embedding_model=embedding_model,
        embedding_variant=variant,
        vector=vector,
        vector_dim=vector_dim,
    )


def _write_embeddings(
    data_dir: Path,
    records: list[EmbeddingRecord],
) -> Path:
    path = data_dir / "embeddings" / f"{_index_name()}.jsonl"
    write_jsonl(path, records)
    return path


def _write_enriched_chunks(data_dir: Path, records: list[Chunk]) -> Path:
    path = data_dir / "chunks" / f"{VIDEO_ID}_{CHUNK_SECONDS}s_enriched.jsonl"
    write_jsonl(path, records)
    return path


def _chunk_for_embedding(embedding: EmbeddingRecord) -> Chunk:
    return Chunk(
        chunk_id=embedding.chunk_id,
        video_id=embedding.video_id,
        chunk_index=int(embedding.start_time // CHUNK_SECONDS),
        start_time=embedding.start_time,
        end_time=embedding.end_time,
        chunk_seconds=float(CHUNK_SECONDS),
        chunking_strategy="fixed_window",
        combined_text_all=f"chunk text for {embedding.chunk_id}",
    )


def _index_dir(data_dir: Path) -> Path:
    return data_dir / "indexes" / _index_name()


def _manifest_path(data_dir: Path) -> Path:
    return _index_dir(data_dir) / "vector_store_manifest.json"


def _open_collection(data_dir: Path):
    import chromadb

    client = chromadb.PersistentClient(path=str(_index_dir(data_dir)))
    return client.get_collection(name=_index_name())


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def test_store_vectors_success(data_dir: Path):
    records = [_embedding(0), _embedding(1)]
    _write_embeddings(data_dir, records)
    chunks = [_chunk_for_embedding(record) for record in records]
    _write_enriched_chunks(data_dir, chunks)

    manifest = store_vectors(
        VIDEO_ID,
        data_dir=data_dir,
        chunk_seconds=CHUNK_SECONDS,
        variant=VARIANT,
    )

    assert manifest == VectorStoreManifest(
        video_id=VIDEO_ID,
        chunk_seconds=float(CHUNK_SECONDS),
        embedding_variant=VARIANT,
        backend="chroma",
        index_path=_index_dir(data_dir).as_posix(),
        num_vectors=2,
        vector_dim=VECTOR_DIM,
    )
    assert _manifest_path(data_dir).exists()

    collection = _open_collection(data_dir)
    stored = collection.get(include=["metadatas", "documents"])
    assert set(stored["ids"]) == {record.chunk_id for record in records}
    for metadata in stored["metadatas"]:
        assert metadata["video_id"] == VIDEO_ID
        assert "chunk_id" in metadata
        assert "start_time" in metadata
        assert "end_time" in metadata
    assert all(doc.startswith("chunk text for ") for doc in stored["documents"])


def test_store_vectors_validates_consistent_vector_dimensions(data_dir: Path):
    records = [
        _embedding(0, vector=[0.1, 0.2, 0.3, 0.4], vector_dim=4),
        _embedding(1, vector=[0.5, 0.6, 0.7], vector_dim=None),
    ]
    _write_embeddings(data_dir, records)

    with pytest.raises(ValueError, match="inconsistent vector dimensions"):
        store_vectors(VIDEO_ID, data_dir=data_dir)


def test_store_vectors_validates_consistent_embedding_variant(data_dir: Path):
    records = [
        _embedding(0),
        _embedding(1, variant="transcript_only"),
    ]
    _write_embeddings(data_dir, records)

    with pytest.raises(ValueError, match="embedding_variant"):
        store_vectors(VIDEO_ID, data_dir=data_dir)


def test_store_vectors_fails_on_missing_embedding_file(data_dir: Path):
    with pytest.raises(FileNotFoundError, match="missing embedding file"):
        store_vectors(VIDEO_ID, data_dir=data_dir)


def test_store_vectors_fails_on_empty_embedding_file(data_dir: Path):
    _write_embeddings(data_dir, [])

    with pytest.raises(ValueError, match="embedding file is empty"):
        store_vectors(VIDEO_ID, data_dir=data_dir)


def test_store_vectors_fails_if_index_exists_without_overwrite(data_dir: Path):
    records = [_embedding(0)]
    _write_embeddings(data_dir, records)
    store_vectors(VIDEO_ID, data_dir=data_dir)

    _write_embeddings(data_dir, records)

    with pytest.raises(FileExistsError, match="vector index already exists"):
        store_vectors(VIDEO_ID, data_dir=data_dir)


def test_store_vectors_overwrite_replaces_existing_index(data_dir: Path):
    first = [_embedding(0)]
    second = [_embedding(0), _embedding(1)]
    _write_embeddings(data_dir, first)
    store_vectors(VIDEO_ID, data_dir=data_dir)

    _write_embeddings(data_dir, second)
    manifest = store_vectors(
        VIDEO_ID,
        data_dir=data_dir,
        overwrite=True,
    )

    assert manifest.num_vectors == 2
    collection = _open_collection(data_dir)
    assert set(collection.get()["ids"]) == {record.chunk_id for record in second}


def test_store_vectors_writes_valid_manifest(data_dir: Path):
    records = [_embedding(0), _embedding(1)]
    _write_embeddings(data_dir, records)

    store_vectors(VIDEO_ID, data_dir=data_dir)

    manifest = read_json(_manifest_path(data_dir), VectorStoreManifest)
    assert manifest.num_vectors == 2
    assert manifest.vector_dim == VECTOR_DIM
    assert manifest.backend == "chroma"

    rc = validate_main(
        [str(_manifest_path(data_dir)), "--type", "vector_store_manifest"]
    )
    assert rc == 0


def test_store_vectors_cli_success(data_dir: Path, capsys):
    _write_embeddings(data_dir, [_embedding(0)])

    rc = main(
        [
            "--video-id",
            VIDEO_ID,
            "--data-dir",
            str(data_dir),
            "--chunk-seconds",
            str(CHUNK_SECONDS),
            "--variant",
            VARIANT,
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "Wrote vector index:" in captured.out
    assert _manifest_path(data_dir).exists()


def test_store_vectors_cli_failure(data_dir: Path, capsys):
    rc = main(
        [
            "--video-id",
            VIDEO_ID,
            "--data-dir",
            str(data_dir),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "FAIL  FileNotFoundError:" in captured.err


def test_store_vectors_works_without_enriched_chunks(data_dir: Path):
    records = [_embedding(0)]
    _write_embeddings(data_dir, records)

    manifest = store_vectors(VIDEO_ID, data_dir=data_dir)

    assert manifest.num_vectors == 1
    collection = _open_collection(data_dir)
    stored = collection.get(include=["documents"])
    assert stored["documents"] == [""]
