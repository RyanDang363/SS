from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from video_rag.api import app as api_app
from video_rag.api.app import create_app
from video_rag.io_utils import write_json
from video_rag.schemas import (
    AnswerCitation,
    AnswerResult,
    RetrievalResult,
    VideoManifest,
)


VIDEO_ID = "lecture_001"


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(data_dir=tmp_path / "data"))


def _write_manifest(data_dir: Path, video_id: str = VIDEO_ID) -> None:
    write_json(
        data_dir / "manifests" / video_id / "video_manifest.json",
        VideoManifest(
            video_id=video_id,
            title="Lecture",
            source_path=f"data/videos/{video_id}.mp4",
            original_filename="lecture.mp4",
            created_at="2026-01-01T00:00:00Z",
        ),
    )


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{VIDEO_ID}_chunk_0000",
        video_id=VIDEO_ID,
        score=0.12,
        start_time=0.0,
        end_time=30.0,
        transcript_text="The transcript answer.",
        ocr_text="Slide text",
        vlm_caption="A slide is visible.",
        combined_text="The transcript answer. Slide text.",
        frame_paths=[f"data/frames/{VIDEO_ID}/frame_000000.jpg"],
    )


def test_health_endpoint(tmp_path: Path):
    client = _client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_video_registers_manifest(tmp_path: Path):
    client = _client(tmp_path)

    response = client.post(
        "/videos",
        params={"title": "Lecture", "video_id": VIDEO_ID},
        files={"file": ("lecture.mp4", b"fake video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_id"] == VIDEO_ID
    assert (tmp_path / payload["manifest_path"]).exists()


def test_get_video_returns_manifest_and_artifact_flags(tmp_path: Path):
    data_dir = tmp_path / "data"
    _write_manifest(data_dir)

    response = _client(tmp_path).get(f"/videos/{VIDEO_ID}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["video"]["video_id"] == VIDEO_ID
    assert payload["artifacts"]["manifest"] is True
    assert payload["artifacts"]["transcript"] is False
    assert payload["validation"] is None


def test_index_endpoint_creates_successful_background_job(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_manifest(data_dir)
    calls = []

    def fake_run_pipeline(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(api_app, "run_pipeline", fake_run_pipeline)

    response = _client(tmp_path).post(
        f"/videos/{VIDEO_ID}/index",
        json={
            "transcription_provider": "mock",
            "embedding_provider": "mock",
            "skip_ocr": True,
            "skip_captions": True,
            "skip_vector_store": True,
            "skip_validation": True,
        },
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job_response = _client(tmp_path).get(f"/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "succeeded"
    assert calls[0]["video_id"] == VIDEO_ID
    assert calls[0]["data_dir"] == data_dir
    assert calls[0]["transcription_provider"] == "mock"
    assert calls[0]["embedding_provider"] == "mock"
    assert calls[0]["skip_ocr"] is True


def test_index_endpoint_records_background_failure(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    _write_manifest(data_dir)

    def fake_run_pipeline(**kwargs):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(api_app, "run_pipeline", fake_run_pipeline)

    response = _client(tmp_path).post(f"/videos/{VIDEO_ID}/index", json={})

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = _client(tmp_path).get(f"/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "pipeline failed" in job["error"]


def test_index_missing_video_returns_404(tmp_path: Path):
    response = _client(tmp_path).post(f"/videos/{VIDEO_ID}/index", json={})

    assert response.status_code == 404


def test_search_endpoint_returns_retrieval_results(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path / "data")

    def fake_retrieve(*args, **kwargs):
        return [_retrieval_result()]

    monkeypatch.setattr(api_app, "retrieve", fake_retrieve)

    response = _client(tmp_path).post(
        f"/videos/{VIDEO_ID}/search",
        json={"question": "What was said?", "provider": "mock", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["chunk_id"] == f"{VIDEO_ID}_chunk_0000"
    assert payload[0]["frame_paths"] == [f"data/frames/{VIDEO_ID}/frame_000000.jpg"]


def test_answer_endpoint_returns_grounded_answer(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path / "data")

    def fake_answer_video(*args, **kwargs):
        result = _retrieval_result()
        return AnswerResult(
            video_id=VIDEO_ID,
            question="What was said?",
            answer="The transcript answer. [00:00-00:30]",
            citations=[
                AnswerCitation(
                    chunk_id=result.chunk_id,
                    video_id=VIDEO_ID,
                    start_time=result.start_time,
                    end_time=result.end_time,
                    frame_paths=result.frame_paths,
                )
            ],
            retrieval_results=[result],
            model="mock-answer",
            provider="mock",
        )

    monkeypatch.setattr(api_app, "answer_video", fake_answer_video)

    response = _client(tmp_path).post(
        f"/videos/{VIDEO_ID}/answer",
        json={
            "question": "What was said?",
            "retrieval_provider": "mock",
            "answer_provider": "mock",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "The transcript answer. [00:00-00:30]"
    assert payload["citations"][0]["chunk_id"] == f"{VIDEO_ID}_chunk_0000"
