"""FastAPI app for upload, indexing, search, answers, and the local UI."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from video_rag.index.register_video import register_video
from video_rag.index.run_pipeline import run_pipeline
from video_rag.io_utils import read_json, write_json
from video_rag.schemas import AnswerResult, IndexValidationReport, VideoManifest
from video_rag.search.answer import answer_video
from video_rag.search.retrieve import retrieve

PathLike = str | Path

JobStatus = Literal["queued", "running", "succeeded", "failed"]
STATIC_DIR = Path(__file__).resolve().parent / "static"


class IndexRequest(BaseModel):
    transcription_provider: str = "openai"
    language: str | None = None
    interval_seconds: int = Field(default=5, gt=0)
    frames_per_caption: int = Field(default=3, gt=0)
    chunk_seconds: float = Field(default=30, gt=0)
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_variant: str = "transcript_ocr_vlm"
    embedding_batch_size: int = Field(default=64, gt=0)
    vector_backend: str = "chroma"
    skip_ocr: bool = False
    skip_captions: bool = False
    skip_vector_store: bool = False
    skip_validation: bool = False


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    chunk_seconds: float = Field(default=30, gt=0)
    variant: str = "transcript_ocr_vlm"
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    top_k: int = Field(default=5, gt=0)


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    chunk_seconds: float = Field(default=30, gt=0)
    variant: str = "transcript_ocr_vlm"
    retrieval_provider: str = "openai"
    retrieval_model: str = "text-embedding-3-small"
    answer_provider: str = "openai"
    answer_model: str = "gpt-4o-mini"
    top_k: int = Field(default=5, gt=0)


class JobRecord(BaseModel):
    job_id: str
    video_id: str
    status: JobStatus
    stage: str
    created_at: str
    updated_at: str
    error: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _job_path(data_root: Path, job_id: str) -> Path:
    return data_root / "jobs" / f"{job_id}.json"


def _manifest_path(data_root: Path, video_id: str) -> Path:
    return data_root / "manifests" / video_id / "video_manifest.json"


def _validation_path(
    data_root: Path,
    video_id: str,
    chunk_seconds: int | float = 30,
    variant: str = "transcript_ocr_vlm",
) -> Path:
    label = f"{float(chunk_seconds):g}"
    return data_root / "validation" / f"{video_id}_{label}s_{variant}_validation.json"


def _read_job(data_root: Path, job_id: str) -> JobRecord:
    path = _job_path(data_root, job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return read_json(path, JobRecord)


def _write_job(data_root: Path, job: JobRecord) -> None:
    write_json(_job_path(data_root, job.job_id), job)


def _update_job(
    data_root: Path,
    job_id: str,
    *,
    status: JobStatus,
    stage: str,
    error: str | None = None,
) -> None:
    job = _read_job(data_root, job_id)
    updated = job.model_copy(
        update={
            "status": status,
            "stage": stage,
            "updated_at": _now(),
            "error": error,
        }
    )
    _write_job(data_root, updated)


def _run_index_job(
    data_root: Path,
    job_id: str,
    video_id: str,
    request: IndexRequest,
) -> None:
    try:
        _update_job(data_root, job_id, status="running", stage="indexing")
        run_pipeline(
            video_id=video_id,
            data_dir=data_root,
            transcription_provider=request.transcription_provider,
            language=request.language,
            interval_seconds=request.interval_seconds,
            frames_per_caption=request.frames_per_caption,
            chunk_seconds=request.chunk_seconds,
            embedding_provider=request.embedding_provider,
            embedding_model=request.embedding_model,
            embedding_variant=request.embedding_variant,
            embedding_batch_size=request.embedding_batch_size,
            vector_backend=request.vector_backend,
            skip_ocr=request.skip_ocr,
            skip_captions=request.skip_captions,
            skip_vector_store=request.skip_vector_store,
            skip_validation=request.skip_validation,
        )
    except Exception as e:
        _update_job(
            data_root,
            job_id,
            status="failed",
            stage="failed",
            error=f"{e.__class__.__name__}: {e}",
        )
        return
    _update_job(data_root, job_id, status="succeeded", stage="complete")


def _artifact_flags(data_root: Path, video_id: str) -> dict[str, bool]:
    return {
        "manifest": _manifest_path(data_root, video_id).exists(),
        "metadata": (data_root / "manifests" / video_id / "media_metadata.json").exists(),
        "audio": (data_root / "audio" / f"{video_id}.wav").exists(),
        "transcript": (data_root / "transcripts" / f"{video_id}.jsonl").exists(),
        "frames": (data_root / "frames" / video_id / "frame_manifest.jsonl").exists(),
        "ocr": (data_root / "ocr" / f"{video_id}.jsonl").exists(),
        "captions": (data_root / "captions" / f"{video_id}.jsonl").exists(),
        "chunks": (data_root / "chunks" / f"{video_id}_30s_enriched.jsonl").exists(),
        "validation": _validation_path(data_root, video_id).exists(),
    }


def _resolve_static_file(asset_path: str) -> Path:
    root = STATIC_DIR.resolve()
    candidate = (root / asset_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="static asset not found") from e
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="static asset not found")
    return candidate


def _resolve_artifact_file(data_root: Path, artifact_path: str) -> Path:
    parts = Path(artifact_path).parts
    if parts and parts[0] == data_root.name:
        relative = Path(*parts[1:])
    else:
        relative = Path(artifact_path)
    root = data_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="artifact not found") from e
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return candidate


def create_app(data_dir: PathLike = "data") -> FastAPI:
    """Create the RAGGERS FastAPI app."""
    data_root = Path(data_dir)
    app = FastAPI(title="RAGGERS API")

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(_resolve_static_file("index.html"))

    @app.get("/static/{asset_path:path}", include_in_schema=False)
    def static_asset(asset_path: str) -> FileResponse:
        return FileResponse(_resolve_static_file(asset_path))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/videos")
    def upload_video(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
        video_id: str | None = Form(default=None),
        mode: Literal["copy", "symlink"] = Form(default="copy"),
    ) -> dict[str, str]:
        title = _blank_to_none(title)
        video_id = _blank_to_none(video_id)
        uploads_dir = data_root / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "upload.mp4").name
        upload_path = uploads_dir / f"{uuid4().hex}_{safe_name}"
        with upload_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        try:
            manifest = register_video(
                upload_path,
                title=title or Path(safe_name).stem,
                video_id=video_id,
                mode=mode,
                data_dir=data_root,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"{e.__class__.__name__}: {e}"
            ) from e
        return {
            "video_id": manifest.video_id,
            "manifest_path": (
                data_root / "manifests" / manifest.video_id / "video_manifest.json"
            ).as_posix(),
        }

    @app.get("/videos/{video_id}")
    def get_video(video_id: str) -> dict:
        path = _manifest_path(data_root, video_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"video not found: {video_id}")
        manifest = read_json(path, VideoManifest)
        validation_path = _validation_path(data_root, video_id)
        validation: dict | None = None
        if validation_path.exists():
            validation = read_json(validation_path, IndexValidationReport).model_dump(
                mode="json"
            )
        return {
            "video": manifest.model_dump(mode="json"),
            "artifacts": _artifact_flags(data_root, video_id),
            "validation": validation,
        }

    @app.post("/videos/{video_id}/index")
    def start_index(
        video_id: str,
        request: IndexRequest,
        background_tasks: BackgroundTasks,
    ) -> JobRecord:
        if not _manifest_path(data_root, video_id).exists():
            raise HTTPException(status_code=404, detail=f"video not found: {video_id}")
        job = JobRecord(
            job_id=uuid4().hex,
            video_id=video_id,
            status="queued",
            stage="queued",
            created_at=_now(),
            updated_at=_now(),
        )
        _write_job(data_root, job)
        background_tasks.add_task(_run_index_job, data_root, job.job_id, video_id, request)
        return job

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> JobRecord:
        return _read_job(data_root, job_id)

    @app.get("/artifacts/{artifact_path:path}")
    def get_artifact(artifact_path: str) -> FileResponse:
        return FileResponse(_resolve_artifact_file(data_root, artifact_path))

    @app.post("/videos/{video_id}/search")
    def search_video(video_id: str, request: SearchRequest) -> list[dict]:
        try:
            results = retrieve(
                video_id,
                request.question,
                data_dir=data_root,
                chunk_seconds=request.chunk_seconds,
                variant=request.variant,
                provider=request.provider,
                model=request.model,
                top_k=request.top_k,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except (ValueError, RuntimeError, OSError) as e:
            raise HTTPException(
                status_code=400, detail=f"{e.__class__.__name__}: {e}"
            ) from e
        return [result.model_dump(mode="json") for result in results]

    @app.post("/videos/{video_id}/answer")
    def answer(video_id: str, request: AnswerRequest) -> AnswerResult:
        try:
            return answer_video(
                video_id,
                request.question,
                data_dir=data_root,
                chunk_seconds=request.chunk_seconds,
                variant=request.variant,
                retrieval_provider=request.retrieval_provider,
                retrieval_model=request.retrieval_model,
                answer_provider=request.answer_provider,
                answer_model=request.answer_model,
                top_k=request.top_k,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except (ValueError, RuntimeError, OSError) as e:
            raise HTTPException(
                status_code=400, detail=f"{e.__class__.__name__}: {e}"
            ) from e

    return app


app = create_app()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_rag.api.app",
        description="Run the RAGGERS FastAPI server.",
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host.")
    p.add_argument("--port", type=int, default=8000, help="Bind port.")
    p.add_argument("--data-dir", default="data", help="Artifact root.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as e:
        print(
            'FAIL  RuntimeError: install the API extra: pip install -e ".[api]"',
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        create_app(data_dir=args.data_dir),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
