from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from video_rag.index import run_pipeline as run_pipeline_module
from video_rag.index.run_pipeline import main, run_pipeline


@pytest.fixture
def stage_calls(monkeypatch):
    """Replace every stage with a recording stub. Returns the call log."""
    calls: list[tuple[str, dict]] = []

    def make_stub(name: str, return_value=None):
        def stub(*args, **kwargs):
            calls.append((name, {"args": args, "kwargs": kwargs}))
            return return_value

        return stub

    fake_manifest = SimpleNamespace(video_id="vid_from_register")

    monkeypatch.setattr(
        run_pipeline_module,
        "register_video",
        make_stub("register_video", return_value=fake_manifest),
    )
    monkeypatch.setattr(run_pipeline_module, "probe_media", make_stub("probe_media"))
    monkeypatch.setattr(run_pipeline_module, "extract_audio", make_stub("extract_audio"))
    monkeypatch.setattr(
        run_pipeline_module, "transcribe_audio", make_stub("transcribe_audio", return_value=[])
    )
    monkeypatch.setattr(
        run_pipeline_module, "sample_frames", make_stub("sample_frames", return_value=[])
    )
    monkeypatch.setattr(run_pipeline_module, "run_ocr", make_stub("run_ocr", return_value=[]))
    monkeypatch.setattr(
        run_pipeline_module, "caption_frames", make_stub("caption_frames", return_value=[])
    )
    monkeypatch.setattr(
        run_pipeline_module, "build_chunks", make_stub("build_chunks", return_value=[])
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "build_search_text",
        make_stub("build_search_text", return_value=[]),
    )
    monkeypatch.setattr(
        run_pipeline_module, "embed_chunks", make_stub("embed_chunks", return_value=[])
    )
    monkeypatch.setattr(
        run_pipeline_module, "store_vectors", make_stub("store_vectors")
    )
    monkeypatch.setattr(
        run_pipeline_module,
        "validate_index",
        make_stub(
            "validate_index",
            return_value=SimpleNamespace(status="passed", errors=[], warnings=[]),
        ),
    )
    return calls


# --- input modes ------------------------------------------------------------


def test_video_path_mode_runs_register_first(stage_calls, tmp_path: Path):
    fake_video = tmp_path / "input.mp4"
    fake_video.write_bytes(b"")

    result = run_pipeline(video_path=fake_video, data_dir=tmp_path / "data")

    assert result == "vid_from_register"
    names = [name for name, _ in stage_calls]
    assert names == [
        "register_video",
        "probe_media",
        "extract_audio",
        "transcribe_audio",
        "sample_frames",
        "run_ocr",
        "caption_frames",
        "build_chunks",
        "build_search_text",
        "embed_chunks",
        "store_vectors",
        "validate_index",
    ]


def test_video_id_mode_skips_register(stage_calls, tmp_path: Path):
    data = tmp_path / "data"
    manifest = data / "manifests" / "lec_001" / "video_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    result = run_pipeline(video_id="lec_001", data_dir=data)

    assert result == "lec_001"
    names = [name for name, _ in stage_calls]
    assert "register_video" not in names
    assert names == [
        "probe_media",
        "extract_audio",
        "transcribe_audio",
        "sample_frames",
        "run_ocr",
        "caption_frames",
        "build_chunks",
        "build_search_text",
        "embed_chunks",
        "store_vectors",
        "validate_index",
    ]


def test_video_id_mode_missing_manifest_raises_before_any_stage(
    stage_calls, tmp_path: Path
):
    with pytest.raises(FileNotFoundError, match="missing video manifest"):
        run_pipeline(video_id="ghost", data_dir=tmp_path / "data")
    assert stage_calls == []


def test_requires_exactly_one_of_video_or_video_id(stage_calls, tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        run_pipeline(data_dir=tmp_path / "data")

    fake_video = tmp_path / "v.mp4"
    fake_video.write_bytes(b"")
    with pytest.raises(ValueError, match="exactly one"):
        run_pipeline(video_path=fake_video, video_id="x", data_dir=tmp_path / "data")
    assert stage_calls == []


# --- argument forwarding ----------------------------------------------------


def test_video_id_from_register_flows_downstream(stage_calls, tmp_path: Path):
    fake_video = tmp_path / "input.mp4"
    fake_video.write_bytes(b"")

    run_pipeline(video_path=fake_video, data_dir=tmp_path / "data")

    for name, payload in stage_calls[1:]:
        # video_id is the first positional or the `video_id` kwarg
        kw = payload["kwargs"]
        args = payload["args"]
        actual = kw.get("video_id") if "video_id" in kw else args[0]
        assert actual == "vid_from_register", f"{name} got video_id={actual!r}"


def test_every_stage_receives_overwrite_true(stage_calls, tmp_path: Path):
    data = tmp_path / "data"
    manifest = data / "manifests" / "lec" / "video_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    run_pipeline(video_id="lec", data_dir=data)

    for name, payload in stage_calls:
        if name == "validate_index":
            continue
        assert payload["kwargs"].get("overwrite") is True, f"{name} missing overwrite=True"


def test_knobs_forwarded_to_correct_stages(stage_calls, tmp_path: Path):
    data = tmp_path / "data"
    manifest = data / "manifests" / "lec" / "video_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    run_pipeline(
        video_id="lec",
        data_dir=data,
        transcription_provider="mock",
        language="en",
        interval_seconds=7,
        frames_per_caption=4,
        chunk_seconds=20,
        embedding_provider="mock",
        embedding_model="fake-model",
        embedding_variant="transcript_only",
        embedding_batch_size=8,
        vector_backend="chroma",
    )

    by_name = {name: payload["kwargs"] for name, payload in stage_calls}
    assert by_name["transcribe_audio"]["provider"] == "mock"
    assert by_name["transcribe_audio"]["language"] == "en"
    assert by_name["sample_frames"]["interval_seconds"] == 7
    assert by_name["caption_frames"]["frames_per_caption"] == 4
    assert by_name["build_chunks"]["chunk_seconds"] == 20
    assert by_name["build_search_text"]["chunk_seconds"] == 20
    assert by_name["embed_chunks"]["chunk_seconds"] == 20
    assert by_name["embed_chunks"]["provider"] == "mock"
    assert by_name["embed_chunks"]["model"] == "fake-model"
    assert by_name["embed_chunks"]["variant"] == "transcript_only"
    assert by_name["embed_chunks"]["batch_size"] == 8
    assert by_name["store_vectors"]["backend"] == "chroma"
    assert by_name["store_vectors"]["variant"] == "transcript_only"
    assert by_name["validate_index"]["variant"] == "transcript_only"


def test_optional_visual_and_validation_stages_can_be_skipped(stage_calls, tmp_path: Path):
    data = tmp_path / "data"
    manifest = data / "manifests" / "lec" / "video_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    run_pipeline(
        video_id="lec",
        data_dir=data,
        skip_ocr=True,
        skip_captions=True,
        skip_vector_store=True,
        skip_validation=True,
    )

    names = [name for name, _ in stage_calls]
    assert "run_ocr" not in names
    assert "caption_frames" not in names
    assert "store_vectors" not in names
    assert "validate_index" not in names
    assert "build_chunks" in names
    assert "build_search_text" in names
    assert "embed_chunks" in names


def test_nonzero_overlap_rejected_until_downstream_filenames_support_it(
    stage_calls, tmp_path: Path
):
    data = tmp_path / "data"
    manifest = data / "manifests" / "lec" / "video_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    with pytest.raises(ValueError, match="overlap_seconds=0"):
        run_pipeline(video_id="lec", data_dir=data, overlap_seconds=5)

    assert stage_calls == []


def test_failed_index_validation_aborts_with_runtime_error(stage_calls, monkeypatch, tmp_path: Path):
    data = tmp_path / "data"
    manifest = data / "manifests" / "lec" / "video_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    monkeypatch.setattr(
        run_pipeline_module,
        "validate_index",
        lambda *a, **k: SimpleNamespace(
            status="failed", errors=["missing vectors"], warnings=[]
        ),
    )

    with pytest.raises(RuntimeError, match="index validation failed"):
        run_pipeline(video_id="lec", data_dir=data)


# --- fail-fast --------------------------------------------------------------


def test_stage_failure_aborts_remaining_stages(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def record(name):
        def stub(*a, **k):
            calls.append(name)

        return stub

    def boom(*a, **k):
        calls.append("extract_audio")
        raise RuntimeError("ffmpeg missing")

    fake_manifest = SimpleNamespace(video_id="lec")
    monkeypatch.setattr(
        run_pipeline_module,
        "register_video",
        lambda *a, **k: (calls.append("register_video"), fake_manifest)[1],
    )
    monkeypatch.setattr(run_pipeline_module, "probe_media", record("probe_media"))
    monkeypatch.setattr(run_pipeline_module, "extract_audio", boom)
    monkeypatch.setattr(run_pipeline_module, "transcribe_audio", record("transcribe_audio"))
    monkeypatch.setattr(run_pipeline_module, "sample_frames", record("sample_frames"))
    monkeypatch.setattr(run_pipeline_module, "run_ocr", record("run_ocr"))
    monkeypatch.setattr(run_pipeline_module, "caption_frames", record("caption_frames"))
    monkeypatch.setattr(run_pipeline_module, "build_chunks", record("build_chunks"))
    monkeypatch.setattr(run_pipeline_module, "build_search_text", record("build_search_text"))
    monkeypatch.setattr(run_pipeline_module, "embed_chunks", record("embed_chunks"))
    monkeypatch.setattr(run_pipeline_module, "store_vectors", record("store_vectors"))
    monkeypatch.setattr(run_pipeline_module, "validate_index", record("validate_index"))

    fake_video = tmp_path / "input.mp4"
    fake_video.write_bytes(b"")

    with pytest.raises(RuntimeError, match="ffmpeg missing"):
        run_pipeline(video_path=fake_video, data_dir=tmp_path / "data")

    assert calls == ["register_video", "probe_media", "extract_audio"]


# --- CLI --------------------------------------------------------------------


def test_cli_video_id_mode(stage_calls, tmp_path: Path):
    data = tmp_path / "data"
    manifest = data / "manifests" / "lec" / "video_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")

    rc = main(
        [
            "--video-id",
            "lec",
            "--data-dir",
            str(data),
            "--provider",
            "mock",
            "--interval-seconds",
            "10",
            "--embedding-provider",
            "mock",
            "--embedding-variant",
            "transcript_only",
            "--skip-vector-store",
            "--skip-validation",
        ]
    )
    assert rc == 0
    by_name = {name: payload["kwargs"] for name, payload in stage_calls}
    assert by_name["transcribe_audio"]["provider"] == "mock"
    assert by_name["sample_frames"]["interval_seconds"] == 10
    assert by_name["embed_chunks"]["provider"] == "mock"
    assert by_name["embed_chunks"]["variant"] == "transcript_only"


def test_cli_requires_video_or_video_id(stage_calls):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_cli_rejects_both_video_and_video_id(stage_calls, tmp_path: Path):
    fake_video = tmp_path / "v.mp4"
    fake_video.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        main(["--video", str(fake_video), "--video-id", "x"])
    assert exc.value.code != 0
