from __future__ import annotations

from pathlib import Path

from video_rag.api.app import _build_parser


ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_installs_runtime_dependencies():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "apt-get install" in dockerfile
    assert "ffmpeg" in dockerfile
    assert "pip install --no-cache-dir -e" in dockerfile
    assert ".[api,transcribe,embed,answer,vectorstore]" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'CMD ["python", "-m", "video_rag.api.app"]' in dockerfile


def test_render_blueprint_uses_persistent_data_disk():
    render_yaml = (ROOT / "render.yaml").read_text(encoding="utf-8")

    assert "runtime: docker" in render_yaml
    assert "healthCheckPath: /health" in render_yaml
    assert "mountPath: /app/data" in render_yaml
    assert "OPENAI_API_KEY" in render_yaml


def test_api_parser_reads_hosting_environment(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8765")
    monkeypatch.setenv("DATA_DIR", "/tmp/raggers-data")

    args = _build_parser().parse_args([])

    assert args.host == "0.0.0.0"
    assert args.port == 8765
    assert args.data_dir == "/tmp/raggers-data"
