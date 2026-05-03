from __future__ import annotations

from pathlib import Path

import pytest

from video_rag.validate import main

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def test_cli_valid_video_manifest(capsys):
    rc = main([str(EXAMPLES / "video_manifest.example.json"), "--type", "video_manifest"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("OK")


def test_cli_valid_media_metadata(capsys):
    rc = main([str(EXAMPLES / "media_metadata.example.json"), "--type", "media_metadata"])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.startswith("OK")


def test_cli_bad_video_manifest(capsys):
    rc = main([str(EXAMPLES / "bad_video_manifest.example.json"), "--type", "video_manifest"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "FAIL" in captured.out


def test_cli_bad_media_metadata(capsys):
    rc = main([str(EXAMPLES / "bad_media_metadata.example.json"), "--type", "media_metadata"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "FAIL" in captured.out


def test_cli_missing_file(capsys, tmp_path: Path):
    missing = tmp_path / "nope.json"
    rc = main([str(missing), "--type", "video_manifest"])
    captured = capsys.readouterr()
    assert rc != 0
    assert "FAIL" in captured.out


def test_cli_unknown_type_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main(["whatever.json", "--type", "not_a_real_type"])
    assert exc.value.code != 0
