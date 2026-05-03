"""JSON / JSONL helpers for RAGGERS artifacts.

All readers validate via Pydantic. All writers create parent directories.
JSONL readers ignore blank lines and report file path + line number on errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

PathLike = str | Path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: PathLike, obj: BaseModel | dict) -> None:
    """Write a single JSON object to ``path``."""
    p = Path(path)
    _ensure_parent(p)
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    else:
        data = obj
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: PathLike, model: Type[T]) -> T:
    """Read a JSON file and validate it against ``model``."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"{p}: validation failed\n{e}") from e


def write_jsonl(path: PathLike, records: Iterable[BaseModel | dict]) -> None:
    """Write an iterable of records as JSONL (one object per line)."""
    p = Path(path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            data = rec.model_dump(mode="json") if isinstance(rec, BaseModel) else rec
            f.write(json.dumps(data, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: PathLike, model: Type[T]) -> Iterator[T]:
    """Yield validated records from a JSONL file. Blank lines are skipped."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {e.msg}") from e
            try:
                yield model.model_validate(raw)
            except ValidationError as e:
                raise ValueError(f"{p}:{lineno}: validation failed\n{e}") from e
