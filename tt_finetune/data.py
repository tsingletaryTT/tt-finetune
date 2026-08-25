"""Public dataset sources resolved without exposing TT input tensors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .errors import TTDependencyError


@dataclass(frozen=True)
class DatasetSource:
    source: Any
    split: str = "train"
    name: str | None = None
    data_files: Any = None


def _local_rows(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        rows = [
            {"text": line.rstrip("\n")}
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"dataset is empty: {path}")
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            for key in ("data", "train", "rows"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"JSON dataset must contain a non-empty row list: {path}")
        return payload
    if suffix in {".jsonl", ".ndjson"}:
        return path
    raise ValueError(
        f"Unsupported local dataset extension {suffix!r}; use JSON, JSONL, or text"
    )


def resolve_dataset(source: Any, *, default_split: str = "train") -> Any:
    """Resolve local/Hugging Face sources while preserving in-memory datasets."""

    if source is None:
        return None
    spec = source if isinstance(source, DatasetSource) else DatasetSource(source, default_split)
    value = spec.source
    if not isinstance(value, (str, Path)):
        return value
    path = Path(value).expanduser()
    if path.is_file():
        return _local_rows(path.resolve())
    text = str(value)
    if path.suffix.lower() in {".json", ".jsonl", ".ndjson", ".txt", ".text"}:
        raise FileNotFoundError(path)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise TTDependencyError(
            "Loading a Hugging Face dataset ID requires the optional `datasets` "
            "package. Install tt-finetune with `pip install 'tt-finetune[data]'`."
        ) from error
    return load_dataset(
        text,
        spec.name,
        split=spec.split,
        data_files=spec.data_files,
    )


__all__ = ["DatasetSource", "resolve_dataset"]
