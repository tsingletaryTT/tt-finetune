#!/usr/bin/env python3
"""Prepare a pinned public instruction dataset for silicon endurance runs.

The script intentionally performs no tokenization.  It produces the ordinary
``prompt``/``response`` JSONL contract consumed by the public ``TTTrainer`` so
that tokenizer, packing, bucketing, and prefetch behavior remain part of the
product qualification rather than being hidden in dataset preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any


DATASET_ID = "yahma/alpaca-cleaned"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_REVISION = "a4a0620c7c65decc93395b78f0e6994664a8662e"
PARQUET_SHA256 = "d11b9d13a0f13c77924f0d6ceeb83baaebb3aa3eb141fce8bd2c8e033bedf420"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    instruction = str(record.get("instruction") or "").strip()
    context = str(record.get("input") or "").strip()
    response = str(record.get("output") or "").strip()
    if not instruction or not response:
        raise ValueError(f"empty instruction/response at source row {source_index}")
    prompt = instruction
    if context:
        prompt = f"{instruction}\n\nInput:\n{context}"
    return {
        "prompt": prompt,
        "response": response,
        "source_index": source_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--heldout-rows", type=int, default=512)
    arguments = parser.parse_args()

    source = arguments.parquet.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if _sha256(source) != PARQUET_SHA256:
        raise RuntimeError("source parquet does not match the pinned dataset shard")

    import pyarrow.parquet as pq

    table = pq.read_table(source, columns=["instruction", "input", "output"])
    records = table.to_pylist()
    if not 0 < arguments.heldout_rows < len(records):
        raise ValueError("heldout-rows must leave at least one training row")

    indices = list(range(len(records)))
    random.Random(arguments.seed).shuffle(indices)
    heldout_indices = set(indices[: arguments.heldout_rows])
    train_rows = [
        _row(record, index)
        for index, record in enumerate(records)
        if index not in heldout_indices
    ]
    heldout_rows = [
        _row(records[index], index) for index in indices[: arguments.heldout_rows]
    ]

    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "alpaca_train.jsonl"
    heldout_path = output / "alpaca_heldout.jsonl"
    for path, rows in ((train_path, train_rows), (heldout_path, heldout_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    manifest = {
        "format": "tt-finetune-real-workload-dataset-v1",
        "source": {
            "dataset_id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "parquet_filename": "default/train/0000.parquet",
            "parquet_sha256": PARQUET_SHA256,
            "source_rows": len(records),
        },
        "preparation": {
            "seed": arguments.seed,
            "heldout_selection": "first N indices after Python random.Random(seed).shuffle",
            "schema": "prompt_response",
            "input_rendering": "instruction + optional double-newline Input section",
            "tokenization": "deferred to public TTTrainer",
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "rows": len(train_rows),
                "sha256": _sha256(train_path),
            },
            "heldout": {
                "path": str(heldout_path),
                "rows": len(heldout_rows),
                "sha256": _sha256(heldout_path),
            },
        },
    }
    manifest_path = output / "DATASET_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
