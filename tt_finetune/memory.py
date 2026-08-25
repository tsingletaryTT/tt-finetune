"""Blackhole device-memory reports and low-memory accounting helpers.

TT-XLA does not currently implement ``PJRT_Device_MemoryStats``.  TT-Metal's
allocator reporter is therefore the authoritative source available to the
public trainer.  It records an allocator snapshot whenever a program is
compiled.  The report is deliberately labelled as a compile-boundary peak:
buffers allocated only after compilation are not visible and must not be
misrepresented as an exact runtime high-water mark.
"""

from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any


_DETAIL_TOTAL = re.compile(r"^\s*Total (allocatable|allocated|free) \(B\):\s*$")
_OOM = re.compile(
    r"Out of Memory: Not enough space to allocate (?P<request>\d+) B .*?"
    r"bank size is (?P<bank_size>\d+) B \(allocated: (?P<allocated>\d+) B, "
    r"free: (?P<free>\d+) B, largest free block: (?P<largest>\d+) B\)",
    re.DOTALL,
)


def _integer(value: str) -> int:
    return int(value.strip().replace(",", ""))


def _summary_snapshots(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    snapshots: list[dict[str, Any]] = []
    current_program: int | None = None
    memory_index = 0
    kinds = ("dram_per_bank", "l1_per_core", "l1_small_per_core")
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            raw_program = (row.get("Program ID") or "").strip()
            if raw_program:
                current_program = int(raw_program)
                memory_index = 0
            if current_program is None or memory_index >= len(kinds):
                continue
            snapshots.append(
                {
                    "program_id": current_program,
                    "memory": kinds[memory_index],
                    "total_bytes": _integer(
                        row[" Total Allocatable Size (B)"]
                    ),
                    "allocated_bytes": _integer(
                        row[" Total Allocated (B)"]
                    ),
                    "free_bytes": _integer(row[" Total Free (B)"]),
                    "largest_free_block_bytes": _integer(
                        row[" Largest Free Block (B)"]
                    ),
                }
            )
            memory_index += 1
    return snapshots


def _detailed_snapshots(path: Path) -> list[dict[str, Any]]:
    """Parse aggregate allocator totals from the human-oriented detail CSV."""

    if not path.is_file():
        return []
    output: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_key: str | None = None
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.reader(source):
            first = row[0].strip() if row else ""
            second = row[1].strip() if len(row) > 1 else ""
            if second in {"DRAM", "L1", "L1_SMALL"} and (
                first.isdigit() or (not first and current is not None)
            ):
                if current is not None:
                    output.append(current)
                current = {
                    "program_id": (
                        int(first) if first else int(current["program_id"])
                    ),
                    "memory": second.lower(),
                }
                pending_key = None
                continue
            if current is None:
                continue
            match = _DETAIL_TOTAL.match(second)
            if match:
                pending_key = f"{match.group(1)}_bytes"
                if len(row) > 2 and row[2].strip():
                    current[pending_key] = _integer(row[2])
                    pending_key = None
                continue
            if pending_key and first:
                current[pending_key] = _integer(first)
                pending_key = None
    if current is not None:
        output.append(current)
    return output


def parse_tt_metal_memory_reports(log_root: str | Path) -> dict[str, Any]:
    """Return a stable, honest summary of TT-Metal allocator snapshots."""

    root = Path(log_root)
    reports = root / "generated" / "reports"
    summary_path = reports / "program_memory_usage_summary.csv"
    detail_path = reports / "program_detailed_memory_usage.csv"
    summary = _summary_snapshots(summary_path)
    detailed = _detailed_snapshots(detail_path)
    dram = [item for item in detailed if item["memory"] == "dram"]
    l1 = [item for item in detailed if item["memory"] == "l1"]
    per_bank = [
        item for item in summary if item["memory"] == "dram_per_bank"
    ]

    peak_dram = max(
        dram, key=lambda item: item.get("allocated_bytes", -1), default=None
    )
    peak_l1 = max(
        l1, key=lambda item: item.get("allocated_bytes", -1), default=None
    )
    worst_fragmentation = None
    fragmentation_program = None
    for item in per_bank:
        free = item["free_bytes"]
        value = (
            0.0
            if free == 0
            else 1.0 - item["largest_free_block_bytes"] / free
        )
        if worst_fragmentation is None or value > worst_fragmentation:
            worst_fragmentation = value
            fragmentation_program = item["program_id"]

    return {
        "available": bool(summary or detailed),
        "measurement_contract": (
            "tt_metal_allocator_program_compile_snapshots_v1"
        ),
        "is_exact_runtime_high_water_mark": False,
        "limitation": (
            "TT-Metal writes allocator snapshots at program compilation; "
            "buffers allocated only after compilation are not captured. "
            "PJRT_Device_MemoryStats is unimplemented in the current TT-XLA "
            "plugin. OOM allocator diagnostics remain authoritative at a "
            "failed allocation boundary."
        ),
        "program_count": len({item["program_id"] for item in detailed}),
        "peak_compile_snapshot_dram": peak_dram,
        "peak_compile_snapshot_l1": peak_l1,
        "worst_per_bank_fragmentation_fraction": worst_fragmentation,
        "worst_per_bank_fragmentation_program_id": fragmentation_program,
        "minimum_largest_free_dram_block_per_bank_bytes": (
            min(item["largest_free_block_bytes"] for item in per_bank)
            if per_bank else None
        ),
        "report_files": {
            "summary": str(summary_path) if summary_path.is_file() else None,
            "detail": str(detail_path) if detail_path.is_file() else None,
            "l1": str(reports / "program_l1_usage_summary.csv")
            if (reports / "program_l1_usage_summary.csv").is_file()
            else None,
        },
    }


def tensor_storage_bytes(parameter: Any) -> int:
    """Logical tensor storage, independent of device tiling/padding."""

    return int(parameter.numel()) * int(parameter.element_size())


def parse_tt_oom_boundary(text: str) -> dict[str, Any] | None:
    """Extract the allocator state at a genuine TT-Metal OOM boundary."""

    match = _OOM.search(text)
    if match is None:
        return None
    values = {name: int(value) for name, value in match.groupdict().items()}
    values.update(
        {
            "contract": "tt_metal_failed_allocation_boundary_v1",
            "bank_count": 8,
            "allocated_aggregate_bytes": values["allocated"] * 8,
            "free_aggregate_bytes": values["free"] * 8,
            "largest_free_block_aggregate_upper_bound_bytes": (
                values["largest"] * 8
            ),
            "fragmentation_fraction_per_bank": (
                0.0
                if values["free"] == 0
                else 1.0 - values["largest"] / values["free"]
            ),
        }
    )
    return values


__all__ = [
    "parse_tt_metal_memory_reports",
    "parse_tt_oom_boundary",
    "tensor_storage_bytes",
]
