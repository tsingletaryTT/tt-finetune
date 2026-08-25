"""Phase-aliased circular-buffer allocation for a resident layer backward.

The accepted MLP, RMSNorm and attention schedules execute monotonically, so
their circular buffers must not be added together. This planner pairs the
largest lifetime from each non-overlapping phase into one physical L1 backing
slot. A backing slot may expose both BF16 and FP32 logical CB formats; TT-Metal
supports that representation and artifact-v3 preserves it through PJRT. It is
small on purpose: source remapping and device barriers consume this plan, but
the planner itself has no TT-XLA dependency and is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from math import lcm
from typing import Mapping, Sequence

_TILE_BYTES = {"bf16": 2048, "fp32": 4096}


@dataclass(frozen=True)
class PhaseCBRequirement:
    name: str
    dtype: str
    tiles: int

    def __post_init__(self) -> None:
        if self.dtype not in _TILE_BYTES:
            raise ValueError(f"unsupported CB dtype: {self.dtype}")
        if self.tiles <= 0:
            raise ValueError("CB capacity must be positive")


@dataclass(frozen=True)
class PhysicalCBSlot:
    index: int
    bytes: int
    formats: tuple[tuple[int, str], ...]
    lifetimes: tuple[tuple[str, str, str, int, int], ...]


@dataclass(frozen=True)
class PhaseAliasedCBPlan:
    slots: tuple[PhysicalCBSlot, ...]
    phase_mapping: Mapping[str, Mapping[str, int]]
    phase_alias_slots: Mapping[str, Mapping[str, int]]
    physical_slot_count: int
    logical_cb_count: int
    l1_bytes_per_core: int
    phase_unaliased_bytes: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LogicalCBFormatPlan:
    logical_index: int
    alias_slot: int
    dtype: str
    tile_capacity: int
    page_bytes: int
    physical_bytes: int


def append_dedicated_cb(
    plan: PhaseAliasedCBPlan,
    *,
    phase: str,
    name: str,
    dtype: str = "bf16",
    tiles: int = 1,
    maximum_slots: int = 64,
) -> PhaseAliasedCBPlan:
    """Append a non-aliased control/lifetime slot to an existing plan."""

    requirement = PhaseCBRequirement(name, dtype, tiles)
    if phase in plan.phase_mapping:
        raise ValueError(f"dedicated CB phase already exists: {phase}")
    logical = plan.logical_cb_count
    if logical + 1 > maximum_slots:
        raise ValueError(
            f"dedicated CB needs {logical + 1} logical IDs, limit is "
            f"{maximum_slots}"
        )
    slot_index = plan.physical_slot_count
    size = tiles * _TILE_BYTES[dtype]
    slot = PhysicalCBSlot(
        index=slot_index,
        bytes=size,
        formats=((logical, dtype),),
        lifetimes=((phase, name, dtype, tiles, logical),),
    )
    mapping = {key: dict(value) for key, value in plan.phase_mapping.items()}
    aliases = {key: dict(value) for key, value in plan.phase_alias_slots.items()}
    mapping[phase] = {name: logical}
    aliases[phase] = {name: slot_index}
    unaliased = dict(plan.phase_unaliased_bytes)
    unaliased[phase] = size
    return PhaseAliasedCBPlan(
        slots=plan.slots + (slot,),
        phase_mapping=mapping,
        phase_alias_slots=aliases,
        physical_slot_count=slot_index + 1,
        logical_cb_count=logical + 1,
        l1_bytes_per_core=plan.l1_bytes_per_core + size,
        phase_unaliased_bytes=unaliased,
    )


def logical_cb_format_plans(
    plan: PhaseAliasedCBPlan,
) -> tuple[LogicalCBFormatPlan, ...]:
    """Flatten physical slots into canonical artifact-v3 logical configs."""

    formats = []
    for slot in plan.slots:
        for logical_index, dtype in slot.formats:
            page_bytes = _TILE_BYTES[dtype]
            if slot.bytes % page_bytes:
                raise ValueError(
                    f"physical slot {slot.index} is not aligned for {dtype}"
                )
            formats.append(
                LogicalCBFormatPlan(
                    logical_index=logical_index,
                    alias_slot=slot.index,
                    dtype=dtype,
                    tile_capacity=slot.bytes // page_bytes,
                    page_bytes=page_bytes,
                    physical_bytes=slot.bytes,
                )
            )
    formats.sort(key=lambda item: item.logical_index)
    if [item.logical_index for item in formats] != list(range(len(formats))):
        raise ValueError("logical CB indices must be canonical and contiguous")
    return tuple(formats)


def install_phase_aliased_cb_plan(
    compiled: object,
    plan: PhaseAliasedCBPlan,
) -> None:
    """Install the canonical logical formats and v3 aliases on a TT-Lang op."""

    from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

    configs = []
    aliases = []
    for item in logical_cb_format_plans(plan):
        config = CompilerAllocatedDFBConfig(
            dfb_index=item.logical_index,
            num_tiles=1,
            data_format="float32" if item.dtype == "fp32" else "bfloat16",
            block_count=item.tile_capacity,
        )
        config._cb_index = item.logical_index
        configs.append(config)
        aliases.append(item.alias_slot)
    compiled.cb_configs = configs
    compiled.cb_alias_slots = aliases


def install_segmented_phase_aliased_cb_plans(
    compiled: object,
    segments: Sequence[tuple[PhaseAliasedCBPlan, object]],
) -> None:
    """Install disjoint-core CB plans into one TT-Lang program artifact.

    Hardware CB IDs are local to a core, so c0 may have different descriptors
    on projection, middle and edge ranges. Descriptor argument positions and
    physical alias labels are global in the serialized artifact, however, and
    are therefore concatenated with deterministic per-segment offsets.
    """

    if not segments:
        raise ValueError("at least one CB segment is required")
    from ttl.dataflow_buffer import CompilerAllocatedDFBConfig

    configs = []
    aliases = []
    ranges = []
    alias_offset = 0
    for plan, core_range in segments:
        for item in logical_cb_format_plans(plan):
            config = CompilerAllocatedDFBConfig(
                dfb_index=item.logical_index,
                num_tiles=1,
                data_format=("float32" if item.dtype == "fp32" else "bfloat16"),
                block_count=item.tile_capacity,
            )
            config._cb_index = item.logical_index
            configs.append(config)
            aliases.append(alias_offset + item.alias_slot)
            ranges.append(core_range)
        alias_offset += plan.physical_slot_count
    compiled.cb_configs = configs
    compiled.cb_alias_slots = aliases
    compiled.cb_core_ranges = ranges


def plan_phase_aliased_cbs(
    phases: Mapping[str, Sequence[PhaseCBRequirement]],
    *,
    maximum_slots: int = 64,
    maximum_l1_bytes: int | None = None,
) -> PhaseAliasedCBPlan:
    """Pair descending byte capacities across non-overlapping epochs.

    Each physical slot contains at most one requirement from any phase.
    Different dtypes receive different logical hardware CB indices but may
    share the same backing address because their lifetimes do not overlap.
    Pairing requirements by descending byte size minimizes the sum of slot
    maxima.
    """

    if not phases:
        raise ValueError("at least one CB phase is required")
    normalized: dict[str, list[PhaseCBRequirement]] = {}
    unaliased: dict[str, int] = {}
    for phase, requirements in phases.items():
        if not phase:
            raise ValueError("phase names must be non-empty")
        names = [requirement.name for requirement in requirements]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate CB requirement in phase {phase}")
        normalized[phase] = sorted(
            requirements,
            key=lambda item: (
                -(item.tiles * _TILE_BYTES[item.dtype]),
                item.dtype,
                item.name,
            ),
        )
        unaliased[phase] = sum(
            requirement.tiles * _TILE_BYTES[requirement.dtype]
            for requirement in requirements
        )

    slots: list[PhysicalCBSlot] = []
    mapping: dict[str, dict[str, int]] = {phase: {} for phase in normalized}
    alias_mapping: dict[str, dict[str, int]] = {phase: {} for phase in normalized}
    next_logical_index = 0
    slot_count = max(len(requirements) for requirements in normalized.values())
    for rank in range(slot_count):
        requirements_at_rank = [
            (phase, requirements[rank])
            for phase, requirements in normalized.items()
            if rank < len(requirements)
        ]
        dtype_to_logical: dict[str, int] = {}
        for dtype in sorted({item.dtype for _, item in requirements_at_rank}):
            dtype_to_logical[dtype] = next_logical_index
            next_logical_index += 1

        alignment = lcm(*(_TILE_BYTES[item.dtype] for _, item in requirements_at_rank))
        capacity_bytes = max(
            item.tiles * _TILE_BYTES[item.dtype] for _, item in requirements_at_rank
        )
        capacity_bytes = ((capacity_bytes + alignment - 1) // alignment) * alignment
        slot_index = len(slots)
        lifetimes = []
        for phase, requirement in requirements_at_rank:
            logical_index = dtype_to_logical[requirement.dtype]
            mapping[phase][requirement.name] = logical_index
            alias_mapping[phase][requirement.name] = slot_index
            lifetimes.append(
                (
                    phase,
                    requirement.name,
                    requirement.dtype,
                    requirement.tiles,
                    logical_index,
                )
            )
        slots.append(
            PhysicalCBSlot(
                index=slot_index,
                bytes=capacity_bytes,
                formats=tuple(
                    (logical_index, dtype)
                    for dtype, logical_index in sorted(
                        dtype_to_logical.items(), key=lambda item: item[1]
                    )
                ),
                lifetimes=tuple(lifetimes),
            )
        )

    if next_logical_index > maximum_slots:
        raise ValueError(
            f"resident layer needs {next_logical_index} logical CBs, limit is "
            f"{maximum_slots}"
        )
    l1_bytes = sum(slot.bytes for slot in slots)
    if maximum_l1_bytes is not None and l1_bytes > maximum_l1_bytes:
        raise ValueError(
            f"resident layer CB plan needs {l1_bytes} L1 bytes, limit is "
            f"{maximum_l1_bytes}"
        )
    return PhaseAliasedCBPlan(
        slots=tuple(slots),
        phase_mapping=mapping,
        phase_alias_slots=alias_mapping,
        physical_slot_count=len(slots),
        logical_cb_count=next_logical_index,
        l1_bytes_per_core=l1_bytes,
        phase_unaliased_bytes=unaliased,
    )


def _minimum_cost_injection(costs: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Return a deterministic minimum-cost row-to-column injection.

    This is the rectangular Hungarian algorithm. Keeping it here avoids a
    scipy dependency in the runtime planner while still allowing a 57-slot
    Blackhole plan to be solved in milliseconds.
    """

    row_count = len(costs)
    if row_count == 0:
        return ()
    column_count = len(costs[0])
    if row_count > column_count:
        raise ValueError("assignment needs at least as many slots as requirements")
    if any(len(row) != column_count for row in costs):
        raise ValueError("assignment cost matrix must be rectangular")

    u = [0] * (row_count + 1)
    v = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)
    infinity = 1 << 62
    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum = [infinity] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate - 1] - u[active_row] - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = previous_column[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment):
        raise RuntimeError("incomplete CB assignment")
    return tuple(assignment)


def plan_constrained_phase_aliased_cbs(
    phases: Mapping[str, Sequence[PhaseCBRequirement]],
    *,
    maximum_slots: int = 64,
    maximum_l1_bytes: int | None = None,
) -> PhaseAliasedCBPlan:
    """Pack non-overlapping phases under both L1 and logical-CB limits.

    Size-rank pairing is optimal when formats do not matter. A Transformer
    layer is different: its 57-CB attention phase leaves only seven logical
    IDs for cross-format aliases. This planner anchors the largest phase and
    uses deterministic minimum-cost matching for each later phase. It sweeps
    the cost of introducing a second format and phase order, then selects the
    smallest plan satisfying the hardware's logical-CB limit.
    """

    if not phases:
        raise ValueError("at least one CB phase is required")
    normalized: dict[str, tuple[PhaseCBRequirement, ...]] = {}
    unaliased: dict[str, int] = {}
    for phase, requirements in phases.items():
        if not phase:
            raise ValueError("phase names must be non-empty")
        items = tuple(requirements)
        names = [item.name for item in items]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate CB requirement in phase {phase}")
        normalized[phase] = tuple(
            sorted(
                items,
                key=lambda item: (
                    -(item.tiles * _TILE_BYTES[item.dtype]),
                    item.dtype,
                    item.name,
                ),
            )
        )
        unaliased[phase] = sum(item.tiles * _TILE_BYTES[item.dtype] for item in items)

    anchor = min(
        normalized,
        key=lambda phase: (-len(normalized[phase]), phase),
    )
    remaining = tuple(phase for phase in normalized if phase != anchor)
    # Zero explores the pure L1 optimum; larger values progressively reserve
    # scarce logical IDs. Values are bytes so they remain shape-independent.
    format_penalties = (
        0,
        2_048,
        8_192,
        32_768,
        131_072,
        524_288,
        2_097_152,
        8_388_608,
    )
    best: (
        tuple[
            tuple[int, int, tuple[str, ...]],
            list[dict[str, object]],
            dict[str, dict[str, int]],
        ]
        | None
    ) = None
    best_unconstrained_logical = 1 << 30

    for phase_order in permutations(remaining):
        for format_penalty in format_penalties:
            slots: list[dict[str, object]] = []
            phase_slot_mapping: dict[str, dict[str, int]] = {
                phase: {} for phase in normalized
            }
            for requirement in normalized[anchor]:
                slot_index = len(slots)
                size = requirement.tiles * _TILE_BYTES[requirement.dtype]
                slots.append(
                    {
                        "bytes": size,
                        "formats": {requirement.dtype},
                        "lifetimes": [(anchor, requirement)],
                    }
                )
                phase_slot_mapping[anchor][requirement.name] = slot_index

            signature_mapping: dict[
                tuple[tuple[str, str, int], ...], dict[str, int]
            ] = {
                tuple(
                    (item.name, item.dtype, item.tiles) for item in normalized[anchor]
                ): dict(phase_slot_mapping[anchor])
            }
            for phase in phase_order:
                requirements = normalized[phase]
                signature = tuple(
                    (item.name, item.dtype, item.tiles) for item in requirements
                )
                reused = signature_mapping.get(signature)
                if reused is not None:
                    phase_slot_mapping[phase] = dict(reused)
                    by_name = {item.name: item for item in requirements}
                    for name, slot_index in reused.items():
                        slots[slot_index]["lifetimes"].append((phase, by_name[name]))
                    continue

                costs: list[list[int]] = []
                for requirement in requirements:
                    requirement_bytes = (
                        requirement.tiles * _TILE_BYTES[requirement.dtype]
                    )
                    row = []
                    for slot in slots:
                        formats = set(slot["formats"])
                        new_formats = formats | {requirement.dtype}
                        alignment = lcm(*(_TILE_BYTES[dtype] for dtype in new_formats))
                        raw_capacity = max(int(slot["bytes"]), requirement_bytes)
                        capacity = (
                            (raw_capacity + alignment - 1) // alignment
                        ) * alignment
                        row.append(
                            capacity
                            - int(slot["bytes"])
                            + (
                                format_penalty
                                if requirement.dtype not in formats
                                else 0
                            )
                        )
                    costs.append(row)
                assignment = _minimum_cost_injection(costs)
                for requirement, slot_index in zip(requirements, assignment):
                    slot = slots[slot_index]
                    formats = set(slot["formats"]) | {requirement.dtype}
                    alignment = lcm(*(_TILE_BYTES[dtype] for dtype in formats))
                    requirement_bytes = (
                        requirement.tiles * _TILE_BYTES[requirement.dtype]
                    )
                    raw_capacity = max(int(slot["bytes"]), requirement_bytes)
                    slot["bytes"] = (
                        (raw_capacity + alignment - 1) // alignment
                    ) * alignment
                    slot["formats"] = formats
                    slot["lifetimes"].append((phase, requirement))
                    phase_slot_mapping[phase][requirement.name] = slot_index
                signature_mapping[signature] = dict(phase_slot_mapping[phase])

            logical_count = sum(len(set(slot["formats"])) for slot in slots)
            best_unconstrained_logical = min(best_unconstrained_logical, logical_count)
            if logical_count > maximum_slots:
                continue
            l1_bytes = sum(int(slot["bytes"]) for slot in slots)
            key = (l1_bytes, logical_count, phase_order)
            if best is None or key < best[0]:
                best = (
                    key,
                    [
                        {
                            "bytes": int(slot["bytes"]),
                            "formats": set(slot["formats"]),
                            "lifetimes": list(slot["lifetimes"]),
                        }
                        for slot in slots
                    ],
                    phase_slot_mapping,
                )

    if best is None:
        raise ValueError(
            "resident layer cannot satisfy the logical CB limit; best "
            f"candidate needs {best_unconstrained_logical}, limit is "
            f"{maximum_slots}"
        )
    _, selected_slots, selected_phase_slots = best
    l1_bytes = sum(int(slot["bytes"]) for slot in selected_slots)
    if maximum_l1_bytes is not None and l1_bytes > maximum_l1_bytes:
        raise ValueError(
            f"resident layer CB plan needs {l1_bytes} L1 bytes, limit is "
            f"{maximum_l1_bytes}"
        )

    logical_mapping: dict[str, dict[str, int]] = {phase: {} for phase in normalized}
    alias_mapping: dict[str, dict[str, int]] = {phase: {} for phase in normalized}
    physical_slots: list[PhysicalCBSlot] = []
    next_logical = 0
    for slot_index, slot in enumerate(selected_slots):
        dtype_to_logical = {}
        for dtype in sorted(set(slot["formats"])):
            dtype_to_logical[dtype] = next_logical
            next_logical += 1
        lifetimes = []
        for phase, requirement in slot["lifetimes"]:
            logical = dtype_to_logical[requirement.dtype]
            logical_mapping[phase][requirement.name] = logical
            alias_mapping[phase][requirement.name] = slot_index
            lifetimes.append(
                (
                    phase,
                    requirement.name,
                    requirement.dtype,
                    requirement.tiles,
                    logical,
                )
            )
        physical_slots.append(
            PhysicalCBSlot(
                index=slot_index,
                bytes=int(slot["bytes"]),
                formats=tuple(
                    (logical, dtype)
                    for dtype, logical in sorted(
                        dtype_to_logical.items(), key=lambda item: item[1]
                    )
                ),
                lifetimes=tuple(lifetimes),
            )
        )

    # The chosen phase-to-slot map is reconstructed from lifetimes above;
    # assert it agrees with the optimizer's selected injection.
    assert alias_mapping == selected_phase_slots
    return PhaseAliasedCBPlan(
        slots=tuple(physical_slots),
        phase_mapping=logical_mapping,
        phase_alias_slots=alias_mapping,
        physical_slot_count=len(physical_slots),
        logical_cb_count=next_logical,
        l1_bytes_per_core=l1_bytes,
        phase_unaliased_bytes=unaliased,
    )


def compact_rmsnorm_backward_requirements(
    hidden_tiles: int,
) -> tuple[PhaseCBRequirement, ...]:
    """Physical requirements of the silicon-qualified compact exact path."""

    if hidden_tiles <= 0:
        raise ValueError("hidden tile count must be positive")
    bf16 = {
        "x": hidden_tiles,
        "dy": hidden_tiles,
        "gamma": hidden_tiles,
        "weighted": hidden_tiles,
        "output": hidden_tiles,
        # These one-tile slots include row broadcast and lifecycle scratch.
        "row_broadcast": 1,
        "lifecycle_scratch_0": 1,
        "lifecycle_scratch_1": 1,
        "lifecycle_scratch_2": 1,
        "lifecycle_scratch_3": 1,
        "lifecycle_scratch_4": 1,
        "lifecycle_scratch_5": 1,
    }
    fp32 = {
        "rstd": hidden_tiles,
        "product": hidden_tiles,
        "work_a": hidden_tiles,
        "work_b": hidden_tiles,
        "scaler": 1,
        "correlation": 1,
        "coefficient": 1,
        "direct_window": 2,
        "correction_window": 2,
    }
    return tuple(
        PhaseCBRequirement(name, "bf16", tiles) for name, tiles in bf16.items()
    ) + tuple(PhaseCBRequirement(name, "fp32", tiles) for name, tiles in fp32.items())
