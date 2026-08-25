"""Source-safe composition of accepted TT-Metal phase kernels.

TT-Metal permits only one kernel per processor class on an overlapping core
range. A resident layer therefore cannot concatenate programs; it must place
the already-qualified phase bodies behind one `kernel_main`. This module does
the mechanical part without changing arithmetic: remap CB compile arguments,
operation tensor indices and per-kernel common arguments, isolate symbols in a
phase namespace, and emit an explicit ordered wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ResidentKernelPhase:
    name: str
    source: str
    cb_arguments: Mapping[int, int]
    tensor_indices: Mapping[int, int]
    common_arguments: Mapping[int, int]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError(f"invalid C++ phase name: {self.name!r}")
        if self.source.count("void kernel_main()") != 1:
            raise ValueError(f"phase {self.name} must contain exactly one kernel_main")


def _replace_numeric_calls(
    source: str,
    *,
    function: str,
    mapping: Mapping[int, int],
    phase: str,
) -> str:
    pattern = re.compile(rf"{re.escape(function)}\((\d+)\)")

    def replace(match: re.Match[str]) -> str:
        local = int(match.group(1))
        if local not in mapping:
            raise ValueError(f"phase {phase} has no mapping for {function}({local})")
        return f"{function}({mapping[local]})"

    return pattern.sub(replace, source)


def _remap_phase_source(
    phase: ResidentKernelPhase,
    *,
    total_cbs: int,
) -> tuple[tuple[str, ...], str]:
    source = phase.source

    compile_arg = re.compile(r"get_compile_time_arg_val\((\d+)\)")

    def replace_compile_arg(match: re.Match[str]) -> str:
        local = int(match.group(1))
        if local not in phase.cb_arguments:
            raise ValueError(
                f"phase {phase.name} has no CB mapping for argument {local}"
            )
        return f"get_compile_time_arg_val({phase.cb_arguments[local]})"

    source = compile_arg.sub(replace_compile_arg, source)
    source = _replace_numeric_calls(
        source,
        function="get_common_arg_val<uint32_t>",
        mapping=phase.common_arguments,
        phase=phase.name,
    )

    accessor = re.compile(
        r"get_tensor_accessor_args_cta_offset<\s*(\d+),\s*(\d+)>\(\)"
        r"(?:\s*,\s*(\d+)(>\(\)))?"
    )

    def replace_accessor(match: re.Match[str]) -> str:
        local_tensor = int(match.group(1))
        if local_tensor not in phase.tensor_indices:
            raise ValueError(
                f"phase {phase.name} has no operation tensor mapping for "
                f"tensor {local_tensor}"
            )
        result = (
            "get_tensor_accessor_args_cta_offset<"
            f"{phase.tensor_indices[local_tensor]}, {total_cbs}>()"
        )
        local_common = match.group(3)
        if local_common is not None:
            local_position = int(local_common)
            if local_position not in phase.common_arguments:
                raise ValueError(
                    f"phase {phase.name} has no common-argument mapping for "
                    f"TensorAccessorArgs tensor {local_position}"
                )
            result += f", {phase.common_arguments[local_position]}{match.group(4)}"
        return result

    source = accessor.sub(replace_accessor, source)
    # A resident owner deliberately combines several already-qualified phase
    # bodies.  Letting the JIT inline them into one enormous kernel_main makes
    # Blackhole compilation super-linear (the RMS->attention owner occupied a
    # host compiler core for more than ten minutes).  A Tensix-local function
    # call is negligible beside a phase containing hundreds of tile ops, and
    # preserves the phase's reduction/rounding body exactly.  The explicit
    # noinline boundary also prevents cross-phase optimization from changing
    # lifecycle-sensitive CB ordering.
    source = source.replace(
        "void kernel_main()",
        "static __attribute__((noinline)) void run()",
        1,
    )

    lines = source.splitlines()
    # Accepted generated kernels can contain a late profiler include after
    # helper declarations. Splitting at the *last* include leaks those early
    # declarations out of the phase namespace and causes helper collisions
    # when two owners are composed. Hoist includes individually and namespace
    # every other source line, including anonymous helper namespaces.
    preamble = tuple(line for line in lines if line.startswith("#include"))
    body = "\n".join(
        line for line in lines if not line.startswith("#include")
    ).strip()
    wrapped = (
        f"namespace resident_layer_phase_{phase.name} {{\n"
        f"{body}\n"
        f"}}  // namespace resident_layer_phase_{phase.name}\n"
    )
    return preamble, wrapped


def compose_resident_kernel_phases(
    phases: Sequence[ResidentKernelPhase],
    *,
    total_cbs: int,
    barriers_after: Mapping[str, str] | None = None,
) -> str:
    """Compose ordered phase bodies behind one physical `kernel_main`.

    `barriers_after` contains literal device-side fence code. The caller owns
    its semaphore generation contract because reader, compute and writer use
    different valid barrier implementations.
    """

    if not phases:
        raise ValueError("at least one resident kernel phase is required")
    # `total_cbs` is the number of serialized descriptors/compile arguments,
    # not the largest hardware CB ID. Disjoint core ranges may each describe
    # the same c0..c63 identities, so a whole-layer program legitimately has
    # more than 64 descriptor entries while every `_cb_index` remains <64.
    if total_cbs <= 0 or total_cbs > 1024:
        raise ValueError("total_cbs must be in the descriptor range 1..1024")
    names = [phase.name for phase in phases]
    if len(names) != len(set(names)):
        raise ValueError("resident kernel phase names must be unique")
    barriers = dict(barriers_after or {})
    unknown_barriers = set(barriers) - set(names)
    if unknown_barriers:
        raise ValueError(
            f"barrier refers to unknown phases: {sorted(unknown_barriers)}"
        )

    includes: list[str] = []
    seen_includes: set[str] = set()
    bodies: list[str] = []
    for phase in phases:
        preamble, body = _remap_phase_source(phase, total_cbs=total_cbs)
        for line in preamble:
            if line.startswith("#include"):
                if line in seen_includes:
                    continue
                seen_includes.add(line)
            includes.append(line)
        bodies.append(body)

    wrapper = ["void kernel_main() {"]
    for phase in phases:
        wrapper.append(f"    resident_layer_phase_{phase.name}::run();")
        barrier = barriers.get(phase.name)
        if barrier:
            wrapper.extend(f"    {line}" for line in barrier.splitlines())
    wrapper.append("}")
    return "\n".join(includes + [""] + bodies + wrapper + [""])
