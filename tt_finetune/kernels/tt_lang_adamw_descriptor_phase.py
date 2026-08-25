"""Single-phase executor for qualified AdamW descriptors.

This is a diagnostic and qualification surface, not a trainer backend.  It
lets each pre-serialized physical child execute through the normal PJRT custom
call ABI without artifact-v4 intermediates, so a device hang can be assigned
to either a child program or sequence-owned tensor lifetime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .adamw_descriptor_bundle import (
    PHASE_ORDER,
    load_exact_adamw_descriptor_bundle,
)
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import remap_serialized_child


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=32)
def build_tt_lang_adamw_descriptor_phase(
    *, descriptor_bundle: str, phase: str
) -> Callable:
    """Return one format-3 custom call executing exactly one descriptor."""

    if phase not in PHASE_ORDER:
        raise ValueError(f"unknown AdamW descriptor phase: {phase!r}")
    bundle = load_exact_adamw_descriptor_bundle(descriptor_bundle)

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime only
            raise RuntimeError(
                "AdamW descriptor phase executor requires TT-XLA"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=(1, 1))
    def _probe(input0, input1, input2, output):
        # The probe is compiled before its physical program is replaced, so
        # its dummy copy must still be type-correct.  Denominator/epsilon
        # produce F32; moments/parameter produce BF16.
        source_input = input1 if phase in {"denominator", "epsilon"} else input0
        source = ttl.make_dataflow_buffer_like(
            source_input, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(source_input[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as result:
                    result.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as tile:
                transaction = ttl.copy(tile, output[0:1, 0:1])
                transaction.wait()

    local_to_outer = {
        "moments": {0: 0, 1: 1, 2: 3},
        "denominator": {0: 0, 1: 1, 2: 3},
        "epsilon": {0: 0, 1: 1, 2: 3},
        "parameter": {0: 0, 1: 1, 2: 2, 3: 3},
    }[phase]
    child = remap_serialized_child(
        bundle.descriptors[phase], local_to_outer
    )
    variant = f"{phase}.r{bundle.rows}.{bundle.hashes[phase][:12]}.v1"

    @tt_lang_operation(
        operation_id=f"tt_finetune.adamw_descriptor_phase.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"adamw-descriptor-phase-{variant}",
    )
    def operation(input0, input1, input2, output):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(input0, input1, input2, output)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.serialized_program_override = child
            break
        return result

    operation.artifact_format = 3
    operation.physical_programs = 1
    operation.phase = phase
    operation.rows = bundle.rows
    operation.gradient_page_offset = bundle.gradient_prefix_pages
    operation.gradient_pages = bundle.gradient_pages
    operation.descriptor_bundle = str(Path(descriptor_bundle))
    operation.descriptor_hash = bundle.hashes[phase]
    operation.safe_default = False
    operation.experimental_backend = "adamw_descriptor_phase_diagnostic_v1"
    operation._ttl_program = _probe
    return operation
