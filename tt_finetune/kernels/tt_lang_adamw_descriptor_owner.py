"""One public AdamW call backed by pre-serialized physical descriptors."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .adamw_descriptor_bundle import load_exact_adamw_descriptor_bundle
from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_adamw_descriptor_owner(
    *, descriptor_bundle: str
) -> Callable:
    """Build four ordered AdamW children without resolving them in PJRT."""

    bundle = load_exact_adamw_descriptor_bundle(descriptor_bundle)
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError(
                "AdamW descriptor owner requires the TT-XLA environment"
            ) from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=(1, 1))
    def _probe(
        state_in,
        gradient_in,
        decay_factor_in,
        denominator_scale_in,
        scaled_epsilon_in,
        moments_template_in,
        denominator_template_in,
        state_out,
    ):
        source = ttl.make_dataflow_buffer_like(
            state_in, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            state_out, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(state_in[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as output:
                    output.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as output:
                tx = ttl.copy(output, state_out[0:1, 0:1])
                tx.wait()

    digest = "-".join(bundle.hashes[name][:8] for name in bundle.hashes)
    variant = (
        f"r{bundle.rows}.off{bundle.gradient_prefix_pages}."
        f"pages{bundle.gradient_pages}.g{bundle.granularity}.{digest}.v1"
    )

    @tt_lang_operation(
        operation_id=f"tt_finetune.adamw_descriptor_owner.{variant}",
        arg_roles=("in",) * 7 + ("out",),
        version_tag=f"adamw-descriptor-owner-{variant}",
    )
    def operation(
        state_in,
        gradient_in,
        decay_factor_in,
        denominator_scale_in,
        scaled_epsilon_in,
        moments_template_in,
        denominator_template_in,
        state_out,
    ):
        tensors = (
            state_in,
            gradient_in,
            decay_factor_in,
            denominator_scale_in,
            scaled_epsilon_in,
            moments_template_in,
            denominator_template_in,
            state_out,
        )
        # External IDs 0..7, compiler-owned IDs 8..10.
        children = bundle.remap({
            "state_in": 0,
            "gradient": 1,
            "decay_factor": 2,
            "denominator_scale": 3,
            "scaled_epsilon": 4,
            "moments": 8,
            "denominator_raw": 9,
            "denominator": 10,
            "state_out": 7,
        })
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(*tensors)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                compiled.internal_tensors = [
                    {"source_operand": 5, "memory": "clone"},
                    {"source_operand": 6, "memory": "clone"},
                    {"source_operand": 6, "memory": "clone"},
                ]
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = 4
    operation.compiler_owned_internal_tensors = 3
    operation.compiler_owned_l1_tensors = 0
    operation.rows = bundle.rows
    operation.gradient_page_offset = bundle.gradient_prefix_pages
    operation.gradient_pages = bundle.gradient_pages
    operation.granularity = bundle.granularity
    operation.descriptor_bundle = str(Path(descriptor_bundle))
    operation.descriptor_hashes = dict(bundle.hashes)
    operation.safe_default = False
    operation.experimental_backend = "adamw_descriptor_owner_v1"
    operation._ttl_program = _probe
    return operation
