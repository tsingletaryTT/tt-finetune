"""One-call exact RMSNorm/residual backward chain with private L1 handoffs.

This is the numerical/control skeleton surrounding the accepted MLP and
attention backward owners.  It qualifies four ordered compact programs under
one PJRT boundary before those two large owners are inserted between them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_exact_residual_add import build_tt_lang_exact_residual_add
from .tt_lang_program_composition import (
    compile_serialized_child,
    remap_serialized_child,
)
from .tt_lang_owner_aligned_copy import build_tt_lang_owner_aligned_copy
from .tt_lang_rmsnorm_backward import build_tt_lang_rmsnorm_backward


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_resident_norm_residual_backward_chain(
    *, rows: int, columns: int, stage: str = "final",
    private_memory: str = "l1",
) -> Callable:
    if stage not in {"post_rms", "private_copy", "copy_add", "residual_add", "input_rms", "final"}:
        raise ValueError(f"unsupported norm/residual chain stage: {stage!r}")
    if private_memory not in {"l1", "clone"}:
        raise ValueError("private_memory must be 'l1' or 'clone'")
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident norm/residual chain requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    rms = build_tt_lang_rmsnorm_backward(
        sequence_length=rows, hidden_size=columns, compact_l1=True
    )
    add = build_tt_lang_exact_residual_add(
        rows=rows, columns=columns, owner_aligned=True
    )
    copy = build_tt_lang_owner_aligned_copy(rows=rows, columns=columns)

    @ttl.operation(grid="full")
    def _probe(
        post_x, mlp_dx, post_weight, post_rstd, seed,
        residual_gradient, input_x, input_weight, input_rstd, output,
    ):
        source = ttl.make_dataflow_buffer_like(mlp_dx, shape=(1, 1), block_count=1)
        destination = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(mlp_dx[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with destination.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with destination.wait() as tile:
                tx = ttl.copy(tile, output[0:1, 0:1])
                tx.wait()

    @tt_lang_operation(
        operation_id=f"tt_finetune.resident_norm_residual_chain.{rows}x{columns}.{stage}.v1",
        arg_roles=("in",) * 9 + ("out",),
        version_tag=f"resident-norm-residual-chain-{rows}x{columns}-v1",
    )
    def operation(
        post_x, mlp_dx, post_weight, post_rstd, seed,
        residual_gradient, input_x, input_weight, input_rstd, output,
    ):
        # Compile accepted children first; the outer probe must compile last so
        # the TT-Lang resolve driver returns the object carrying the sequence.
        post_rms = compile_serialized_child(
            rms, (post_x, mlp_dx, post_weight, post_rstd, seed, output)
        )
        residual_add = compile_serialized_child(
            add, (mlp_dx, residual_gradient, output)
        )
        private_copy = compile_serialized_child(copy, (mlp_dx, output))
        input_rms = compile_serialized_child(
            rms, (input_x, residual_gradient, input_weight, input_rstd, seed, output)
        )
        final_add = compile_serialized_child(add, (input_x, residual_gradient, output))

        # Public IDs 0..9; compiler-owned BF16 hidden states 10..12.
        post_output = 9 if stage == "post_rms" else 10
        children = [remap_serialized_child(
            post_rms, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: post_output}
        )]
        if stage == "private_copy":
            children.append(remap_serialized_child(private_copy, {0: 10, 1: 9}))
        if stage == "copy_add":
            children.append(remap_serialized_child(private_copy, {0: 10, 1: 11}))
            children.append(remap_serialized_child(
                residual_add, {0: 11, 1: 5, 2: 9}
            ))
        if stage in {"residual_add", "input_rms", "final"}:
            # TT-Metal starts the next compact program per RISC.  A raw
            # owner-aligned copy is the exact BF16 epoch transition between
            # RMSNorm's FP32 compute state and the residual binary engine;
            # both tensors remain compiler-owned in L1.
            children = [remap_serialized_child(
                post_rms, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 10}
            )]
            children.append(remap_serialized_child(
                private_copy, {0: 10, 1: 11}
            ))
            residual_output = 9 if stage == "residual_add" else 12
            children.append(remap_serialized_child(
                residual_add, {0: 11, 1: 5, 2: residual_output}
            ))
        if stage in {"input_rms", "final"}:
            children.append(remap_serialized_child(
                private_copy, {0: 12, 1: 13}
            ))
            if stage == "input_rms":
                children.append(remap_serialized_child(
                    input_rms, {0: 6, 1: 13, 2: 7, 3: 8, 4: 4, 5: 9}
                ))
            else:
                children.append(remap_serialized_child(
                    input_rms, {0: 6, 1: 13, 2: 7, 3: 8, 4: 4, 5: 14}
                ))
                children.append(remap_serialized_child(
                    private_copy, {0: 14, 1: 15}
                ))
                children.append(remap_serialized_child(
                    final_add, {0: 15, 1: 5, 2: 9}
                ))

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(
            post_x, mlp_dx, post_weight, post_rstd, seed,
            residual_gradient, input_x, input_weight, input_rstd, output,
        )
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                private_count = {
                    "post_rms": 0,
                    "private_copy": 1, "copy_add": 2, "residual_add": 2,
                    "input_rms": 4, "final": 6,
                }[stage]
                compiled.internal_tensors = [
                    {"source_operand": 1, "memory": "l1"}
                    for _ in range(private_count)
                ]
                for spec in compiled.internal_tensors:
                    spec["memory"] = private_memory
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = {
        "post_rms": 1, "private_copy": 2, "copy_add": 3, "residual_add": 3,
        "input_rms": 5, "final": 7
    }[stage]
    operation.compiler_owned_l1_tensors = {
        "post_rms": 0, "private_copy": 1, "copy_add": 2, "residual_add": 2,
        "input_rms": 4, "final": 6,
    }[stage]
    operation.stage = stage
    operation.private_memory = private_memory
    operation.safe_default = False
    return operation
