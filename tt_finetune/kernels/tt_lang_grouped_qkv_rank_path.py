"""Canonical grouped Q/V/K LoRA rank matmul for SmolLM2 attention."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_program_composition import compiled_ttl_program


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=8)
def build_tt_lang_grouped_qkv_rank_path(
    *,
    tokens: int,
    hidden_features: int,
    logical_rank: int,
) -> Callable:
    """Build the untouched ``X @ cat(QA, VA, KA).T`` TTNN program.

    The untouched graph asks TTNN to auto-select with an all-DRAM result.  An
    L1 private result otherwise selects a different one-K reduction tree, so
    reproduce the captured two-K all-DRAM schedule explicitly.
    """

    logical_width = 3 * logical_rank
    if tokens <= 0 or tokens % 32:
        raise ValueError("grouped Q/V/K rank tokens must be tile aligned")
    if hidden_features <= 0 or hidden_features % 32:
        raise ValueError("grouped Q/V/K hidden width must be tile aligned")
    if not 1 <= logical_rank or logical_width > 32:
        raise ValueError("Smol grouped Q/V/K rank must fit one physical tile")

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("grouped Q/V/K rank path requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid="full")
    def _probe(hidden, grouped_a, output):
        source = ttl.make_dataflow_buffer_like(
            hidden, shape=(1, 1), block_count=1
        )
        result = ttl.make_dataflow_buffer_like(
            output, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(hidden[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with result.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with result.wait() as tile:
                transaction = ttl.copy(tile, output[0:1, 0:1])
                transaction.wait()

    variant = f"t{tokens}.h{hidden_features}.r{logical_rank}.qvk.v2"

    @tt_lang_operation(
        operation_id=f"tt_finetune.grouped_qkv_rank_path.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"grouped-qkv-rank-path-{variant}",
    )
    def operation(hidden, grouped_a, output):
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(hidden, grouped_a, output)
        compiled = compiled_ttl_program(_probe)
        compiled.program_sequence = [
            {
                "operation": "ttnn.matmul",
                "operand_indices": [0, 1, 2],
                "transpose_a": False,
                "transpose_b": True,
                "program_config": {
                    "grid": [11, 10],
                    "in0_block_w": 2,
                    "out_subblock_h": 1,
                    "out_subblock_w": 1,
                    "out_block_h": 1,
                    "out_block_w": 1,
                    "per_core_m": 1,
                    "per_core_n": 1,
                    "fuse_batch": False,
                    "mcast_in0": False,
                    "gather_in0": False,
                    "untilize_out": False,
                },
                "compute_config": {
                    "math_fidelity": "HiFi4",
                    "math_approx_mode": True,
                    "fp32_dest_acc_en": True,
                    "packer_l1_acc": False,
                    "dst_full_sync_en": False,
                },
            }
        ]
        return result

    operation.artifact_format = 4
    operation.physical_programs = 1
    operation.public_custom_calls = 1
    operation.output_shape = (tokens, logical_width)
    operation.safe_default = False
    operation.experimental_backend = "grouped_qkv_rank_path_v2"
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_grouped_qkv_rank_path"]
