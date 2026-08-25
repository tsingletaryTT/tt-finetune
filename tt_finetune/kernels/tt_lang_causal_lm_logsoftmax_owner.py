"""One public exact max/exp/sum/log owner for the causal-LM edge."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_causal_lm_log import build_tt_lang_causal_lm_log
from .tt_lang_causal_lm_logsoftmax import build_tt_lang_causal_lm_logsoftmax
from .tt_lang_causal_lm_logsum import build_tt_lang_causal_lm_logsum
from .tt_lang_program_composition import (
    compile_serialized_child,
    remap_serialized_child,
    serialized_child_tensor_ids,
)


ttl = None
tt_lang_operation = None


def _remap_used(program: dict, mapping: dict[int, int]) -> dict:
    used = serialized_child_tensor_ids(program)
    return remap_serialized_child(
        program, {tensor: mapping[tensor] for tensor in used}
    )


@lru_cache(maxsize=8)
def build_tt_lang_causal_lm_logsoftmax_owner(
    *, tokens: int, vocabulary_size: int, private_memory: str = "l1",
) -> Callable:
    if private_memory not in {"l1", "clone"}:
        raise ValueError("private_memory must be 'l1' or 'clone'")
    exponential = build_tt_lang_causal_lm_logsoftmax(
        tokens=tokens,
        vocabulary_size=vocabulary_size,
        diagnostic_stage="exponentials",
    )
    reduction = build_tt_lang_causal_lm_logsum(
        tokens=tokens,
        vocabulary_size=vocabulary_size,
        diagnostic_stage="sum",
    )
    logarithm = build_tt_lang_causal_lm_log(tokens=tokens)

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("causal-LM log-softmax owner requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=(1, 1))
    def _probe(logits, exponential_template, row_template, output):
        source = ttl.make_dataflow_buffer_like(logits, shape=(1, 1))
        target = ttl.make_dataflow_buffer_like(output, shape=(1, 1))

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(logits[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as value:
                with target.reserve() as result:
                    result.store(value)

        @ttl.datamovement()
        def write():
            with target.wait() as value:
                tx = ttl.copy(value, output[0:1, 0:1])
                tx.wait()

    variant = f"t{tokens}.v{vocabulary_size}.{private_memory}handoff.v2"

    @tt_lang_operation(
        operation_id=f"tt_finetune.causal_lm_logsoftmax_owner.{variant}",
        arg_roles=("in", "in", "in", "out"),
        version_tag=f"causal-lm-logsoftmax-owner-{variant}",
    )
    def operation(logits, exponential_template, row_template, output):
        exp_child = compile_serialized_child(
            exponential, (logits, exponential_template)
        )
        sum_child = compile_serialized_child(
            reduction, (exponential_template, row_template)
        )
        log_child = compile_serialized_child(
            logarithm, (row_template, output)
        )
        # Public table 0..3; compiler-owned private L1 tensors 4 and 5.
        children = [
            _remap_used(exp_child, {0: 0, 1: 4}),
            _remap_used(sum_child, {0: 4, 1: 5}),
            _remap_used(log_child, {0: 5, 1: 3}),
        ]
        operation._debug_program_sequence = tuple(children)

        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(logits, exponential_template, row_template, output)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                compiled.internal_tensors = [
                    {"source_operand": 1, "memory": private_memory},
                    {"source_operand": 2, "memory": private_memory},
                ]
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = 3
    operation.public_custom_calls = 1
    operation.compiler_owned_internal_tensors = 2
    operation.private_intermediate_memory = private_memory
    operation.safe_default = False
    operation.experimental_backend = "causal_lm_logsoftmax_exact_owner_v1"
    operation._ttl_program = _probe
    return operation


__all__ = ["build_tt_lang_causal_lm_logsoftmax_owner"]
