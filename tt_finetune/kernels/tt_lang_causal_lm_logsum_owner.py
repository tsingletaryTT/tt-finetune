"""One public exact SUM->LOG owner used to qualify the private handoff."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_causal_lm_log import build_tt_lang_causal_lm_log
from .tt_lang_causal_lm_logsum import build_tt_lang_causal_lm_logsum
from .tt_lang_program_composition import (
    compile_serialized_child,
    remap_serialized_child,
    serialized_child_tensor_ids,
)

ttl = None
tt_lang_operation = None


def _remap(program: dict, mapping: dict[int, int]) -> dict:
    return remap_serialized_child(
        program,
        {index: mapping[index] for index in serialized_child_tensor_ids(program)},
    )


@lru_cache(maxsize=8)
def build_tt_lang_causal_lm_logsum_owner(
    *, tokens: int, vocabulary_size: int, private_memory: str = "clone",
) -> Callable:
    if private_memory not in {"clone", "l1"}:
        raise ValueError("private_memory must be clone or l1")
    reduction = build_tt_lang_causal_lm_logsum(
        tokens=tokens,
        vocabulary_size=vocabulary_size,
        diagnostic_stage="sum",
    )
    logarithm = build_tt_lang_causal_lm_log(tokens=tokens)

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        import ttl as ttl_module
        from tt_torch.tt_lang import tt_lang_operation as decorator
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=(1, 1))
    def _probe(exponentials, row_template, output):
        source = ttl.make_dataflow_buffer_like(exponentials, shape=(1, 1))
        target = ttl.make_dataflow_buffer_like(output, shape=(1, 1))

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(exponentials[0:1, 0:1], tile)
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

    variant = f"t{tokens}.v{vocabulary_size}.{private_memory}.v1"

    @tt_lang_operation(
        operation_id=f"tt_finetune.causal_lm_logsum_owner.{variant}",
        arg_roles=("in", "in", "out"),
        version_tag=f"causal-lm-logsum-owner-{variant}",
    )
    def operation(exponentials, row_template, output):
        sum_child = compile_serialized_child(
            reduction, (exponentials, row_template)
        )
        log_child = compile_serialized_child(logarithm, (row_template, output))
        children = [
            _remap(sum_child, {0: 0, 1: 3}),
            _remap(log_child, {0: 3, 1: 2}),
        ]
        _clear_ttl_compile_wrapper_cache(_probe)
        result = _probe(exponentials, row_template, output)
        closure = _probe.__closure__ or ()
        for name, cell in zip(_probe.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                compiled.program_sequence = children
                compiled.internal_tensors = [
                    {"source_operand": 1, "memory": private_memory}
                ]
            break
        return result

    operation.artifact_format = 4
    operation.physical_programs = 2
    operation.public_custom_calls = 1
    operation.compiler_owned_internal_tensors = 1
    operation._ttl_program = _probe
    operation.safe_default = False
    return operation


__all__ = ["build_tt_lang_causal_lm_logsum_owner"]
