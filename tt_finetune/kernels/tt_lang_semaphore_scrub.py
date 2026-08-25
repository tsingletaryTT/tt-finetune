"""All-core semaphore scrub program for resident epoch boundaries."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache
from .tt_lang_frozen_linear_lora import BLACKHOLE_P150A_GRID


ttl = None
tt_lang_operation = None


@lru_cache(maxsize=1)
def build_tt_lang_semaphore_scrub() -> Callable:
    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("semaphore scrub requires TT-XLA") from exc
        ttl, tt_lang_operation = ttl_module, decorator

    @ttl.operation(grid=BLACKHOLE_P150A_GRID)
    def _program(input_tensor, output_tensor):
        source = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=1
        )
        target = ttl.make_dataflow_buffer_like(
            output_tensor, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                transaction = ttl.copy(input_tensor[0:1, 0:1], tile)
                transaction.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with target.reserve() as output:
                    output.store(tile)

        @ttl.datamovement()
        def write():
            with target.wait() as output:
                transaction = ttl.copy(output, output_tensor[0:1, 0:1])
                transaction.wait()

    @tt_lang_operation(
        operation_id="tt_finetune.semaphore_scrub.all16.v3",
        arg_roles=("in", "out"),
        version_tag="semaphore-scrub-all16-v3",
    )
    def operation(input_tensor, output_tensor):
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(input_tensor, output_tensor)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                for index, (kernel_path, thread_type) in enumerate(
                    compiled.kernel_paths
                ):
                    if thread_type == "compute" or "read" not in Path(
                        kernel_path
                    ).name:
                        continue
                    path = Path(compiled.kernel_paths[index][0])
                    source = path.read_text(encoding="utf-8")
                    marker = "void kernel_main() {"
                    scrub = r'''
  // Artifact-v4 physical programs reuse fixed L1 semaphore addresses.  A
  // full program boundary guarantees prior kernels are drained; clear every
  // mailbox on every worker before the next resident epoch publishes state.
  for (uint32_t semaphore = 0; semaphore < 16; ++semaphore) {
    volatile tt_l1_ptr uint32_t* flag =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_semaphore(semaphore));
    noc_semaphore_set(flag, 0);
  }
'''
                    if source.count(marker) != 1:
                        raise RuntimeError("semaphore scrub reader ABI changed")
                    source = source.replace(marker, marker + scrub, 1)
                    path.write_text(source, encoding="utf-8")
            break
        return result

    operation.safe_default = False
    operation.experimental_backend = "resident_semaphore_scrub_v2"
    operation._ttl_program = _program
    return operation
