"""Experimental one-call MLP→attention backward source-composition gate.

This owner is intentionally a qualification step for artifact-v3 aliases,
segmented core placement, numerical descriptor compatibility and the physical
device epoch barrier. It executes the accepted full MLP and full attention
owners in one TT-Metal program while retaining their existing external tensor
ABIs. RMSNorm/residual L1 handoffs replace those external edges in the final
layer owner; this pair is never the safe/default training backend.
"""

from __future__ import annotations

import hashlib
import json
import copy
import ast
from pathlib import Path
from typing import Callable

import torch

from .tt_lang_adamw import _clear_ttl_compile_wrapper_cache

ttl = None
tt_lang_operation = None

_ALL_WORKERS_RANGE = {"start": [0, 0], "end": [10, 9]}
_SEGMENT_ORDER = ("projection_rows", "middle_rows", "edge_column")
_THREAD_ORDER = ("reader", "compute", "writer")


def _bundle_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_tt_lang_resident_layer_backward_pair(
    *,
    source_bundle: str | Path,
    program_sequence: bool = False,
    program_components: tuple[str, ...] = ("mlp", "attention"),
) -> Callable:
    root = Path(source_bundle)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing resident layer source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "accepted_source_abi_gate":
        raise ValueError("resident layer source bundle did not pass its ABI gate")
    if manifest.get("phase_order") != ["mlp", "attention"]:
        raise ValueError("resident layer pair requires MLP then attention phases")
    segments = manifest["segments"]
    if tuple(segments) != _SEGMENT_ORDER:
        raise ValueError("resident layer segment order changed")
    if not program_sequence and program_components != ("mlp", "attention"):
        raise ValueError("program_components is valid only in sequence mode")
    if (
        not program_components
        or any(component not in ("mlp", "attention") for component in program_components)
        or len(set(program_components)) != len(program_components)
    ):
        raise ValueError("program_components must be a unique non-empty MLP/attention tuple")
    if program_sequence:
        if "standalone_programs" not in manifest:
            raise ValueError(
                "program-sequence mode requires accepted standalone programs; "
                "regenerate the resident layer bundle"
            )
        for segment_name in _SEGMENT_ORDER:
            segment = segments[segment_name]
            if "phase_kernel_sources" not in segment:
                raise ValueError(
                    "program-sequence mode requires phase-specific sources; "
                    "regenerate the resident layer bundle"
                )

    global ttl, tt_lang_operation
    if ttl is None or tt_lang_operation is None:
        try:
            import ttl as ttl_module
            from tt_torch.tt_lang import tt_lang_operation as decorator
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("resident layer pair requires TT-XLA") from exc
        ttl = ttl_module
        tt_lang_operation = decorator

    @ttl.operation(
        grid="full",
        fp32_dest_acc_en=True,
        options="--ttl-matmul-full-fp32",
    )
    def _program(
        mlp_down_dy,
        mlp_down_weight,
        mlp_down_b,
        mlp_down_hidden,
        mlp_down_a,
        mlp_product,
        mlp_gate,
        mlp_up,
        mlp_gate_weight,
        mlp_gate_b,
        mlp_gate_hidden,
        mlp_gate_a,
        mlp_up_weight,
        mlp_up_b,
        mlp_up_hidden,
        mlp_up_a,
        mlp_inputs,
        attention_output_dy,
        attention_o_weight,
        attention_o_b,
        attention_o_a,
        attention_cosine,
        attention_sine,
        attention_hidden,
        attention_q_weight,
        attention_q_b,
        attention_q_a,
        attention_k_weight,
        attention_k_b,
        attention_k_a,
        attention_v_weight,
        attention_v_b,
        attention_v_a,
        attention_checkpoint_arena,
        mlp_input_gradient,
        mlp_gate_a_gradient,
        mlp_gate_b_gradient,
        mlp_up_a_gradient,
        mlp_up_b_gradient,
        mlp_down_a_gradient,
        mlp_down_b_gradient,
        attention_output_arena,
    ):
        source = ttl.make_dataflow_buffer_like(mlp_down_dy, shape=(1, 1), block_count=1)
        output = ttl.make_dataflow_buffer_like(
            mlp_input_gradient, shape=(1, 1), block_count=1
        )

        @ttl.datamovement()
        def read():
            with source.reserve() as tile:
                tx = ttl.copy(mlp_down_dy[0:1, 0:1], tile)
                tx.wait()

        @ttl.compute()
        def compute():
            with source.wait() as tile:
                with output.reserve() as target:
                    target.store(tile)

        @ttl.datamovement()
        def write():
            with output.wait() as tile:
                tx = ttl.copy(tile, mlp_input_gradient[0:1, 0:1])
                tx.wait()

    digest = _bundle_digest(root)
    roles = ("in",) * 34 + ("out",) * 8

    sequence_suffix = "-".join(program_components)
    mode = (
        f"sequence-v4-{sequence_suffix}"
        if program_sequence
        else "monolithic-v3"
    )

    @tt_lang_operation(
        operation_id=(
            f"tt_finetune.resident_layer_backward_pair.{mode}.{digest}"
        ),
        arg_roles=roles,
        version_tag=f"resident-layer-backward-pair-{mode}-{digest}",
    )
    def operation(
        mlp_down_dy,
        mlp_down_weight,
        mlp_down_b,
        mlp_down_hidden,
        mlp_down_a,
        mlp_product,
        mlp_gate,
        mlp_up,
        mlp_gate_weight,
        mlp_gate_b,
        mlp_gate_hidden,
        mlp_gate_a,
        mlp_up_weight,
        mlp_up_b,
        mlp_up_hidden,
        mlp_up_a,
        mlp_inputs,
        attention_output_dy,
        attention_o_weight,
        attention_o_b,
        attention_o_a,
        attention_cosine,
        attention_sine,
        attention_hidden,
        attention_q_weight,
        attention_q_b,
        attention_q_a,
        attention_k_weight,
        attention_k_b,
        attention_k_a,
        attention_v_weight,
        attention_v_b,
        attention_v_a,
        attention_checkpoint_arena,
        mlp_input_gradient,
        mlp_gate_a_gradient,
        mlp_gate_b_gradient,
        mlp_up_a_gradient,
        mlp_up_b_gradient,
        mlp_down_a_gradient,
        mlp_down_b_gradient,
        attention_output_arena,
    ):
        arguments = (
            mlp_down_dy,
            mlp_down_weight,
            mlp_down_b,
            mlp_down_hidden,
            mlp_down_a,
            mlp_product,
            mlp_gate,
            mlp_up,
            mlp_gate_weight,
            mlp_gate_b,
            mlp_gate_hidden,
            mlp_gate_a,
            mlp_up_weight,
            mlp_up_b,
            mlp_up_hidden,
            mlp_up_a,
            mlp_inputs,
            attention_output_dy,
            attention_o_weight,
            attention_o_b,
            attention_o_a,
            attention_cosine,
            attention_sine,
            attention_hidden,
            attention_q_weight,
            attention_q_b,
            attention_q_a,
            attention_k_weight,
            attention_k_b,
            attention_k_a,
            attention_v_weight,
            attention_v_b,
            attention_v_a,
            attention_checkpoint_arena,
            mlp_input_gradient,
            mlp_gate_a_gradient,
            mlp_gate_b_gradient,
            mlp_up_a_gradient,
            mlp_up_b_gradient,
            mlp_down_a_gradient,
            mlp_down_b_gradient,
            attention_output_arena,
        )
        _clear_ttl_compile_wrapper_cache(_program)
        result = _program(*arguments)
        closure = _program.__closure__ or ()
        for name, cell in zip(_program.__code__.co_freevars, closure):
            if name != "cache" or not isinstance(cell.cell_contents, dict):
                continue
            for compiled in cell.cell_contents.values():
                from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
                import ttnn

                cb_configs = []
                cb_alias_slots = []
                cb_core_ranges = []
                for segment_name in _SEGMENT_ORDER:
                    segment = segments[segment_name]
                    core_range = segment["core_range"]
                    for item in segment["logical_cb_formats"]:
                        config = CompilerAllocatedDFBConfig(
                            dfb_index=int(item["logical_index"]),
                            num_tiles=1,
                            data_format=(
                                "float32" if item["dtype"] == "fp32" else "bfloat16"
                            ),
                            block_count=int(item["tile_capacity"]),
                        )
                        config._cb_index = int(item["logical_index"])
                        cb_configs.append(config)
                        cb_alias_slots.append(int(item["artifact_alias_slot"]))
                        cb_core_ranges.append(core_range)
                if len(cb_configs) != int(manifest["total_cb_descriptors"]):
                    raise RuntimeError("resident layer CB descriptor count changed")
                compiled.cb_configs = cb_configs
                compiled.cb_alias_slots = cb_alias_slots
                compiled.cb_core_ranges = cb_core_ranges

                paths = []
                kernel_tensor_indices = []
                kernel_core_ranges = []
                kernel_configs = []
                kernel_noc_roles = []
                for segment_name in _SEGMENT_ORDER:
                    segment = segments[segment_name]
                    attention_cbs = segment["phase_logical_cb_maps"]["attention"]
                    for thread in _THREAD_ORDER:
                        path = root / segment["kernel_sources"][thread]
                        if not path.exists():
                            raise RuntimeError(f"missing resident source: {path}")
                        paths.append(
                            (str(path), "compute" if thread == "compute" else "noc")
                        )
                        kernel_tensor_indices.append(
                            segment["kernel_tensor_indices"][thread]
                        )
                        kernel_core_ranges.append(segment["core_range"])
                        if thread == "reader":
                            kernel_configs.append(ttnn.ReaderConfigDescriptor())
                            kernel_noc_roles.append("reader")
                        elif thread == "writer":
                            kernel_configs.append(ttnn.WriterConfigDescriptor())
                            kernel_noc_roles.append("writer")
                        else:
                            compute_config = ttnn.ComputeConfigDescriptor(
                                math_fidelity=ttnn.MathFidelity.HiFi4,
                                math_approx_mode=(segment_name != "projection_rows"),
                                fp32_dest_acc_en=True,
                            )
                            while len(compute_config.unpack_to_dest_mode) < 64:
                                compute_config.unpack_to_dest_mode.append(
                                    ttnn.UnpackToDestMode.Default
                                )
                            if segment_name != "projection_rows":
                                for local_id in (0, 1, 13, 14, 15):
                                    logical = int(attention_cbs[f"cb{local_id}"])
                                    compute_config.unpack_to_dest_mode[logical] = (
                                        ttnn.UnpackToDestMode.UnpackToDestFp32
                                    )
                            kernel_configs.append(compute_config)
                            kernel_noc_roles.append(None)
                compiled.kernel_paths = paths
                compiled.kernel_tensor_indices = kernel_tensor_indices
                compiled.kernel_core_ranges = kernel_core_ranges
                compiled.kernel_configs = kernel_configs
                compiled.kernel_noc_roles = kernel_noc_roles
                compiled.semaphore_configs = [
                    {
                        "id": semaphore,
                        "core_type": "worker",
                        "core_range": _ALL_WORKERS_RANGE,
                        "initial_value": 0,
                    }
                    for semaphore in range(16)
                ]
                compiled.num_pipe_sync_semaphores = 16
                compiled.kernel_arg_specs = [[] for _ in paths]
                if program_sequence:
                    children = []
                    for component in program_components:
                        child = copy.copy(compiled)
                        program = manifest["standalone_programs"][component]
                        child_paths = []
                        for filename, thread in zip(
                            program["kernel_sources"], program["kernel_threads"]
                        ):
                            path = root / filename
                            if not path.exists():
                                raise RuntimeError(
                                    f"missing resident child source: {path}"
                                )
                            child_paths.append(
                                (
                                    str(path),
                                    "compute" if thread == "compute" else "noc",
                                )
                            )
                        child_cb_configs = []
                        for item in program["cb_configs"]:
                            # Captured CompilerAllocatedDFBConfig entries already
                            # carry the serialized allocation fields.  Tensor-backed
                            # configs instead carry the logical tile shape, which the
                            # TT-Lang serializer does *not* inspect directly: it uses
                            # num_tiles * block_count.  Preserve either representation
                            # rather than silently collapsing a multi-tile CB to one
                            # tile (the latter deadlocks once the producer fills it).
                            if "tensor" in item:
                                dtype = (
                                    "float32"
                                    if "float32" in item["tensor"]
                                    else "bfloat16"
                                )
                                shape = tuple(ast.literal_eval(item["shape"]))
                                num_tiles = 1
                                for extent in shape:
                                    num_tiles *= int(extent)
                            else:
                                dtype = str(item["data_format"])
                                num_tiles = int(item["num_tiles"])
                                shape = (num_tiles,)
                            config = CompilerAllocatedDFBConfig(
                                dfb_index=int(item["_cb_index"]),
                                num_tiles=num_tiles,
                                data_format=dtype,
                                block_count=int(item["block_count"]),
                            )
                            config.shape = shape
                            config._cb_index = int(item["_cb_index"])
                            child_cb_configs.append(config)
                        child.cb_configs = child_cb_configs
                        child.cb_alias_slots = None
                        # The probe operation is compiled on tt-lang's small
                        # deviceless placeholder grid.  Per-kernel and per-CB
                        # ranges are explicit below, but the program-level
                        # bounding box is still consumed by TT-Metal when it
                        # builds the physical program.  Preserve the accepted
                        # owners' full Blackhole worker grid; otherwise the
                        # attention island's x=8..10 workers are outside the
                        # inherited 8x8 root even though their kernels exist.
                        child.core_ranges = ttnn.CoreRangeSet(
                            [
                                ttnn.CoreRange(
                                    ttnn.CoreCoord(0, 0),
                                    ttnn.CoreCoord(10, 9),
                                )
                            ]
                        )
                        child.cb_core_ranges = program["cb_core_ranges"]
                        child.kernel_paths = child_paths
                        child.kernel_tensor_indices = program[
                            "kernel_tensor_indices"
                        ]
                        child.kernel_core_ranges = program[
                            "kernel_core_ranges"
                        ]
                        child.kernel_noc_roles = [
                            "reader" if thread == "reader" else (
                                "writer" if thread == "writer" else None
                            )
                            for thread in program["kernel_threads"]
                        ]
                        child.kernel_configs = []
                        for kernel_index, thread in enumerate(
                            program["kernel_threads"]
                        ):
                            if thread == "reader":
                                config = ttnn.ReaderConfigDescriptor()
                            elif thread == "writer":
                                config = ttnn.WriterConfigDescriptor()
                            else:
                                projection = kernel_index == 1
                                config = ttnn.ComputeConfigDescriptor(
                                    math_fidelity=ttnn.MathFidelity.HiFi4,
                                    math_approx_mode=(
                                        component == "attention" and not projection
                                    ),
                                    fp32_dest_acc_en=(
                                        component == "attention" or projection
                                    ),
                                )
                                if component == "attention" and not projection:
                                    while len(config.unpack_to_dest_mode) < 64:
                                        config.unpack_to_dest_mode.append(
                                            ttnn.UnpackToDestMode.Default
                                        )
                                    for local_id in (0, 1, 13, 14, 15):
                                        config.unpack_to_dest_mode[local_id] = (
                                            ttnn.UnpackToDestMode.UnpackToDestFp32
                                        )
                            child.kernel_configs.append(config)
                        child.kernel_arg_specs = [[] for _ in child_paths]
                        children.append(child)
                    compiled.program_sequence = children
            break
        return result

    operation.source_bundle = str(root)
    operation.artifact_format = 4 if program_sequence else 3
    operation.experimental_backend = (
        "resident_layer_backward_pair_sequence_v4"
        if program_sequence
        else "resident_layer_backward_pair_v1"
    )
    operation.safe_default = False
    operation._ttl_program = _program
    return operation
