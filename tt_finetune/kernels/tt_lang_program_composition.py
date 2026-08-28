"""Utilities for composing qualified compact TT-Lang programs.

Artifact-v4 owns one public custom-call tensor table while each physical child
keeps its own CB descriptors and kernels.  These helpers capture an accepted
child through its normal compile path and remap only operation tensor IDs; CB
IDs, common-argument positions, arithmetic and reduction order stay unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


_ACCESSOR = re.compile(
    r"get_tensor_accessor_args_cta_offset<\s*(\d+)\s*,\s*"
    r"([A-Za-z_][A-Za-z0-9_]*|\d+)\s*>\(\)"
)


def _serialized_cache_key(
    *, kind: str, entry: Any, arguments: Sequence[Any]
) -> tuple[str, dict[str, Any]]:
    """Return a stable key for an eagerly composed TT-Lang artifact.

    Composition invokes child implementations before the outer StableHLO graph
    reaches PJRT, so PJRT's executable cache cannot avoid this work.  Tensor
    identity is part of the signature: using one symbolic tensor in two operand
    positions aliases their TT-Lang tensor IDs, while two same-shaped tensors
    remain distinct.
    """

    aliases: dict[int, int] = {}
    signature = []
    for argument in arguments:
        identity = id(argument)
        alias = aliases.setdefault(identity, len(aliases))
        shape = tuple(int(dimension) for dimension in argument.shape)
        signature.append(
            {
                "shape": list(shape),
                "dtype": str(argument.dtype),
                "alias": alias,
            }
        )
    payload = {
        "schema": 1,
        "kind": kind,
        "operation_id": entry.operation_id,
        "version_tag": entry.version_tag,
        "arg_roles": list(entry.arg_roles),
        "arguments": signature,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _load_serialized_cache(
    *, kind: str, entry: Any, arguments: Sequence[Any]
) -> tuple[Path | None, Any | None, dict[str, Any] | None]:
    root = os.environ.get("TT_FINETUNE_SERIALIZED_PROGRAM_CACHE_DIR")
    if not root:
        return None, None, None
    key, signature = _serialized_cache_key(
        kind=kind, entry=entry, arguments=arguments
    )
    path = Path(root) / f"{key}.json"
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError:
        return path, None, signature
    except (OSError, json.JSONDecodeError) as exc:
        print(f"TT_FINETUNE serialized cache ignored {path}: {exc}")
        return path, None, signature
    if document.get("signature") != signature or document.get("kind") != kind:
        print(f"TT_FINETUNE serialized cache signature mismatch: {path}")
        return path, None, signature
    print(f"TT_FINETUNE serialized program cache hit: {path}")
    return path, document.get("artifact"), signature


def _store_serialized_cache(
    *, path: Path | None, kind: str, signature: dict[str, Any] | None,
    artifact: Any,
) -> None:
    if path is None or signature is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"kind": kind, "signature": signature, "artifact": artifact}
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, sort_keys=True))
    temporary.replace(path)
    print(f"TT_FINETUNE stored serialized program cache entry: {path}")


def compiled_ttl_program(ttl_program: Any) -> Any:
    """Return the sole cached compiled program after its owner was invoked."""
    closure = ttl_program.__closure__ or ()
    for name, cell in zip(ttl_program.__code__.co_freevars, closure):
        if name != "cache" or not isinstance(cell.cell_contents, dict):
            continue
        values = tuple(cell.cell_contents.values())
        if len(values) != 1:
            raise RuntimeError(
                "resident child compilation expected one cached specialization, "
                f"found {len(values)}"
            )
        return values[0]
    raise RuntimeError("resident child compilation cache was not found")


def compile_serialized_child(operation: Any, arguments: Sequence[Any]) -> dict:
    """Compile an operation's private TTL program and return artifact JSON."""
    entry = getattr(operation, "_tt_lang_operation_entry", None)
    ttl_program = getattr(operation, "_ttl_program", None)
    if entry is None or ttl_program is None:
        raise TypeError("resident child must expose operation entry and _ttl_program")
    cache_path, cached, signature = _load_serialized_cache(
        kind="child", entry=entry, arguments=arguments
    )
    if isinstance(cached, dict):
        return copy.deepcopy(cached)
    entry.impl(*arguments)
    compiled = compiled_ttl_program(ttl_program)
    from tt_torch.tt_lang import _serialize_compiled_operation

    artifact = json.loads(
        _serialize_compiled_operation(compiled).decode("utf-8")
    )
    _store_serialized_cache(
        path=cache_path, kind="child", signature=signature, artifact=artifact
    )
    return artifact


def compile_serialized_sequence_children(
    operation: Any, arguments: Sequence[Any]
) -> list[dict]:
    """Compile an artifact-v4 owner and extract its compact physical children."""
    entry = getattr(operation, "_tt_lang_operation_entry", None)
    ttl_program = getattr(operation, "_ttl_program", None)
    if entry is None or ttl_program is None:
        raise TypeError("sequence owner must expose operation entry and _ttl_program")
    cache_path, cached, signature = _load_serialized_cache(
        kind="sequence", entry=entry, arguments=arguments
    )
    if isinstance(cached, list) and cached:
        return copy.deepcopy(cached)
    entry.impl(*arguments)
    compiled = compiled_ttl_program(ttl_program)
    sequence = getattr(compiled, "program_sequence", None)
    if not isinstance(sequence, (list, tuple)) or not sequence:
        raise RuntimeError("compiled owner did not produce physical child programs")
    from tt_torch.tt_lang import _serialize_compiled_operation

    children = []
    for index, child in enumerate(sequence):
        if isinstance(child, dict):
            serialized = copy.deepcopy(child)
            # Artifact-v4 can also carry a native TTNN operation.  Unlike a
            # generic TT-Lang child it deliberately has no format_version or
            # kernel table: TTNNLowerTTLangToGeneric lowers this descriptor to
            # a real ttnn.matmul op so the ordinary TTNN program factory owns
            # its complete numerical schedule.
            if serialized.get("operation") in {
                "ttnn.matmul", "ttnn.linear", "ttnn.add", "ttnn.multiply", "ttnn.full",
                "ttnn.typecast", "ttnn.softmax", "ttnn.silu", "ttnn.permute",
            }:
                children.append(serialized)
                continue
            if serialized.get("format_version") not in (1, 2, 3):
                raise RuntimeError(
                    "sequence child "
                    f"{index} has unsupported serialized format "
                    f"{serialized.get('format_version')!r}"
                )
            if "programs" in serialized:
                raise RuntimeError(
                    f"sequence child {index} is itself a nested sequence"
                )
        else:
            serialized = json.loads(
                _serialize_compiled_operation(child).decode("utf-8")
            )
        children.append(serialized)
    _store_serialized_cache(
        path=cache_path,
        kind="sequence",
        signature=signature,
        artifact=children,
    )
    return children


def remap_serialized_child(program: dict, tensor_map: Mapping[int, int]) -> dict:
    """Remap local tensor IDs into an outer external+private tensor table."""
    import copy

    result = copy.deepcopy(program)
    if "programs" in result:
        raise ValueError("nested program sequences cannot be remapped as children")
    mapping = {int(local): int(outer) for local, outer in tensor_map.items()}
    if result.get("operation") in {
        "ttnn.matmul", "ttnn.linear", "ttnn.add", "ttnn.multiply", "ttnn.full",
        "ttnn.typecast", "ttnn.softmax", "ttnn.silu", "ttnn.permute",
    }:
        operands = result.get("operand_indices")
        if not isinstance(operands, list) or not operands:
            raise ValueError("native sequence child requires operand_indices")
        missing = {int(value) for value in operands} - set(mapping)
        if missing:
            raise ValueError(
                f"no outer tensor mapping for native tensors {sorted(missing)}"
            )
        result["operand_indices"] = [
            mapping[int(value)] for value in operands
        ]
        return result
    used: set[int] = set()
    for kernel in result["kernels"]:
        remapped_indices = []
        for local in kernel["tensor_indices"]:
            local = int(local)
            if local not in mapping:
                raise ValueError(f"no outer tensor mapping for local tensor {local}")
            remapped_indices.append(mapping[local])
            used.add(local)
        kernel["tensor_indices"] = remapped_indices

        def replace_accessor(match: re.Match[str]) -> str:
            local = int(match.group(1))
            if local not in mapping:
                raise ValueError(
                    f"source accessor uses unmapped local tensor {local}"
                )
            used.add(local)
            return (
                "get_tensor_accessor_args_cta_offset<"
                f"{mapping[local]}, {match.group(2)}>()"
            )

        kernel["cpp_source"] = _ACCESSOR.sub(
            replace_accessor, kernel["cpp_source"]
        )
    missing = set(mapping) - used
    if missing:
        # Inputs used only as metadata are almost always an ABI mistake for a
        # compact child and become silent stale-state bugs after remapping.
        raise ValueError(f"tensor map contains unused local tensors: {sorted(missing)}")
    return result


def bind_serialized_child(program: dict, tensor_map: Mapping[int, int]) -> dict:
    """Bind one child to a sparse outer table through a compact local ABI.

    Artifact-v4 used to rewrite every child accessor to its parent-global
    tensor number.  The runtime consequently attached the complete parent
    tensor table to every GenericOp, even when a physical program used only a
    handful of tensors.  Large resident owners then overflowed the Tensix
    compile-time-argument data region.

    Keep a dense program-local tensor table instead.  ``operand_indices`` maps
    that table back to the outer external+private table at lowering time.
    """
    used = sorted(serialized_child_tensor_ids(program))
    mapping = {int(local): int(outer) for local, outer in tensor_map.items()}
    missing = set(used) - set(mapping)
    if missing:
        raise ValueError(
            f"no outer tensor mapping for local tensors {sorted(missing)}"
        )
    dense = {local: index for index, local in enumerate(used)}
    result = remap_serialized_child(program, dense)
    result["num_tensors"] = len(used)
    result["operand_indices"] = [mapping[local] for local in used]
    return result


def expand_serialized_child_bindings(program: dict) -> dict:
    """Restore an artifact-v4 child's parent-local tensor IDs.

    TT-Lang may compact a sequence child to a dense tensor table and attach
    ``operand_indices`` that map those dense slots back to its owning
    operation.  A second-level composer must undo that binding before it can
    retarget semantic operands such as packed weights or checkpoint arenas.
    Children without a dense binding are returned unchanged.
    """

    operands = program.get("operand_indices")
    if operands is None:
        return copy.deepcopy(program)
    if not isinstance(operands, list) or not operands:
        raise ValueError("serialized child has an invalid operand_indices table")
    used = serialized_child_tensor_ids(program)
    dense_slots = set(range(len(operands)))
    # Native TTNN children lowered inside an artifact-v4 owner already carry
    # their parent-local tensor IDs in TensorAccessor metadata.  The compiler
    # still attaches ``operand_indices`` for ABI registration, so blindly
    # treating those IDs as dense slots attempts to remap (for example) local
    # tensor 32 through a three-entry table.  Detect that representation and
    # merely retire the redundant binding.  Hand-written TT-Lang children keep
    # the original dense 0..N-1 representation and follow the normal path.
    if not used.issubset(dense_slots):
        parent_ids = {int(value) for value in operands}
        if not used.issubset(parent_ids):
            raise ValueError(
                "serialized child mixes dense and parent-local tensor IDs: "
                f"used={sorted(used)}, operands={operands}"
            )
        # Native children use operand_indices as their actual tensor table,
        # not as a removable dense binding.  Preserve it for the outer
        # artifact serializer and only normalize the table size.
        result = copy.deepcopy(program)
        result["num_tensors"] = max(parent_ids | used) + 1
        return result
    result = remap_serialized_child(
        program,
        {index: int(outer) for index, outer in enumerate(operands)},
    )
    result.pop("operand_indices", None)
    result["num_tensors"] = max(int(value) for value in operands) + 1
    return result


def serialized_child_tensor_ids(program: dict) -> set[int]:
    """Return operation tensor IDs referenced by a compact child's kernels."""
    if program.get("operation") in {
        "ttnn.matmul", "ttnn.linear", "ttnn.add", "ttnn.multiply", "ttnn.full",
        "ttnn.typecast", "ttnn.softmax", "ttnn.silu", "ttnn.permute",
    }:
        operands = program.get("operand_indices")
        if not isinstance(operands, list) or not operands:
            raise ValueError("native sequence child requires operand_indices")
        return {int(tensor) for tensor in operands}
    return {
        int(tensor)
        for kernel in program["kernels"]
        for tensor in kernel["tensor_indices"]
    }


_PACKED_PAGE_ACCESSOR = r'''
template <typename Accessor>
struct TTFineTunePackedPageAccessor : Accessor {
    uint32_t page_offset;

    explicit TTFineTunePackedPageAccessor(
        Accessor accessor, const uint32_t page_offset_) :
        Accessor(accessor), page_offset(page_offset_) {}
    using Accessor::get_noc_addr;
    FORCE_INLINE uint64_t get_noc_addr(
        const uint32_t page_id, const uint32_t byte_offset = 0,
        uint8_t noc = noc_index) const {
        return Accessor::get_noc_addr(
            page_offset + page_id, byte_offset, noc);
    }
};

template <typename Accessor>
TTFineTunePackedPageAccessor<Accessor>
tt_finetune_packed_page_accessor(
    Accessor accessor, const uint32_t page_offset) {
    return TTFineTunePackedPageAccessor<Accessor>(accessor, page_offset);
}

template <typename Accessor>
struct noc_traits_t<TTFineTunePackedPageAccessor<Accessor>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    template <Noc::AddressType address_type>
    static auto src_addr(
        const TTFineTunePackedPageAccessor<Accessor>& src,
        const Noc& noc,
        const src_args_type& args)
        -> std::conditional_t<
            address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        const uint64_t noc_addr = src.get_noc_addr(
            args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        }
        return noc_addr;
    }
    template <Noc::AddressType address_type>
    static auto dst_addr(
        const TTFineTunePackedPageAccessor<Accessor>& dst,
        const Noc& noc,
        const dst_args_type& args)
        -> std::conditional_t<
            address_type == Noc::AddressType::LOCAL_L1, uint32_t, uint64_t> {
        const uint64_t noc_addr = dst.get_noc_addr(
            args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        }
        return noc_addr;
    }
};

'''


def _matching_parenthesis(source: str, start: int) -> int:
    depth = 0
    for position in range(start, len(source)):
        if source[position] == "(":
            depth += 1
        elif source[position] == ")":
            depth -= 1
            if depth == 0:
                return position
    raise ValueError("unterminated TensorAccessor construction")


def retarget_tensor_reads_to_packed_pages(
    program: dict,
    *,
    packed_tensor: int,
    tensor_page_offsets: Mapping[int, int],
) -> dict:
    """Make selected DRAM tensor accessors view slices of one packed tensor.

    Merely adding bytes to a DRAM base address is incorrect for interleaved
    storage because the page ID selects the bank.  The injected accessor keeps
    the packed tensor's own distribution metadata and adds a compile-time page
    offset at every ``get_noc_addr`` call.  Existing readers, helper templates,
    core schedules and arithmetic remain untouched.

    Tensor IDs absent from a particular child are ignored so this transform
    can be applied uniformly to every program in a larger sequence.  Any
    selected tensor that is present but lacks the qualified TensorAccessor
    construction fails closed.
    """

    if packed_tensor < 0:
        raise ValueError("packed tensor ID must be non-negative")
    offsets = {int(tensor): int(offset) for tensor, offset in tensor_page_offsets.items()}
    if any(tensor < 0 or offset < 0 for tensor, offset in offsets.items()):
        raise ValueError("packed tensor IDs and page offsets must be non-negative")
    result = copy.deepcopy(program)
    present = serialized_child_tensor_ids(result) & set(offsets)
    if not present:
        return result

    wrapped: set[int] = set()
    for kernel in result["kernels"]:
        source = kernel["cpp_source"]
        replacements: list[tuple[int, int, str]] = []
        for tensor in sorted(present):
            marker = re.compile(
                r"get_tensor_accessor_args_cta_offset<\s*"
                + str(tensor)
                + r"\s*,"
            )
            for match in marker.finditer(source):
                construction = source.rfind("TensorAccessor(", 0, match.start())
                if construction < 0 or match.start() - construction > 180:
                    # Some generated readers materialize TensorAccessorArgs in
                    # one statement and construct the accessor in the next.
                    # In that form the qualified tensor marker precedes the
                    # TensorAccessor construction instead of being nested in
                    # it.  Accept only the immediately following construction
                    # and verify that it consumes the named args object.
                    statement_start = source.rfind("\n", 0, match.start()) + 1
                    statement_end = source.find(";", match.end())
                    args_statement = source[statement_start:statement_end]
                    args_match = re.search(
                        r"\bauto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
                        r"TensorAccessorArgs<",
                        args_statement,
                    )
                    construction = source.find(
                        "TensorAccessor(", statement_end + 1
                    )
                    if (
                        args_match is None
                        or construction < 0
                        or construction - statement_end > 240
                    ):
                        construction = -1
                    elif args_match.group(1) not in source[
                        construction:construction + 180
                    ]:
                        construction = -1
                if construction < 0:
                    raise ValueError(
                        "packed parameter TensorAccessor construction changed "
                        f"for tensor {tensor}"
                    )
                close = _matching_parenthesis(
                    source, construction + len("TensorAccessor")
                )
                expression = source[construction:close + 1]
                replacement = (
                    "tt_finetune_packed_page_accessor("
                    f"{expression}, {offsets[tensor]})"
                )
                replacements.append((construction, close + 1, replacement))
                # Preserve the wrapper's concrete type. Generated kernels
                # commonly spell the RHS as
                # `TensorAccessor accessor = TensorAccessor(...)`. Replacing
                # only the constructor expression would assign the derived
                # offset-aware wrapper into the base TensorAccessor and slice
                # away `page_offset`, silently turning every packed view into
                # a page-zero read. `auto` retains the wrapper type while
                # leaving direct constructor arguments unchanged.
                declaration_prefix = source[max(0, construction - 240):construction]
                declaration = re.search(
                    r"\bTensorAccessor\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*$",
                    declaration_prefix,
                )
                if declaration is not None:
                    type_start = max(0, construction - 240) + declaration.start()
                    type_stop = type_start + len("TensorAccessor")
                    replacements.append((type_start, type_stop, "auto"))
                wrapped.add(tensor)
        if replacements:
            unique = {(start, stop): value for start, stop, value in replacements}
            if len(unique) != len(replacements):
                raise ValueError("duplicate packed TensorAccessor match")
            for (start, stop), value in sorted(
                unique.items(), reverse=True
            ):
                source = source[:start] + value + source[stop:]
            # A child may read independent packed arenas (for example frozen
            # base weights and mutable LoRA state).  The transform is applied
            # once per arena, so make helper injection idempotent while still
            # wrapping each selected accessor with its own packed tensor and
            # page offset.
            if "struct TTFineTunePackedPageAccessor" not in source:
                insertion = min(
                    position for position in (
                        source.find("namespace"),
                        source.find("void kernel_main"),
                        source.find("__attribute__"),
                    ) if position >= 0
                )
                source = (
                    source[:insertion]
                    + _PACKED_PAGE_ACCESSOR
                    + source[insertion:]
                )
            kernel["cpp_source"] = source

    missing = present - wrapped
    if missing:
        raise ValueError(
            "packed parameter accessors were not found for tensors "
            f"{sorted(missing)}"
        )
    mapping = {
        tensor: packed_tensor if tensor in present else tensor
        for tensor in serialized_child_tensor_ids(result)
    }
    return remap_serialized_child(result, mapping)


def _offset_page_id_in_call(call: str, page_offset: int) -> tuple[str, int]:
    """Add a constant offset to the destination page initializer in one call."""

    marker = re.search(r"\.page_id\s*=\s*", call)
    if marker is None:
        return call, 0
    start = marker.end()
    depth = 0
    stop = start
    while stop < len(call):
        character = call[stop]
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif depth == 0 and character in ",}":
            break
        stop += 1
    if stop == len(call) or depth != 0:
        raise ValueError("unterminated async-write page expression")
    expression = call[start:stop].strip()
    if not expression:
        raise ValueError("empty async-write page expression")
    replacement = f"({page_offset} + ({expression}))"
    return call[:start] + replacement + call[stop:], 1


def retarget_tensor_writes_to_packed_pages(
    program: dict,
    *,
    packed_tensor: int,
    tensor_page_offsets: Mapping[int, int],
) -> dict:
    """Point selected output writers at slices of one packed DRAM tensor.

    Unlike the read transform above, this keeps the qualified TensorAccessor
    type unchanged and shifts the destination ``page_id`` expressions at the
    actual NoC write sites.  Blackhole qualification showed that deriving an
    accessor wrapper for a raw-FP32 destination can alter the published low
    mantissa bits even though the producer CB is exact.  Explicit page IDs
    preserve the accepted writer path byte-for-byte apart from address math.

    The transform is fail-closed: every selected accessor must be used by at
    least one ``Noc::async_write`` destination, and an accessor cannot escape
    to an unqualified use site.
    """

    if packed_tensor < 0:
        raise ValueError("packed tensor ID must be non-negative")
    offsets = {int(tensor): int(offset) for tensor, offset in tensor_page_offsets.items()}
    if any(tensor < 0 or offset < 0 for tensor, offset in offsets.items()):
        raise ValueError("packed tensor IDs and page offsets must be non-negative")
    result = copy.deepcopy(program)
    present = serialized_child_tensor_ids(result) & set(offsets)
    if not present:
        return result

    qualified: dict[int, int] = {tensor: 0 for tensor in present}
    for kernel in result["kernels"]:
        source = kernel["cpp_source"]
        accessors: dict[str, int] = {}
        for tensor in sorted(present):
            marker = re.compile(
                r"get_tensor_accessor_args_cta_offset<\s*"
                + str(tensor)
                + r"\s*,"
            )
            for match in marker.finditer(source):
                construction = source.rfind("TensorAccessor(", 0, match.start())
                if construction < 0 or match.start() - construction > 180:
                    statement_end = source.find(";", match.end())
                    construction = source.find("TensorAccessor(", statement_end + 1)
                    if construction < 0 or construction - statement_end > 240:
                        construction = -1
                if construction < 0:
                    raise ValueError(
                        "packed output TensorAccessor construction changed "
                        f"for tensor {tensor}"
                    )
                prefix = source[max(0, construction - 260):construction]
                declaration = re.search(
                    r"\b(?:const\s+)?(?:auto|TensorAccessor)\s+"
                    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*$",
                    prefix,
                )
                if declaration is None:
                    raise ValueError(
                        "packed output TensorAccessor declaration changed "
                        f"for tensor {tensor}"
                    )
                name = declaration.group(1)
                if name in accessors and accessors[name] != tensor:
                    raise ValueError(f"ambiguous output accessor {name!r}")
                accessors[name] = tensor

        replacements: list[tuple[int, int, str]] = []
        for match in re.finditer(
            r"\b[A-Za-z_][A-Za-z0-9_]*\.async_write\s*\(", source
        ):
            open_parenthesis = source.find("(", match.start())
            close_parenthesis = _matching_parenthesis(source, open_parenthesis)
            call = source[match.start():close_parenthesis + 1]
            matched_names = [
                name for name in accessors
                if re.search(rf",\s*{re.escape(name)}\s*,", call)
            ]
            if not matched_names:
                continue
            if len(matched_names) != 1:
                raise ValueError("ambiguous packed output async-write destination")
            name = matched_names[0]
            rewritten, count = _offset_page_id_in_call(
                call, offsets[accessors[name]]
            )
            if count != 1:
                raise ValueError(
                    f"packed output write through {name!r} lacks one page ID"
                )
            replacements.append((match.start(), close_parenthesis + 1, rewritten))
            qualified[accessors[name]] += 1
        for start, stop, replacement in sorted(replacements, reverse=True):
            source = source[:start] + replacement + source[stop:]
        kernel["cpp_source"] = source

    missing = {tensor: count for tensor, count in qualified.items() if count == 0}
    if missing:
        raise ValueError(
            "packed output accessors had no qualified async writes: "
            f"{sorted(missing)}"
        )
    mapping = {
        tensor: packed_tensor if tensor in present else tensor
        for tensor in serialized_child_tensor_ids(result)
    }
    return remap_serialized_child(result, mapping)


def make_serialized_page_copy_bitwise(
    program: dict,
    *,
    total_pages: int,
    column_tiles: int,
    source_page_offset: int,
    tile_bytes: int = 2048,
) -> dict:
    """Turn the qualified one-tile TT-Lang copy into a raw NoC relay.

    The normal generated copy is reader -> BF16 compute copy -> writer.  That
    is numerically valid for BF16 tensors, but a resident checkpoint also
    stores raw FP32 tiles as pairs of 2048-byte pages.  Passing those bytes
    through the BF16 unpack/pack engine changes some low-half bit patterns.
    The reader CB already contains the exact DRAM bytes, so the writer can
    consume that CB directly and the compute kernel can be removed.
    """

    if total_pages <= 0 or column_tiles <= 0 or source_page_offset < 0:
        raise ValueError("raw page-copy dimensions must be positive")
    if tile_bytes not in (2048, 4096):
        raise ValueError("raw page-copy supports BF16 or FP32 tile bytes")
    result = copy.deepcopy(program)
    kernels = result.get("kernels", [])
    compute_indices = [
        index for index, kernel in enumerate(kernels)
        if kernel.get("thread_type") == "compute"
    ]
    noc_indices = [
        index for index, kernel in enumerate(kernels)
        if kernel.get("thread_type") == "noc"
    ]
    if len(compute_indices) != 1 or len(noc_indices) != 2:
        raise ValueError("qualified page-copy kernel topology changed")
    compute_source = kernels[compute_indices[0]]["cpp_source"]
    if (
        compute_source.count("copy_tile(") != 1
        or compute_source.count("pack_tile<true>(") != 1
    ):
        raise ValueError("qualified page-copy compute relay changed")

    writer_index = next(
        index for index in noc_indices
        if "async_write(" in kernels[index]["cpp_source"]
    )
    reader_index = next(index for index in noc_indices if index != writer_index)
    reader = kernels[reader_index]
    writer = kernels[writer_index]
    if len(reader["tensor_indices"]) != 1 or len(writer["tensor_indices"]) != 1:
        raise ValueError("qualified page-copy tensor ABI changed")
    source_tensor = int(reader["tensor_indices"][0])
    output_tensor = int(writer["tensor_indices"][0])
    reader["tensor_indices"] = [source_tensor, output_tensor]
    reader["cpp_source"] = f'''#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {{
    constexpr uint32_t kGridColumns = 11;
    constexpr uint32_t kTileBytes = {tile_bytes};
    constexpr uint32_t kTotalPages = {total_pages};
    constexpr uint32_t kColumnTiles = {column_tiles};
    constexpr uint32_t kSourcePageOffset = {source_page_offset};
    const uint32_t node =
        get_absolute_logical_y() * kGridColumns + get_absolute_logical_x();
    const auto source = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {source_tensor}, 1>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {output_tensor}, 1>(), 0>(),
        get_common_arg_val<uint32_t>(1), kTileBytes);
    CircularBuffer scratch(get_compile_time_arg_val(0));
    Noc noc0(0);
    for (uint32_t column = 0; column < kColumnTiles; ++column) {{
        const uint32_t page = node * kColumnTiles + column;
        if (page >= kTotalPages) continue;
        const uint32_t address = scratch.get_write_ptr();
        noc0.async_read(
            source, CoreLocalMem<uint32_t>(address), kTileBytes,
            {{.page_id = kSourcePageOffset + page}}, {{}});
        noc0.async_read_barrier();
        noc0.async_write(
            CoreLocalMem<uint32_t>(address), output, kTileBytes, {{}},
            {{.page_id = page}});
        noc0.async_write_barrier();
    }}
}}
'''
    result["kernels"] = [reader]

    cb_configs = result.get("cb_configs", [])
    if cb_configs:
        matching = [
            config for config in cb_configs
            if int(config.get("buffer_index", -1)) == 0
        ]
        if len(matching) != 1:
            raise ValueError("qualified page-copy source CB changed")
        result["cb_configs"] = matching
    return result


def make_serialized_packed_parameter_unpack_bitwise(
    program: dict,
    *,
    entries: Sequence[tuple[int, int, int]],
    tile_bytes: int = 2048,
    grid_columns: int = 11,
    grid_rows: int = 10,
    smol_grouped_qvk: Mapping[str, int] | None = None,
    qwen_grouped_qkv: Mapping[str, int] | None = None,
) -> dict:
    """Expand one qualified copy into one parallel packed-parameter unpack.

    ``entries`` contains ``(source_arena, source_page_offset, page_count)``
    for each destination tensor.  Source arena 0 is the packed frozen arena,
    source arena 1 is the packed trainable arena, and destination tensor IDs
    are assigned densely from 2.  All pages are relayed as raw bytes, so this
    staging step cannot alter BF16 payloads or the numerical contract of a
    following native TTNN operation.

    The operation deliberately uses one NOC kernel and one scratch page.  It
    replaces 26 framework-visible parameter tensors with one on-device
    scatter program per layer while allowing the native TTNN matmul/linear
    factories to continue seeing their original tensor shapes.
    """

    if not entries:
        raise ValueError("packed parameter unpack requires at least one entry")
    if tile_bytes not in (2048, 4096):
        raise ValueError("packed parameter unpack supports BF16 or FP32 pages")
    if grid_columns <= 0 or grid_rows <= 0:
        raise ValueError("packed parameter unpack grid must be positive")
    normalized = [tuple(int(value) for value in entry) for entry in entries]
    if any(
        len(entry) != 3
        or entry[0] not in (0, 1)
        or entry[1] < 0
        or entry[2] <= 0
        for entry in normalized
    ):
        raise ValueError(
            "unpack entries require source arena 0/1, non-negative offset "
            "and positive page count"
        )

    grouped = (
        {str(name): int(value) for name, value in smol_grouped_qvk.items()}
        if smol_grouped_qvk is not None else None
    )
    grouped_destinations: set[int] = set()
    if grouped is not None:
        required = {
            "q_a_source", "v_a_source", "k_a_source",
            "q_b_source", "v_b_source", "k_b_source",
            "a_destination", "q_b_destination",
            "v_b_destination", "k_b_destination",
            "a_pages", "q_b_pages", "v_b_pages", "k_b_pages",
        }
        if set(grouped) != required or any(value < 0 for value in grouped.values()):
            raise ValueError("invalid Smol grouped Q/V/K unpack descriptor")
        grouped_destinations = {
            grouped["a_destination"],
            grouped["q_b_destination"],
            grouped["v_b_destination"],
            grouped["k_b_destination"],
        }
        if any(index >= len(normalized) for index in grouped_destinations):
            raise ValueError("Smol grouped destination is outside unpack entries")
    qwen_grouped = (
        {str(name): int(value) for name, value in qwen_grouped_qkv.items()}
        if qwen_grouped_qkv is not None else None
    )
    if grouped is not None and qwen_grouped is not None:
        raise ValueError("Smol and Qwen grouped unpack modes are exclusive")
    if qwen_grouped is not None:
        required = {
            "q_weight_source", "k_weight_source", "v_weight_source",
            "q_weight_pages", "k_weight_pages", "v_weight_pages",
            "q_bias_source", "k_bias_source", "v_bias_source",
            "q_bias_pages", "k_bias_pages", "v_bias_pages",
            "q_a_source", "k_a_source", "v_a_source", "a_column_tiles",
            "q_b_source", "k_b_source", "v_b_source",
            "q_b_pages", "k_b_pages", "v_b_pages",
            "q_b_destination", "k_b_destination", "v_b_destination",
            "weight_destination", "bias_destination", "a_destination",
        }
        if set(qwen_grouped) != required or any(
            value < 0 for value in qwen_grouped.values()
        ):
            raise ValueError("invalid Qwen grouped Q/K/V unpack descriptor")
        qwen_destinations = {
            qwen_grouped["weight_destination"],
            qwen_grouped["bias_destination"],
            qwen_grouped["a_destination"],
            qwen_grouped["q_b_destination"],
            qwen_grouped["k_b_destination"],
            qwen_grouped["v_b_destination"],
        }
        if any(index >= len(normalized) for index in qwen_destinations):
            raise ValueError("Qwen grouped destination is outside unpack entries")
        grouped_destinations.update(qwen_destinations)

    # Reuse the accepted one-page TT-Lang copy only for its qualified core
    # range, NOC configuration and scratch-CB descriptor.  The generated
    # reader/compute/writer bodies are replaced by the raw relay below.
    result = copy.deepcopy(program)
    kernels = result.get("kernels", [])
    noc_kernels = [
        kernel for kernel in kernels if kernel.get("thread_type") == "noc"
    ]
    compute_kernels = [
        kernel for kernel in kernels if kernel.get("thread_type") == "compute"
    ]
    if len(noc_kernels) != 2 or len(compute_kernels) != 1:
        raise ValueError("qualified parameter-unpack template topology changed")
    reader = next(
        (kernel for kernel in noc_kernels if "async_read(" in kernel.get("cpp_source", "")),
        None,
    )
    if reader is None:
        raise ValueError("qualified parameter-unpack reader changed")

    tensor_count = 2 + len(normalized)
    reader["tensor_indices"] = list(range(tensor_count))
    accessor_lines = []
    for tensor in range(tensor_count):
        accessor_lines.append(f'''    const auto tensor_{tensor} = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {tensor}, 1>(), 0>(),
        get_common_arg_val<uint32_t>({tensor}), kTileBytes);''')
    copy_lines = []
    for destination, (source_arena, source_offset, page_count) in enumerate(
        normalized
    ):
        if destination in grouped_destinations:
            continue
        output = 2 + destination
        copy_lines.append(f'''    for (uint32_t page = node; page < {page_count}; page += kCoreCount) {{
        const uint32_t address = scratch.get_write_ptr();
        noc0.async_read(
            tensor_{source_arena}, CoreLocalMem<uint32_t>(address), kTileBytes,
            {{.page_id = {source_offset} + page}}, {{}});
        noc0.async_read_barrier();
        noc0.async_write(
            CoreLocalMem<uint32_t>(address), tensor_{output}, kTileBytes, {{}},
            {{.page_id = page}});
        noc0.async_write_barrier();
    }}''')
    if grouped is not None:
        a_output = 2 + grouped["a_destination"]
        copy_lines.append(f'''    // SmolLM2 logical rank-24 A = [Q | V | K].  The source
    // tensors each contain eight live BF16 rows in one padded tile row.
    for (uint32_t page = node; page < {grouped["a_pages"]}; page += kCoreCount) {{
        const uint32_t source_address = scratch.get_write_ptr();
        const uint32_t destination_address = source_address + kTileBytes;
        volatile tt_l1_ptr uint32_t* destination_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_address);
        for (uint32_t word = 0; word < kTileBytes / 4; ++word) {{
            destination_words[word] = 0;
        }}
        const uint32_t source_pages[3] = {{
            {grouped["q_a_source"]} + page,
            {grouped["v_a_source"]} + page,
            {grouped["k_a_source"]} + page,
        }};
        const uint32_t destination_face_words[3][2] = {{
            {{0, 128}}, {{64, 192}}, {{256, 384}},
        }};
        for (uint32_t member = 0; member < 3; ++member) {{
            noc0.async_read(
                tensor_1, CoreLocalMem<uint32_t>(source_address), kTileBytes,
                {{.page_id = source_pages[member]}}, {{}});
            noc0.async_read_barrier();
            volatile tt_l1_ptr uint32_t* source_words =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_address);
            for (uint32_t word = 0; word < 64; ++word) {{
                destination_words[destination_face_words[member][0] + word] =
                    source_words[word];
                destination_words[destination_face_words[member][1] + word] =
                    source_words[128 + word];
            }}
        }}
        noc0.async_write(
            CoreLocalMem<uint32_t>(destination_address), tensor_{a_output},
            kTileBytes, {{}}, {{.page_id = page}});
        noc0.async_write_barrier();
    }}''')

        # Each padded B source has live values in columns 0..7.  Build the
        # canonical physical rank-32 members at Q=0, V=8 and K=16 without
        # unpacking/repacking through the math engine.
        for member, column_offset in (("q", 0), ("v", 8), ("k", 16)):
            output = 2 + grouped[f"{member}_b_destination"]
            source_page = grouped[f"{member}_b_source"]
            page_count = grouped[f"{member}_b_pages"]
            destination_face = 0 if column_offset < 16 else 1
            destination_word_in_face = (column_offset % 16) // 2
            copy_lines.append(f'''    for (uint32_t page = node; page < {page_count}; page += kCoreCount) {{
        const uint32_t source_address = scratch.get_write_ptr();
        const uint32_t destination_address = source_address + kTileBytes;
        volatile tt_l1_ptr uint32_t* destination_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_address);
        for (uint32_t word = 0; word < kTileBytes / 4; ++word) {{
            destination_words[word] = 0;
        }}
        noc0.async_read(
            tensor_1, CoreLocalMem<uint32_t>(source_address), kTileBytes,
            {{.page_id = {source_page} + page}}, {{}});
        noc0.async_read_barrier();
        volatile tt_l1_ptr uint32_t* source_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_address);
        for (uint32_t row = 0; row < 32; ++row) {{
            const uint32_t source_face = row < 16 ? 0 : 2;
            const uint32_t source_row = row & 15;
            const uint32_t destination_row_face = row < 16 ? 0 : 2;
            const uint32_t destination_base =
                (destination_row_face + {destination_face}) * 128
                + source_row * 8 + {destination_word_in_face};
            const uint32_t source_base = source_face * 128 + source_row * 8;
            for (uint32_t word = 0; word < 4; ++word) {{
                destination_words[destination_base + word] =
                    source_words[source_base + word];
            }}
        }}
        noc0.async_write(
            CoreLocalMem<uint32_t>(destination_address), tensor_{output},
            kTileBytes, {{}}, {{.page_id = page}});
        noc0.async_write_barrier();
    }}''')
    if qwen_grouped is not None:
        # Qwen's fused prefix consumes one [Q|K|V] frozen weight, bias and A.
        # Weight rows and bias columns are already whole-tile aligned, so
        # their packed pages concatenate directly without touching payloads.
        for kind in ("weight", "bias"):
            output = 2 + qwen_grouped[f"{kind}_destination"]
            destination_offset = 0
            for member in ("q", "k", "v"):
                source_page = qwen_grouped[f"{member}_{kind}_source"]
                page_count = qwen_grouped[f"{member}_{kind}_pages"]
                copy_lines.append(f'''    for (uint32_t page = node; page < {page_count}; page += kCoreCount) {{
        const uint32_t address = scratch.get_write_ptr();
        noc0.async_read(
            tensor_0, CoreLocalMem<uint32_t>(address), kTileBytes,
            {{.page_id = {source_page} + page}}, {{}});
        noc0.async_read_barrier();
        noc0.async_write(
            CoreLocalMem<uint32_t>(address), tensor_{output}, kTileBytes, {{}},
            {{.page_id = {destination_offset} + page}});
        noc0.async_write_barrier();
    }}''')
                destination_offset += page_count

        # Each rank-16 A occupies the upper half of one physical tile row.
        # Form logical rows [Q(0:16), K(16:32), V(32:48)] while preserving
        # the BF16 words exactly. The unused lower half of the V tile is zero.
        a_output = 2 + qwen_grouped["a_destination"]
        a_columns = qwen_grouped["a_column_tiles"]
        copy_lines.append(f'''    for (uint32_t column = node; column < {a_columns}; column += kCoreCount) {{
        const uint32_t source_address = scratch.get_write_ptr();
        const uint32_t destination_address = source_address + kTileBytes;
        volatile tt_l1_ptr uint32_t* destination_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_address);

        // Tile row zero: Q in faces 0/1, K in faces 2/3.
        for (uint32_t word = 0; word < kTileBytes / 4; ++word) {{
            destination_words[word] = 0;
        }}
        const uint32_t qk_sources[2] = {{
            {qwen_grouped["q_a_source"]} + column,
            {qwen_grouped["k_a_source"]} + column,
        }};
        for (uint32_t member = 0; member < 2; ++member) {{
            noc0.async_read(
                tensor_1, CoreLocalMem<uint32_t>(source_address), kTileBytes,
                {{.page_id = qk_sources[member]}}, {{}});
            noc0.async_read_barrier();
            volatile tt_l1_ptr uint32_t* source_words =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_address);
            const uint32_t destination_face = member * 2;
            for (uint32_t face = 0; face < 2; ++face) {{
                for (uint32_t word = 0; word < 128; ++word) {{
                    destination_words[(destination_face + face) * 128 + word] =
                        source_words[face * 128 + word];
                }}
            }}
        }}
        noc0.async_write(
            CoreLocalMem<uint32_t>(destination_address), tensor_{a_output},
            kTileBytes, {{}}, {{.page_id = column}});
        noc0.async_write_barrier();

        // Tile row one: V in faces 0/1, padded rows remain zero.
        for (uint32_t word = 0; word < kTileBytes / 4; ++word) {{
            destination_words[word] = 0;
        }}
        noc0.async_read(
            tensor_1, CoreLocalMem<uint32_t>(source_address), kTileBytes,
            {{.page_id = {qwen_grouped["v_a_source"]} + column}}, {{}});
        noc0.async_read_barrier();
        volatile tt_l1_ptr uint32_t* source_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_address);
        for (uint32_t face = 0; face < 2; ++face) {{
            for (uint32_t word = 0; word < 128; ++word) {{
                destination_words[face * 128 + word] =
                    source_words[face * 128 + word];
            }}
        }}
        noc0.async_write(
            CoreLocalMem<uint32_t>(destination_address), tensor_{a_output},
            kTileBytes, {{}}, {{.page_id = {a_columns} + column}});
        noc0.async_write_barrier();
    }}''')
        # Projection-specific B tensors use the grouped logical-rank-48 ABI
        # padded to rank 64: Q at columns 0..15, K at 16..31 and V at
        # 32..47. Copy the live half-faces without arithmetic.
        for member, column_offset in (("q", 0), ("k", 16), ("v", 32)):
            output = 2 + qwen_grouped[f"{member}_b_destination"]
            source_page = qwen_grouped[f"{member}_b_source"]
            page_count = qwen_grouped[f"{member}_b_pages"]
            destination_column_tile = column_offset // 32
            destination_face_column = (column_offset % 32) // 16
            copy_lines.append(f'''    for (uint32_t row_tile = node; row_tile < {page_count}; row_tile += kCoreCount) {{
        const uint32_t source_address = scratch.get_write_ptr();
        const uint32_t destination_address = source_address + kTileBytes;
        volatile tt_l1_ptr uint32_t* destination_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_address);
        for (uint32_t word = 0; word < kTileBytes / 4; ++word) {{
            destination_words[word] = 0;
        }}
        noc0.async_read(
            tensor_1, CoreLocalMem<uint32_t>(source_address), kTileBytes,
            {{.page_id = {source_page} + row_tile}}, {{}});
        noc0.async_read_barrier();
        volatile tt_l1_ptr uint32_t* source_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source_address);
        for (uint32_t row_face = 0; row_face < 2; ++row_face) {{
            const uint32_t source_face = row_face * 2;
            const uint32_t destination_face =
                row_face * 2 + {destination_face_column};
            for (uint32_t word = 0; word < 128; ++word) {{
                destination_words[destination_face * 128 + word] =
                    source_words[source_face * 128 + word];
            }}
        }}
        noc0.async_write(
            CoreLocalMem<uint32_t>(destination_address), tensor_{output},
            kTileBytes, {{}},
            {{.page_id = row_tile * 2 + {destination_column_tile}}});
        noc0.async_write_barrier();
    }}''')
    reader["cpp_source"] = f'''#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {{
    constexpr uint32_t kGridColumns = {grid_columns};
    constexpr uint32_t kCoreCount = {grid_columns * grid_rows};
    constexpr uint32_t kTileBytes = {tile_bytes};
    const uint32_t node =
        get_absolute_logical_y() * kGridColumns + get_absolute_logical_x();
{chr(10).join(accessor_lines)}
    CircularBuffer scratch(get_compile_time_arg_val(0));
    Noc noc0(0);
{chr(10).join(copy_lines)}
}}
'''
    result["kernels"] = [reader]
    cb_configs = result.get("cb_configs", [])
    if cb_configs:
        scratch = [
            config for config in cb_configs
            if int(config.get("buffer_index", -1)) == 0
        ]
        if len(scratch) != 1:
            raise ValueError("qualified parameter-unpack scratch CB changed")
        scratch[0]["total_size"] = max(
            int(scratch[0].get("total_size", tile_bytes)), 2 * tile_bytes
        )
        result["cb_configs"] = scratch
    result["num_tensors"] = tensor_count
    result.pop("operand_indices", None)
    return result


def make_serialized_strided_page_copy_bitwise(
    program: dict,
    *,
    rows_tiles: int,
    section_column_tiles: int,
    source_row_tiles: int,
    source_section_offset: int,
    source_page_offset: int = 0,
    tile_bytes: int = 2048,
) -> dict:
    """Relay one row-strided tensor section without unpacking its tiles.

    Resident grouped projections store several logical tensors in every token
    row (for example ``gate | up | gate-H | up-H``).  Backward checkpoints use
    tensor-major contiguous slices instead.  This transform performs that
    publication as a raw NoC page permutation, preserving every BF16/FP32 bit
    while keeping the producer's row-packed arena private to the owner.
    """

    if min(rows_tiles, section_column_tiles, source_row_tiles) <= 0:
        raise ValueError("strided page-copy dimensions must be positive")
    if not 0 <= source_section_offset < source_row_tiles:
        raise ValueError("source section must begin inside its packed row")
    if source_section_offset + section_column_tiles > source_row_tiles:
        raise ValueError("source section extends beyond its packed row")
    if source_page_offset < 0:
        raise ValueError("source page offset must be non-negative")

    total_pages = rows_tiles * section_column_tiles
    result = make_serialized_page_copy_bitwise(
        program,
        total_pages=total_pages,
        column_tiles=section_column_tiles,
        source_page_offset=source_page_offset,
        tile_bytes=tile_bytes,
    )
    source = result["kernels"][0]["cpp_source"]
    marker = (
        "        const uint32_t page = node * kColumnTiles + column;\n"
        "        if (page >= kTotalPages) continue;\n"
    )
    if source.count(marker) != 1:
        raise ValueError("qualified raw page-copy loop changed")
    source = source.replace(
        marker,
        marker
        + "        const uint32_t source_page = kSourcePageOffset + "
        + f"(page / kColumnTiles) * {source_row_tiles} + "
        + f"{source_section_offset} + (page % kColumnTiles);\n",
    )
    address = ".page_id = kSourcePageOffset + page"
    if source.count(address) != 1:
        raise ValueError("qualified raw page-copy source address changed")
    source = source.replace(address, ".page_id = source_page")
    result["kernels"][0]["cpp_source"] = source
    return result


def make_serialized_head_to_token_copy_bitwise(
    program: dict,
    *,
    sequence_tiles: int,
    query_heads: int,
    head_dimension_tiles: int,
    source_page_offset: int,
) -> dict:
    """Relay a head-major attention matrix into token-major tile order.

    Attention checkpoints are packed as
    ``[head, sequence_tile, head_dimension_tile]`` while the O projection
    consumes ``[sequence_tile, head * head_dimension_tile]``.  Both layouts
    contain identical BF16 pages, so the exact conversion is a NoC page
    permutation and must not traverse unpack/pack compute.
    """

    if min(sequence_tiles, query_heads, head_dimension_tiles) <= 0:
        raise ValueError("head-to-token copy dimensions must be positive")
    column_tiles = query_heads * head_dimension_tiles
    total_pages = sequence_tiles * column_tiles
    result = make_serialized_page_copy_bitwise(
        program,
        total_pages=total_pages,
        column_tiles=column_tiles,
        source_page_offset=source_page_offset,
    )
    source = result["kernels"][0]["cpp_source"]
    constant_marker = (
        f"    constexpr uint32_t kSourcePageOffset = {source_page_offset};\n"
    )
    if source.count(constant_marker) != 1:
        raise ValueError("qualified raw page-copy constants changed")
    source = source.replace(
        constant_marker,
        constant_marker
        + f"    constexpr uint32_t kSequenceTiles = {sequence_tiles};\n"
        + (
            "    constexpr uint32_t kHeadDimensionTiles = "
            f"{head_dimension_tiles};\n"
        ),
    )
    page_marker = (
        "        const uint32_t page = node * kColumnTiles + column;\n"
        "        if (page >= kTotalPages) continue;\n"
    )
    if source.count(page_marker) != 1:
        raise ValueError("qualified raw page-copy page loop changed")
    source = source.replace(
        page_marker,
        page_marker
        + "        const uint32_t head = column / kHeadDimensionTiles;\n"
        + (
            "        const uint32_t head_column = "
            "column % kHeadDimensionTiles;\n"
        )
        + (
            "        const uint32_t source_page = kSourcePageOffset + "
            "head * kSequenceTiles * kHeadDimensionTiles + "
            "node * kHeadDimensionTiles + head_column;\n"
        ),
    )
    source_marker = ".page_id = kSourcePageOffset + page"
    if source.count(source_marker) != 1:
        raise ValueError("qualified raw page-copy source expression changed")
    source = source.replace(source_marker, ".page_id = source_page")
    result["kernels"][0]["cpp_source"] = source
    return result


def make_serialized_token_to_head_copy_bitwise(
    program: dict,
    *,
    sequence_tiles: int,
    heads: int,
    head_dimension_tiles: int,
    source_page_offset: int,
) -> dict:
    """Relay token-major projection pages into head-major tile order.

    Frozen Q/K/V projection checkpoints use
    ``[sequence_tile, head, head_dimension_tile]`` while the resident
    attention tail consumes
    ``[head, sequence_tile, head_dimension_tile]``.  The payload of every
    BF16 tile is unchanged; only NoC page addresses are permuted.
    """

    if min(sequence_tiles, heads, head_dimension_tiles) <= 0:
        raise ValueError("token-to-head dimensions must be positive")
    pages_per_head = sequence_tiles * head_dimension_tiles
    column_tiles = heads * head_dimension_tiles
    total_pages = sequence_tiles * column_tiles
    result = make_serialized_page_copy_bitwise(
        program,
        total_pages=total_pages,
        column_tiles=head_dimension_tiles,
        source_page_offset=source_page_offset,
    )
    source = result["kernels"][0]["cpp_source"]
    constant_marker = (
        f"    constexpr uint32_t kSourcePageOffset = {source_page_offset};\n"
    )
    if source.count(constant_marker) != 1:
        raise ValueError("qualified raw page-copy constants changed")
    source = source.replace(
        constant_marker,
        constant_marker
        + f"    constexpr uint32_t kSequenceTiles = {sequence_tiles};\n"
        + f"    constexpr uint32_t kHeads = {heads};\n"
        + (
            "    constexpr uint32_t kHeadDimensionTiles = "
            f"{head_dimension_tiles};\n"
        )
        + f"    constexpr uint32_t kPagesPerHead = {pages_per_head};\n",
    )
    page_marker = (
        "        const uint32_t page = node * kColumnTiles + column;\n"
        "        if (page >= kTotalPages) continue;\n"
    )
    if source.count(page_marker) != 1:
        raise ValueError("qualified raw page-copy page loop changed")
    source = source.replace(
        page_marker,
        page_marker
        + "        const uint32_t head = page / kPagesPerHead;\n"
        + "        const uint32_t page_in_head = page % kPagesPerHead;\n"
        + (
            "        const uint32_t token = "
            "page_in_head / kHeadDimensionTiles;\n"
        )
        + (
            "        const uint32_t head_column = "
            "page_in_head % kHeadDimensionTiles;\n"
        )
        + (
            "        const uint32_t source_page = kSourcePageOffset + "
            "token * kHeads * kHeadDimensionTiles + "
            "head * kHeadDimensionTiles + head_column;\n"
        ),
    )
    source_marker = ".page_id = kSourcePageOffset + page"
    if source.count(source_marker) != 1:
        raise ValueError("qualified raw page-copy source expression changed")
    source = source.replace(source_marker, ".page_id = source_page")
    result["kernels"][0]["cpp_source"] = source
    return result


def make_serialized_gqa_head_expand_copy_bitwise(
    program: dict,
    *,
    sequence_tiles: int,
    query_heads: int,
    kv_heads: int,
    head_dimension_tiles: int,
    source_page_offset: int = 0,
    tile_bytes: int = 2048,
) -> dict:
    """Expand compact head-major K/V pages to query-head-major pages.

    This is a pure NoC/L1 page fan-out.  It performs no unpack, compute or
    repack, so each expanded query-head page is bit-identical to the compact
    KV-head page selected by its GQA group.
    """

    if min(sequence_tiles, query_heads, kv_heads, head_dimension_tiles) <= 0:
        raise ValueError("GQA expansion dimensions must be positive")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group_size = query_heads // kv_heads
    pages_per_head = sequence_tiles * head_dimension_tiles
    total_pages = query_heads * pages_per_head
    result = make_serialized_page_copy_bitwise(
        program,
        total_pages=total_pages,
        column_tiles=pages_per_head,
        source_page_offset=source_page_offset,
        tile_bytes=tile_bytes,
    )
    source = result["kernels"][0]["cpp_source"]
    constant_marker = (
        f"    constexpr uint32_t kSourcePageOffset = {source_page_offset};\n"
    )
    if source.count(constant_marker) != 1:
        raise ValueError("qualified raw page-copy constants changed")
    source = source.replace(
        constant_marker,
        constant_marker
        + f"    constexpr uint32_t kPagesPerHead = {pages_per_head};\n"
        + f"    constexpr uint32_t kGroupSize = {group_size};\n",
    )
    page_marker = (
        "        const uint32_t page = node * kColumnTiles + column;\n"
        "        if (page >= kTotalPages) continue;\n"
    )
    if source.count(page_marker) != 1:
        raise ValueError("qualified raw page-copy page loop changed")
    source = source.replace(
        page_marker,
        page_marker
        + "        const uint32_t query_head = page / kPagesPerHead;\n"
        + "        const uint32_t page_in_head = page % kPagesPerHead;\n"
        + "        const uint32_t kv_head = query_head / kGroupSize;\n"
        + (
            "        const uint32_t source_page = kSourcePageOffset + "
            "kv_head * kPagesPerHead + page_in_head;\n"
        ),
    )
    source_marker = ".page_id = kSourcePageOffset + page"
    if source.count(source_marker) != 1:
        raise ValueError("qualified raw page-copy source expression changed")
    source = source.replace(source_marker, ".page_id = source_page")
    result["kernels"][0]["cpp_source"] = source
    return result


def make_serialized_page_fill_zero(
    program: dict,
    *,
    total_pages: int,
    column_tiles: int,
) -> dict:
    """Turn a qualified TT-Lang copy into a CB-credit-free zero fill.

    Full-step checkpoint arenas are compiler-owned clones.  They must be
    initialized before the first forward writer because some checkpoint
    regions are raw FP32 payloads that cannot safely pass through a BF16
    compute relay.  Copying from a framework tensor is also unnecessarily
    fragile: the framework may choose a different distributed memory layout
    for that tensor than for the compiler-owned destination.

    This transformation keeps one qualified data-movement kernel, constructs
    one exact all-zero 2048-byte page in its local CB allocation, and writes
    that page directly to every destination page.  It never reserves, pushes,
    waits on, or pops a circular buffer, so it cannot leak CB credit state into
    the following resident program.  The CB is used only as an allocated L1
    scratch address; no source tensor is read.
    """

    if total_pages <= 0 or column_tiles <= 0:
        raise ValueError("zero-fill dimensions must be positive")
    result = copy.deepcopy(program)
    kernels = result.get("kernels", [])
    compute_indices = [
        index for index, kernel in enumerate(kernels)
        if kernel.get("thread_type") == "compute"
    ]
    noc_indices = [
        index for index, kernel in enumerate(kernels)
        if kernel.get("thread_type") == "noc"
    ]
    if len(compute_indices) != 1 or len(noc_indices) != 2:
        raise ValueError("qualified zero-fill kernel topology changed")
    writer_index = next(
        index for index in noc_indices
        if "async_write(" in kernels[index]["cpp_source"]
    )
    reader_index = next(index for index in noc_indices if index != writer_index)
    writer = kernels[writer_index]
    if len(writer["tensor_indices"]) != 1:
        raise ValueError("qualified zero-fill destination ABI changed")
    output_tensor = int(writer["tensor_indices"][0])
    reader = kernels[reader_index]
    reader["tensor_indices"] = [output_tensor]
    reader["cpp_source"] = f'''#include <cstdint>
#include "api/compile_time_args.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {{
    constexpr uint32_t kGridColumns = 11;
    constexpr uint32_t kTileBytes = 2048;
    constexpr uint32_t kTileWords = kTileBytes / sizeof(uint32_t);
    constexpr uint32_t kTotalPages = {total_pages};
    constexpr uint32_t kColumnTiles = {column_tiles};
    const uint32_t node =
        get_absolute_logical_y() * kGridColumns + get_absolute_logical_x();
    const auto output = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {output_tensor}, 1>(), 0>(),
        get_common_arg_val<uint32_t>(0), kTileBytes);
    CircularBuffer scratch(get_compile_time_arg_val(0));
    const uint32_t address = scratch.get_write_ptr();
    volatile tt_l1_ptr uint32_t* words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    for (uint32_t word = 0; word < kTileWords; ++word) {{
        words[word] = 0;
    }}
    Noc noc0(0);
    for (uint32_t column = 0; column < kColumnTiles; ++column) {{
        const uint32_t page = node * kColumnTiles + column;
        if (page >= kTotalPages) continue;
        noc0.async_write(
            CoreLocalMem<uint32_t>(address), output, kTileBytes, {{}},
            {{.page_id = page}});
    }}
    noc0.async_write_barrier();
}}
'''
    result["kernels"] = [reader]

    cb_configs = result.get("cb_configs", [])
    if cb_configs:
        matching = [
            config for config in cb_configs
            if int(config.get("buffer_index", -1)) == 0
        ]
        if len(matching) != 1:
            raise ValueError("qualified zero-fill scratch CB changed")
        result["cb_configs"] = matching
    return result


def retarget_full_mlp_checkpoint_arena(
    program: dict,
    *,
    sequence_tiles: int,
    hidden_tiles: int,
    intermediate_tiles: int,
    gate_page_offset: int,
    up_page_offset: int,
    gate_h_page_offset: int,
    up_h_page_offset: int,
    normalized_input_page_offset: int,
) -> dict:
    """Make the exact MLP backward child consume the full-layer arena.

    The accepted MLP owner originally reads a row-packed checkpoint with the
    layout ``gate | up | gate-H | up-H | normalized-X``.  The full resident
    forward owner deliberately keeps those same values in separate contiguous
    regions of one larger attention/layer arena.  Rewriting only reader page
    IDs preserves every compute instruction, reduction order and BF16 packing
    boundary while removing the checkpoint repack/DRAM boundary.

    This helper is intentionally strict: every known address expression must
    occur exactly as many times as the qualified v31 child emits it.  A source
    change therefore fails compilation instead of silently reading stale L1.
    """
    if min(
        sequence_tiles, hidden_tiles, intermediate_tiles,
        gate_page_offset, up_page_offset, gate_h_page_offset,
        up_h_page_offset, normalized_input_page_offset,
    ) < 0:
        raise ValueError("MLP arena dimensions and offsets must be non-negative")
    if not sequence_tiles or not hidden_tiles or not intermediate_tiles:
        raise ValueError("MLP arena tile dimensions must be positive")

    result = copy.deepcopy(program)
    old_row_tiles = 2 * intermediate_tiles + 2 + hidden_tiles
    replacements = (
        (
            f"{{.page_id = m * {old_row_tiles} + "
            f"{2 * intermediate_tiles} + member}}",
            "{.page_id = (member == 0 ? "
            f"{gate_h_page_offset} : {up_h_page_offset}) + m}}",
            1,
        ),
        (
            f"{{.page_id = m * {old_row_tiles} + "
            f"{2 * intermediate_tiles + 2} + node}}",
            f"{{.page_id = {normalized_input_page_offset} + "
            f"m * {hidden_tiles} + node}}",
            1,
        ),
        (
            "{.page_id = m * product_row_tiles + k}",
            f"{{.page_id = {gate_page_offset} + "
            "m * product_i_tiles + k}",
            2,
        ),
        (
            "{.page_id = m * product_row_tiles + product_i_tiles + k}",
            f"{{.page_id = {up_page_offset} + "
            "m * product_i_tiles + k}",
            1,
        ),
        (
            f"{{.page_id = (job / {intermediate_tiles}) * "
            f"{old_row_tiles} + (job % {intermediate_tiles})}}",
            f"{{.page_id = {gate_page_offset} + job}}",
            1,
        ),
        (
            f"{{.page_id = (job / {intermediate_tiles}) * "
            f"{old_row_tiles} + {intermediate_tiles} + "
            f"(job % {intermediate_tiles})}}",
            f"{{.page_id = {up_page_offset} + job}}",
            1,
        ),
    )
    sources = [kernel["cpp_source"] for kernel in result["kernels"]]
    for old, new, expected in replacements:
        count = sum(source.count(old) for source in sources)
        if count != expected:
            raise ValueError(
                "qualified full-MLP checkpoint address changed: "
                f"expected {expected} occurrence(s) of {old!r}, found {count}"
            )
        sources = [source.replace(old, new) for source in sources]
    for kernel, source in zip(result["kernels"], sources):
        kernel["cpp_source"] = source
    return result


def _replace_accessor_page_expression(
    source: str,
    *,
    accessor: str,
    old_expression: str,
    new_expression: str,
) -> tuple[str, int]:
    """Rewrite one TensorAccessor write without touching its arithmetic."""

    pattern = re.compile(
        rf"(\b{re.escape(accessor)}\b\s*,[\s\S]{{0,160}}?"
        rf"\{{\.page_id\s*=\s*){re.escape(old_expression)}(\s*\}})"
    )
    return pattern.subn(rf"\g<1>{new_expression}\g<2>", source)


def _function_extent(source: str, signature: str) -> tuple[int, int]:
    start = source.find(signature)
    if start < 0:
        raise ValueError(f"qualified function {signature!r} was not found")
    brace = source.find("{", start)
    depth = 0
    for position in range(brace, len(source)):
        depth += (source[position] == "{") - (source[position] == "}")
        if depth == 0:
            return start, position + 1
    raise ValueError(f"qualified function {signature!r} is unterminated")


def retarget_full_mlp_gradient_outputs(
    program: dict,
    *,
    gate_a_page_offset: int,
    gate_b_page_offset: int,
    up_a_page_offset: int,
    up_b_page_offset: int,
    down_a_page_offset: int,
    down_b_page_offset: int,
) -> dict:
    """Point six exact MLP final writers at dense arena page ranges.

    Tensor-ID remapping is deliberately separate. A parent owner maps all six
    output operands to the same arena after this helper changes only their
    page addresses. CB lifecycles, reductions and BF16 packing are unchanged.
    """

    offsets = (
        gate_a_page_offset, gate_b_page_offset,
        up_a_page_offset, up_b_page_offset,
        down_a_page_offset, down_b_page_offset,
    )
    if any(value < 0 for value in offsets):
        raise ValueError("gradient arena page offsets must be non-negative")
    result = copy.deepcopy(program)
    contracts = (
        ("gate_a_output", "node", f"{gate_a_page_offset} + node"),
        ("gate_b_output", "n_tile", f"{gate_b_page_offset} + n_tile"),
        ("up_a_output", "node", f"{up_a_page_offset} + node"),
        ("up_b_output", "n_tile", f"{up_b_page_offset} + n_tile"),
        ("down_a_output", "k", f"{down_a_page_offset} + k"),
        ("down_b_output", "n", f"{down_b_page_offset} + n"),
    )
    counts = {name: 0 for name, _, _ in contracts}
    for kernel in result["kernels"]:
        source = kernel["cpp_source"]
        for accessor, old, new in contracts:
            source, count = _replace_accessor_page_expression(
                source,
                accessor=accessor,
                old_expression=old,
                new_expression=new,
            )
            counts[accessor] += count
        kernel["cpp_source"] = source
    wrong = {name: count for name, count in counts.items() if count != 1}
    if wrong:
        raise ValueError(
            "qualified full-MLP final-writer contract changed: "
            f"{wrong}"
        )
    return result


def retarget_full_attention_gradient_outputs(
    program: dict,
    *,
    o_a_page_offset: int,
    o_b_page_offset: int,
    q_a_page_offset: int,
    k_a_page_offset: int,
    v_a_page_offset: int,
    q_b_page_offset: int,
    k_b_page_offset: int,
    v_b_page_offset: int,
    logical_rank: int = 32,
) -> dict:
    """Compact eight attention gradients but retain page-zero dHidden scratch."""

    offsets = (
        o_a_page_offset, o_b_page_offset,
        q_a_page_offset, k_a_page_offset, v_a_page_offset,
        q_b_page_offset, k_b_page_offset, v_b_page_offset,
    )
    if any(value < 0 for value in offsets):
        raise ValueError("gradient arena page offsets must be non-negative")
    if logical_rank not in (8, 16, 32):
        raise ValueError("attention gradient packing requires rank 8, 16, or 32")
    result = copy.deepcopy(program)
    matching_kernels = []
    for kernel in result["kernels"]:
        source = kernel["cpp_source"]
        if all(
            marker in source
            for marker in (
                "FORCE_INLINE void v_writer_phase()",
                "FORCE_INLINE void k_writer_phase()",
                "FORCE_INLINE void q_writer_phase()",
                "Final output-projection adapter gradients",
            )
        ):
            matching_kernels.append(kernel)
    if len(matching_kernels) != 1:
        raise ValueError(
            "qualified full-attention final writer count changed: "
            f"{len(matching_kernels)}"
        )
    kernel = matching_kernels[0]
    source = kernel["cpp_source"]

    for old, new in (
        ("(m_tiles + 1 + n) * arena_cols", f"{o_b_page_offset} + n"),
        ("m_tiles * arena_cols + k", f"{o_a_page_offset} + k"),
    ):
        old_text = "{.page_id = " + old + "}"
        count = source.count(old_text)
        if count != 1:
            raise ValueError(
                "qualified output-projection page contract changed: "
                f"{old!r} occurs {count} times"
            )
        source = source.replace(old_text, "{.page_id = " + new + "}", 1)

    phase_offsets = {
        "v_writer_phase": (v_b_page_offset, v_a_page_offset),
        "k_writer_phase": (k_b_page_offset, k_a_page_offset),
        "q_writer_phase": (q_b_page_offset, q_a_page_offset),
    }
    page_pattern = re.compile(r"\{\.page_id\s*=\s*([^}]*)\}")
    for name, (db_offset, da_offset) in phase_offsets.items():
        start, stop = _function_extent(source, f"FORCE_INLINE void {name}()")
        body = source[start:stop]
        matches = [
            match for match in page_pattern.finditer(body)
            if match.group(1).strip() != "job"
        ]
        if len(matches) != 2:
            raise ValueError(
                f"qualified {name} final page contract changed: "
                f"found {len(matches)} non-scratch writes"
            )
        replacements = (
            f"{{.page_id = {db_offset} + n}}",
            f"{{.page_id = {da_offset} + k}}",
        )
        for match, replacement in zip(reversed(matches), reversed(replacements)):
            body = body[:match.start()] + replacement + body[match.end():]

        # A normal rank-16 TT tensor has one physical 32-column tile, but its
        # right two faces are outside the logical shape.  The standalone
        # owner returns that logical tensor, so PJRT never observes those
        # lanes.  Once the writer is retargeted into the rank-32 packed
        # gradient arena, however, clipping deliberately reduces the complete
        # physical tile.  Preserve the reference packer's zero-padding
        # contract explicitly instead of allowing the unused matmul lanes to
        # become part of the norm.  This is a writer/layout fix only: the 16
        # logical columns and their numerical kernel are untouched.
        if logical_rank == 16:
            db_pop = "        db.pop_front(1);\n    }"
            if body.count(db_pop) != 1:
                raise ValueError(
                    f"qualified {name} dB writer loop contract changed"
                )
            zero_padding = f'''        db.pop_front(1);
    }}

    // The packed optimizer ABI owns a full rank tile. Zero the two right
    // 16x16 faces which are padding for logical rank 16.
    constexpr uint32_t rank_face_bytes = 16 * 16 * sizeof(uint16_t);
    zero_l1_buf(
        reinterpret_cast<tt_l1_ptr uint32_t*>(db.get_read_ptr()),
        rank_face_bytes);
    for (uint32_t local_n = 0; local_n < n_tiles_per_node; ++local_n) {{
        const uint32_t n = node * n_tiles_per_node + local_n;
        if (n >= n_tiles) continue;
        noc1.async_write(
            CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
            rank_face_bytes, {{}},
            {{.page_id = {db_offset} + n, .offset_bytes = rank_face_bytes}});
        noc1.async_write(
            CoreLocalMem<uint32_t>(db.get_read_ptr()), arena,
            rank_face_bytes, {{}},
            {{.page_id = {db_offset} + n,
              .offset_bytes = 3 * rank_face_bytes}});
    }}
    noc1.async_write_barrier();'''
            body = body.replace(db_pop, zero_padding, 1)
        source = source[:start] + body + source[stop:]
    kernel["cpp_source"] = source
    return result


def retarget_full_mlp_forward_trainable_parameters(
    program: dict,
    *,
    gate_a_tensor: int,
    gate_b_tensor: int,
    up_a_tensor: int,
    up_b_tensor: int,
    down_a_tensor: int,
    down_b_tensor: int,
    gate_a_pack_base: int,
    gate_b_pack_base: int,
    up_a_pack_base: int,
    up_b_pack_base: int,
    down_a_pack_base: int,
    down_b_pack_base: int,
) -> dict:
    """Retarget full-layer forward LoRA reads to canonical tensor operands.

    Frozen gamma/mask/base weights remain in the immutable packed parameter
    tensor. Only gate/up/down A/B reads are redirected. This lets the normal
    PEFT Parameters be the single trainable representation, while preserving
    the accepted TT-Metal arithmetic and all L1/NoC scheduling.
    """
    result = copy.deepcopy(program)
    candidates = [
        kernel for kernel in result["kernels"]
        if "resident_layer_gate_up_reader::run" in kernel["cpp_source"]
        and "resident_layer_swiglu_down_reader::run" in kernel["cpp_source"]
    ]
    if len(candidates) != 1:
        raise ValueError(
            "full-layer forward trainable reader count changed: "
            f"{len(candidates)}"
        )
    kernel = candidates[0]
    source = kernel["cpp_source"]
    old_indices = list(kernel["tensor_indices"])
    new_tensors = (
        gate_a_tensor, gate_b_tensor, up_a_tensor,
        up_b_tensor, down_a_tensor, down_b_tensor,
    )
    if any(tensor in old_indices for tensor in new_tensors):
        raise ValueError("trainable MLP tensor IDs collide with reader ABI")
    first_common = len(old_indices)
    kernel["tensor_indices"].extend(new_tensors)

    accessor = re.search(
        r"get_tensor_accessor_args_cta_offset<\s*20\s*,\s*(\d+)\s*>",
        source,
    )
    if accessor is None:
        raise ValueError("packed full-layer parameter accessor changed")
    total_cbs = int(accessor.group(1))

    gate_start = source.index("namespace resident_layer_gate_up_reader")
    gate_end = source.index(
        "}  // namespace resident_layer_gate_up_reader", gate_start
    )
    down_start = source.index("namespace resident_layer_swiglu_down_reader")
    down_end = source.index(
        "}  // namespace resident_layer_swiglu_down_reader", down_start
    )
    gate = source[gate_start:gate_end]
    down = source[down_start:down_end]

    gate_signature = re.compile(
        r"template <typename PackedAccessor>(\s*static __attribute__\(\("
        r"noinline, noclone,.*?\)\)\s*)void run\(\s*"
        r"const PackedAccessor& packed_parameters\) \{",
        re.DOTALL,
    )
    gate, count = gate_signature.subn(
        r"template <typename PackedAccessor, typename GateAAccessor, "
        r"typename GateBAccessor, typename UpAAccessor, typename UpBAccessor>"
        r"\1void run(\n"
        r"    const PackedAccessor& packed_parameters,\n"
        r"    const GateAAccessor& gate_a_parameters,\n"
        r"    const GateBAccessor& gate_b_parameters,\n"
        r"    const UpAAccessor& up_a_parameters,\n"
        r"    const UpBAccessor& up_b_parameters) {",
        gate,
        count=1,
    )
    if count != 1:
        raise ValueError("gate/up reader run signature changed")
    for old, new, expected in (
        (
            f"const uint32_t a_base = member ?\n"
            f"                    {up_a_pack_base} : {gate_a_pack_base};",
            "const auto& a_parameters = member ?\n"
            "                    up_a_parameters : gate_a_parameters;",
            1,
        ),
        (
            f"const uint32_t a_base = up_member ?\n"
            f"                {up_a_pack_base} : {gate_a_pack_base};",
            "const auto& a_parameters = up_member ?\n"
            "                up_a_parameters : gate_a_parameters;",
            1,
        ),
        (
            f"const uint32_t b_base = up_member ?\n"
            f"                {up_b_pack_base} : {gate_b_pack_base};",
            "const auto& b_parameters = up_member ?\n"
            "                up_b_parameters : gate_b_parameters;",
            1,
        ),
    ):
        if gate.count(old) != expected:
            raise ValueError(f"gate/up packed declaration changed: {old!r}")
        gate = gate.replace(old, new, expected)
    if gate.count("{.page_id = a_base + k}") != 2:
        raise ValueError("gate/up A page expressions changed")
    gate = gate.replace("{.page_id = a_base + k}", "{.page_id = k}")
    gate, a_reads = re.subn(
        r"noc0\.async_read\(\s*packed_parameters,(.*?)"
        r"\{\.page_id = k\}, \{\}\);",
        r"noc0.async_read(\n                        a_parameters,\1"
        r"{.page_id = k}, {});",
        gate,
        flags=re.DOTALL,
    )
    if a_reads != 2:
        raise ValueError(f"gate/up A read count changed: {a_reads}")
    if gate.count("{.page_id = b_base + member_n}") != 1:
        raise ValueError("gate/up B page expression changed")
    gate = gate.replace(
        "packed_parameters, CoreLocalMem<uint32_t>(b_cb.get_write_ptr()),\n"
        "                kTileBytes, {.page_id = b_base + member_n}, {});",
        "b_parameters, CoreLocalMem<uint32_t>(b_cb.get_write_ptr()),\n"
        "                kTileBytes, {.page_id = member_n}, {});",
        1,
    )
    if "b_base + member_n" in gate:
        raise ValueError("gate/up B read was not retargeted")

    down_signature = re.compile(
        r"template <typename PackedAccessor>(\s*static __attribute__\(\("
        r"noinline, noclone,.*?\)\)\s*)void run\("
        r"const PackedAccessor& packed_parameters\) \{",
        re.DOTALL,
    )
    down, count = down_signature.subn(
        r"template <typename PackedAccessor, typename DownAAccessor, "
        r"typename DownBAccessor>\1void run(\n"
        r"    const PackedAccessor& packed_parameters,\n"
        r"    const DownAAccessor& down_a_parameters,\n"
        r"    const DownBAccessor& down_b_parameters) {",
        down,
        count=1,
    )
    if count != 1:
        raise ValueError("down reader run signature changed")
    old_down_a = f"{{.page_id = {down_a_pack_base} + i + inner}}"
    if down.count(old_down_a) != 1:
        raise ValueError("down A packed page expression changed")
    down = down.replace(old_down_a, "{.page_id = i + inner}", 1)
    down, count = re.subn(
        r"noc0\.async_read\(\s*packed_parameters,(.*?)"
        r"\{\.page_id = i \+ inner\}, \{\}\);",
        r"noc0.async_read(\n                    down_a_parameters,\1"
        r"{.page_id = i + inner}, {});",
        down,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("down A read was not retargeted")
    old_down_b = f"{{.page_id = {down_b_pack_base} + h}}"
    if down.count(old_down_b) != 1:
        raise ValueError("down B packed page expression changed")
    down = down.replace(
        "packed_parameters, CoreLocalMem<uint32_t>(b.get_write_ptr()),\n"
        f"            kTileBytes, {old_down_b}, {{}});",
        "down_b_parameters, CoreLocalMem<uint32_t>(b.get_write_ptr()),\n"
        "            kTileBytes, {.page_id = h}, {});",
        1,
    )
    if old_down_b in down:
        raise ValueError("down B read was not retargeted")

    source = source[:gate_start] + gate + source[gate_end:]
    # Gate replacement changes the down substring offsets; locate it again.
    down_start = source.index("namespace resident_layer_swiglu_down_reader")
    down_end = source.index(
        "}  // namespace resident_layer_swiglu_down_reader", down_start
    )
    source = source[:down_start] + down + source[down_end:]

    declaration = (
        "    const auto layer_parameters = TensorAccessor(TensorAccessorArgs<\n"
    )
    call_start = source.index(declaration)
    gate_call = "    resident_layer_gate_up_reader::run(layer_parameters);"
    down_call = "    resident_layer_swiglu_down_reader::run(layer_parameters);"
    if source.count(gate_call) != 1 or source.count(down_call) != 1:
        raise ValueError("full-layer MLP reader calls changed")
    names = ("gate_a", "gate_b", "up_a", "up_b", "down_a", "down_b")
    accessors = []
    for offset, (name, tensor) in enumerate(zip(names, new_tensors)):
        position = first_common + offset
        accessors.append(
            f"    const auto {name}_parameters = TensorAccessor("
            "TensorAccessorArgs<\n"
            "        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
            f"{tensor}, {total_cbs}>(), {position}>(),\n"
            f"        get_common_arg_val<uint32_t>({position}), 2048);\n"
        )
    source = source.replace(
        gate_call,
        "".join(accessors)
        + "    resident_layer_gate_up_reader::run(\n"
        "        layer_parameters, gate_a_parameters, gate_b_parameters,\n"
        "        up_a_parameters, up_b_parameters);",
        1,
    )
    source = source.replace(
        down_call,
        "    resident_layer_swiglu_down_reader::run(\n"
        "        layer_parameters, down_a_parameters, down_b_parameters);",
        1,
    )
    kernel["cpp_source"] = source
    return result


def extract_resident_full_mlp_forward_child(
    program: dict,
    *,
    normalized_tensor: int,
    residual_tensor: int,
) -> dict:
    """Extract the accepted MLP continuation from a full-layer artifact.

    The historical full-layer program co-locates attention and MLP on the
    same 60-core island.  Its MLP arithmetic may use either the historical
    register-paired SwiGLU or the exact decomposed-BF16 continuation, but its
    input arrives through a projection-writer CB epoch that cannot follow the
    newly recovered TTNN RMSNorm schedule safely.  This transformation keeps
    the gate/up, paired SwiGLU, down and residual helpers unchanged, replaces
    only that transport with a TensorAccessor read from a compiler-owned L1
    tensor, reads the pre-MLP residual from a second compiler-owned L1
    tensor, and removes the preceding attention kernels.
    """

    result = copy.deepcopy(program)
    if result.get("format_version") not in (1, 2, 3):
        raise ValueError("MLP extraction requires one physical child")
    kernels = result.get("kernels", [])
    reader_candidates = [
        kernel for kernel in kernels
        if "namespace resident_layer_gate_up_reader" in kernel.get(
            "cpp_source", ""
        )
        and "namespace resident_layer_swiglu_down_reader" in kernel.get(
            "cpp_source", ""
        )
    ]
    compute_candidates = [
        kernel for kernel in kernels
        if (
            "run_resident_layer_paired_gate_up_swiglu();" in kernel.get(
                "cpp_source", ""
            )
            or "run_resident_layer_gate_up_compute();" in kernel.get(
                "cpp_source", ""
            )
        )
        and "resident_layer_swiglu_down_compute::run();" in kernel.get(
            "cpp_source", ""
        )
    ]
    writer_candidates = [
        kernel for kernel in kernels
        if "resident_layer_gate_up_writer::run_full_mlp_tail" in kernel.get(
            "cpp_source", ""
        )
    ]
    if not (
        len(reader_candidates) == len(compute_candidates)
        == len(writer_candidates) == 1
    ):
        raise ValueError(
            "full-layer MLP physical kernel set changed: "
            f"reader={len(reader_candidates)}, "
            f"compute={len(compute_candidates)}, "
            f"writer={len(writer_candidates)}"
        )
    reader, compute, writer = (
        reader_candidates[0], compute_candidates[0], writer_candidates[0]
    )

    reader_source = reader["cpp_source"]
    # This extracted child is a new physical TT-Metal program.  The previous
    # attention child has therefore retired before any of these RISCs start,
    # but the reused L1 CB control words still contain the preceding child's
    # counters and pointers.  The monolithic source waits for received==acked
    # because its attention and MLP epochs share one physical program.  That
    # wait is invalid after extraction: aliased attention CBs can retain
    # counters from the opposite producer/consumer direction and can never
    # satisfy the MLP reader's local equality test.  Reset the already-drained
    # epoch directly.  The reconstructed writer below does not release the
    # compute RISC until the reader publishes init_ready, so no other RISC can
    # observe a partially rebased interface.
    rebase_start_marker = (
        "static __attribute__((noinline, noclone)) void\n"
        "rebase_empty_producer_cb(uint32_t cb_id) {"
    )
    rebase_start = reader_source.index(rebase_start_marker)
    rebase_stop = reader_source.index(
        "\n// Arm a reusable all-reader generation", rebase_start
    )
    extracted_rebase = (
        "static __attribute__((noinline, noclone)) void\n"
        "rebase_empty_producer_cb(uint32_t cb_id) {\n"
        "    // A preceding physical-program boundary proves the old epoch\n"
        "    // retired.  Do not wait on direction-aliased attention counters.\n"
        "    *get_cb_tiles_received_ptr(cb_id) = 0;\n"
        "    *get_cb_tiles_acked_ptr(cb_id) = 0;\n"
        "    auto& interface = get_local_cb_interface(cb_id);\n"
        "    const uint32_t base = interface.fifo_limit - interface.fifo_size;\n"
        "    interface.fifo_rd_ptr = base;\n"
        "    interface.fifo_wr_ptr = base;\n"
        "    interface.tiles_acked_received_init = 0;\n"
        "    interface.fifo_wr_tile_ptr = 0;\n"
        "}\n"
    )
    reader_source = (
        reader_source[:rebase_start]
        + extracted_rebase
        + reader_source[rebase_stop:]
    )
    accessor_match = re.search(
        r"get_tensor_accessor_args_cta_offset<\s*20\s*,\s*(\d+)\s*>",
        reader_source,
    )
    if accessor_match is None:
        raise ValueError("MLP packed-parameter accessor changed")
    total_cbs = int(accessor_match.group(1))
    common_position = len(reader.get("tensor_indices", []))
    if normalized_tensor in reader.get("tensor_indices", []):
        common_position = reader["tensor_indices"].index(normalized_tensor)
    else:
        reader["tensor_indices"].append(normalized_tensor)
    residual_common_position = len(reader.get("tensor_indices", []))
    if residual_tensor in reader.get("tensor_indices", []):
        residual_common_position = reader["tensor_indices"].index(
            residual_tensor
        )
    else:
        reader["tensor_indices"].append(residual_tensor)

    retargeted_signature = re.compile(
        r"template <typename PackedAccessor, typename GateAAccessor, "
        r"typename GateBAccessor, typename UpAAccessor, "
        r"typename UpBAccessor>(\s*static __attribute__\(\("
        r"noinline, noclone,.*?\)\)\s*)void run\(\s*"
        r"const PackedAccessor& packed_parameters,",
        re.DOTALL,
    )
    reader_source, retargeted_count = retargeted_signature.subn(
        r"template <typename PackedAccessor, typename NormalizedAccessor, "
        r"typename GateAAccessor, typename GateBAccessor, "
        r"typename UpAAccessor, typename UpBAccessor>\1void run(\n"
        r"    const PackedAccessor& packed_parameters,\n"
        r"    const NormalizedAccessor& normalized_input,",
        reader_source,
        count=1,
    )
    signature = re.compile(
        r"template <typename PackedAccessor>(\s*static __attribute__\(\("
        r"noinline, noclone,.*?\)\)\s*)void run\(\s*"
        r"const PackedAccessor& packed_parameters\) \{",
        re.DOTALL,
    )
    reader_source, count = signature.subn(
        r"template <typename PackedAccessor, typename NormalizedAccessor>"
        r"\1void run(\n"
        r"    const PackedAccessor& packed_parameters,\n"
        r"    const NormalizedAccessor& normalized_input) {",
        reader_source,
        count=1,
    )
    if count + retargeted_count != 1:
        raise ValueError("MLP gate/up reader signature changed")

    down_signature = re.compile(
        r"template <typename PackedAccessor, typename DownAAccessor, "
        r"typename DownBAccessor>(\s*static __attribute__\(\("
        r"noinline, noclone,.*?\)\)\s*)void run\(\s*"
        r"const PackedAccessor& packed_parameters,\s*"
        r"const DownAAccessor& down_a_parameters,\s*"
        r"const DownBAccessor& down_b_parameters\) \{",
        re.DOTALL,
    )
    reader_source, down_signature_count = down_signature.subn(
        r"template <typename PackedAccessor, typename ResidualAccessor, "
        r"typename DownAAccessor, typename DownBAccessor>\1void run(\n"
        r"    const PackedAccessor& packed_parameters,\n"
        r"    const ResidualAccessor& residual_input,\n"
        r"    const DownAAccessor& down_a_parameters,\n"
        r"    const DownBAccessor& down_b_parameters) {",
        reader_source,
        count=1,
    )
    if down_signature_count != 1:
        raise ValueError("MLP down reader signature changed")

    down_template_marker = (
        "template <typename PackedAccessor, typename ResidualAccessor, "
        "typename DownAAccessor, typename DownBAccessor>"
    )
    down_template_start = reader_source.index(down_template_marker)
    reader_barrier_helper = (
        "static __attribute__((noinline, noclone)) void\n"
        "reader_device_barrier(uint32_t node) {\n"
        "    auto* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        "        get_semaphore(13));\n"
        "    auto* release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        "        get_semaphore(11));\n"
        "    noc_semaphore_set(release, 0);\n"
        "    noc_semaphore_inc(worker_l1(0, get_semaphore(13)), 1);\n"
        "    noc_async_atomic_barrier();\n"
        "    if (node == 0) {\n"
        "        noc_semaphore_wait_min(count, kWorkers);\n"
        "        noc_semaphore_set(count, 0);\n"
        "        noc_semaphore_set(release, 1);\n"
        "        for (uint32_t target = 1; target < kWorkers; ++target)\n"
        "            noc_async_write(\n"
        "                get_semaphore(11),\n"
        "                worker_l1(target, get_semaphore(11)),\n"
        "                sizeof(uint32_t));\n"
        "        noc_async_write_barrier();\n"
        "    }\n"
        "    noc_semaphore_wait(release, 1);\n"
        "    noc_semaphore_set(release, 0);\n"
        "}\n\n"
    )
    reader_source = (
        reader_source[:down_template_start]
        + reader_barrier_helper
        + reader_source[down_template_start:]
    )

    residual_route_start = reader_source.index(
        "    // The projection writer retained one residual row in CB13"
    )
    residual_route_stop_marker = '    WAYPOINT("DRR1");\n'
    residual_route_stop = reader_source.index(
        residual_route_stop_marker, residual_route_start
    ) + len(residual_route_stop_marker)
    direct_residual_load = (
        "    // The compact attention owner retains the pre-MLP residual in\n"
        "    // compiler-owned L1.  Read it directly; the old semaphore-8\n"
        "    // mailbox belonged to the removed monolithic attention writer.\n"
        '    WAYPOINT("DRR0");\n'
        "    for (uint32_t local = 0; local < output_jobs_per_core; ++local) {\n"
        "        const uint32_t job = node + local * kWorkers;\n"
        "        if (job >= output_jobs) continue;\n"
        "        noc0.async_read(\n"
        "            residual_input,\n"
        "            CoreLocalMem<uint32_t>(\n"
        "                residual.get_write_ptr() + local * kTileBytes),\n"
        "            kTileBytes, {.page_id = job}, {});\n"
        "    }\n"
        "    if (valid_outputs) {\n"
        "        noc0.async_read_barrier();\n"
        "        residual.push_back(valid_outputs);\n"
        "    }\n"
        '    WAYPOINT("DRR1");\n'
    )
    reader_source = (
        reader_source[:residual_route_start]
        + direct_residual_load
        + reader_source[residual_route_stop:]
    )

    down_wait_start = reader_source.index(
        "    auto* reader_phase = reinterpret_cast<volatile tt_l1_ptr "
        "uint32_t*>(\n        get_semaphore(5));",
        residual_route_start - 5000,
    )
    down_wait_stop_marker = '    WAYPOINT("DRS1");\n'
    down_wait_stop = reader_source.index(
        down_wait_stop_marker, down_wait_start
    ) + len(down_wait_stop_marker)
    reader_owned_phase_barrier = (
        "    auto* reader_phase = reinterpret_cast<volatile tt_l1_ptr "
        "uint32_t*>(\n        get_semaphore(5));\n"
        "    auto* local_writer_done = reinterpret_cast<volatile tt_l1_ptr "
        "uint32_t*>(\n        get_semaphore(4));\n"
        '    WAYPOINT("DRS0");\n'
        "    while (true) {\n"
        "        invalidate_l1_cache();\n"
        "        if (*local_writer_done >= 1) break;\n"
        "    }\n"
        "    noc_semaphore_set(local_writer_done, 0);\n"
        "    // Reader-owned completion barrier: every reader enters only\n"
        "    // after its co-located writer finished all gate/up routes.  The\n"
        "    // reader RISC's NoC0 60-to-one protocol is already used and\n"
        "    // qualified by begin_generation(); keeping the writer RISC out\n"
        "    // of this barrier avoids Blackhole NoC1/ack cycles.\n"
        "    auto* completion_count = reinterpret_cast<\n"
        "        volatile tt_l1_ptr uint32_t*>(get_semaphore(13));\n"
        "    noc_semaphore_inc(worker_l1(0, get_semaphore(13)), 1);\n"
        "    noc_async_atomic_barrier();\n"
        "    if (node == 0) {\n"
        "        noc_semaphore_wait_min(completion_count, kWorkers);\n"
        "        noc_semaphore_set(completion_count, 0);\n"
        "        noc_semaphore_set(reader_phase, 2);\n"
        "        for (uint32_t target = 1; target < kWorkers; ++target)\n"
        "            noc_async_write(\n"
        "                get_semaphore(5),\n"
        "                worker_l1(target, get_semaphore(5)),\n"
        "                sizeof(uint32_t));\n"
        "        noc_async_write_barrier();\n"
        "    }\n"
        "    while (true) {\n"
        "        invalidate_l1_cache();\n"
        "        if (*reader_phase >= 2) break;\n"
        "    }\n"
        '    WAYPOINT("DRS1");\n'
    )
    reader_source = (
        reader_source[:down_wait_start]
        + reader_owned_phase_barrier
        + reader_source[down_wait_stop:]
    )

    reader_source = reader_source.replace(
        "    uint32_t row_counts[m_tiles] = {};\n", "", 1
    ).replace(
        "        ++row_counts[row];\n", "", 1
    )
    product_fence_start = reader_source.index(
        "    if (valid_products) {\n"
        "        noc_async_write_barrier();",
        reader_source.index('    WAYPOINT("DRP1");'),
    )
    product_fence_stop_marker = '    WAYPOINT("DRP2");\n'
    product_fence_stop = reader_source.index(
        product_fence_stop_marker, product_fence_start
    ) + len(product_fence_stop_marker)
    product_device_fence = (
        "    if (valid_products) {\n"
        "        noc_async_write_barrier();\n"
        "        product_out.pop_front(valid_products);\n"
        "    }\n"
        "    // Every product payload is now visible in its target's fixed\n"
        "    // L1 window. One device-wide reader fence replaces hundreds of\n"
        "    // per-target atomics and their self-target corner case.\n"
        "    reader_device_barrier(node);\n"
        "    if (valid_outputs)\n"
        "        product_resident.push_back(valid_outputs * i_tiles);\n"
        '    WAYPOINT("DRP2");\n'
    )
    reader_source = (
        reader_source[:product_fence_start]
        + product_device_fence
        + reader_source[product_fence_stop:]
    )

    h_fence_start = reader_source.index(
        "        noc_async_write_barrier();\n"
        "        for (uint32_t h = 0; h < h_tiles; ++h)\n"
        "            noc_semaphore_inc(",
        reader_source.index('        WAYPOINT("DRH1");'),
    )
    h_fence_stop_marker = "        h_resident.push_back(valid_outputs);\n    }\n"
    h_fence_stop = reader_source.index(
        h_fence_stop_marker, h_fence_start
    ) + len(h_fence_stop_marker)
    h_device_fence = (
        "        noc_async_write_barrier();\n"
        "        h_multicast.pop_front(1);\n"
        "    }\n"
        "    // The four row owners have published every rank-H tile.\n"
        "    reader_device_barrier(node);\n"
        "    if (valid_outputs)\n"
        "        h_resident.push_back(valid_outputs);\n"
    )
    reader_source = (
        reader_source[:h_fence_start]
        + h_device_fence
        + reader_source[h_fence_stop:]
    )

    route_start = reader_source.index(
        "    volatile tt_l1_ptr uint32_t* x_payload_ready = nullptr;"
    )
    route_stop_marker = '    WAYPOINT("GRX1");\n'
    route_stop = reader_source.index(route_stop_marker, route_start) + len(
        route_stop_marker
    )
    direct_load = (
        "    if (cached_rows) {\n"
        "        x_cache.reserve_back(cached_rows * hidden_tiles);\n"
        "        for (uint32_t row = first_row; row <= last_row; ++row) {\n"
        "            const uint32_t slot = row - first_row;\n"
        "            for (uint32_t k = 0; k < hidden_tiles; ++k)\n"
        "                noc0.async_read(\n"
        "                    normalized_input,\n"
        "                    CoreLocalMem<uint32_t>(\n"
        "                        x_cache.get_write_ptr() +\n"
        "                        (slot * hidden_tiles + k) * kTileBytes),\n"
        "                    kTileBytes,\n"
        "                    {.page_id = row * hidden_tiles + k}, {});\n"
        "        }\n"
        "        noc0.async_read_barrier();\n"
        "        x_cache.push_back(cached_rows * hidden_tiles);\n"
        "    }\n"
        "    noc_semaphore_set(init_ready, 1);\n"
        '    WAYPOINT("GRX1");\n'
    )
    reader_source = (
        reader_source[:route_start] + direct_load + reader_source[route_stop:]
    )
    kernel_start = reader_source.index("__attribute__((optimize(\"Os\"))) void kernel_main()")
    gate_call = "    resident_layer_gate_up_reader::run("
    call_start = reader_source.rindex(gate_call)
    down_call = "    resident_layer_swiglu_down_reader::run("
    down_start = reader_source.index(down_call, call_start)
    down_end = reader_source.index(";", down_start) + 1
    calls = reader_source[call_start:down_end]
    calls = calls.replace(
        "resident_layer_gate_up_reader::run(\n"
        "        layer_parameters, gate_a_parameters, gate_b_parameters,\n"
        "        up_a_parameters, up_b_parameters);",
        "resident_layer_gate_up_reader::run(\n"
        "        layer_parameters, normalized_input, gate_a_parameters,\n"
        "        gate_b_parameters, up_a_parameters, up_b_parameters);",
        1,
    )
    if "normalized_input" not in calls:
        # Non-retargeted packed-state child has the one-argument call.
        calls = calls.replace(
            "resident_layer_gate_up_reader::run(layer_parameters);",
            "resident_layer_gate_up_reader::run(\n"
            "        layer_parameters, normalized_input);",
            1,
        )
    if "normalized_input" not in calls:
        raise ValueError("MLP gate/up call could not be retargeted")
    calls, residual_call_count = re.subn(
        r"resident_layer_swiglu_down_reader::run\(\s*"
        r"layer_parameters,\s*down_a_parameters,\s*"
        r"down_b_parameters\);",
        "resident_layer_swiglu_down_reader::run(\n"
        "        layer_parameters, residual_input, down_a_parameters,\n"
        "        down_b_parameters);",
        calls,
        count=1,
    )
    if residual_call_count != 1:
        raise ValueError("MLP down reader call could not be retargeted")
    normalized_accessor = (
        "    const auto normalized_input = TensorAccessor(TensorAccessorArgs<\n"
        "        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
        f"{normalized_tensor}, {total_cbs}>(), {common_position}>(),\n"
        f"        get_common_arg_val<uint32_t>({common_position}), 2048);\n"
    )
    residual_accessor = (
        "    const auto residual_input = TensorAccessor(TensorAccessorArgs<\n"
        "        tensor_accessor::detail::get_tensor_accessor_args_cta_offset<"
        f"{residual_tensor}, {total_cbs}>(), {residual_common_position}>(),\n"
        "        get_common_arg_val<uint32_t>("
        f"{residual_common_position}), 2048);\n"
    )
    reader_source = (
        reader_source[:kernel_start]
        + "__attribute__((optimize(\"Os\"))) void kernel_main() {\n"
        + "    const uint32_t x = get_absolute_logical_x();\n"
        + "    if (x < 5) return;\n"
        + "    resident_layer_gate_up_reader::begin_generation();\n"
        + normalized_accessor
        + residual_accessor
        + reader_source[
            reader_source.rfind(
                "    const auto layer_parameters = TensorAccessor(",
                kernel_start,
                call_start,
            ):call_start
        ]
        + calls
        + "\n}\n"
    )
    reader["cpp_source"] = reader_source

    compute_source = compute["cpp_source"]
    compute_start = compute_source.index(
        "__attribute__((optimize(\"Os\"))) void kernel_main()"
    )
    decomposed_gate_up = (
        compute_source.find(
            "run_resident_layer_gate_up_compute();", compute_start
        ) >= 0
    )
    gate_up_compute_call = (
        "    run_resident_layer_gate_up_compute();\n"
        if decomposed_gate_up else
        "    run_resident_layer_paired_gate_up_swiglu();\n"
    )
    compute["cpp_source"] = (
        compute_source[:compute_start]
        + "__attribute__((optimize(\"Os\"))) void kernel_main() {\n"
        + "    CircularBuffer invocation_release(28);\n"
        + "    CircularBuffer invocation_acknowledgement(31);\n"
        + "    invocation_release.wait_front(1);\n"
        + "    invocation_release.pop_front(1);\n"
        + gate_up_compute_call
        + "    resident_layer_swiglu_down_compute::run();\n"
        + "    invocation_acknowledgement.reserve_back(1);\n"
        + "    invocation_acknowledgement.push_back(1);\n"
        + "}\n"
    )

    writer_source = writer["cpp_source"]
    writer_barrier_start = writer_source.index('        WAYPOINT("GWS1");\n')
    writer_barrier_stop_marker = '        WAYPOINT("GWS2");\n'
    writer_barrier_stop = writer_source.index(
        writer_barrier_stop_marker, writer_barrier_start
    ) + len(writer_barrier_stop_marker)
    local_writer_publication = (
        '        WAYPOINT("GWS1");\n'
        "        // Publish completion only to the co-located reader.  That\n"
        "        // reader joins the proven all-reader NoC0 barrier before\n"
        "        // any down-phase FIFO is exposed.\n"
        "        noc_semaphore_set(\n"
        "            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        "                get_semaphore(kRouteWriterCount)), 1);\n"
        '        WAYPOINT("GWS2");\n'
    )
    writer_source = (
        writer_source[:writer_barrier_start]
        + local_writer_publication
        + writer_source[writer_barrier_stop:]
    )
    writer_start = writer_source.index(
        "__attribute__((optimize(\"Os\"))) void kernel_main()"
    )
    arena_decl_start = writer_source.index(
        "    const auto arena = TensorAccessor(", writer_start
    )
    arena_decl_end = writer_source.index("    Noc noc1(1);", arena_decl_start)
    arena_declaration = writer_source[arena_decl_start:arena_decl_end]
    writer["cpp_source"] = (
        writer_source[:writer_start]
        + "__attribute__((optimize(\"Os\"))) void kernel_main() {\n"
        + "    const uint32_t x = get_absolute_logical_x();\n"
        + "    if (x < 5) return;\n"
        + "    constexpr uint32_t tile_bytes = 2048;\n"
        + arena_declaration
        + "    Noc noc1(1);\n"
        + "    CircularBuffer invocation_release(28);\n"
        + "    CircularBuffer invocation_acknowledgement(31);\n"
        + "    // The reader owns startup reset of every reused CB epoch.\n"
        + "    // Hold the compute RISC until that reset and the direct L1\n"
        + "    // normalized-input publication are both complete.\n"
        + "    auto* init_ready = reinterpret_cast<\n"
        + "        volatile tt_l1_ptr uint32_t*>(get_semaphore(2));\n"
        + '    WAYPOINT("MWIR");\n'
        + "    while (true) {\n"
        + "        invalidate_l1_cache();\n"
        + "        if (*init_ready == 1) break;\n"
        + "    }\n"
        + '    WAYPOINT("MWI1");\n'
        + "    invocation_release.reserve_back(1);\n"
        + "    invocation_release.push_back(1);\n"
        + "    resident_layer_gate_up_writer::run(noc1, arena);\n"
        + "    resident_layer_gate_up_writer::run_full_mlp_tail(noc1, arena);\n"
        + "    invocation_acknowledgement.wait_front(1);\n"
        + "    invocation_acknowledgement.pop_front(1);\n"
        + "}\n"
    )

    keep = {id(reader), id(compute), id(writer)}
    result["kernels"] = [kernel for kernel in kernels if id(kernel) in keep]
    if len(result["kernels"]) != 3:
        raise RuntimeError("MLP extraction did not retain exactly three kernels")
    return result


def fuse_rmsnorm_residual_child(
    program: dict, *, hidden_tiles: int, residual_tensor_id: int = 6
) -> dict:
    """Append exact BF16 residual add to a compact RMSNorm-backward child.

    RMSNorm's accepted final value is first packed to BF16 CB18.  The added
    operation then consumes that rounded tile and the residual from BF16 CB15,
    packs the exact TTNN add result to CB16, and leaves all reduction/rounding
    points before it unchanged.
    """
    import copy

    result = copy.deepcopy(program)
    if result.get("format_version") not in (1, 2, 3):
        raise ValueError("RMSNorm residual fusion requires a compact child")
    reader = next(
        kernel for kernel in result["kernels"]
        if kernel["kernel_config"]["type"] == "ReaderKernelConfig"
    )
    compute = next(
        kernel for kernel in result["kernels"]
        if kernel["kernel_config"]["type"] == "ComputeKernelConfig"
    )
    writer = next(
        kernel for kernel in result["kernels"]
        if kernel["kernel_config"]["type"] == "WriterKernelConfig"
    )

    if residual_tensor_id in reader["tensor_indices"]:
        raise ValueError("residual tensor ID collides with RMSNorm child ABI")
    residual_position = len(reader["tensor_indices"])
    reader["tensor_indices"].append(residual_tensor_id)
    reader_accessor_marker = "    Noc noc0(0);\n"
    if reader["cpp_source"].count(reader_accessor_marker) != 1:
        raise ValueError("RMSNorm reader Noc marker changed")
    residual_setup = f'''    const uint32_t residual_addr =
        get_common_arg_val<uint32_t>({residual_position});
    const auto residual_tensor = TensorAccessor(
        TensorAccessorArgs<
            tensor_accessor::detail::get_tensor_accessor_args_cta_offset<
                {residual_tensor_id}, total_cbs>(), {residual_position}>(),
        residual_addr, bf16_tile_bytes);

'''
    reader["cpp_source"] = reader["cpp_source"].replace(
        reader_accessor_marker, residual_setup + reader_accessor_marker, 1
    )
    read_barrier = "    noc0.async_read_barrier();\n"
    if reader["cpp_source"].count(read_barrier) != 1:
        raise ValueError("RMSNorm reader barrier marker changed")
    residual_read = '''    CircularBuffer residual(get_compile_time_arg_val(15));
    residual.reserve_back(hidden_tiles);
    for (uint32_t tile = 0; tile < hidden_tiles; ++tile) {
        noc0.async_read(
            residual_tensor,
            CoreLocalMem<uint32_t>(
                residual.get_write_ptr() + tile * bf16_tile_bytes),
            bf16_tile_bytes,
            {.page_id = node * hidden_tiles + tile}, {});
    }
'''
    reader["cpp_source"] = reader["cpp_source"].replace(
        read_barrier, residual_read + read_barrier, 1
    ).replace(
        "    rstd.push_back(hidden_tiles);\n",
        "    rstd.push_back(hidden_tiles);\n"
        "    residual.push_back(hidden_tiles);\n",
        1,
    )

    final_marker = '''        binary_tiles<1, true>(
            cb_direct, cb_correction, cb_output, 1, true);
'''
    if compute["cpp_source"].count(final_marker) != 1:
        raise ValueError("compact RMSNorm final-add marker changed")
    compute["cpp_source"] = compute["cpp_source"].replace(
        "    constexpr uint32_t cb_output = get_compile_time_arg_val(18);\n",
        "    constexpr uint32_t cb_output = get_compile_time_arg_val(18);\n"
        "    constexpr uint32_t cb_residual = get_compile_time_arg_val(15);\n"
        "    constexpr uint32_t cb_residual_output = get_compile_time_arg_val(16);\n",
        1,
    ).replace(
        final_marker,
        final_marker
        + "        exact_bf16_add_tile(\n"
        + "            cb_output, cb_residual, cb_residual_output);\n",
        1,
    )

    writer_marker = "constexpr uint32_t cb_output = get_compile_time_arg_val(18);"
    if writer["cpp_source"].count(writer_marker) != 1:
        raise ValueError("RMSNorm writer output marker changed")
    writer["cpp_source"] = writer["cpp_source"].replace(
        writer_marker,
        "constexpr uint32_t cb_output = get_compile_time_arg_val(16);",
        1,
    )
    for cb_id in (15, 16):
        cb = next(
            item for item in result["cb_configs"]
            if int(item["buffer_index"]) == cb_id
        )
        cb["num_tiles"] = 1
        cb["block_count"] = hidden_tiles
        cb["total_size"] = int(cb["page_size"]) * hidden_tiles
    return result
