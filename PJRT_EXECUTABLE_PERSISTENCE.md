# PJRT executable persistence gap

## Observed behavior

`tt-finetune` can retain a loaded executable in its package-owned persistent
runtime and safely reuse it for compatible jobs in the same PJRT process. A
fresh process can reuse TT-Metal kernel artifacts and the semantic measured-plan
cache, but it still rebuilds the complete PJRT executable.

The public workload metrics expose this directly:

- `compile.cross_process_executable_reuse` is `false`;
- `compile.bucket_first_use` remains non-zero in a fresh process using the same
  lower-level cache;
- compatible same-process jobs report a persistent-runtime cache hit while
  receiving fresh model, optimizer, RNG, and dataset state.

This is a startup limitation, not a training-state cache. `tt-finetune` does not
pretend that a lower-level kernel-cache hit is a deserialized model executable.

## Minimal missing primitive

The bundled TT PJRT plugin contains its executable serialization handler
(`tt::pjrt::internal::onExecutableSerialize`) and a serialized executable image
type. The same binary binds `PJRT_Executable_DeserializeAndLoad` through its
undefined-operation path instead of a TT implementation. Therefore Torch-XLA's
persistent compilation cache has no backend operation that can reconstruct and
load the serialized TT executable in a new client process.

The upstream fix is a real TT implementation of:

```text
PJRT_Executable_DeserializeAndLoad
```

It must validate at least the TT-XLA/TT-MLIR/TT-Metal build contract, Blackhole
architecture, mesh/device assignment, compiler options, executable ABI, and
resource metadata before loading. A format or runtime mismatch must fail closed
and trigger normal compilation; it must never load an incompatible image.

## Reproducer

Run [`blackhole_training/reproduce_pjrt_persistence.sh`](blackhole_training/reproduce_pjrt_persistence.sh)
from a clean wheel environment with a local SmolLM2 snapshot and the supplied
training JSONL. It launches two fresh Python/PJRT processes with the same
semantic geometry and lower-level cache contract. Both results retain a
non-zero `estimated_compile_seconds`; the second may be faster because TT-Metal
artifacts are reusable, but it is not a complete-executable cache hit.

The same script records plugin-symbol evidence without modifying the runtime.
No workaround that reuses optimizer/model state or an ownerless service is an
acceptable substitute for this missing PJRT primitive.
