# Troubleshooting

Start with:

```bash
tt-finetune doctor --json
```

## Device not found

Confirm the Tenstorrent driver is loaded, `/dev/tenstorrent` is visible to the
current user, firmware is compatible, and required hugepages are mounted.

## Runtime not found

Reinstall the complete platform wheel and verify that its `_runtime` resources
were not removed by an archive or antivirus tool. `TT_FINETUNE_WORKSPACE` is an
advanced development override for a separately qualified runtime bundle; it is
not required for the normal wheel. Do not set `PYTHONPATH` to a development
checkout.

## Out of memory

Keep `memory_mode="auto"`, then reduce sequence length, batch size, rank, or
model size. `TTTrainer.supports(...)` reports the estimated peak and a smaller
admitted geometry where one is known.

## Unsupported architecture or target

Inspect the capability error and canonical target list. Model names never
force a backend. Unknown attention, position, MLP, expert, or adapter semantics
fail closed.

## Compilation failure

Check runtime/compiler compatibility and available host memory, invalidate a
stale cache when the reported source contract changed, and rerun with
`verbose=True`. Compilation time is expected on the first new geometry.

## A fresh process compiles again

This is expected when the lower-level cache cannot satisfy the request. Current
PJRT cannot deserialize a complete executable across processes. Use persistent
runtime mode for safe same-process reuse.

## Another training job is already active

A P150a persistent runtime admits one training owner. Finish or interrupt the
reported job before starting another. If the owner crashes, its connection
lease terminates the package-owned runtime so the card is released; resume from
the latest completed checkpoint. Do not start a second service or delete its
socket while the reported owner is still alive.

## TLB-window allocation failure

Only one process may own the P150a mappings used by TT-Metal. Current
`tt-finetune` releases an idle package-owned persistent runtime before an
isolated profiling or memory-measurement job and refuses to interrupt an active
training owner. If `tt_tlb_alloc` still fails, check for an unrelated TT
process using the device, finish that job, and retry. Do not delete runtime
sockets or kill an active owner without first preserving its checkpoint.

## Resume rejected

The checkpoint data or model contract differs from the current run. Restore the
same tokenizer, source fingerprint, packing/bucketing settings, rank, targets,
optimizer configuration, and model revision, or start a new run.
