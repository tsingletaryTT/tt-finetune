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

## Resume rejected

The checkpoint data or model contract differs from the current run. Restore the
same tokenizer, source fingerprint, packing/bucketing settings, rank, targets,
optimizer configuration, and model revision, or start a new run.
