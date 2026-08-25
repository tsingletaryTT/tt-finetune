# Troubleshooting

## Device not found

Run `tt-finetune doctor`. If `/dev/tenstorrent` is missing, install or load the
compatible Tenstorrent kernel driver and verify firmware compatibility.

## Permission denied

The current user needs read/write access to the Tenstorrent character device.
Fix the host udev/group policy, log in again, and rerun the doctor command.

## Hugepages missing

Configure and mount the hugepages required by the Tenstorrent runtime. The
doctor accepts `/dev/hugepages-1G` or `/dev/hugepages`.

## Unsupported Python or host

This wheel requires CPython 3.12, Linux x86-64, and glibc 2.34 or newer. Create
a new CPython 3.12 virtual environment rather than forcing installation into a
different interpreter.

## Compilation failure

First run `tt-finetune doctor --smoke`. Then check that the model is supported:

```bash
tt-finetune supports MODEL_ID --rank 8 --batch-size 1 --sequence-length 128
```

Use `verbose=True` only when collecting diagnostics. Ordinary operation keeps
the internal compiler log quiet.

## Out of memory

Reduce sequence length, batch size, LoRA rank, or targets. For a qualified
large dense model, try `memory_mode="bf8_frozen"` or
`memory_mode="bf4_frozen"`. These modes change the frozen-weight numerical
contract and are not QLoRA.

## A compile repeats in a new process

This is expected when the executable is not alive in the persistent runtime.
The bundled PJRT does not safely deserialize executables in a fresh process.
Use `runtime_mode="persistent"` for repeated compatible jobs in one service;
measured execution plans remain reusable through the semantic cache.

## Cache invalidation

Plans are keyed by architecture semantics, tensor geometry, adapter topology,
precision/memory mode, optimizer, data geometry, and runtime fingerprint.
Changing any of these creates a new cache identity. Inspect entries with
`tt-finetune cache`.

