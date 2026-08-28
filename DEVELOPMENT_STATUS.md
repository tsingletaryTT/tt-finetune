# Development status after v0.2.0

This document summarizes the compiler/runtime milestone completed after the
public `v0.2.0` release. The release tag and wheel remain unchanged so existing
links, installs and external evaluations continue to refer to the exact build
that was originally published.

## Milestone result

The deterministic execution/compiler stage completed with:

- SmolLM2 R8/B1/S128: 60.828 ms median synchronized step;
- Qwen2.5-0.5B R16/B1/S128: 125.493 ms, improved from 134.341 ms;
- Granite sparse MoE E32/K8 R8/B1/S128: 4.673 s, improved from 21.021 s;
- Qwen2.5-7B mixed BF8/BF4 R8/B1/S32: 455.097 ms after removing the
  pathological materialized forward/backward boundary;
- 108/108 owned regressions passing.

The compiler selects execution from canonical model semantics, tensor geometry,
R/B/S, adapter topology, precision/memory policy and P150a hardware properties.
Normal training does not benchmark candidate plans online, and model/repository
names do not select kernels.

## Generalization

The qualification matrix now covers MHA/GQA/MQA, separate and fused QKV,
RMSNorm/LayerNorm/QK normalization, SwiGLU and ordinary GELU MLPs, irregular
rank/batch/sequence geometry, long-sequence execution, compressed frozen
weights and materially different sparse-MoE layouts.

Independent semantic holdouts include:

- Mixtral-style 8-expert/top-2 routing;
- Qwen2MoE-style routing with a gated shared expert;
- Falcon-style MQA with asymmetric fused QKV.

These holdouts exposed reusable compiler defects that were fixed without
checkpoint-name routing, including compiled shared-expert control flow and
fused-QKV backward scatter lowering.

## Rejected experiments remain rejected

Several attractive microbenchmarks were deliberately not promoted:

- a Smol full-layer owner is lifecycle-stable but numerically incorrect in the
  down-projection continuation;
- a larger exact Qwen fused-attention program reduces program count but is
  slower end-to-end;
- short Phi native streamed attention is extremely fast in isolation but slower
  and numerically different in the complete step;
- packed gradient clipping gives measurable latency wins but changes the
  accepted optimizer-state contract.

The acceptance rule remains complete synchronized step plus correctness, not
kernel speed or program-count reduction alone.

## Next major target

The next milestone is compile/startup closure: precompile model-independent
pieces, define reusable execution archetypes, build persistent compiled-artifact
coverage offline, and reduce normal-user JIT toward zero. That work is intended
to lead directly into multi-P150a execution.

For installation and reproducibility today, continue to use the unchanged
`v0.2.0` release until the next wheel is built and clean-environment qualified.
