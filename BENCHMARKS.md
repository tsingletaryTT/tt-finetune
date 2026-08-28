# Benchmarks

All reported performance uses one Tenstorrent Blackhole P150a. Compilation is
excluded. Unless an artifact states otherwise, measurements use four warmups
followed by at least twenty synchronized complete training steps.

> **Release note:** the public `v0.2.0` wheel remains unchanged. The results in
> the development section below were qualified after that release and are being
> prepared for the next repository/release update.

## Current development authorities

| Workload | Previous authority | Current development authority | Change |
| --- | ---: | ---: | ---: |
| SmolLM2-style GQA, R8/B1/S128 | 67.143 ms | 60.828 ms | 9.41% faster |
| Qwen2.5-0.5B, R16/B1/S128 | 134.341 ms | 125.493 ms | 6.59% faster |
| Granite sparse MoE, 32 experts/top-8, R8/B1/S128 | 21.021 s | 4.673 s | 4.50x / 77.77% lower latency |
| Qwen2.5-7B mixed BF8/BF4, R8/B1/S32 | 51.293 s | 455.097 ms | 112.71x cliff recovery |

The 7B result removes a pathological materialized forward/backward boundary and
is intentionally reported separately from normal model-wide speedup claims.

The accepted Smol result is 60.828 ms (p10–p90 58.483–64.838 ms). A larger
single-owner experiment was lifecycle-stable but numerically invalid in the
MLP down-projection continuation, so it remains disabled rather than being
promoted for a headline sub-60 result.

The accepted Qwen2.5-0.5B result is 125.493 ms (p10–p90 125.413–125.629 ms).
A larger exact fused-attention candidate reduced physical program count but
measured 130.481 ms end-to-end, so the deterministic compiler correctly keeps
the faster safe resident path.

## Sparse MoE

Granite-3.1-1B-A400M, rank 8, batch 1, sequence 128, 24 routed layers, 32
experts and top-8 routing now uses a geometry-selected
`dynamic_selected_block_batched` schedule. Routing, packing, selected-expert
execution and restoration remain on device; there is no host token routing
loop, all-expert token evaluation, or capacity/drop shortcut.

| Complete synchronized step | Median | p10 | p90 |
| --- | ---: | ---: | ---: |
| Previous selected-block sparse schedule | 21.021 s | 20.964 s | 21.075 s |
| Current selected block-batched schedule | 4.673 s | 4.664 s | 4.680 s |

The current phase medians are approximately 2.709 s forward, 1.345 s backward,
452.6 ms gradient clipping, 155.3 ms optimizer and 4.7 ms cross entropy.

A packed-gradient-clipping experiment reached 4.103 s, but changed 88/132
AdamW state tensors after the first clipped update. It is frozen and not
compiler-selectable. Full-step latency and evolving-state correctness are both
required for promotion.

## Architecture and holdout coverage

The current compiler/runtime qualification matrix covers:

- MHA, GQA and MQA attention;
- separate QKV, fused QKV and asymmetric fused-QKV MQA;
- RMSNorm, LayerNorm and Q/K normalization;
- separate/fused SwiGLU and ordinary GELU MLPs;
- adapter ranks 8, 16, 32 and 73;
- batches 1, 2 and 3;
- sequences 32, 96, 128, 256 and qualified long-sequence paths through 4096;
- BF16 and mixed BF8/BF4 frozen-weight execution;
- sparse layouts including 8 experts/top-2, gated shared experts, and 32
  experts/top-8.

Independent semantic holdouts include Mixtral-style routed experts,
shared-expert Qwen2MoE, and Falcon-style MQA with asymmetric fused QKV. These
holdouts found and fixed generic compiler defects rather than model-name
special cases, including shared-expert compiled control-flow specialization and
an additive fused-QKV backward scatter that TT-MLIR misclassified as a paged
cache update.

The current owned regression gate is 108/108 passing tests.

## Large-model low-memory boundary recovery

Qwen2.5-7B, rank 8, batch 1, sequence 32, using the deterministic mixed BF8/BF4
frozen-weight contract:

| Schedule | Median step |
| --- | ---: |
| Materialized forward/backward boundary | 51.293 s |
| Reduced-weight fused boundary | 455.097 ms |

The 112.71x result removes a pathological frozen-weight phase boundary. It is
reported separately and is not a general speedup expectation. Independent
fresh-process runs remain finite and deterministic under the explicit reduced
frozen-weight contract.

## Data pipeline

On the accepted packed text workload, deterministic bucketing, safe packing,
bounded tokenization, and prefetch improved useful-token throughput by 2.952x
against the matched unplanned pipeline while preserving exact resume.

## Measurement policy

Performance claims are based on synchronized complete training steps. Isolated
kernel or component timings are diagnostic only. A candidate is not promoted
because it has fewer physical programs or a faster primitive: it must preserve
the required numerical/evolving-state contract and improve the matched
end-to-end step.

The public `v0.2.0` release remains the installation authority until the next
wheel is built and clean-environment qualified.
