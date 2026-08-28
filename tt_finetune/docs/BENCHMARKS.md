# Benchmarks

All reported performance uses one Tenstorrent Blackhole P150a. Compilation is
excluded. Unless an artifact states otherwise, measurements use four warmups
followed by at least twenty synchronized complete training steps.

## Fixed-geometry execution-engine qualification

| Model semantics | Rank / batch / sequence | Median step |
| --- | ---: | ---: |
| SmolLM2-style GQA | 8 / 1 / 128 | 60.828 ms |
| Qwen2-style GQA | 16 / 1 / 128 | 125.493 ms |

These fixed-shape resident-runtime paths retain the accepted exact
evolving-state contract, including loss, LoRA gradients and parameters,
clipping, and AdamW moments. They are backend qualification results, not a
claim that every architecture-general public `TTTrainer` workload selects the
same packed resident ABI. Real dataset, bucketing, checkpointing, logging and
public-worker measurements are reported separately.

## Real public-API workloads

These runs use a real instruction dataset, automatic target discovery,
deterministic sequence planning, checkpointing, and the normal
`FastTTModel`/`TTTrainer` interface. Compilation is excluded from the step
statistics.

| Model | Rank / batch / sequence | Steps | Median | p10–p90 | Useful tokens/s |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 8 / 1 / 128 | 5,000 | 575.919 ms | 494.918–582.111 ms | 173.981 |
| TinyLlama-1.1B | 16 / 1 / 256 | 1,000 | 659.430 ms | 624.465–671.340 ms | 250.857 |
| Qwen2.5-7B, BF4 frozen weights | 8 / 1 / 32 | 100 | 442.052 ms | 439.617–444.786 ms | 72.263 |
| Granite-3.1-1B-A400M, 32 experts/top-8 | 8 / 1 / 128 | 20 | 4.673 s | 4.664–4.680 s | 27.39 |

The small-model distribution includes real periodic checkpoint work in its
long-run tail. The medium workload improved held-out mean NLL from 1.21752 to
1.17856 and changed all six deterministic generation probes. The large run
uses the published deterministic reduced-frozen-weight contract. The sparse
run uses the accepted device-selected, block-batched schedule. It reduced the
previous matched 21.021-second median to 4.673 seconds (4.50x) without
evaluating all experts or using a host token-routing loop. Sparse-MoE
utilization remains a documented performance target.

## Large-model low-memory boundary recovery

Qwen2.5-7B, rank 8, batch 1, sequence 32, Q/V targets, using the deterministic
mixed BF8/BF4 frozen-weight contract:

| Schedule | Median step |
| --- | ---: |
| Materialized forward/backward boundary | 51.293 s |
| Reduced-weight fused boundary | 455.097 ms |

The 112.71x result removes a pathological frozen-weight phase boundary. It is
reported separately and is not a general speedup expectation. Two independent
fresh-process 20-step runs had finite and identical losses, trainable state,
and logical optimizer state under the explicit reduced-weight contract.

## Data pipeline

On the accepted packed text workload, deterministic bucketing, safe packing,
bounded tokenization, and prefetch improved useful-token throughput by 2.952x
against the matched unplanned pipeline while preserving exact resume.

The release evidence directory contains the machine-readable geometry table,
static/automatic/oracle comparisons, p10–p90 distributions, tuning time, cache
status, and correctness contract for each planner qualification case.
