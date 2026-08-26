# Benchmarks

All reported performance uses one Tenstorrent Blackhole P150a. Compilation is
excluded. Unless an artifact states otherwise, measurements use four warmups
followed by at least twenty synchronized complete training steps.

## Representative exact BF16 training

| Model semantics | Rank / batch / sequence | Median step |
| --- | ---: | ---: |
| SmolLM2-style GQA | 8 / 1 / 128 | 67.143 ms |
| Qwen2-style GQA | 16 / 1 / 128 | 134.232 ms |

These paths retain the accepted exact evolving-state contract, including loss,
LoRA gradients and parameters, clipping, and AdamW moments.

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
