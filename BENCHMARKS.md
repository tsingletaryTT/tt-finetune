# Benchmarks

All numbers below were measured on one Tenstorrent Blackhole P150a. Steady
timings exclude compilation and warmup and use synchronized complete training
steps. Results are workload-specific and are not universal speedup claims.

## Resident dense training

| Model and geometry | Median | p10-p90 | Throughput |
|---|---:|---:|---:|
| SmolLM2-135M, LoRA R8, B1, S128, BF16 | 67.125 ms | 65.559-69.515 ms | 1,906.9 tokens/s |
| Qwen2.5-0.5B, LoRA R16, B1, S128, BF16 | 134.408 ms | 134.217-134.558 ms | 952.3 tokens/s |

Each result used four warmups and 20 measured steps. Both models separately
passed 20 evolving optimizer steps with exact loss, gradients, parameters,
and AdamW state against their accepted reference contract.

## Public API representative models

| Semantics | Geometry | Median | p10-p90 |
|---|---|---:|---:|
| GPT-2 fused Conv1D + LayerNorm | R8/B1/S32/BF16 | 42.518 ms | 42.078-42.934 ms |
| Pythia stable LayerNorm | R8/B1/S32/BF16 | 674.802 ms | 649.845-678.129 ms |
| Qwen RMSNorm + GQA + SwiGLU | R8/B1/S32/BF16 | 583.914 ms | 576.980-591.141 ms |

These are matched 20-step steady-state authorities from the public training
path, not isolated kernel measurements.

## Large-model low-memory schedule

Qwen2.5-7B, LoRA R8, B1, S32, `q_proj` + `v_proj`, deterministic mixed
block-float frozen weights:

- median: 455.097 ms
- p10-p90: 453.561-456.367 ms
- throughput: 70.315 tokens/s
- four warmups and 20 measured steps
- two independent 20-step evolving runs reproduced loss, trainable tensors,
  and logical optimizer state exactly inside the selected reduced-weight
  contract

The accepted schedule is 112.7 times faster than a historical 51.293-second
path that materialized 197 saved frozen-weight views at an unnecessary graph
boundary. That ratio demonstrates recovery from a pathological boundary; it
must not be interpreted as expected acceleration for arbitrary models.

## Compilation reuse

On a tiny GPT-2 qualification, the first compiled step took 41.797 seconds.
Reusing the executable in the same persistent process reduced the next
equivalent first step to 1.125 seconds, saving 40.672 seconds. Cross-process
executable deserialization is not claimed.

## Data pipeline

For a mixed-length tiny GPT-2 workload at R4/B1/S128, deterministic sequence
bucketing, packing, bounded tokenization, and prefetch raised useful-token
throughput from 630.229 to 1,860.442 tokens/s (2.952x). Four warmups and 16
steps were measured. Interrupted and resumed 20-step runs produced identical
training-state hashes.

