# Features

## Public training workflow

- `FastTTModel.from_pretrained()` for Hub IDs and local checkpoints
- `TTTrainer` with automatic architecture, target, memory, and execution plans
- local JSON, JSONL, text, Python row lists, and Hugging Face datasets
- deterministic tokenization, shuffling, sequence buckets, and safe causal
  packing
- bounded parallel tokenization, DataLoader prefetch, and useful-token
  throughput reporting
- checkpoints with optimizer, progress, and data-pipeline state
- deterministic continuation under the supported execution contract

## Adapter methods

- LoRA
- rsLoRA
- LoRA+
- DoRA
- IA3 for semantically valid separate projections
- semantic target policies and explicit PEFT-compatible target names
- layer selection and named adapter lifecycle
- standard PEFT save, inspect, reload, continue-training, and dense merge

## Blackhole execution

- BF16 training
- BF8 and BF4 storage for frozen weights in qualified low-memory paths
- native resident Transformer schedules for qualified geometries
- packed exact optimizer and clipping paths
- architecture- and geometry-derived planning without model-name kernel routing
- measured plan selection and semantic plan cache
- compile buckets and safe same-process executable reuse
- serialized device profiling on a single P150a

## Sparse models

- qualified bounded sparse-expert execution
- routed and shared expert adapter state
- lossless TT extension data beside the standard PEFT component when sparse
  expert tensors cannot be represented by ordinary PEFT alone

