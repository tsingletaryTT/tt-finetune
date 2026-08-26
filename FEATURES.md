# Features

## Adapter training

- LoRA and rsLoRA
- LoRA+ optimizer groups
- DoRA
- IA3
- automatic canonical target discovery or explicit target selection
- named adapter load, selection, training, save, inspection, and supported merge
- standard PEFT adapter export and reload

## Training

- Hugging Face model IDs and local checkpoints
- automatic architecture discovery and Blackhole memory admission
- BF16 and qualified reduced frozen-weight memory modes
- deterministic checkpoint/resume, including optimizer and data position
- gradient accumulation, clipping, AdamW, logging, and checkpoint cadence
- dense causal LMs and the documented sparse-MoE contract, including
  capacity-free device routing into selected 32-row expert blocks

## Data pipeline

- in-memory examples, Hugging Face datasets, JSON, JSONL, NDJSON, and text
- tokenizer loading, truncation, attention masks, and position behavior
- deterministic shuffling and exact data-position resume
- safe causal-text packing, sequence buckets, bounded parallel tokenization,
  and DataLoader prefetch
- useful-token throughput and padding-efficiency reporting

## Compilation and planning

- static capability and memory query before compilation
- bounded measured full-step execution-plan search
- semantic measured-plan cache
- compile buckets and same-process executable reuse
- explicit compilation versus training timing
- safe static fallback whenever tuning finds no valid improvement
