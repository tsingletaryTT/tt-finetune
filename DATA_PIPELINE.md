# Data pipeline

Pass a Python sequence, Hugging Face dataset object or ID, local JSON/JSONL,
NDJSON, or plain-text file. Common `messages`, `prompt`/`response`, and `text`
records are detected automatically. `DataConfig` exposes custom columns,
tokenizer behavior, truncation, packing, length buckets, worker count,
prefetch, and deterministic seed.

The automatic packing policy joins only plain causal text separated by EOS. It
does not concatenate independent chat or completion records when doing so
would change masking semantics. The worker creates token IDs, masks, position
behavior, and the smallest admitted static bucket; users do not build
Blackhole tensors.

Checkpoint/resume records the source fingerprint, seed, epoch, exact batch
position, packing policy, buckets, and tokenizer contract. A mismatch fails
with an actionable error rather than resuming on a different data stream.

`trainer.data_plan()` reports useful-token fraction, padding waste, candidate
batch/sequence/accumulation geometries, projected memory, and the selected
pipeline. Training metrics include raw step latency and useful tokens per
second.
