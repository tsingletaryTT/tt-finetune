# tt-finetune

`tt-finetune` is a Pythonic parameter-efficient trainer for one Tenstorrent
Blackhole P150a. It presents a Hugging Face/PEFT-style workflow while the
architecture planner, TT-XLA/PJRT runtime, TT-Metal kernels, persistent compile
service, and low-memory frozen-weight implementation stay behind the public API.

## Quick start

```python
from tt_finetune import FastTTModel, TTTrainer

model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    max_seq_length=128,
)
trainer = TTTrainer(
    model=model,
    dataset=[
        {"prompt": "Name a primary color.", "response": "Blue."},
        {"prompt": "What is 2 + 2?", "response": "4"},
    ],
    rank=8,
    max_steps=20,
    learning_rate=2e-4,
    output_dir="./qwen-lora-run",
)
trainer.train()
trainer.save_pretrained("./qwen-adapter")
```

The longer Unsloth-compatible `FastLanguageModel` and `SFTTrainer` interface
remains supported. Standard LoRA, rsLoRA, LoRA+, DoRA, IA3, selected canonical
roles/layers, and independently named adapters share the same trainer surface.

Before compiling a new geometry, query the public static planner:

```python
support = TTTrainer.supports(
    model="Qwen/Qwen2.5-0.5B",
    rank=16,
    batch_size=1,
    sequence_length=2048,
)
print(support.to_dict())
```

The result includes canonical semantics, selected targets and execution path,
memory admission, estimated bytes, and an actionable rejection reason. Its
`evidence` field distinguishes a static prediction from silicon qualification.

Continue training a standard PEFT adapter directly:

```python
model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    adapter_path="./qwen-adapter",
    max_seq_length=128,
)
trainer = TTTrainer(model=model, dataset=dataset, max_steps=500)
trainer.train()
```

## Install

Install the supplied platform wheel. Its dependency metadata installs the
compatible host-side data tools:

```bash
python -m pip install ./dist/tt_finetune-0.2.0-cp312-cp312-linux_x86_64.whl
tt-finetune doctor
```

The platform wheel bundles the compatible TT-XLA/PJRT plugin, TTNN, TT-Metal
user-space libraries, compiler/runtime libraries, Python bindings, custom ops,
and kernel resources. The Python process and `output_dir` may live in any
working directory; no source checkout or `PYTHONPATH` is required. An external
runtime bundle can still be selected explicitly for development, but ordinary
use does not require one and model names never select a backend.

The host must provide the Tenstorrent kernel driver, compatible firmware,
hugepages, and `/dev/tenstorrent` access. The command `tt-finetune doctor`
reports which prerequisite is missing. This project does not claim PyPI
publication.

## Datasets

Pass an in-memory sequence or Hugging Face `Dataset`, a Hub dataset ID, or a
local `.json`, `.jsonl`, `.ndjson`, or text file. The common schemas are
inferred:

- `{"messages": [...]}`
- `{"prompt": "...", "response": "..."}`
- `{"text": "..."}`

Use `DataConfig` for custom field names, deterministic truncation, EOS handling,
sequence packing, length buckets, CPU tokenization workers, and bounded
prefetch. The default `"auto"` policy packs plain causal text only when EOS
separators make that semantic safe; it does not concatenate independent chat
or completion records. The worker creates token IDs, attention masks, minimal
tile-aligned padding, and model-specific positional behavior; users do not
create TT tensors.

`trainer.data_plan()` inspects the source distribution, deterministic epoch
order, packing/padding efficiency, worker pipeline, projected memory, and a
bounded B/S/gradient-accumulation candidate portfolio without initializing the
device. Analytical candidates are clearly marked unmeasured until the
full-step profiler qualifies them.

## Checkpoint and resume

```python
from tt_finetune import TrainingArguments

args = TrainingArguments(
    output_dir="./run",
    max_steps=100,
    save_strategy="steps",
    save_steps=10,
    save_optimizer_state=True,
)
trainer = TTTrainer(model=model, dataset=dataset, args=args)
trainer.train(resume_from_checkpoint=True)
# or trainer.train(resume_from_checkpoint="./run/checkpoints/checkpoint.pt")
```

Checkpoints contain adapter weights, AdamW state, global/micro step, epoch, the
deterministic batch offset, and the exact tokenization/packing/bucketing
contract. Prefetched batches are pure transformations and are reconstructed
from the same epoch plan on resume. A mismatched dataset contract fails with an
actionable error instead of silently continuing on different examples.
`save_pretrained` produces a normal PEFT adapter directory for standard reload.

## Compilation

The first static geometry compiles before steady-state training. User metrics
separate compile estimate from step time and record a stable cache identity.
Lower-level build artifacts can be reused, but PJRT executable deserialization
is not implemented, so a fresh process may still compile. The trainer never
reuses model/optimizer state across isolated processes merely to avoid compile.

For repeated notebook jobs, set `runtime_mode="persistent"` in
`TrainingArguments`. A long-lived local worker may retain compatible compiled
executables, but each job receives fresh model, optimizer, RNG, dropout, and
dataset state. A full service restart still cannot deserialize a complete PJRT
executable; that limitation is reported explicitly rather than hidden.

## Memory modes

`memory_mode="auto"` is the normal default. The planner retains BF16 when it
fits, then may select native BF8 or a mixed BF8/BF4 frozen-weight representation
to admit a larger model while adapter and optimizer state remain in training
precision. Qwen2.5-7B R8/B1/S32 has completed evolving training on one P150a
through this path. Reduced frozen-weight runs use a separate deterministic,
finite numerical contract; they are not claimed bit-exact to BF16.

Use `TTTrainer.supports(...)` before compilation to inspect the selected memory
mode, estimated peak, and an actionable smaller geometry when admission fails.

## Adapter methods

The shared public adapter contract supports:

- LoRA and rsLoRA scaling;
- LoRA+ A/B learning-rate groups;
- DoRA magnitude/direction training;
- IA3 K/V and feed-forward scaling;
- canonical attention, MLP, Q/V, arbitrary-role, and layer targeting;
- independent named adapter load, selection, training, save, inspection, and
  inference merge where standard PEFT supports it.

Prefix/prompt tuning and data-dependent simultaneous adapter routing fail
closed because they change the resident attention or graph-state ABI. They are
not approximated by checkpoint-specific wrappers.

See the [installation guide](docs/INSTALL.md), [supported
configurations](docs/SUPPORTED.md), [known limitations](docs/LIMITATIONS.md),
[benchmarks](docs/BENCHMARKS.md), and [roadmap](docs/PRODUCT_ROADMAP.md).
Runnable examples include
[minimal LoRA](examples/minimal_lora.py),
[dataset/checkpoint/resume](examples/dataset_and_resume.py), and
[standard PEFT reload](examples/adapter_reload.py), and
[rsLoRA/LoRA+/DoRA/IA3 configuration](examples/adapter_methods.py).
