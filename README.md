# tt-finetune preview

`tt-finetune` is a Python library for parameter-efficient causal-language-model
training on one Tenstorrent Blackhole P150a. It provides a familiar
Hugging Face-style API while the package handles architecture discovery,
target selection, memory admission, compilation, execution planning, data
preparation, checkpointing, and PEFT export.

The preview wheel includes its compatible Tenstorrent user-space stack. It
does not require a TT-Metal, TT-XLA, or TT-MLIR source checkout and it does not
use a container at runtime. The host still needs the Tenstorrent kernel driver,
device permissions, compatible firmware, and hugepages.

```python
from tt_finetune import FastTTModel, TTTrainer

model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    max_seq_length=128,
)

trainer = TTTrainer(
    model=model,
    dataset="train.jsonl",
    rank=8,
    max_steps=20,
    output_dir="./run",
)

trainer.train()
trainer.save_pretrained("./adapter")
```

Start with [QUICKSTART.md](QUICKSTART.md). See [SUPPORTED.md](SUPPORTED.md)
before using an architecture not listed as qualified, and run
`tt-finetune doctor` after installation. The exact release checks and artifact
identity are recorded in [QUALIFICATION.md](QUALIFICATION.md).
