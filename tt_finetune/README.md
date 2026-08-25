# tt-finetune

`tt-finetune` provides a familiar Python API for LoRA and related PEFT
training on a single Tenstorrent Blackhole P150a.  This preview wheel bundles
the compatible Tenstorrent user-space runtime; the host kernel driver,
firmware, device access and hugepages remain system prerequisites.

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
)
trainer.train()
trainer.save_pretrained("./adapter")
```

Run `tt-finetune doctor` after installation.  Use
`tt-finetune doctor --smoke` for a tiny real compile and device execution.
