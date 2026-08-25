from pathlib import Path

from tt_finetune import FastTTModel, TTTrainer, inspect_adapter


here = Path(__file__).resolve().parent
model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B", max_seq_length=128
)
trainer = TTTrainer(
    model=model,
    dataset=here / "train.jsonl",
    rank=8,
    max_steps=2,
    output_dir="./export-run",
)
trainer.train()
path = trainer.save_pretrained("./peft-adapter")
print(inspect_adapter(path))

# Standard PEFT metadata is enough to construct a continued-training plan.
continued_model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    max_seq_length=128,
    adapter=path,
)
print(continued_model.inspect_adapters())

