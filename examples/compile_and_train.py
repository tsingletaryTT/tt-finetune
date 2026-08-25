from pathlib import Path

from tt_finetune import FastTTModel, TTTrainer


here = Path(__file__).resolve().parent
model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B", max_seq_length=128
)
trainer = TTTrainer(
    model=model,
    dataset=here / "train.jsonl",
    rank=8,
    max_steps=20,
    output_dir="./compiled-run",
    auto_tune=True,
)

print(trainer.plan())
print(trainer.compile())
trainer.train()
trainer.save_pretrained("./compiled-adapter")

