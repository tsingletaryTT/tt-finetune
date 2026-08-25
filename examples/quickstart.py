from pathlib import Path

from tt_finetune import FastTTModel, TTTrainer


HERE = Path(__file__).resolve().parent
OUTPUT = Path.cwd() / "quickstart-output"

model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    max_seq_length=128,
)

trainer = TTTrainer(
    model=model,
    dataset=HERE / "train.jsonl",
    rank=8,
    max_steps=20,
    logging_steps=1,
    output_dir=str(OUTPUT / "run"),
)

print(trainer.preflight())
trainer.train()
print("adapter:", trainer.save_pretrained(OUTPUT / "adapter"))

