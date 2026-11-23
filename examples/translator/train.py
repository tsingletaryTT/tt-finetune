"""LoRA-tune SmolLM2-135M on the translator data with tt-finetune (one leased chip)."""
import sys
from pathlib import Path
from tt_finetune import FastTTModel, TTTrainer

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "translator-run"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200

model = FastTTModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", max_seq_length=128)
trainer = TTTrainer(
    model=model,
    dataset=HERE / "train.jsonl",
    rank=16,
    max_steps=STEPS,
    learning_rate=2e-4,
    logging_steps=25,
    output_dir=str(OUT / "run"),
)
print(trainer.preflight(), flush=True)
trainer.train()
print("adapter:", trainer.save_pretrained(OUT / "adapter"), flush=True)
