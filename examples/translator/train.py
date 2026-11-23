"""LoRA-tune SmolLM2-135M on the translator data with tt-finetune (one chip).

Usage: train.py OUT_DIR [STEPS=1200] [RANK=16] [LR=2e-4]
"""
import os
import sys
from pathlib import Path
from tt_finetune import FastTTModel, TTTrainer

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "translator-run"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
RANK = int(sys.argv[3]) if len(sys.argv) > 3 else 16
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-4

model = FastTTModel.from_pretrained("HuggingFaceTB/SmolLM2-135M", max_seq_length=128)
trainer = TTTrainer(
    model=model,
    dataset=HERE / "train.jsonl",
    rank=RANK,
    max_steps=STEPS,
    learning_rate=LR,
    logging_steps=25,
    verbose=bool(os.environ.get("TT_FT_VERBOSE")),  # show the worker subprocess output on failure
    output_dir=str(OUT / "run"),
    workspace=str(OUT / "workspace"),  # per-run: concurrent jobs must not share the default manifest path
)
print(trainer.preflight(), flush=True)
trainer.train()
print("adapter:", trainer.save_pretrained(OUT / "adapter"), flush=True)
