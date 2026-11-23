# Feasibility test: does a 7B coder LoRA step compile, fit and run at a given sequence length?
# Usage: train7b.py SEQ STEPS OUT_DIR
import sys, time
from pathlib import Path
from tt_finetune import FastTTModel, TTTrainer
SEQ, STEPS, OUT = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
DATA = "/tmp/claude-1000/-home-ttuser-code-tt-finetune/3a465160-5228-49e4-96b1-fc069b82ae50/scratchpad/long_text.jsonl"
t0 = time.time()
model = FastTTModel.from_pretrained((sys.argv[5] if len(sys.argv) > 5 else "Qwen/Qwen2.5-Coder-7B-Instruct"), max_seq_length=SEQ)
print("loaded", round(time.time() - t0), "s", flush=True)
trainer = TTTrainer(model=model, dataset=DATA, rank=16, max_steps=STEPS, learning_rate=1e-4, logging_steps=1,
                    memory_mode=(sys.argv[4] if len(sys.argv) > 4 else "auto"), output_dir=str(OUT / "run"), workspace=str(OUT / "workspace"), verbose=True)
print(trainer.preflight(), flush=True)
trainer.train()
print("DONE", round(time.time() - t0), "s", flush=True)
