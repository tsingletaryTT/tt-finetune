from pathlib import Path

from tt_finetune import FastTTModel, TTTrainer, TrainingArguments


here = Path(__file__).resolve().parent
model = FastTTModel.from_pretrained(
    "Qwen/Qwen2.5-0.5B", max_seq_length=128
)
args = TrainingArguments(
    output_dir="./resume-run",
    max_steps=20,
    save_strategy="steps",
    save_steps=5,
    logging_steps=1,
)
trainer = TTTrainer(model=model, dataset=here / "train.jsonl", rank=8, args=args)

# Simulate an interruption after ten optimizer steps, then continue from the
# latest complete checkpoint including optimizer and data-pipeline state.
trainer.train(stop_after=10)
trainer.train(resume_from_checkpoint=True)
trainer.save_pretrained("./resumed-adapter")

