# Quick start

Use CPython 3.12 on a Linux x86-64 host with an accessible Blackhole P150a.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ./dist/tt_finetune-0.2.0-cp312-cp312-linux_x86_64.whl
tt-finetune doctor --smoke
python examples/quickstart.py
```

The first use of a new model geometry compiles the training graph. The CLI
and trainer distinguish compilation from steady-state training. The demo
writes the normal PEFT adapter to `./quickstart-output/adapter`.

The model download requires Hugging Face network access unless it is already
cached. Set `HF_TOKEN` if the selected model is gated.

