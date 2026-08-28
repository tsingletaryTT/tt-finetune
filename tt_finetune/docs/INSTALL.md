# Installation

`tt-finetune` targets a single Tenstorrent Blackhole P150a on 64-bit Linux.
Install the supplied wheel into a clean Python virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./tt_finetune-0.2.1rc1-cp312-cp312-linux_x86_64.whl
tt-finetune doctor
```

The final wheel filename may include a more specific platform tag. Use the
filename present in the release `dist/` directory.

The host must provide the Tenstorrent kernel driver, access to
`/dev/tenstorrent`, compatible firmware, and the configured hugepage mount.
The wheel carries the matching TT-XLA/PJRT, TTNN, TT-Metal user-space, compiler,
and kernel resources. Run `tt-finetune doctor --json` to see the exact bundled
paths and every missing system prerequisite.

The optional Hugging Face data and adapter tools are pinned in the release
compatibility lock. The project is not claimed to be published on PyPI.

## Verify the installation

From a directory outside the source checkout:

```bash
python -c 'import tt_finetune; print(tt_finetune.__version__)'
tt-finetune doctor
python quickstart.py
```

No `PYTHONPATH`, source checkout, or repository-relative working directory is
required by the Python product layer.
