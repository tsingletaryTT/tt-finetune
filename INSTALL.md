# Installation

`tt-finetune` targets a single Tenstorrent Blackhole P150a on 64-bit Linux.
Install the current platform wheel into a clean CPython 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install https://github.com/danielisraeli2409-jpg/tt-finetune/releases/download/v0.2.1/tt_finetune-0.2.1-cp312-cp312-linux_x86_64.whl
tt-finetune doctor --smoke
```

Release page: <https://github.com/danielisraeli2409-jpg/tt-finetune/releases/latest>

The host must provide the Tenstorrent kernel driver, access to
`/dev/tenstorrent`, compatible firmware, and the configured hugepage mount.
The wheel carries the matching TT-XLA/PJRT, TTNN, TT-Metal user-space, compiler,
and kernel resources. Run `tt-finetune doctor --json` to see the exact bundled
paths and every missing system prerequisite.

The optional Hugging Face data and adapter tools are pinned in the release
compatibility lock. The project is not claimed to be published on PyPI.

## Verify the installation

After installation, run from any directory outside the source checkout:

```bash
python -c 'import tt_finetune; print(tt_finetune.__version__)'
tt-finetune doctor --smoke
```

If you cloned the repository for the examples, you can then run:

```bash
python examples/quickstart.py
```

No `PYTHONPATH`, source checkout, or repository-relative working directory is
required by the installed Python package.
