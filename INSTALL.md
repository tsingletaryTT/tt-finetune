# Installation

## Supported preview host

- Linux x86-64 with glibc 2.34 or newer
- CPython 3.12
- one Tenstorrent Blackhole P150a
- compatible Tenstorrent kernel driver and firmware
- read/write access to `/dev/tenstorrent/*`
- configured hugepages mounted at `/dev/hugepages-1G` or `/dev/hugepages`

Create a clean environment and install the wheel:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install ./dist/tt_finetune-0.1.0rc1-cp312-cp312-linux_x86_64.whl
tt-finetune doctor
```

The wheel bundles the matching PJRT plugin, TT-XLA components, TTNN Python
bindings, TT-Metal user-space runtime and kernel resources, TT-MLIR compiler
runtime, and tt-finetune custom sources. Ordinary Python dependencies are
installed from the configured Python package index.

No `PYTHONPATH`, environment activation script, development checkout, or
container is required. The package uses resources relative to its installed
location.

Use `tt-finetune doctor --smoke` to compile and execute a small operation on
the actual device. This is a stronger check than library loading alone.

