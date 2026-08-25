# Release qualification

This preview was built from the stable trainer source and qualified as a
relocatable binary wheel for one Tenstorrent Blackhole P150a.

## Artifact

- Package: `tt-finetune 0.1.0rc1`
- Wheel: `dist/tt_finetune-0.1.0rc1-cp312-cp312-linux_x86_64.whl`
- Wheel SHA256: `469c60b89369c6636ca542a5d833d1634370637f155a20717cd61a8a92e25bd9`
- Release source hash: `f31304cb1ff31e55e9477cf6902b7e565d00a748ec5073521f8d23cff4960950`
- Wheel size: 374,666,218 bytes
- Installed package size: 1,404,838,662 bytes (about 1.4 GiB)

## Accepted checks

- Clean virtual environment install with `PYTHONPATH` unset.
- Package import and runtime discovery from an arbitrary working directory.
- Source repository hidden during relocation qualification.
- TTNN, PJRT plugin, compiler and native libraries loaded from the installed
  package.
- Real P150a operation passed with exact output.
- Public trainer completed one step, saved a checkpoint, resumed to step two,
  exported a standard PEFT adapter and reloaded it.
- Owned stable release regression: 571 passed, 6 deselected, 0 failed.
- Wheel audit: no Python bytecode, development paths or absolute RPATH/RUNPATH
  entries.

Machine-readable evidence is in `evidence/`.

The wheel does not include unfinished dynamic sparse scheduling or unfinished
planner geometry experiments. Those were preserved separately from the
release source.
