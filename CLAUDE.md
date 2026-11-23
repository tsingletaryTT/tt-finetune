# tt-finetune (fork) - project notes for Claude

Fork of danielisraeli2409-jpg/tt-finetune (v0.2.1) at tsingletaryTT/tt-finetune. Remotes: `origin` = upstream,
`fork` = ours. Goal: learn what tt-finetune can do on this box (two p300c boards), then try a TT-Lang LoRA proxy
experiment (`docs/superpowers/specs/2026-11-23-ttlang-lora-proxy-design.md`, draft).

## Log
- 2026-11-23: asked to pick a small bounded project, compare with tt-train, and fork so it runs on p300c.
  Coordinated with tt-orchard-57 (chip leases; filed tsingletaryTT/tt-orchard#815 for a one-shot kernel probe).
- 2026-11-23: v0.2.1 wheel `doctor --smoke` failed on the p300c host. Cause was not the board. The wheel's
  tt-metal was built for SFPI 7.67.0 and JIT-compiles kernels with the host `/opt/tenstorrent/sfpi` (7.96.0
  after the morning reboot, tt-kmd 2.12.0), whose -Werror rejects four idioms in the old headers
  (`__builtin_rvtt_sfpiadd_i` macro in exp.h/exp2.h; deleted `sfpi::lut(vUInt...)` in sigmoid_appx.h/tanh.h).
  `tools/patch_wheel_for_host_sfpi.py` patches the installed wheel; after it `doctor --smoke` passes
  (device add result exact, ~5.5 s). Only blackhole headers are patched; the fix stops at the first failing
  kernel, so larger kernels (real training) may hit more.
- A dead end worth remembering: pointing `TT_METAL_RUNTIME_ROOT` at an overlay with `runtime/sfpi` of an older
  SFPI does nothing, because the runtime checks `/opt/tenstorrent/sfpi` first.
- Benign noise on p300c: `TT_FATAL: Chip N logical eth core ... connects to a remote mmio device` is logged
  during device open and the run still succeeds.

## Rules for this box
- Lease before any device open: `gozer run --chips 1 ...`. A p300c lease grants the whole board (2 chips).
- `gh` calls to the tenstorrent org fail with a SAML 403 (token revoked); tsingletaryTT repos work.
- The wheel lives in `~/.venvs/tt-finetune-smoke/.venv` (CPython 3.12). Re-run the patch tool after reinstalling.
- 2026-11-23: command-translator proof (`examples/translator/`). SmolLM2-135M, rank 16, 1200 steps, batch 1,
  seq 128, one p300c board, ~0.55 s/step (~225 tokens/s) after the header patch. Held-out set (204 requests with
  unseen phrasings and slot values): base model 0/204 exact, tuned 85/204 (41.7%). Loss 3.78 (step 25) to ~0.6.
  Strong on intents with one fixed command (`gozer queue/history/services` 12/12), weak where the model invents
  subcommands (`gozer withdraw`) or mixes gozer with tt-smi. Training took about 14 min; scoring runs on CPU.
  Results in `examples/translator/results/`. Not tried: more epochs, rank, completion-only loss.
- 2026-11-23: using more chips. One job cannot span chips (single-device). One job per CHIP fails (pinning one chip
  of a p300c pair gives a CUSTOM cluster and a fabric mesh descriptor error), so the unit is one job per BOARD:
  4 chips run 2 jobs. Two concurrent jobs break on two shared paths: the pipeline manifest in
  `~/.cache/tt-finetune/workspace` (fix: `workspace=` per job) and `/tmp/tt_pjrt_system_descriptor` (from
  `tempfile.gettempdir()`; second job fails at checkpoint export with "Error code: 13"; fix: short private
  `TMPDIR` per job; a long path gives "AF_UNIX path too long"). With both fixes the pair ran 100 steps and saved
  adapters (one pair run). Step time: solo 0.546 s, pair 0.575 to 0.592 s (about 6 to 8% slower, ~1.85x throughput).
  `examples/translator/run_two_boards.sh` does it. A rank-32 job alone on board 2 (chips 2,3) also works.
- 2026-11-23: rank 16 at 2000 steps scores 85/204, the same count as 1200 steps (102 answers differ, +18/-18):
  the score has levelled off near 42% for this data. Hex IDs get mangled (`1a0c7` for `1a0ce7`).
- Gozer note: SIGTERM to `gozer run` does not stop its children; kill the child PIDs, then `gozer reconcile`
  clears the stale lease. `TT_FT_VERBOSE=1` makes `train.py` pass `verbose=True` to show worker errors.
- Findings page (private artifact): docs/tt-finetune-p300c.html in this repo.
