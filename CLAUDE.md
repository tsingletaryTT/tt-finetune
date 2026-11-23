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
