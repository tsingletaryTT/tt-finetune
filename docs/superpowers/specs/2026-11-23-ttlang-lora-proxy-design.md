# TT-Lang LoRA proxy experiment (DRAFT for orchard review)

Status: draft, not approved. Written 2026-11-23 by tt-finetune-38 for Taylor Singletary and tt-orchard-57.
Nothing here has been run. Every number marked "proposal" is for orchard to change.

## Intent

Find out whether a LoRA trained with tt-finetune can raise the simulator pass rate of a code model on
TT-Lang kernel tasks. The long-term interest is Qwen3-Coder-Next (orchard's coder tier). That model is
probably out of reach for tt-finetune today (see Constraints), so this experiment uses a smaller proxy
model and measures whether the approach works at all.

## Goals

1. A measured before/after on orchard's kernel-probe ladder: does a LoRA-tuned proxy pass rungs the base
   proxy fails?
2. A comparison of two levers on the same tasks: fine-tuning (LoRA) and prompting (orchard's `--library`
   worked-kernel reference). If the library alone closes the gap, LoRA is not worth the cost.
3. A reusable data set and evaluation split that can be applied to Coder-Next later.
4. A record of whether tt-finetune runs on this box's p300c chips.

## Non-goals

- Making Qwen3-Coder-Next better. No claim about it comes out of this experiment.
- Serving the tuned adapter through orchard's TT vLLM bundle.
- Hardware-phase (on-chip) kernel validation. Simulator only, as in the ladder pre-registration.

## Anti-goals (a pass that would still be a failure)

- Any ladder fixture kernel (`tests/fixtures/kernel-loop/kernels/`) or near copy of one in the training data.
- Training data produced by an outside model. Orchard's rule is no outside model; this keeps it.
- A pass-rate gain that appears only because the base proxy was run under a weaker prompt than the tuned one.

## Constraints

- tt-finetune is qualified for one P150a only. This box has two p300c boards (4 chips). Step 0 below
  tests one chip before anything else is built.
- Qwen3-Coder-Next is, as far as I remember, an 80B sparse MoE with hybrid linear attention. tt-finetune lists
  state-space models as unsupported and fails closed on unknown MoE routing. Check its `config.json` before
  relying on this.
- Orchard will hold all 4 chips in short leases until about 2026-11-23 evening. All hardware work queues
  behind that through `gozer`, 1 chip at a time.
- TT-Lang changes quickly. Training examples must come from the pinned pair that the ladder used
  (tt-lang `fec38c6de`, tt-metal `0e9d200db976`, wheel `1.1.10.dev20261026`).

## Approaches considered

| | A. LoRA on a 1.5B coder | B. LoRA on a 7B coder, BF4 frozen weights | C. LoRA on Qwen3-Coder-30B-A3B |
|---|---|---|---|
| Fit in tt-finetune | Dense, supported | Dense, BF4 path is qualified (Qwen2.5-7B, seq 32 only) | Qwen3-MoE fixtures have silicon evidence; 30B memory fit is unknown |
| Closeness to Coder-Next | Low | Low | Highest (same family, MoE) |
| Baseline pass rate on the ladder | Probably 0 on every rung, which cannot show a gain | Possibly above 0 on rungs 1-4 | Possibly above 0 |
| Cost per run | Minutes | Hours at the sequence lengths kernels need | Unknown, likely long |
| Risk | Eval floor at zero | Long sequences untested on the BF4 path | Memory admission may reject it |

Proposal: start with B, falling back to A only for pipeline debugging. Run `TTTrainer.supports(...)` for B
and C at sequence length 2048 before choosing; keep C as the follow-up if B shows a gain. The pick is not
made until the planner reports the geometry fits.

## Design

### Step 0: gate (about 1 hour)
`tt-finetune doctor --smoke` on one p300c chip under a gozer lease, then the 135M command-translator run
as a pipeline check. If this fails, stop and report; the rest depends on it.

### Evaluation
- Task source: orchard's ladder, 9 rungs (`defaults.KERNEL_PROBE_LADDER`), same references and same
  simulator bars (PCC at least 0.999, relative L2 at most 0.01, slope 0.98 to 1.02, max abs error bound),
  6 input sets per attempt.
- Mode: one-shot generation (prompt in, kernel file out, simulator check), k samples per rung. The agent
  loop (40 turns with tools) is a poor test for a small proxy. This needs a one-shot entry point in the
  probe; open question 1.
- Split: train on examples that resemble rungs 1-4 (elementwise, scalar, broadcast add). Hold out rungs
  5-9 (`rsqrt`, `row_sum`, `center`, `rmsnorm`, `linear_scan`) entirely. Rungs 1-4 are also scored, as the
  "learned what it was taught" check.
- Samples: proposal, 20 per rung per arm, temperature 0.7, seeds fixed and shared across arms so results
  pair by seed.
- Arms, all on the same proxy and the same prompt text apart from the stated difference:
  1. base, no library
  2. base, `--library` worked kernels in the prompt
  3. LoRA, no library
  4. LoRA, with library

### Training data (all simulator-checked)
1. Upstream tt-lang `examples/` (463 Python files) and `test/` programs from the pinned commit, filtered to
   kernels that pass on the simulator. Most of the 13k test files are not kernels; the filter decides.
2. Simulator-passing attempts from orchard's coder logs (Qwen3-Coder-Next), for rungs 1-4 only. This is
   self-distillation from a model orchard already runs. The logs exist; use needs the operator's permission
   (open question 1). Never `tests/fixtures/kernel-loop/kernels`, and nothing from rungs 5-9.
3. Prompt format matches the probe's prompt so the model sees the same framing at test time.

Contamination check before any training: n-gram overlap (proposal: 8-token shingles) between every training
example and every ladder fixture, rungs 1-9. Drop anything above a set overlap. The check and its output are
written to the run record.

### Training
tt-finetune `TTTrainer`, rank 16 (proposal), LoRA on attention and MLP projections, sequence bucket up to
2048 tokens, loss on the completion only. Held-out validation NLL on a split of the training examples is
logged for overfitting. Chip leased with `gozer run --chips 1`.

### Success rule (proposal; orchard to set the final thresholds before any run)
Pre-registered in the same style as `2026-10-28-kernel-probe-ladder.md`, written before training starts.
- Primary: pooled pass count on held-out rungs 5-9, arm 3 against arm 1, paired by seed, 100 attempts per
  arm. Success is a difference whose exact paired test gives p below 0.05 and at least 10 more passes.
- Lever comparison: arm 3 against arm 2, same test. If arm 2 matches arm 3, the report says the library is
  the cheaper lever.
- Report all arms and every rung, including zeros. A result of no gain is a valid outcome.

## What this decides about Coder-Next
- If LoRA gains on the proxy and the library does not match it: the data and the split are the starting
  point for a Coder-Next run once the architecture is supported (or the 30B-A3B sibling as a stepping stone).
- If the library matches the LoRA or nothing moves: spend effort on the prompt and library instead.
- Either way the run record states what carried over and what did not (model family, size, agent loop).

## Orchard's answers (2026-11-23, from orchard's docs and code; orchard had not read this draft)
1. One-shot mode does not exist. `kernel-probe` always runs the agent loop. The simulator check
   (`orchard/kernelladder.py` plus `references.py`) can be called on its own. A one-shot entry point needs a
   small change in `orchard/kernelprobe.py`; orchard offered to scope it if an issue is filed. Not yet filed.
2. Logs: protocol and results in `docs/run-logs/2026-10-28-kernel-probe-ladder.md`; 14-attempt data with
   kernels and simulator verdicts in `docs/run-logs/2026-11-21-opencode-v2-vs-orchard-harness-data` (on main);
   older per-attempt directories under `~/orchard-runs`, not indexed. Orchard sees no technical objection to
   using Coder-Next output as training data. Whether it is allowed is the operator's decision (open, below).
   Hard rules from orchard: keep rungs 5-9 out of training, and do not train on
   `tests/fixtures/kernel-loop/kernels` (those are the reference solutions).
3. Library arm: it must use current orchard main. The old cheat sheet told the coder to use
   `ttl.math.fill` / broadcast, which both chip compilers refuse; the fix (`ttl.block.*`) is in PRs 802, 803, 812.
   A simulator pass is not a chip pass. PR 802 adds a no-device compile check with the chip package's own
   compiler. This design adds that check as a second, reported column.
4. Thresholds: orchard's evidence is that arm differences at 14 attempts were not distinguishable (p about 0.5),
   so 100 paired attempts per arm with 10 or more extra passes is a sensible bar. Pre-register before running.
5. Window: orchard's 4-chip leases end in about 1-2 hours from its message; orchard will message when released.

## Coder-Next baseline (read from the ladder log, results section, partial run)
Agent loop, simulator only, `--library none`, temperature 0.7, stopped early by the operator, decode speed
fell during the batch. Counts, not rates:

| Rung | Op | Result |
|---|---|---|
| 1-5 | add, mul, scale_shift, add_row, rsqrt | 1 of 1 pilot each; no batch |
| 6 | row_sum | pilot failed; batch 6 of 9 |
| 7 | center | pilot passed; batch 1 of 6 |
| 8 | rmsnorm | 0 of 3 (stopped by the operator) |
| 9 | linear_scan | not run |

Consequence: the baseline is an agent loop and the proxy runs one-shot, so the proxy's numbers cannot be
compared with these. They show that rungs 1-5 are easy for the big coder and 7-9 are hard. A proxy that
scores 0 on rungs 1-4 one-shot means the LoRA has to teach the basics first; rungs 1-4 are reported
separately so that case is visible.

## Open questions
1. Operator (Taylor): may Coder-Next's simulator-passing kernels (rungs 1-4) be used as LoRA training data?
   Without them the data is upstream examples and tests only.
2. Operator: file an orchard issue for a one-shot kernel-probe mode? The main evaluation cannot run without
   it, unless this repo calls `kernelladder.py` directly.
3. Operator: approve step 0 (`doctor --smoke`, one chip) once orchard releases its leases.
4. Training-data filter: tt-lang examples and tests that use `ttl.math.fill` or broadcast must be dropped or
   rewritten, because both chip compilers refuse them. The compile check decides which examples survive.

## Left to the builder
Exact prompt template, filter script details, shingle size, learning rate and step count, report format.
