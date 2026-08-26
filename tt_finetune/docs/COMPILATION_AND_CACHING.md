# Compilation and caching

The first use of a new model semantics, rank, batch, sequence, target, memory,
and execution-plan contract compiles before training. User metrics separate
compile time from synchronized steady-state step time.

During a persistent-runtime job, `tt-finetune` reports the active startup
stage (runtime initialization, model loading, optimizer setup, dataset
preparation, schedule setup, device compilation/bucket warmup, and steady
training). Long stages emit a heartbeat once per minute. The same information
is written atomically to `training_metrics.json` under `startup_profile`, so a
job can be monitored without interpreting internal compiler logs.

Sequence bucketing is also a compile-shape contract: batches are padded to the
declared bucket ceilings rather than producing every tile-rounded length. A
persistent `trainer.compile()` job deterministically round-robins real batches
from every populated bucket instead of hoping a rare shape appears early in a
shuffled epoch. Empty declared buckets are reported rather than fabricated.
Metrics report every first-use bucket latency separately under
`compile.bucket_first_use`; these startup samples are excluded from the
synchronized steady-state summary. The visible initial compile stage ends
after the first complete step—later bucket first-use events do not relabel
ordinary intervening training as compilation.

The planner uses two distinct caches:

- a semantic profile cache records measured full-step plans and invalidates
  them when the executable source or contract changes;
- the TT compiler cache reuses portable lower-level artifacts supported by the
  installed runtime.

The persistent runtime mode can reuse a complete executable inside one process
while creating fresh model, optimizer, RNG, dropout, and dataset state for each
job. It never reuses training state as a shortcut. A service restart must
compile or rebuild the executable because current PJRT does not safely
deserialize it across processes.

The exact missing backend operation and a two-process reproducer are documented
in `PJRT_EXECUTABLE_PERSISTENCE.md`.

Fresh measured-profile candidates require isolated PJRT processes. The trainer
releases an idle package-owned persistent service before that search and starts
a new one for the winner. It refuses to terminate a service with an active
training job and reports that conflict directly.

Only one training owner may use a single-card persistent service at a time. A
second request fails promptly with the active job identifier instead of waiting
behind an unknown device workload. The service also holds a lease on the
owning client connection: if that client or its safety supervisor exits, the
package-owned service terminates the abandoned accelerator job and releases
the card. Restarting from the latest completed checkpoint is then the recovery
path; an ownerless job is never allowed to continue silently.

Measured tuning always includes the accepted static plan and falls back to it
when no candidate is faster or a candidate violates correctness. Candidate
winners come from synchronized complete training steps; microbenchmarks may
only prune the search. The selected attention, PEFT, sparse-MoE, optimizer, and
graph-partition choices are part of both the worker ABI and persistent
executable cache key; a cached executable is never reused for a different
measured execution contract.
