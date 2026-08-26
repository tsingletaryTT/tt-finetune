# Compilation and caching

The first use of a new model semantics, rank, batch, sequence, target, memory,
and execution-plan contract compiles before training. User metrics separate
compile time from synchronized steady-state step time.

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

Measured tuning always includes the accepted static plan and falls back to it
when no candidate is faster or a candidate violates correctness. Candidate
winners come from synchronized complete training steps; microbenchmarks may
only prune the search.
