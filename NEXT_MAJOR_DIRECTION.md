# Recommended next major direction

## Decision

Prioritize a fully native, higher-utilization dynamic sparse-MoE training
runtime before expanding into another broad training mode.

There is one immediate product-performance prerequisite before that major
capability begins: connect the architecture-general Hugging Face worker to the
already-qualified packed resident execution ABI whenever the semantic and
memory planner admits it. The real SmolLM2 endurance run reports the safe
TT-XLA autograd fallback and a 575.919 ms median step, while the matched
fixed-geometry resident engine is 67.143 ms. This is not a request for another
kernel or architecture backend; it is completion of the public execution
bridge around already-proven mathematics. Leaving that gap open would make a
new feature faster in isolation while ordinary users still miss most of the
available dense performance.

The bridge must remain semantics-driven, retain the current safe fallback, and
be accepted on complete public training steps. Once it is closed, dynamic MoE
is the next major capability/performance direction for the reasons below.

The real Granite 3.1 MoE workload now proves the missing semantic capability:
32 experts, top-8 routing, exact no-drop assignment packing, device routing,
selected 32-row expert blocks, finite evolving training, and a normal exported
adapter. It also exposes the clearest remaining performance gap. At rank 8,
batch 1, and sequence 128, the accepted run has a 21.021-second median step,
5.804 useful tokens/s, 66.3 W mean board power, and 106 W peak board power.
That is a system-level utilization problem, not a reason to tune another small
optimizer kernel.

The next implementation should enlarge the resident boundary around routing,
assignment packing, expert gate/up/down forward and backward, gradient scatter,
and selected-expert reductions. The planner should choose block sizes,
expert/core groups, residency, and prefetch from semantics and routing geometry.
It must preserve exact no-drop routing, avoid host token loops, and retain the
current safe fallback for unsupported geometries.

## Alternatives considered

| Direction | Evidence-based assessment | Priority |
|---|---|---:|
| Broader/faster dynamic MoE | Functionally proven, but the 400M-active real model remains strongly underutilized at 21.021 s/step. The largest measured product gap is already isolated here. | 1 |
| Multi-P150a training | High eventual value for capacity, long context, and throughput. It adds distributed correctness, communication, checkpoint, and planner contracts before the single-card sparse path is efficient. | 2 |
| QLoRA / stronger 4-bit training | Useful for quality and capacity, but the existing deterministic BF8/BF4 frozen-weight path already admits and trains Qwen2.5-7B on one card. True quantized-training semantics remain separate work. | 3 |
| New architecture classes | Dense semantic discovery already covers materially different RMSNorm/LayerNorm, MHA/GQA/MQA, fused/separate projections, and ordinary/gated MLPs. Add new classes when a concrete workload requires them. | 4 |
| Full-parameter fine-tuning | Requires a much larger optimizer, gradient, memory, and distributed-state contract and is least aligned with the proven PEFT product. | 5 |

## Exit criteria for the recommendation

The dynamic sparse-MoE direction should not be declared complete from an
isolated kernel win. Acceptance should require real checkpoint training with:

- synchronized full-step improvement and materially higher board utilization;
- no all-expert evaluation and no host token-routing loop;
- finite forward, gradients, parameters, and optimizer state;
- deterministic resume under the published numerical contract;
- model-name-independent planner choices across different expert counts,
  top-k values, ranks, batches, and sequence lengths;
- safe fallback for unsupported routing, capacity, or layout semantics.

After that boundary is efficient, multi-P150a training becomes the strongest
next expansion because it can reuse a credible single-card execution unit.
