# Roadmap

The current single-P150a execution/compiler milestone is complete. The next
major work is intentionally ordered around product impact rather than adding
more model-specific fast paths.

## 1. Compile and startup closure

The next primary target is to remove compilation from the normal user path as
far as possible.

Planned work includes:

- separate model-independent kernels/runtime pieces that can ship precompiled;
- define reusable execution archetypes from canonical semantics and physical
  geometry rather than checkpoint identity;
- precompile broad architecture/geometry families offline;
- persist compiled artifacts under deterministic ABI/plan fingerprints;
- reuse artifacts across processes and service restarts where the runtime ABI
  permits it;
- reduce the remaining unavoidable JIT to only genuinely unseen execution
  specializations;
- optimize the residual compiler path itself after precompilation/cache reuse
  has removed avoidable work;
- build an offline qualification/calibration matrix across diverse dense,
  compressed and sparse Transformer families.

The intended user experience is that common models bind weights and runtime
geometry to an existing executable family instead of recompiling the training
system from scratch.

## 2. Multi-chip execution

The compile/qualification work naturally leads into multi-device training.
After the single-device artifact/runtime layer is mature, planned work includes:

- multi-P150a execution planning;
- tensor/pipeline/expert parallel execution families selected from geometry;
- device-resident collectives and communication-aware scheduling;
- multi-chip sparse-MoE expert placement and routing;
- deterministic checkpoint/resume across multi-device layouts;
- performance qualification on complete synchronized training steps.

## 3. Continued architecture coverage

Further architecture work will be driven by new semantic/resource bottlenecks
found by holdouts rather than by checkpoint names. Likely areas include:

- additional fused projection layouts and attention variants;
- wider dynamic sparse-expert geometries;
- longer-context scheduling and memory ownership;
- selected multimodal and state-space architectures after their semantics and
  memory contracts are defined;
- additional parameter-efficient training methods where they preserve a clear
  public execution contract.

Roadmap items are not current support claims. The public `v0.2.0` wheel remains
the installation authority until a new release is built and independently
qualified.
