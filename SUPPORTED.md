# Supported configurations

`tt-finetune` accepts decoder-only causal language models whose discovered
semantics can be mapped to the operations below and whose requested training
geometry fits one P150a.

## Qualified dense semantics

- RMSNorm and LayerNorm
- MHA, GQA, and MQA
- separate, fused, and supported partially fused Q/K/V projections
- standard and scaled RoPE where the architecture descriptor recognizes it
- ordinary MLP and gated/SwiGLU MLP
- linear and Conv1D-style projection storage
- tied or untied embeddings and supported bias variants
- BF16 LoRA ranks including tile-friendly and irregular tails
- batch sizes 1–4 and sequence buckets through 4096 in the qualified matrix

Real-checkpoint qualification includes representatives from Qwen2/Qwen2.5,
Qwen3, Llama-compatible models, Pythia/GPT-NeoX, GPT-Neo, OPT, BLOOM, XGLM,
CodeGen, StableLM, and GPT-2. Qualification applies to the listed checkpoint
and geometry evidence, not automatically to every model published under those
family names.

## Sparse MoE

The qualified sparse path covers decoder-only expert layers with top-k routing,
SwiGLU experts, assignment weights, supported auxiliary loss, selected-expert
execution, and documented shared-expert variants. Mixtral, Qwen2-MoE, and
Qwen3-MoE semantic fixtures have silicon evidence. Unknown routing,
capacity/drop semantics, or expert layouts fail closed rather than silently
using dense semantics.

The capacity-free dynamic path is qualified on Granite 3.1 MoE with 32 experts,
top-8 routing, BF16, rank 8, and 128-token batches. Router decisions, exact
per-expert positions, packing, and restoration remain on the device. Every
selected assignment is retained; only occupied 32-row expert blocks enter the
sparse expert projections. This path does not gather an expert matrix per
assignment, evaluate every expert for every token, or route tokens in a Python
host loop. Its published numerical contract is deterministic selected-block
BF16 with HiFi4/FP32 reductions, not bit identity to the dense fallback.

## Memory admission

The planner prefers BF16 when it fits. Qualified larger-model paths may use
native BF8 or mixed BF8/BF4 frozen-weight storage while adapters, gradients,
and optimizer state retain the training contract. Reduced frozen-weight runs
are deterministic and finite but are not claimed bit-exact to BF16.

Use `TTTrainer.supports(...)` or `tt-finetune supports ...` for an explicit
decision, estimated peak memory, selected path, and actionable rejection.
