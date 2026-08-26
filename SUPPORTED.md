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

## Memory admission

The planner prefers BF16 when it fits. Qualified larger-model paths may use
native BF8 or mixed BF8/BF4 frozen-weight storage while adapters, gradients,
and optimizer state retain the training contract. Reduced frozen-weight runs
are deterministic and finite but are not claimed bit-exact to BF16.

Use `TTTrainer.supports(...)` or `tt-finetune supports ...` for an explicit
decision, estimated peak memory, selected path, and actionable rejection.
