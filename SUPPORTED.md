# Supported configurations

Support is decided from model semantics and tensor geometry. A model name is
never used to select a kernel or backend. `TTTrainer.supports(...)` and
`tt-finetune supports MODEL` provide a compilation-free prediction; a
prediction is not the same as a completed silicon qualification.

## Qualified architecture semantics

- decoder-only causal language models
- RMSNorm and stable LayerNorm
- MHA and GQA attention
- separate Q/K/V, fused QKV, and Conv1D-style projections
- standard rotary positions and learned absolute positions
- SwiGLU/gated MLP and ordinary GELU MLP
- tied and untied language-model heads
- dense and bounded sparse-MoE execution

Representative silicon-qualified paths include Qwen 2/2.5-style RMSNorm GQA
models, GPT-2-style fused Conv1D models, Pythia/GPT-NeoX-style stable
LayerNorm models, and Granite sparse-MoE. The dense generalization matrix
accepted every checkpoint that physically fit on one P150a; four larger
checkpoints were excluded only by device-memory capacity.

## Qualified geometries and methods

- LoRA ranks including 4, 8, and 16 in public full-model qualification; the
  lower-level geometry engine also covers larger and non-tile-aligned ranks
- batch 1 public silicon qualification and broader planner validation
- sequence lengths 32 and 128 for full training, plus a sequence-256
  long-attention smoke
- LoRA, rsLoRA, LoRA+, DoRA, and IA3 on their valid semantic targets
- BF16 and qualified mixed BF8/BF4 frozen-weight configurations

Configurations outside this evidence may still work through the general
planner or safe fallback, but should be treated as expected rather than
qualified until a device run succeeds.

Query an intended job before compiling:

```bash
tt-finetune supports Qwen/Qwen2.5-0.5B \
  --rank 8 --batch-size 1 --sequence-length 128
```

