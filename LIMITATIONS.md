# Known limitations

- One Blackhole P150a only; multi-device and distributed training are not
  included.
- Parameter-efficient training only. Full-parameter fine-tuning and QLoRA are
  not part of the current contract.
- PJRT complete-executable deserialization across fresh processes is not
  available. Lower-level compiler caches and measured plans are reusable, and
  a persistent same-process service can retain an executable safely.
- Reduced frozen-weight training has its own deterministic numerical contract;
  it is not BF16 bit identity.
- Sparse-MoE support is limited to admitted routing, expert, capacity, and
  auxiliary-loss semantics. Unsupported variants fail with a capability error.
- Prefix/prompt tuning and data-dependent simultaneous adapter routing are not
  implemented because they alter attention state or graph ABI.
- Encoder-only, encoder-decoder, multimodal, and Mamba/state-space models are
  not supported.
- Checkpoints that exceed single-card DRAM after supported reductions are
  rejected as capacity limitations.
- Compilation can be substantial for a new semantic geometry. It is excluded
  from steady-state performance claims and is reported separately.
