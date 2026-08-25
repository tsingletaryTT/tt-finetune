# Known limitations

- One Blackhole P150a only. Multi-device and multi-host training are not
  implemented.
- Parameter-efficient training only. Full-weight fine-tuning and QLoRA are
  not supported.
- Cross-process PJRT executable deserialization is unavailable in the bundled
  runtime. Measured semantic plans persist across processes, while compiled
  executables can be reused only inside the same persistent Python service.
- A new model/geometry may therefore require compilation on first use.
- Dynamic no-capacity sparse-expert scheduling is not included. The release
  uses the qualified bounded sparse schedule and safe sparse fallback.
- Sparse expert tensors require a lossless TT extension alongside normal PEFT
  files; ordinary PEFT alone cannot represent all expert-specific state.
- Prompt tuning, prefix tuning, full fine-tuning, Mamba/state-space models,
  multimodal models, and arbitrary remote custom modeling code are not
  qualified.
- Long-context qualification is narrower than short-sequence qualification.
- Low-memory BF8/BF4 modes reduce frozen-weight representation only. They do
  not claim BF16 bit equality; the qualified contract is deterministic,
  finite training with exact repeatability inside the selected precision
  contract.
- Static support queries cannot prove that a model fits after every compiler
  allocation. Genuine device-memory exhaustion remains possible.

