# Third-party notices

This preview redistributes compatible user-space components needed to run
tt-finetune without development checkouts.

| Component | Upstream | License |
|---|---|---|
| PJRT plugin for Tenstorrent / TT-XLA | https://github.com/tenstorrent/tt-xla | Apache-2.0 |
| TT-Metal and TTNN user-space runtime | https://github.com/tenstorrent/tt-metal | Apache-2.0 |
| TT-Lang | https://github.com/tenstorrent/tt-lang | Apache-2.0 |
| PyTorch/XLA | https://github.com/pytorch/xla | BSD-style license |

The corresponding license texts are retained inside the wheel's distribution
metadata and bundled runtime tree. Convenience copies are included beside
this notice as `LICENSE-APACHE-2.0.txt` and `LICENSE-TORCH-XLA.txt`.

Ordinary Python dependencies installed by pip retain their own package
metadata and licenses.

