# Compatibility contract

| Component | Preview requirement |
|---|---|
| Python | CPython 3.12 |
| Operating system | Linux x86-64, glibc 2.34+ |
| Accelerator | Tenstorrent Blackhole P150a, single card |
| torch | 2.11.0+cpu (pinned official PyTorch CPU wheel) |
| transformers | 4.57.1 |
| peft | 0.19.1 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |
| datasets | 5.0.0 |
| accelerate | 1.14.0 |
| PJRT/TTNN Python support dependencies | Versions pinned in `requirements-lock.txt` |
| numpy | 2.5.1 |
| PyYAML | 6.0.3 |
| loguru | 0.7.3 |

The wheel contains the matching Tenstorrent user-space components. The kernel
driver and firmware are host-managed prerequisites and are intentionally not
installed by pip.
