from __future__ import annotations

import torch

from tt_finetune.worker import _tensor_digest


def test_tensor_digest_ignores_torch_compile_wrapper_prefix():
    value = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    direct = _tensor_digest(torch, [("model.adapter", value)])
    compiled = _tensor_digest(
        torch, [("_orig_mod.model.adapter", value.clone())]
    )
    assert direct == compiled

