"""Public helpers for tokenizer-extended TT foundations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


def build_token_foundation(
    *,
    base_model: str,
    token_spec: str | Path,
    output_dir: str | Path,
    dtype: str = "bfloat16",
    reuse_if_valid: bool = True,
) -> Path:
    """Build the source notebook's clean token-extended Hugging Face base.

    Token initialization is a host-side model transformation, not a device
    training step. The returned model is then consumed by
    :class:`MultiAdapterSFTTrainer` for appended-row warmup on Blackhole.
    """
    output = Path(output_dir).expanduser().resolve()
    contract = output / "token_contract.json"
    if reuse_if_valid and (output / "config.json").is_file() and contract.is_file():
        payload = json.loads(contract.read_text(encoding="utf-8"))
        if (
            payload.get("base_model") == base_model
            and payload.get("new_token_count", 0) > 0
            and payload.get("foundation_vocab_size", 0)
            > payload.get("base_vocab_size", 0)
        ):
            return output
        raise ValueError(
            f"Existing foundation does not match the requested base: {output}"
        )
    if output.exists():
        raise FileExistsError(
            f"Foundation output already exists but is incomplete: {output}"
        )

    workspace = Path(__file__).resolve().parents[1]
    spec = Path(token_spec).expanduser().resolve()
    try:
        relative_output = output.relative_to(workspace)
        relative_spec = spec.relative_to(workspace)
    except ValueError as error:
        raise ValueError(
            f"Foundation paths must be inside workspace {workspace}"
        ) from error
    generated = workspace / "blackhole_training" / "generated" / "foundations"
    generated.mkdir(parents=True, exist_ok=True)
    config_path = generated / f"{output.name}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_model": base_model,
                "output_dir": str(workspace / relative_output),
                "token_spec": str(workspace / relative_spec),
                "dtype": dtype,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(workspace / "blackhole_training" / "build_foundation.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        cwd=workspace,
    )
    if not (output / "config.json").is_file() or not contract.is_file():
        raise RuntimeError(f"Foundation build did not produce a valid model: {output}")
    return output
