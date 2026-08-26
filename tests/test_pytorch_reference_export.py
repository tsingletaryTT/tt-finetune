from __future__ import annotations

import json
from pathlib import Path

from blackhole_training.run_pytorch_reference import (
    _arguments,
    _export_adapter,
)


def test_reference_export_uses_latest_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    (checkpoint_dir / "checkpoint_history.json").write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "path": "/missing/development/path/checkpoint.pt",
                        "name": checkpoint.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    observed: list[str] = []

    def fake_run(command, check):
        assert check is True
        observed.extend(command)
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        (destination / "adapter_config.json").write_text("{}")
        (destination / "adapter_model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr("subprocess.run", fake_run)
    payload = {
        "model_name": "/models/base",
        "model_metadata_name": "org/base",
        "adapter_type": "lora",
        "lora_target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
        "lora_bias": "none",
        "lora_task_type": "CAUSAL_LM",
        "lora_init_weights": True,
        "lora_r": 8,
        "lora_alpha": 16,
    }

    destination = _export_adapter(payload, tmp_path)

    assert destination == (tmp_path / "peft_adapter").resolve()
    assert str(checkpoint.resolve()) in observed
    assert observed[observed.index("--targets") + 1 :][:2] == [
        "q_proj",
        "v_proj",
    ]


def test_reference_cli_exposes_schedule_preserving_stop_after(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pytorch_reference",
            "--resolved-config",
            "config.yaml",
            "--output",
            "out",
            "--steps",
            "5000",
            "--stop-after",
            "500",
        ],
    )

    arguments = _arguments()

    assert arguments.steps == 5000
    assert arguments.stop_after == 500
