"""Host-only dataset inspection used by the public geometry-aware planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    arguments = parser.parse_args()

    from blacksmith.datasets.torch.dataset_utils import get_dataset
    from blacksmith.experiments.torch.qwen.configs import TrainingConfig

    raw = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    # Dataset inspection is deliberately host-only.  It does not initialize
    # PJRT, a device, the model weights, or optimizer state.
    config = TrainingConfig(**raw)
    dataset = get_dataset(config=config, split="train", collate_fn=None)
    report = dataset.pipeline_report(epoch=0)
    report["epoch_replay"] = []
    for epoch in range(max(1, arguments.epochs)):
        plan = dataset.epoch_plan(epoch)
        replay = dataset.epoch_plan(epoch)
        report["epoch_replay"].append({
            "epoch": epoch,
            "order_hash": plan.order_hash,
            "replay_order_hash": replay.order_hash,
            "exact_reconstruction": plan.order_hash == replay.order_hash,
            "batch_count": len(plan.batches),
        })
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
