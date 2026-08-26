"""Small command-line surface for installation checks and YAML training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .environment import RuntimeEnvironment, host_data_dependencies


def main() -> None:
    parser = argparse.ArgumentParser(prog="tt-finetune")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="check the Blackhole runtime")
    doctor.add_argument("--workspace")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument(
        "--smoke", action="store_true",
        help="compile and execute a tiny operation on the attached device",
    )
    train = commands.add_parser("train", help="train from a public YAML file")
    train.add_argument("config", type=Path)
    train.add_argument("--workspace")
    train.add_argument("--resume", nargs="?", const=True, default=False)
    supports = commands.add_parser(
        "supports", help="query static model/geometry support without compiling"
    )
    supports.add_argument("model")
    supports.add_argument("--rank", type=int, default=8)
    supports.add_argument("--batch-size", type=int, default=1)
    supports.add_argument("--sequence-length", type=int, default=128)
    supports.add_argument("--targets", nargs="+", default=["auto"])
    supports.add_argument("--precision", default="bfloat16")
    cache = commands.add_parser(
        "cache", help="inspect measured execution-plan profiles"
    )
    cache.add_argument("--workspace")
    cache.add_argument("--path", type=Path)
    cache.add_argument("--json", action="store_true")
    adapter = commands.add_parser(
        "adapter", help="inspect or merge a standard PEFT adapter"
    )
    adapter_commands = adapter.add_subparsers(
        dest="adapter_command", required=True
    )
    inspect = adapter_commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    merge = adapter_commands.add_parser("merge")
    merge.add_argument("path", type=Path)
    merge.add_argument("--base-model", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "doctor":
        environment = RuntimeEnvironment.discover(args.workspace)
        payload = {
            **environment.to_dict(),
            "host_data_dependencies": host_data_dependencies(),
        }
        if args.smoke and environment.ready:
            from .environment import run_device_smoke

            payload["device_smoke"] = run_device_smoke()
            if payload["device_smoke"]["status"] != "passed":
                payload["ready"] = False
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print("tt-finetune runtime:", "ready" if payload["ready"] else "not ready")
            print("mode:", environment.runtime_mode or "none")
            print("python:", environment.runtime_python or "none")
            if args.smoke:
                smoke = payload.get("device_smoke", {})
                print("device smoke:", smoke.get("status", "not run"))
                if smoke.get("error"):
                    print("-", smoke["error"])
            for problem in environment.problems:
                print("-", problem)
        raise SystemExit(0 if payload["ready"] else 1)

    if args.command == "supports":
        from .trainer import TTTrainer

        targets = args.targets[0] if len(args.targets) == 1 else args.targets
        result = TTTrainer.supports(
            model=args.model,
            rank=args.rank,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            target_modules=targets,
            precision=args.precision,
        )
        print(json.dumps(result.to_dict(), indent=2))
        raise SystemExit(0 if result.supported else 2)

    if args.command == "cache":
        from .runtime.autotuner import ExecutionProfileCache

        root = Path(
            args.workspace
            or os.environ.get("TT_FINETUNE_WORKSPACE")
            or Path(__file__).resolve().parents[1]
        ).expanduser().resolve()
        path = (
            args.path.expanduser().resolve()
            if args.path is not None
            else Path.home() / ".cache/tt-finetune/profiles/profiles.json"
        )
        payload = {
            **ExecutionProfileCache(path).describe(),
            "executable_persistence": {
                "cross_process_deserialization": False,
                "same_process_reuse": True,
                "reason": (
                    "the installed PJRT runtime does not expose safe "
                    "executable deserialization"
                ),
            },
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Measured profiles: {payload['profile_count']}")
            print(f"Cache: {payload['cache_path']}")
            for item in payload["entries"]:
                print(
                    f"- {item['profile_key']} | "
                    f"{item['graph_partition']} | "
                    f"{item['median_ms']} ms | compile estimate "
                    f"{item.get('estimated_compile_seconds')} s"
                )
            print(
                "Cross-process executable deserialization: unavailable; "
                "measured plans and same-process executables remain reusable"
            )
        return

    if args.command == "adapter":
        from .adapters import inspect_adapter, merge_adapter

        if args.adapter_command == "inspect":
            print(json.dumps(inspect_adapter(args.path).to_dict(), indent=2))
            return
        output = merge_adapter(
            base_model=args.base_model,
            adapter=args.path,
            output_dir=args.output,
        )
        print(json.dumps({"merged_model": str(output)}, indent=2))
        return

    from blackhole_training import TTLoRATrainer

    trainer = TTLoRATrainer.from_yaml(args.config, workspace=args.workspace)
    result = trainer.train(resume=args.resume)
    print(json.dumps({
        "output_dir": str(result.output_dir),
        "adapter_dir": str(result.adapter_dir),
        "metrics": str(result.metrics_path),
    }, indent=2))


if __name__ == "__main__":
    main()
