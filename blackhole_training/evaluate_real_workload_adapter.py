#!/usr/bin/env python3
"""Independent CPU/PyTorch evaluation for a real-workload adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path, limit: int) -> list[dict[str, str]]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result.append({
                    "prompt": str(row["prompt"]),
                    "response": str(row["response"]),
                })
                if len(result) == limit:
                    break
    if not result:
        raise ValueError("evaluation dataset is empty")
    return result


def _render(tokenizer, row: dict[str, str]) -> tuple[str, str]:
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["response"]},
    ]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return prompt, full
    prompt = f"User: {row['prompt']}\nAssistant:"
    eos = tokenizer.eos_token or ""
    return prompt, f"{prompt} {row['response']}{eos}"


def _tokenize(tokenizer, row: dict[str, str], max_length: int):
    prompt, full = _render(tokenizer, row)
    input_ids = tokenizer(
        full, add_special_tokens=False, truncation=False
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, truncation=False
    )["input_ids"]
    prompt_length = len(prompt_ids)
    if len(input_ids) > max_length:
        removed = len(input_ids) - max_length
        input_ids = input_ids[removed:]
        prompt_length = max(0, prompt_length - removed)
    if (
        tokenizer.eos_token_id is not None
        and len(input_ids) < max_length
        and (not input_ids or input_ids[-1] != tokenizer.eos_token_id)
    ):
        input_ids.append(tokenizer.eos_token_id)
    labels = list(input_ids)
    labels[:prompt_length] = [-100] * prompt_length
    return prompt, input_ids, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=3407)
    arguments = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = arguments.model.expanduser().resolve()
    dataset_path = arguments.dataset.expanduser().resolve()
    adapter_path = (
        arguments.adapter.expanduser().resolve()
        if arguments.adapter is not None else None
    )
    torch.manual_seed(arguments.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    load_seconds = time.perf_counter() - started

    rows = _rows(dataset_path, arguments.rows)
    total_nll = 0.0
    supervised_tokens = 0
    per_example = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            _, ids, labels = _tokenize(
                tokenizer, row, arguments.max_length
            )
            input_tensor = torch.tensor([ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_tensor)
            label_tensor = torch.tensor([labels], dtype=torch.long)
            logits = model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
            ).logits.float()
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = label_tensor[:, 1:].contiguous()
            nll = functional.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            count = int((shifted_labels != -100).sum())
            value = float(nll)
            total_nll += value
            supervised_tokens += count
            per_example.append({
                "index": index,
                "nll": value,
                "supervised_tokens": count,
                "mean_nll": value / count if count else None,
            })

        generations = []
        for index, row in enumerate(rows[: arguments.generations]):
            prompt, _ = _render(tokenizer, row)
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            input_limit = max(1, arguments.max_length - arguments.max_new_tokens)
            if prompt_ids.shape[1] > input_limit:
                prompt_ids = prompt_ids[:, -input_limit:]
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=arguments.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[0, prompt_ids.shape[1]:]
            generations.append({
                "index": index,
                "prompt": row["prompt"],
                "reference": row["response"],
                "generated_token_ids": continuation.tolist(),
                "generated_text": tokenizer.decode(
                    continuation, skip_special_tokens=True
                ),
            })

    mean_nll = total_nll / supervised_tokens
    payload = {
        "format": "tt-finetune-independent-evaluation-v1",
        "status": "accepted" if math.isfinite(mean_nll) else "failed",
        "authority": "PyTorch Transformers/PEFT CPU evaluation",
        "model": str(model_path),
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "examples": len(rows),
        },
        "adapter": (
            {
                "path": str(adapter_path),
                "config_sha256": _sha256(adapter_path / "adapter_config.json"),
                "weights_sha256": _sha256(adapter_path / "adapter_model.safetensors"),
            }
            if adapter_path is not None else None
        ),
        "configuration": {
            "max_length": arguments.max_length,
            "max_new_tokens": arguments.max_new_tokens,
            "seed": arguments.seed,
            "dtype": "bfloat16",
            "deterministic_generation": True,
        },
        "load_seconds": load_seconds,
        "total_negative_log_likelihood": total_nll,
        "supervised_tokens": supervised_tokens,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
        "per_example": per_example,
        "generations": generations,
    }
    destination = arguments.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
