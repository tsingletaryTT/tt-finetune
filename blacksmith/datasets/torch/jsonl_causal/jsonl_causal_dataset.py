# SPDX-FileCopyrightText: (c) 2026
#
# SPDX-License-Identifier: Apache-2.0
"""Generic JSONL dataset for chat, prompt-response, and plain causal LM data."""

import json
import math
import os
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Dict

from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from tt_finetune.data_pipeline import (
    DATA_PLAN_VERSION,
    bucket_for_length,
    data_contract_hash,
    normalize_sequence_buckets,
    plan_data_geometries,
    plan_epoch_batches,
)


class _PlannedBatchSampler:
    """Small picklable batch sampler backed by an immutable epoch plan."""

    def __init__(self, batches):
        self.batches = tuple(tuple(batch.indices) for batch in batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _DynamicPaddingCollator:
    def __init__(
        self,
        tokenizer,
        ignored_index,
        max_length,
        bucketed,
        callback,
        boundaries=None,
    ):
        self.bucketed = bool(bucketed)
        self.boundaries = (
            normalize_sequence_buckets(max_length, boundaries)
            if self.bucketed
            else (int(max_length),)
        )
        self.collators = {
            boundary: DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding="max_length",
                max_length=boundary,
                label_pad_token_id=ignored_index,
            )
            for boundary in self.boundaries
        }
        self.callback = callback

    def __call__(self, batch):
        boundary = (
            bucket_for_length(
                max(len(item["input_ids"]) for item in batch),
                self.boundaries,
            )
            if self.bucketed
            else self.boundaries[-1]
        )
        collated = self.collators[boundary](batch)
        return self.callback(collated) if self.callback is not None else collated


class JSONLCausalDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        self.config = config
        self.split = split
        self.collate_fn = collate_fn
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.model_revision,
            trust_remote_code=self.config.trust_remote_code,
            padding_side="right",
            use_fast=self.config.use_fast_tokenizer,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.loaded_rows = 0
        self.dropped_rows = 0
        self.dropped_unsupervised_rows = 0
        self.lengths = []
        self.source_lengths = []
        self._epoch_cache = {}
        self._epoch_plan_cache = {}
        self._tokenization_seconds = 0.0
        self._tokenization_workers_used = 1
        self._prepare_dataset()

    def _source_paths(self):
        values = self.config.train_files if self.split == "train" else self.config.eval_files
        if not values:
            value = self.config.train_file if self.split == "train" else self.config.eval_file
            values = [value] if value else []
        if not values:
            raise ValueError(
                f"{self.split}_file or {self.split}_files must be set for dataset_id=jsonl_causal"
            )
        paths = [Path(value) for value in values]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"JSONL {self.split} files do not exist: {missing}")
        return paths

    def _transform_text(self, text):
        text = str(text or "")
        if self.config.text_transform == "none":
            return text
        if self.config.text_transform != "router_moa_tokens":
            raise ValueError(f"Unknown text_transform={self.config.text_transform!r}")

        replacements = [
            ("Conversation state:", "<|conversation_state|>"),
            ("ASSISTANT_PIPE:", "<|assistant_pipe|>"),
            ("TOOL_RESULT:", "<|tool_result|>"),
            ("UPSTREAM_DECISION_SOURCE:", "<|upstream_decision_source|>"),
            ("UPSTREAM_MODE_ACTION:", "<|upstream_mode_action|>"),
            ("UPSTREAM_PARAMS:", "<|upstream_params|>"),
        ]
        for raw, token in replacements:
            text = text.replace(raw, token)

        mode_tokens = {
            "s": "<|ma_s|>",
            "m": "<|ma_m|>",
            "ts": "<|ma_ts|>",
            "tm": "<|ma_tm|>",
            "f": "<|ma_f|>",
            "c": "<|ma_c|>",
            "q": "<|ma_q|>",
        }

        def encode_mode_action(match):
            mode = match.group(1)
            payload = match.group(2)
            if payload == "<NONE>":
                return mode_tokens[mode] + "<|none|>"
            actions = [
                action.strip()
                for action in payload.split(",")
                if action.strip()
            ]
            return mode_tokens[mode] + "".join(
                f"{action}|" for action in actions
            )

        text = re.sub(
            r"MA\|(ts|tm|s|m|f|c|q)\|([^\s\r\n]+)",
            encode_mode_action,
            text,
        )
        text = text.replace("PV|", "<|pv|>")
        text = text.replace("SP|", "<|sp|>")
        text = text.replace("<NONE>", "<|none|>")
        text = text.replace("<EMPTY>", "<|empty|>")
        for raw, token in {
            "|I": "<|emo_i|>",
            "|H": "<|emo_h|>",
            "|J": "<|emo_j|>",
            "|S": "<|emo_s|>",
        }.items():
            text = text.replace(raw, token)
        return text

    def _chat_messages(self, row):
        dataset_format = self.config.dataset_format
        if dataset_format == "chat":
            messages = row[self.config.messages_field]
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{self.config.messages_field!r} must be a non-empty list")
            return [
                {
                    "role": str(message.get("role", "")),
                    "content": self._transform_text(message.get("content", "")),
                }
                for message in messages
            ]
        if dataset_format == "prompt_response":
            messages = []
            if self.config.dataset_system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": self._transform_text(self.config.dataset_system_prompt),
                    }
                )
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": self._transform_text(row[self.config.prompt_field]),
                    },
                    {
                        "role": "assistant",
                        "content": self._transform_text(row[self.config.response_field]),
                    },
                ]
            )
            return messages
        if dataset_format == "text":
            return None
        raise ValueError(
            f"Unsupported dataset_format={dataset_format!r}; "
            "expected chat, prompt_response, or text"
        )

    def _render_without_chat_template(self, messages):
        """Serialize chat data for base models whose tokenizer has no chat template."""
        separator = self.config.fallback_role_separator
        history = []
        for message in messages[:-1]:
            role = message["role"].strip().capitalize() or "User"
            history.append(f"{role}: {message['content']}")
        history.append(self.config.fallback_assistant_prefix)
        prompt_text = separator.join(history)
        response = messages[-1]["content"]
        eos = self.tokenizer.eos_token or ""
        full_text = f"{prompt_text} {response}{eos}"
        return prompt_text, full_text

    def _tokenize_row(self, row: Dict) -> Dict:
        messages = self._chat_messages(row)
        if messages is None:
            full_text = self._transform_text(row[self.config.text_field])
            prompt_text = ""
        else:
            if messages[-1].get("role") != "assistant":
                raise ValueError("Chat and prompt_response rows must end with an assistant message")
            if self.tokenizer.chat_template:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.config.chat_template_enable_thinking,
                )
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=self.config.chat_template_enable_thinking,
                )
            else:
                prompt_text, full_text = self._render_without_chat_template(messages)

        # Tokenize without truncation first so assistant-only SFT can choose
        # the correct side to discard.  Ordinary tokenizer right-truncation
        # can remove the entire answer when a chat template and prompt fill
        # max_length, yielding an all-ignore label row and an infinite/NaN
        # mean cross-entropy.  For assistant-only loss the useful suffix is
        # the answer, so retain the final max_length tokens and adjust the
        # prompt boundary.  Plain causal-LM training keeps the conventional
        # right-truncation contract.
        encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        input_ids = list(encoding["input_ids"])
        attention_mask = list(encoding["attention_mask"])
        prompt_length = 0
        if self.config.assistant_only_loss and prompt_text:
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            prompt_length = len(prompt_ids)

        if (
            self.config.dataset_truncation
            and len(input_ids) > self.config.max_length
        ):
            if self.config.assistant_only_loss and prompt_text:
                removed = len(input_ids) - self.config.max_length
                input_ids = input_ids[removed:]
                attention_mask = attention_mask[removed:]
                prompt_length = max(0, prompt_length - removed)
            else:
                input_ids = input_ids[: self.config.max_length]
                attention_mask = attention_mask[: self.config.max_length]

        labels = list(input_ids)
        if self.config.assistant_only_loss and prompt_text:
            labels[:prompt_length] = [self.config.ignored_index] * prompt_length

        tokenized = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if (
            self.config.dataset_append_eos_token
            and self.tokenizer.eos_token_id is not None
            and len(tokenized["input_ids"]) < self.config.max_length
            and (
                not tokenized["input_ids"]
                or tokenized["input_ids"][-1] != self.tokenizer.eos_token_id
            )
        ):
            tokenized["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized["attention_mask"].append(1)
            tokenized["labels"].append(self.tokenizer.eos_token_id)
        return tokenized

    @staticmethod
    def _automatic_workers(value, *, item_count, purpose):
        if value != "auto":
            return int(value)
        if item_count < (32 if purpose == "tokenize" else 64):
            return 1 if purpose == "tokenize" else 0
        available = max(1, os.cpu_count() or 1)
        return min(8 if purpose == "tokenize" else 4, max(1, available // 4))

    @property
    def packing_enabled(self):
        policy = self.config.dataset_packing
        if policy == "auto":
            # Concatenating independent chat/completion records changes their
            # attention semantics unless a block-diagonal mask is available.
            # Plain causal text with an EOS separator is a stream and is safe.
            return bool(
                self.config.dataset_format == "text"
                and self.config.dataset_append_eos_token
            )
        return bool(policy)

    @property
    def bucketing_enabled(self):
        policy = self.config.dataset_sequence_bucketing
        if policy == "auto":
            lengths = (
                [len(row["input_ids"]) for row in self.unpacked_dataset]
                if hasattr(self, "unpacked_dataset")
                else [
                    min(int(length), int(self.config.max_length))
                    for length in self.source_lengths
                ]
            )
            return len(set(
                math.ceil(length / 32) * 32 for length in lengths
            )) > 1
        return bool(policy)

    def _pack(self, rows, *, epoch=0):
        """Deterministic best-fit packing without splitting an example."""

        order = list(range(len(rows)))
        if self.split == "train":
            generator = torch.Generator().manual_seed(
                int(self.config.seed) + int(epoch)
            )
            order = torch.randperm(len(rows), generator=generator).tolist()
        bins = []
        for source_index in order:
            row = rows[source_index]
            length = len(row["input_ids"])
            candidates = [
                (remaining - length, index)
                for index, (remaining, _parts) in enumerate(bins)
                if remaining >= length
            ]
            if candidates:
                _, bin_index = min(candidates)
                remaining, parts = bins[bin_index]
                parts.append((source_index, row))
                bins[bin_index] = (remaining - length, parts)
            else:
                bins.append((
                    self.config.max_length - length,
                    [(source_index, row)],
                ))
        packed = []
        provenance = []
        for _remaining, parts in bins:
            input_ids = []
            labels = []
            sources = []
            for source_index, row in parts:
                input_ids.extend(row["input_ids"])
                labels.extend(row["labels"])
                sources.append(source_index)
            packed.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })
            provenance.append(tuple(sources))
        return packed, tuple(provenance)

    def _bounded_tokenize(self, entries, workers):
        if workers <= 1:
            for path, line_number, row in entries:
                try:
                    yield self._tokenize_row(row)
                except Exception as error:
                    raise ValueError(
                        f"Could not process {path} line {line_number}: {error}"
                    ) from error
            return

        def execute(entry):
            path, line_number, row = entry
            try:
                return self._tokenize_row(row)
            except Exception as error:
                raise ValueError(
                    f"Could not process {path} line {line_number}: {error}"
                ) from error

        # Keep at most 2x workers outstanding so a very large JSONL does not
        # turn parallel tokenization into an unbounded host-memory queue.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = deque()
            iterator = iter(entries)
            for _ in range(workers * 2):
                try:
                    pending.append(executor.submit(execute, next(iterator)))
                except StopIteration:
                    break
            while pending:
                yield pending.popleft().result()
                try:
                    pending.append(executor.submit(execute, next(iterator)))
                except StopIteration:
                    pass

    def _prepare_dataset(self):
        sample_limit = (
            self.config.max_train_samples if self.split == "train" else self.config.max_eval_samples
        )
        split_per_file_limit = (
            self.config.max_train_samples_per_file
            if self.split == "train"
            else self.config.max_eval_samples_per_file
        )
        per_file_limit = split_per_file_limit or self.config.max_samples_per_file
        entries = []
        for path in self._source_paths():
            rows_from_file = 0
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if sample_limit is not None and self.loaded_rows >= sample_limit:
                        break
                    if per_file_limit is not None and rows_from_file >= per_file_limit:
                        break
                    if not line.strip():
                        continue
                    self.loaded_rows += 1
                    rows_from_file += 1
                    try:
                        row = json.loads(line)
                    except Exception as error:
                        raise ValueError(
                            f"Could not process {path} line {line_number}: {error}"
                        ) from error
                    entries.append((path, line_number, row))
            if sample_limit is not None and self.loaded_rows >= sample_limit:
                break

        workers = self._automatic_workers(
            self.config.dataset_tokenization_workers,
            item_count=len(entries),
            purpose="tokenize",
        )
        self._tokenization_workers_used = workers
        started = perf_counter()
        dataset = []
        for tokenized in self._bounded_tokenize(entries, workers):
            length = len(tokenized["input_ids"])
            self.source_lengths.append(length)
            has_supervised_target = any(
                label != self.config.ignored_index
                for label in tokenized["labels"][1:]
            )
            if (
                self.config.assistant_only_loss
                and not has_supervised_target
            ):
                self.dropped_rows += 1
                self.dropped_unsupervised_rows += 1
            elif length <= self.config.max_length:
                dataset.append(tokenized)
            else:
                self.dropped_rows += 1
        self._tokenization_seconds = perf_counter() - started
        if not dataset:
            raise ValueError(
                f"No {self.split} rows fit max_length={self.config.max_length}; "
                f"loaded={self.loaded_rows}, dropped={self.dropped_rows}"
            )
        self.unpacked_dataset = dataset
        if self.packing_enabled:
            self.dataset, self._initial_pack_provenance = self._pack(
                dataset, epoch=0
            )
        else:
            self.dataset = dataset
            self._initial_pack_provenance = tuple((index,) for index in range(len(dataset)))
        self.lengths = [len(row["input_ids"]) for row in self.dataset]
        self._contract_hash = data_contract_hash(
            tokenized_rows=self.unpacked_dataset,
            configuration={
                "format": self.config.dataset_format,
                "max_length": self.config.max_length,
                "assistant_only_loss": self.config.assistant_only_loss,
                "packing": self.packing_enabled,
                "bucketing": self.bucketing_enabled,
                "sequence_buckets": self.config.dataset_sequence_buckets,
                "seed": self.config.seed,
                "append_eos": self.config.dataset_append_eos_token,
            },
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _epoch_dataset(self, epoch):
        if not self.packing_enabled:
            return self.dataset, self._initial_pack_provenance
        if epoch not in self._epoch_cache:
            self._epoch_cache[epoch] = self._pack(
                self.unpacked_dataset, epoch=epoch
            )
        return self._epoch_cache[epoch]

    def epoch_plan(self, epoch=0):
        if epoch not in self._epoch_plan_cache:
            rows, _provenance = self._epoch_dataset(epoch)
            lengths = [len(row["input_ids"]) for row in rows]
            boundaries = (
                normalize_sequence_buckets(
                    self.config.max_length,
                    self.config.dataset_sequence_buckets,
                )
                if self.bucketing_enabled
                else None
            )
            self._epoch_plan_cache[epoch] = plan_epoch_batches(
                lengths,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                seed=self.config.seed,
                epoch=epoch,
                shuffle=self.split == "train",
                bucketed=self.bucketing_enabled,
                boundaries=boundaries,
                drop_last=True,
            )
        return self._epoch_plan_cache[epoch]

    @property
    def data_contract_hash(self):
        return self._contract_hash

    def pipeline_report(self, *, epoch=0):
        rows, provenance = self._epoch_dataset(epoch)
        plan = self.epoch_plan(epoch)
        raw_tokens = sum(len(row["input_ids"]) for row in self.unpacked_dataset)
        packed_tokens = sum(len(row["input_ids"]) for row in rows)
        workers = self._automatic_workers(
            self.config.dataset_dataloader_workers,
            item_count=len(rows),
            purpose="dataloader",
        )
        sorted_lengths = sorted(self.source_lengths)

        def percentile(fraction):
            if not sorted_lengths:
                return 0
            position = fraction * (len(sorted_lengths) - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            weight = position - lower
            return (
                sorted_lengths[lower] * (1.0 - weight)
                + sorted_lengths[upper] * weight
            )
        data_geometry_plan = plan_data_geometries(
            self.source_lengths,
            batch_size=self.config.batch_size,
            sequence_length=self.config.max_length,
            gradient_accumulation_steps=(
                self.config.gradient_accumulation_steps
            ),
            seed=self.config.seed,
            packing_enabled=self.packing_enabled,
            packing_semantically_safe=bool(
                self.config.dataset_format == "text"
                and self.config.dataset_append_eos_token
            ),
            preparation_seconds_per_batch=(
                self._tokenization_seconds / max(1, len(self.source_lengths))
            ),
        )
        return {
            "algorithm": DATA_PLAN_VERSION,
            "contract_hash": self._contract_hash,
            "split": self.split,
            "loaded_rows": self.loaded_rows,
            "accepted_rows": len(self.unpacked_dataset),
            "effective_examples": len(rows),
            "raw_tokens": raw_tokens,
            "packed_tokens": packed_tokens,
            "shape_distribution": {
                "minimum_tokens": min(sorted_lengths),
                "median_tokens": percentile(0.5),
                "p90_tokens": percentile(0.9),
                "maximum_tokens": max(sorted_lengths),
                "mean_tokens": sum(sorted_lengths) / len(sorted_lengths),
                "tile_length_histogram": {
                    str(boundary): sum(
                        1 for length in sorted_lengths
                        if math.ceil(length / 32) * 32 == boundary
                    )
                    for boundary in sorted(set(
                        math.ceil(length / 32) * 32
                        for length in sorted_lengths
                    ))
                },
            },
            "packing": {
                "requested": self.config.dataset_packing,
                "enabled": self.packing_enabled,
                "safe_auto_reason": (
                    "plain causal text with EOS separators"
                    if self.packing_enabled and self.config.dataset_packing == "auto"
                    else "explicit user policy"
                    if self.config.dataset_packing != "auto"
                    else "chat/completion attention isolation is unavailable"
                ),
                "pack_count": len(rows),
                "source_order_hash": data_contract_hash(
                    tokenized_rows=rows,
                    configuration={"epoch": epoch, "provenance": provenance},
                ),
                "utilization": packed_tokens / max(1, len(rows) * self.config.max_length),
            },
            "bucketing": {
                "requested": self.config.dataset_sequence_bucketing,
                "enabled": self.bucketing_enabled,
                "boundaries": list(normalize_sequence_buckets(
                    self.config.max_length,
                    self.config.dataset_sequence_buckets,
                )) if self.bucketing_enabled else [
                    math.ceil(self.config.max_length / 32) * 32
                ],
                "epoch_order_hash": plan.order_hash,
                "batch_count": len(plan.batches),
                "useful_tokens": plan.useful_tokens,
                "padded_tokens": plan.padded_tokens,
                "padding_fraction": plan.padding_fraction,
            },
            "tokenization": {
                "workers": self._tokenization_workers_used,
                "seconds": self._tokenization_seconds,
                "bounded_outstanding_per_worker": 2,
                "order_preserved": True,
            },
            "prefetch": {
                "dataloader_workers": workers,
                "batches_per_worker": (
                    self.config.dataset_prefetch_batches if workers else 0
                ),
                "bounded": True,
                "checkpoint_state": (
                    "reconstructed from deterministic epoch+batch offset; "
                    "prefetched tensors are pure and carry no RNG state"
                ),
            },
            "automatic_data_geometry_plan": data_geometry_plan,
        }

    def _get_dataloader(self, *, epoch: int = 0):
        rows, _provenance = self._epoch_dataset(epoch)
        plan = self.epoch_plan(epoch)
        return self._dataloader_from_batches(rows, plan.batches)

    def _dataloader_from_batches(self, rows, batches):
        workers = self._automatic_workers(
            self.config.dataset_dataloader_workers,
            item_count=len(rows),
            purpose="dataloader",
        )
        arguments = {
            "dataset": rows,
            "batch_sampler": _PlannedBatchSampler(batches),
            "collate_fn": _DynamicPaddingCollator(
                self.tokenizer,
                self.config.ignored_index,
                self.config.max_length,
                self.bucketing_enabled,
                self.collate_fn,
                self.config.dataset_sequence_buckets,
            ),
            "num_workers": workers,
            "persistent_workers": workers > 0,
        }
        if workers > 0:
            arguments["prefetch_factor"] = self.config.dataset_prefetch_batches
            # Dataset workers are created after the parent has initialized
            # PJRT. Forking that process can duplicate runtime threads and
            # device handles, so use clean spawned workers for host-only
            # collation/prefetch.
            arguments["multiprocessing_context"] = "spawn"
        return DataLoader(**arguments)

    def get_compile_warmup_dataloader(
        self,
        boundaries,
        *,
        repeats: int = 4,
        epoch: int = 0,
    ):
        """Return a bounded loader that covers every populated compile shape.

        Ordinary epoch order is deliberately shuffled. Relying on its first
        few batches can leave a rare sequence bucket cold for hundreds of
        training steps. This qualification-only loader uses real batches and
        the normal collator, but places one batch from each populated bucket
        in a deterministic round-robin schedule. Its model/optimizer state is
        discarded by the persistent compile warmup job.
        """

        if repeats <= 0:
            raise ValueError("compile warmup repeats must be positive")
        declared = set(normalize_sequence_buckets(
            self.config.max_length,
            self.config.dataset_sequence_buckets,
        )) if self.bucketing_enabled else {
            math.ceil(self.config.max_length / 32) * 32
        }
        requested = tuple(dict.fromkeys(int(value) for value in boundaries))
        unsupported = sorted(set(requested) - declared)
        if unsupported:
            raise ValueError(
                "compile warmup requested undeclared sequence buckets: "
                f"{unsupported}"
            )
        rows, _provenance = self._epoch_dataset(epoch)
        by_boundary = {boundary: [] for boundary in requested}
        for batch in self.epoch_plan(epoch).batches:
            if batch.sequence_length in by_boundary:
                by_boundary[batch.sequence_length].append(batch)
        populated = tuple(
            boundary for boundary in requested if by_boundary[boundary]
        )
        if not populated:
            raise ValueError(
                "compile warmup found no real batches for the requested "
                "sequence buckets"
            )
        selected = []
        for repeat in range(repeats):
            for boundary in populated:
                candidates = by_boundary[boundary]
                selected.append(candidates[repeat % len(candidates)])
        self.compile_warmup_schedule = {
            "requested_buckets": list(requested),
            "populated_buckets": list(populated),
            "empty_buckets": sorted(set(requested) - set(populated)),
            "repeats": repeats,
            "steps": len(selected),
        }
        return self._prepare_test_dataloader(
            self._dataloader_from_batches(rows, selected)
        )

    def get_dataloader_for_epoch(self, epoch: int):
        return self._prepare_test_dataloader(
            self._get_dataloader(epoch=epoch)
        )
