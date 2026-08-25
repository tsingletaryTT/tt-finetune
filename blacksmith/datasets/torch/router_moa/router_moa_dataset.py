# SPDX-FileCopyrightText: (c) 2026
#
# SPDX-License-Identifier: Apache-2.0
"""Router MoA chat dataset for causal-LM adapter training."""

import json
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig


class RouterMoADataset(BaseDataset):
    """Load one Router MoA stage and train only on the assistant completion."""

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        self.config = config
        self.split = split
        self.collate_fn = collate_fn
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="right",
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]
        self._prepare_dataset()

    def _source_path(self) -> Path:
        value = self.config.train_file if self.split == "train" else self.config.eval_file
        if not value:
            raise ValueError(f"{self.split}_file must be set for dataset_id=router_moa")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Router MoA {self.split} file does not exist: {path}")
        return path

    def _tokenize_function(self, example: Dict) -> Dict:
        messages = example["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Every Router MoA row must end with an assistant message")

        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )

        input_ids = encoding["input_ids"]
        labels = list(input_ids)
        prompt_len = len(prompt_encoding["input_ids"])
        labels[:prompt_len] = [self.config.ignored_index] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
            "len": len(input_ids),
        }

    def _prepare_dataset(self):
        sample_limit = (
            self.config.max_train_samples if self.split == "train" else self.config.max_eval_samples
        )
        dataset = []
        with self._source_path().open(encoding="utf-8") as source:
            for row_number, line in enumerate(source, start=1):
                if sample_limit is not None and row_number > sample_limit:
                    break
                if not line.strip():
                    continue
                example = json.loads(line)
                tokenized = self._tokenize_function({"messages": example["messages"]})
                if tokenized["len"] <= self.config.max_length:
                    tokenized.pop("len")
                    dataset.append(tokenized)

        if len(dataset) == 0:
            raise ValueError(
                f"No Router MoA {self.split} rows fit max_length={self.config.max_length}. "
                "Increase max_length or inspect the selected stage."
            )
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

    def _get_dataloader(self) -> DataLoader:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.config.max_length,
            label_pad_token_id=self.config.ignored_index,
        )
        if self.collate_fn is not None:
            total_collate_fn = lambda batch: self.collate_fn(data_collator(batch))
        else:
            total_collate_fn = data_collator

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=total_collate_fn,
            shuffle=self.split == "train",
            drop_last=True,
        )
