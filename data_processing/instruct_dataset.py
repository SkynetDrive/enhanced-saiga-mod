import json
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from tqdm import tqdm


class InstructDataset(Dataset):
    def __init__(
            self,
            original_records: List[Dict],
            tokenizer: AutoTokenizer,
            max_source_tokens_count: int,
            max_target_tokens_count: int,
            templates_path: str,
            sample_rate: float = 1.0,
            only_target_loss: bool = True,
            input_type: str = "causal",
            target_field: str = "output",
            source_field: str = "input",
            use_padding: bool = False
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.only_target_loss = only_target_loss
        self.input_type = input_type
        self.target_field = target_field
        self.source_field = source_field
        self.use_padding = use_padding
        self.is_printed = False

        with open(templates_path) as r:
            self.templates = json.load(r)

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_record(record)
            if tensors is None:
                continue
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_record(self, record):
        instruction = record["instruction"]
        inp = record[self.source_field]
        out = record[self.target_field]
        if inp.strip() != "":
            templates = self.templates["prompts_input"]
            prompt_template = random.choice(templates)
            source = prompt_template.format(instruction=instruction.strip(), inp=inp.strip())
        else:
            templates = self.templates["prompts_no_input"]
            prompt_template = random.choice(templates)
            source = prompt_template.format(instruction=instruction.strip())
        target = out.strip()
        if not self.is_printed:
            print("Source and target examples")
            print(source)
            print(target)
            self.is_printed = True
        if self.input_type == "causal":
            return self.convert_causal(source, target)
        elif self.input_type == "seq2seq":
            return self.convert_seq2seq(source, target)
        else:
            assert False

    def convert_causal(self, source, target=None):
        source_tokens = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True
        )["input_ids"]
        if self.tokenizer.bos_token_id:
            source_tokens.insert(0, self.tokenizer.bos_token_id)
        input_ids = source_tokens[:]
        actual_length = len(input_ids)
        max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2
        if target is not None:
            target_tokens = self.tokenizer(
                target,
                add_special_tokens=F