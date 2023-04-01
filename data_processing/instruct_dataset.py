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
            self.rec