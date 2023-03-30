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
            input_type: st