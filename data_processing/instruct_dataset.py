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
            tokenizer