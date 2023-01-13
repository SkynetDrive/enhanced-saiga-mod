import json
import re
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
from data_processing.bad_substrings import has_bad_ss
from datasketch import MinHash, MinHashLSH, LeanMinHash
from itertools import tee


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


def build_char_system_messages(char):
    name = char["name"]
    context = char["context"]
    greeting = char["greeting"]
    example_dialogue = char["example_dialogue"]

    context = f"Т