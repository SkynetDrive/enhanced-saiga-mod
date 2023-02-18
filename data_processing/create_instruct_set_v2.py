import os
import json
import re
import random
import sys
from datasets import load_dataset
from tqdm import tqdm
from itertools import tee
from datasketch import MinHash, MinHashLSH, LeanMinHash
from data_processing.bad_substrings import has_bad_ss
from joblib import Parallel, delayed


def calc_max_length(records):
    return max([sum([len(m["content"]) for m in r["messages"]]) for r in records])


def re_tokenize(text):
    return re.findall(r'[а-яё-]+|[a-z-]+|\d+|\S', text, re.I)


def ngrams