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


def ngrams(sequence, n):
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def calc_fingerprint(text, ngram_size: int = 1, num_perm: int = 128):
    tokens = re_tokenize(text)
    if ngram_size > 1:
        tokens = {" ".join(t) for t in ngrams(tokens, ngram_size)}
    tokens = [token.encode('utf-8') for token in tokens]

    minhash = MinHash(num_perm=num_perm)
    minhash.update_batch(tokens)

    lean_minhash = LeanMinHash(minhash)
    buf = bytearray(lean_minhash.bytesize())
    lean_minhash.serialize(buf)

    return buf


def undup_alpaca(alpaca_records, num_perm: int = 32, threshold: float = 0.3, debug: bool = False):
    with Parallel(n_jobs=os.cpu_count()) as parallel:
        fingerprints = parallel(
            delayed(calc_fingerprint)(record["messages"][0]["content"], 1, num_perm)
            for record in tqdm(alpaca_records, desc="Fingerprinting")
        )

    for idx, record in tqdm(enumerate(alpaca_records)):
        record["minhash"] = fingerprints[idx]

    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm
    )

    filtered_records = []
    for idx, record in tqdm(enumerate(alpaca_records), desc="Undup"):
        minhash = LeanMinHash.deserialize(