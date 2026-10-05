"""Учебный поток «Ткани»: сырые байты из нескольких источников.

  wiki_ru / wiki_en — живой язык (Википедия), последние 2 МБ каждого языка — валидация
  code              — исходники стандартной библиотеки Python (настоящий код)
  tasks             — задачи-упражнения (учитель), без тестовых ключей и без уровня 5;
                      с вероятностью tool_prob учитель показывает решение «руками» (через Python)
"""
import glob
import os
import random

import numpy as np
import torch

from .bench import tasks as T

DATA = T.DATA
VAL_BYTES = 2 * 1024 * 1024


def _load(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)


def python_corpus(limit=16 * 1024 * 1024):
    out = os.path.join(DATA, "py_stdlib.txt")
    if not os.path.exists(out):
        root = os.path.dirname(os.__file__)
        files = sorted(glob.glob(os.path.join(root, "*.py")))
        size = 0
        with open(out, "wb") as f:
            for p in files:
                b = open(p, "rb").read()
                f.write(b + b"\n\x00\n")
                size += len(b)
                if size > limit:
                    break
    return out


class Mixture:
    def __init__(self, seq_len, weights=None, seed=0, tool_prob=0.0):
        self.T = seq_len
        self.tool_prob = tool_prob
        self.rng = random.Random(seed)
        self.src = {
            "wiki_ru": _load(os.path.join(DATA, "wiki_ru.txt")),
            "wiki_en": _load(os.path.join(DATA, "wiki_en.txt")),
            "code": _load(python_corpus()),
        }
        self.weights = weights or {"wiki_ru": 0.2, "wiki_en": 0.2, "code": 0.1, "tasks": 0.5}
        self.tasks = T.train_stream(seed)

    def split(self, name, val):
        a = self.src[name]
        return a[-VAL_BYTES:] if val else a[:-VAL_BYTES]

    def sample_text(self, name, val=False):
        a = self.split(name, val)
        i = self.rng.randrange(0, len(a) - self.T - 1)
        return a[i:i + self.T + 1]

    def sample_tasks(self):
        buf = bytearray()
        while len(buf) < self.T + 1:
            t = next(self.tasks)
            use_tool = t.tool is not None and self.rng.random() < self.tool_prob
            buf += (t.tool_text if use_tool else t.text).encode("utf-8")
        return np.frombuffer(bytes(buf[:self.T + 1]), dtype=np.uint8)

    def batch(self, bsz):
        names = self.rng.choices(list(self.weights), weights=list(self.weights.values()), k=bsz)
        rows = [self.sample_tasks() if n == "tasks" else self.sample_text(n) for n in names]
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    def val_batches(self, name, n, bsz, seed=1):
        """Фиксированные валидационные окна для честного сравнения bits-per-byte."""
        rng = random.Random(seed)
        a = self.split(name, True)
        for _ in range(n):
            rows = []
            for _ in range(bsz):
                i = rng.randrange(0, len(a) - self.T - 1)
                rows.append(a[i:i + self.T + 1])
            yield torch.from_numpy(np.stack(rows).astype(np.int64))
