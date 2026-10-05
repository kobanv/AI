"""Скачивает корпуса RU/EN и реальные бенчмарки в data/ (не коммитится).

python -m tkan.download
"""
import gzip
import io
import json
import os
import urllib.request

import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HF = "https://huggingface.co/datasets"

CORPORA = {
    # язык: (url parquet-шарда Википедии, сколько МБ текста взять)
    "ru": (f"{HF}/wikimedia/wikipedia/resolve/main/20231101.ru/train-00004-of-00021.parquet", 60),
    "en": (f"{HF}/wikimedia/wikipedia/resolve/main/20231101.en/train-00005-of-00041.parquet", 60),
}

BENCH = {
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "mgsm_en.tsv": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_en.tsv",
    "mgsm_ru.tsv": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_ru.tsv",
    "humaneval.jsonl.gz": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
}


def fetch(url, path):
    if os.path.exists(path):
        return
    print("скачиваю", url)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def corpus(lang):
    url, mb = CORPORA[lang]
    out = os.path.join(ROOT, f"wiki_{lang}.txt")
    if os.path.exists(out):
        return out
    raw = os.path.join(ROOT, f"wiki_{lang}.parquet")
    fetch(url, raw)
    table = pq.read_table(raw, columns=["text"])
    limit, size = mb * 1024 * 1024, 0
    with open(out, "w", encoding="utf-8") as f:
        for text in table.column("text").to_pylist():
            text = text.strip()
            if len(text) < 500:
                continue
            f.write(text + "\n\x00\n")  # \x00 — разделитель документов
            size += len(text.encode("utf-8"))
            if size >= limit:
                break
    os.remove(raw)
    print(f"{out}: {size / 1e6:.1f} МБ")
    return out


TEACHER_FILES = ["config.json", "generation_config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "LICENSE", "model.safetensors"]


def teacher(repo="Qwen/Qwen3-0.6B", name="qwen3-0.6b"):
    """Ядро-донор для пересадки знаний (Apache-2.0)."""
    out = os.path.join(ROOT, name)
    os.makedirs(out, exist_ok=True)
    for f in TEACHER_FILES:
        fetch(f"https://huggingface.co/{repo}/resolve/main/{f}", os.path.join(out, f))
    return out


def main():
    import sys
    os.makedirs(ROOT, exist_ok=True)
    if "--teacher" in sys.argv:
        teacher()
    for name, url in BENCH.items():
        fetch(url, os.path.join(ROOT, name))
    with gzip.open(os.path.join(ROOT, "humaneval.jsonl.gz"), "rt") as f:
        n = sum(1 for _ in f)
    print("HumanEval задач:", n)
    for lang in CORPORA:
        corpus(lang)


if __name__ == "__main__":
    main()
