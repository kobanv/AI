"""Собирает итоги прогонов (только маленькие файлы: таблицы, дневники, json) в results/<метка>/.

python -m tkan.collect --tag colab_2026-10-05
"""
import argparse
import glob
import os
import shutil

from .train import RUNS

ROOT = os.path.dirname(RUNS)
KEEP = ("*.md", "*.json", "diary.jsonl", "log.jsonl", "eye_log.jsonl", "voice_log.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = os.path.join(ROOT, "results", args.tag)
    n = 0
    for run in sorted(glob.glob(os.path.join(RUNS, "*"))):
        for pat in KEEP:
            for f in glob.glob(os.path.join(run, pat)):
                if os.path.getsize(f) > 2_000_000:
                    continue
                dst = os.path.join(out, os.path.basename(run), os.path.basename(f))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(f, dst)
                n += 1
    print(f"собрано файлов: {n} → {out}")


if __name__ == "__main__":
    main()
