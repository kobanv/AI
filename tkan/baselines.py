"""Эталоны для сравнения: открытые модели на том же экзамене.

Формат экзамена модели не знаком, поэтому даём 3 примера из обучающей части того же домена
(few-shot) — честное «объяснение формата», без утечки тестовых задач.

python -m tkan.baselines --model data/qwen3-0.6b --n 10
"""
import argparse
import itertools
import json
import os
import time

import torch

from .bench import exam as E
from .bench import tasks as T
from .train import RUNS


class FewShot:
    """3 примера формата из обучающей части того же домена и языка (одинаково для всех эталонов)."""

    def __init__(self, shots=3):
        self.shots = {}
        stream = T.train_stream(seed=4242)
        need = {(d, l): shots for d in T.DOMAINS for l in T.LANGS}
        for t in itertools.islice(stream, 20000):
            k = (t.domain, t.lang)
            if need.get(k, 0) > 0:
                self.shots.setdefault(k, []).append(t.text)
                need[k] -= 1
        self.current = None

    def prefix(self, prompt):
        lang = "ru" if prompt.startswith("Вопрос") else "en"
        dom = self.current or "math"
        return "".join(self.shots.get((dom, lang), [])) + prompt


class HFAgent(FewShot):
    def __init__(self, path, shots=3):
        super().__init__(shots)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()

    @torch.no_grad()
    def answer(self, prompt, max_new):
        text = self.prefix(prompt)
        ids = self.tok(text, return_tensors="pt")["input_ids"]
        out = self.model.generate(ids, max_new_tokens=max(8, max_new // 2), do_sample=False,
                                  output_scores=True, return_dict_in_generate=True,
                                  stop_strings=["\n\n"], tokenizer=self.tok)
        new = out.sequences[0, ids.shape[1]:]
        lp = sum(float(torch.log_softmax(s[0], -1)[t]) for s, t in zip(out.scores, new))
        return self.tok.decode(new, skip_special_tokens=True), float(torch.exp(torch.tensor(lp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/qwen3-0.6b")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    agent = HFAgent(args.model)
    name = os.path.basename(args.model.rstrip("/"))
    out = os.path.join(RUNS, "baseline_" + name)
    os.makedirs(out, exist_ok=True)
    tasks = T.test_set(n_per_cell=args.n)
    rows, t0 = [], time.time()
    for t in tasks:
        agent.current = t.domain
        rows += E.run(agent, [t])
    with open(os.path.join(out, "exam.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = E.table(rows, f"{name} (3-shot), {time.time() - t0:.0f} с")
    open(os.path.join(out, "exam.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
