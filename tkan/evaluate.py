"""Оценка обученной «Ткани»: экзамен при разном числе циклов мысли r, язык, нарезка, образцы речи.

python -m tkan.evaluate --run A_learned --r 1 2 4 6 8 12
"""
import argparse
import json
import os
import time

import torch

from .agent import ToolAgent
from .bench import exam as E
from .bench import tasks as T
from .data import Mixture
from .model import Config, Tkan
from .train import RUNS, bits_per_byte

SEGMENT_SAMPLES = [
    "Москва — столица России, крупнейший по численности населения город страны.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    "Вопрос: Сколько будет 47 + 38?\nОтвет: 85",
]

FREE_PROMPTS = [
    "Вопрос: Привет! Расскажи, что ты умеешь.\nОтвет:",
    "Question: Hello! What can you do?\nAnswer:",
    "Санкт-Петербург был основан",
    "The theory of evolution",
]


class Agent:
    def __init__(self, model, r):
        self.model, self.r = model, r

    def answer(self, prompt, max_new):
        out, conf = self.model.generate(prompt.encode("utf-8"), max_new=max_new, r=self.r)
        return out.decode("utf-8", errors="replace"), conf


def load(run):
    ck = torch.load(os.path.join(RUNS, run, "model.pt"), map_location="cpu")
    model = Tkan(Config(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def segment(model, text):
    x = torch.tensor([list(text.encode("utf-8"))])
    _, b, _ = model(x, r=model.cfg.r_train_max)
    raw = text.encode("utf-8")
    parts, cur = [], bytearray()
    for i, byte in enumerate(raw):
        if b[0, i] and cur:
            parts.append(cur.decode("utf-8", errors="replace"))
            cur = bytearray()
        cur.append(byte)
    parts.append(cur.decode("utf-8", errors="replace"))
    return "|".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--r", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    ap.add_argument("--n", type=int, default=10, help="задач на ячейку домен×уровень")
    ap.add_argument("--real", type=int, default=10, help="задач MGSM на язык")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tools", action="store_true", help="разрешить модели запускать Python")
    ap.add_argument("--skip_bpb", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model, ck = load(args.run)
    out_dir = os.path.join(RUNS, args.run)
    mix = Mixture(512)
    report = {"run": args.run, "params": model.n_params(), "bytes_seen": ck["bytes"], "cfg": ck["cfg"]}

    # язык: bits-per-byte на отложенной Википедии и коде при разном r
    report["bpb"] = {}
    for r in ([] if args.skip_bpb else sorted(set([1, model.cfg.r_train_max] + args.r))):
        report["bpb"][r] = {name: bits_per_byte(model, mix, name, n=8, r=r) for name in ("wiki_ru", "wiki_en", "code")}
        print("bpb r=", r, report["bpb"][r], flush=True)

    # нарезка, которую модель нашла сама
    report["segments"] = [segment(model, s) for s in SEGMENT_SAMPLES]
    for s in report["segments"]:
        print(s)

    # экзамен при разном числе циклов мысли
    tasks = T.test_set(n_per_cell=args.n)
    report["exam"] = {}
    tag = "_tools" if args.tools else ""
    for r in args.r:
        t0 = time.time()
        agent = ToolAgent(model, r) if args.tools else Agent(model, r)
        rows = E.run(agent, tasks, log_path=os.path.join(out_dir, f"exam{tag}_r{r}.jsonl"))
        acc, cal = E.summarize(rows)
        report["exam"][r] = {"acc": {"|".join(map(str, k)): v for k, v in acc.items()}, "ece": cal}
        print(f"r={r}: всего {100 * acc[('ALL', 'both', 'all')]:.1f}%  ru {100 * acc[('ALL', 'ru', 'all')]:.1f}%  "
              f"en {100 * acc[('ALL', 'en', 'all')]:.1f}%  ECE {cal:.3f}  ({time.time() - t0:.0f} с)", flush=True)
        with open(os.path.join(out_dir, f"exam{tag}_r{r}.md"), "w") as f:
            f.write(E.table(rows, f"{args.run}, r={r}"))

    # реальный бенчмарк (честно: на таком масштабе ожидаем ~0)
    if args.real:
        best_r = max(args.r, key=lambda r: report["exam"][r]["acc"]["ALL|both|all"])
        real = T.load_mgsm("ru", args.real) + T.load_mgsm("en", args.real)
        rows = E.run(ToolAgent(model, best_r) if args.tools else Agent(model, best_r), real)
        acc, _ = E.summarize(rows)
        report["mgsm"] = {"r": best_r, "ru": acc.get(("ALL", "ru", "all")), "en": acc.get(("ALL", "en", "all")),
                          "sample": rows[0]["out"][:200]}
        print("MGSM", report["mgsm"], flush=True)

    # свободная речь
    report["samples"] = {}
    for p in FREE_PROMPTS:
        out, _ = model.generate(p.encode("utf-8"), max_new=160, r=model.cfg.r_train_max)
        report["samples"][p] = out.decode("utf-8", errors="replace")
        print("----", p, "\n", report["samples"][p], flush=True)

    json.dump(report, open(os.path.join(out_dir, f"report{tag}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
