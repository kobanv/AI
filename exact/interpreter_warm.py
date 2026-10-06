"""Контроль к эксперименту 3: помогают ли повторения, если исключить трудность оптимизации.

Исполнитель с K = 4 стартует из обученного однопроходного (K = 1): общие веса и код первого шага копируются,
у шагов 2–4 «вентиль» скрытого слоя выставлен в ноль (γ = −1), так что в старте он считает ровно то же.
Дальше обе версии — K = 1 и K = 4 — дообучаются одинаковое число шагов на тех же данных.
Если K = 4 не становится лучше K = 1, повторения на этом масштабе качество не возвращают.

python -m exact.interpreter_warm --a 12
"""
import argparse
import json
import os
import random
import time

import torch

from .interpreter import Executor, collect, evaluate, train
from .probe import MODEL, OUT, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=12)
    ap.add_argument("--span", type=int, default=4)
    ap.add_argument("--H", type=int, default=4096)
    ap.add_argument("--base_local", type=int, default=300)
    ap.add_argument("--base_loop", type=int, default=80)
    ap.add_argument("--more_local", type=int, default=200)
    ap.add_argument("--more_loop", type=int, default=80)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    a, b = args.a, args.a + args.span
    out_dir = os.path.join(OUT, f"interp_{a}_{b}")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "data.pt")
    if os.path.exists(cache):
        train_data, test_data = torch.load(cache)
    else:
        train_data = collect(model, texts(tok, 192, 128, seed=11), a, b)
        test_data = collect(model, texts(tok, 24, 128, seed=12), a, b, keep_logits=True)
        torch.save((train_data, test_data), cache)
    d = model.config.hidden_size
    log = open(os.path.join(out_dir, "warm_log.jsonl"), "w")
    t0 = time.time()

    base = Executor(d, args.H, b - a, 1)
    train(model, base, train_data, a, b, args.base_local, args.base_loop, 256, 4, 2e-3, log)
    res = {"k1_base": evaluate(model, base, test_data, a, b)}
    print("K=1 база:", json.dumps(res["k1_base"]), flush=True)

    k4 = Executor(d, args.H, b - a, 4)
    with torch.no_grad():
        k4.gate_up[0].weight.copy_(base.gate_up[0].weight)
        k4.down[0].weight.copy_(base.down[0].weight)
        k4.scale.copy_(base.scale)
        k4.alpha[:, 0] = base.alpha[:, 0]
        k4.beta[:, 0] = base.beta[:, 0]
        k4.gamma[:, 0] = base.gamma[:, 0]
        k4.gamma[:, 1:] = -1.0                       # шаги 2–4 выключены: в старте K=4 ≡ K=1
    res["k4_warm_start"] = evaluate(model, k4, test_data, a, b)
    print("K=4 в старте (должно совпасть с K=1):", json.dumps(res["k4_warm_start"]), flush=True)

    # одинаковое дообучение обеих версий (масштаб выхода не сбрасываем — продолжаем с достигнутого)
    for name, ex in (("k1_more", base), ("k4_more", k4)):
        train(model, ex, train_data, a, b, args.more_local, args.more_loop, 256, 4, 5e-4, log, init_scale=False)
        res[name] = evaluate(model, ex, test_data, a, b)
        print(name, json.dumps(res[name]), flush=True)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(os.path.join(out_dir, "warm_results.json"), "w"), indent=1)
    print("готово", flush=True)


if __name__ == "__main__":
    main()
