"""Обучение «Ткани» на CPU/GPU.

python -m tkan.train --name tkan0 --minutes 90
"""
import argparse
import json
import math
import os
import time

import torch

from .data import Mixture
from .model import Config, Tkan

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


@torch.no_grad()
def bits_per_byte(model, mix, name, n=8, bsz=8, r=None):
    model.eval()
    tot, cnt, chunks = 0.0, 0, 0
    dev = model.head.weight.device
    for x in mix.val_batches(name, n, bsz):
        x = x.to(dev)
        logits, b, _ = model(x[:, :-1], r=r)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum")
        tot += float(ce)
        cnt += x[:, 1:].numel()
        chunks += int(b.sum())
    model.train()
    return tot / cnt / math.log(2), cnt / max(1, chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tkan0")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--chunking", default="learned")
    ap.add_argument("--r_max", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--d_main", type=int, default=320)
    ap.add_argument("--d_byte", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tools", type=float, default=0.0, help="доля демонстраций с инструментом")
    ap.add_argument("--r_min", type=int, default=1)
    ap.add_argument("--s0_noise", type=float, default=0.0)
    ap.add_argument("--mix", default="wiki_ru=0.2,wiki_en=0.2,code=0.1,tasks=0.5")
    ap.add_argument("--domains", default=",".join(__import__("tkan.bench.tasks", fromlist=["x"]).DOMAINS))
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    cfg = Config(chunking=args.chunking, r_train_max=args.r_max, r_train_min=args.r_min, s0_noise=args.s0_noise,
                 d_main=args.d_main, d_byte=args.d_byte)
    model = Tkan(cfg).to(args.device)
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}
    mix = Mixture(args.seq, weights=weights, seed=args.seed, tool_prob=args.tools,
                  domains=tuple(args.domains.split(",")), levels=tuple(int(x) for x in args.levels.split(",")))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"параметров: {model.n_params() / 1e6:.2f}M, нарезка: {cfg.chunking}", flush=True)

    budget = args.minutes * 60
    t0 = time.time()
    step, seen = 0, 0
    log = open(os.path.join(out, "log.jsonl"), "w")
    while True:
        elapsed = time.time() - t0
        if elapsed > budget:
            break
        frac = elapsed / budget
        lr = args.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = mix.batch(args.bsz).to(args.device)
        ce, rl, b = model.loss(x)
        loss = ce + cfg.ratio_loss_w * rl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        seen += x.numel()
        if step % 50 == 0:
            rec = {"step": step, "min": round(elapsed / 60, 1), "bytes": seen, "ce_bits": ce.item() / math.log(2),
                   "bytes_per_chunk": x.numel() / max(1, int(b.sum())), "lr": lr}
            if step % 500 == 0 and args.mix.split(",")[0].split("=")[1] != "0":
                for name in ("wiki_ru", "wiki_en", "code"):
                    rec[f"val_{name}"], rec[f"bpc_{name}"] = bits_per_byte(model, mix, name, n=4)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(), "step": step, "bytes": seen, "args": vars(args)},
               os.path.join(out, "model.pt"))
    final = {"step": step, "bytes": seen, "minutes": (time.time() - t0) / 60, "params": model.n_params()}
    for name in ("wiki_ru", "wiki_en", "code"):
        final[f"val_{name}"], final[f"bpc_{name}"] = bits_per_byte(model, mix, name, n=16)
    json.dump(final, open(os.path.join(out, "final.json"), "w"), indent=1)
    print("ИТОГ", json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
