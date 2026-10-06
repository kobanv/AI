"""Эксперимент 3: «компактный интерпретатор» — один общий исполнитель с кодами операций вместо нескольких MoE-блоков.

Гипотеза (пользователь): значительную часть вычислений модели можно выразить короткими программами
для одного общего обученного исполнителя. Проверяем на участке из 4 подряд идущих MoE-блоков granite.

Исполнитель для MoE-блока слоя l (u — вход блока после нормы, выход — приближение смеси экспертов F_l(u)):
    z₀ = 0
    z_{k+1} = z_k + down( SiLU(gate(x_k)) · up(x_k) · (1 + γ_{l,k}) ),   x_k = norm(u + z_k)·(1 + α_{l,k}) + β_{l,k}
    F̂_l(u) = s_l · z_K
Общие веса (gate/up/down) одни на все слои и шаги; «программа» слоя — K кодов (α, β, γ) по ~7 тыс. чисел.
Больше шагов K — больше вычислений теми же весами.

Контроль: отдельный маленький блок на каждый слой с тем же общим числом параметров, K = 1.

Обучение:
  1. локально — на парах (u_l, F_l(u_l)) из исходной модели;
  2. «на своих состояниях» — участок целиком: настоящие слои внимания получают состояния, созданные
     исполнителем; цель — состояние на выходе участка у исходной модели.
Оценка: ошибка участка, KL итоговых предсказаний к исходной модели, совпадение лучшего токена,
биты на токен на отложенном тексте, экзамен (3 примера формата), образцы свободной генерации.

python -m exact.interpreter --a 12
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .probe import MODEL, OUT, texts


def rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class Executor(nn.Module):
    def __init__(self, d, H, n_layers, K, shared=True):
        super().__init__()
        self.K, self.shared, self.n_layers = K, shared, n_layers
        nb = 1 if shared else n_layers
        self.gate_up = nn.ModuleList([nn.Linear(d, 2 * H, bias=False) for _ in range(nb)])
        self.down = nn.ModuleList([nn.Linear(H, d, bias=False) for _ in range(nb)])
        for m in self.down:
            nn.init.zeros_(m.weight)
        self.alpha = nn.Parameter(torch.zeros(n_layers, K, d))
        self.beta = nn.Parameter(torch.zeros(n_layers, K, d))
        self.gamma = nn.Parameter(torch.zeros(n_layers, K, H))
        self.scale = nn.Parameter(torch.ones(n_layers))

    def forward(self, u, l):
        blk = 0 if self.shared else l
        z = torch.zeros_like(u)
        for k in range(self.K):
            x = rms(u + z) * (1 + self.alpha[l, k]) + self.beta[l, k]
            g, v = self.gate_up[blk](x).chunk(2, -1)
            z = z + self.down[blk](F.silu(g) * v * (1 + self.gamma[l, k]))
        return z * self.scale[l]

    def unique_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith(("gate_up", "down")))

    def code_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith(("gate_up", "down")))


class Slot(nn.Module):
    """Подставляется вместо MoE-блока слоя l."""

    def __init__(self, ex, l):
        super().__init__()
        self.ex, self.l = ex, l

    def forward(self, x):
        return self.ex(x.float(), self.l).to(x.dtype)


# ---------------------------------------------------------------- данные
@torch.no_grad()
def collect(model, seqs, a, b, keep_logits=False):
    """Для каждой последовательности: h_a, h_b, входы и выходы MoE-блоков участка (и логиты — для теста)."""
    caps = {"u": [], "f": []}
    hooks = []
    for l in range(a, b):
        moe = model.model.layers[l].block_sparse_moe
        def hook(m, inp, out, l=l):          # ничего не возвращает: выход блока не подменяется
            caps["u"].append((l, inp[0][0].clone()))
            caps["f"].append((l, out[0].clone()))
        hooks.append(moe.register_forward_hook(hook))
    data = []
    for _, ids in seqs:
        caps["u"].clear(), caps["f"].clear()
        out = model(torch.tensor([ids]), output_hidden_states=True)
        hs = out.hidden_states
        data.append({"ids": ids, "ha": hs[a][0].clone(), "hb": hs[b][0].clone(),
                     "u": torch.stack([u for _, u in caps["u"]]), "f": torch.stack([f for _, f in caps["f"]]),
                     "logits": out.logits[0].float() if keep_logits else None})
    for h in hooks:
        h.remove()
    return data


def span_forward(model, ha, a, b):
    """Участок слоёв a..b−1 вручную (как в модели), с тем, что сейчас стоит на месте MoE-блоков."""
    h = ha[None]
    rot = model.model.rotary_emb(h, torch.arange(h.shape[1])[None])
    for l in range(a, b):
        h = model.model.layers[l](h, attention_mask=None, position_embeddings=rot)
        h = h[0] if isinstance(h, tuple) else h
    return h[0]


def install(model, ex, a, b):
    saved = [model.model.layers[l].block_sparse_moe for l in range(a, b)]
    for i, l in enumerate(range(a, b)):
        model.model.layers[l].block_sparse_moe = Slot(ex, i)
    return saved


def restore(model, saved, a):
    for i, m in enumerate(saved):
        model.model.layers[a + i].block_sparse_moe = m


# ---------------------------------------------------------------- обучение
def train(model, ex, train_data, a, b, local_steps, loop_steps, bsz_rows, bsz_seq, lr, log, init_scale=True):
    opt = torch.optim.AdamW(ex.parameters(), lr=lr, weight_decay=0.0)
    n_l = b - a
    U = torch.cat([d["u"] for d in train_data], 1).float()          # (n_l, N, d)
    Fo = torch.cat([d["f"] for d in train_data], 1).float()
    fnorm = Fo.pow(2).mean((1, 2))                                    # (n_l,)
    if init_scale:
        with torch.no_grad():                                         # масштаб выхода — по эталону
            ex.scale.copy_(Fo.pow(2).mean((1, 2)).sqrt())
    t0 = time.time()
    total = local_steps + loop_steps
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr * min(1, (step + 1) / 30) * 0.5 * (1 + math.cos(math.pi * step / total))
        if step == local_steps and loop_steps:
            saved = install(model, ex, a, b)                          # дальше участок считает исполнитель
        if step < local_steps:                                        # 1. локально, на идеальных входах
            idx = torch.randint(0, U.shape[1], (bsz_rows,))
            loss = sum(((ex(U[l, idx], l) - Fo[l, idx]).pow(2).mean() / fnorm[l]) for l in range(n_l)) / n_l
        else:                                                         # 2. на собственных состояниях участка
            batch = random.sample(train_data, bsz_seq)
            loss = 0
            for d in batch:
                hb = span_forward(model, d["ha"], a, b).float()
                ref = d["hb"].float()
                loss = loss + (hb - ref).pow(2).sum() / (ref - d["ha"].float()).pow(2).sum()
            loss = loss / bsz_seq
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ex.parameters(), 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            rec = {"step": step + 1, "phase": "local" if step < local_steps else "closed_loop",
                   "loss": float(loss), "min": round((time.time() - t0) / 60, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
    if loop_steps:
        restore(model, saved, a)
    return (time.time() - t0) / 60


@torch.no_grad()
def evaluate(model, ex, test_data, a, b):
    """Ошибка участка на своих состояниях и последствия для итоговых предсказаний."""
    span_err, kl, agree, bits_ref, bits_new, n = 0.0, 0.0, 0, 0.0, 0.0, 0
    saved = install(model, ex, a, b) if ex is not None else None
    for d in test_data:
        hb = span_forward(model, d["ha"], a, b).float()
        span_err += float((hb - d["hb"].float()).pow(2).sum() / (d["hb"].float() - d["ha"].float()).pow(2).sum())
        ids = torch.tensor([d["ids"]])
        lg = model(ids).logits[0].float()
        ref = d["logits"]
        lp, lr_ = lg.log_softmax(-1), ref.log_softmax(-1)
        kl += float(F.kl_div(lp, lr_, log_target=True, reduction="sum"))
        agree += int((lg.argmax(-1) == ref.argmax(-1)).sum())
        tgt = ids[0, 1:]
        bits_new += float(-lp[:-1].gather(1, tgt[:, None]).sum()) / math.log(2)
        bits_ref += float(-lr_[:-1].gather(1, tgt[:, None]).sum()) / math.log(2)
        n += len(d["ids"])
    if saved is not None:
        restore(model, saved, a)
    m = len(test_data)
    return {"span_rel_err": span_err / m, "kl_per_token": kl / n, "top1_agree": agree / n,
            "bits_per_token": bits_new / (n - m), "bits_per_token_original": bits_ref / (n - m)}


class Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class ZeroEx(nn.Module):
    def forward(self, u, l):
        return torch.zeros_like(u)


@torch.no_grad()
def exam(model, tok, ex, a, b, n):
    """Экзамен с теми же 3 примерами формата, что у эталонов (подмножество: n задач на ячейку)."""
    from tkan.baselines import FewShot
    from tkan.bench import exam as E
    from tkan.bench import tasks as T
    shots = FewShot()
    saved = install(model, ex, a, b) if ex is not None else None
    rows = []
    for t in T.test_set(n_per_cell=n):
        shots.current = t.domain
        text = shots.prefix(t.prompt)
        ids = tok(text, return_tensors="pt")["input_ids"]
        out = model.generate(ids, max_new_tokens=max(8, E.max_new_for(t) // 2), do_sample=False,
                             stop_strings=["\n\n"], tokenizer=tok)
        ans = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        rows.append({"domain": t.domain, "lang": t.lang, "level": t.level, "ok": bool(E.is_correct(t, ans)),
                     "conf": 0.5, "prompt": t.prompt, "out": E.extract(t, ans), "ref": t.answer})
    samples = {}
    for p in ["Москва — это", "The theory of evolution", "def quicksort(arr):"]:
        ids = tok(p, return_tensors="pt")["input_ids"]
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
        samples[p] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    if saved is not None:
        restore(model, saved, a)
    acc, _ = E.summarize(rows)
    return {"exam_ru": acc[("ALL", "ru", "all")], "exam_en": acc[("ALL", "en", "all")],
            "exam_by_domain": {d: acc.get((d, "ru", "all"), 0) / 2 + acc.get((d, "en", "all"), 0) / 2
                               for d in T.DOMAINS}, "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, default=12)
    ap.add_argument("--span", type=int, default=4)
    ap.add_argument("--variants", default="indep_1024_k1,shared_4096_k1,shared_4096_k2,shared_4096_k4,shared_2048_k4")
    ap.add_argument("--n_train", type=int, default=192)
    ap.add_argument("--n_test", type=int, default=24)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--local_steps", type=int, default=300)
    ap.add_argument("--loop_steps", type=int, default=100)
    ap.add_argument("--exam_n", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
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
    t0 = time.time()
    train_data = collect(model, texts(tok, args.n_train, args.seq, seed=11), a, b)
    test_data = collect(model, texts(tok, args.n_test, args.seq, seed=12), a, b, keep_logits=True)
    d = model.config.hidden_size
    moe_params = sum(p.numel() for l in range(a, b) for p in model.model.layers[l].block_sparse_moe.parameters())
    active = (b - a) * model.config.num_experts_per_tok * 3 * d * model.config.intermediate_size
    print(f"данные собраны за {(time.time() - t0) / 60:.1f} мин; параметров MoE в участке {moe_params / 1e6:.1f}M, "
          f"активных умножений на токен {active / 1e6:.1f}M", flush=True)
    results = {"span": [a, b], "moe_params": moe_params, "moe_active_macs": active}
    results["original"] = {**evaluate(model, None, test_data, a, b), **exam(model, tok, None, a, b, args.exam_n)}
    print("исходная модель:", json.dumps(results["original"], ensure_ascii=False)[:400], flush=True)
    results["removed"] = evaluate(model, ZeroEx(), test_data, a, b)
    print("MoE участка выключены:", json.dumps(results["removed"]), flush=True)
    log = open(os.path.join(out_dir, "train_log.jsonl"), "w")
    for v in args.variants.split(","):
        kind, H, K = v.split("_")
        H, K = int(H), int(K[1:])
        ex = Executor(d, H, b - a, K, shared=(kind == "shared"))
        log.write(json.dumps({"variant": v}) + "\n")
        mins = train(model, ex, train_data, a, b, args.local_steps, args.loop_steps, 256, 4, 2e-3, log)
        r = evaluate(model, ex, test_data, a, b)
        r.update(exam(model, tok, ex, a, b, args.exam_n))
        r.update({"unique_params": ex.unique_params(), "code_params": ex.code_params(),
                  "compression_vs_moe": moe_params / ex.unique_params(),
                  "macs_per_token": (b - a) * K * 3 * d * H if kind == "shared" else (b - a) * 3 * d * H,
                  "train_minutes": mins})
        results[v] = r
        print(v, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()
                             if k not in ("samples", "exam_by_domain")}, ensure_ascii=False), flush=True)
        json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), ensure_ascii=False, indent=1)
    print(f"готово за {(time.time() - t0) / 60:.0f} мин", flush=True)


if __name__ == "__main__":
    main()
