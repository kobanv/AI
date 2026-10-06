"""Эксперимент 1 «точного движка»: можно ли доказать результат MoE-слоя, не читая большую часть весов.

Идея (по предложению пользователя):
  состояние x раскладывается как x = U a + r, где U — главные направления реальных состояний (из калибровки);
  W x = (W U) a + W r: компактную часть (W U) держим на GPU, остаток W r читаем только при необходимости.
  Строгая граница остатка: |w_i · r| = |(P⊥ w_i) · r| ≤ ‖P⊥ w_i‖ · ‖r‖, где P⊥ w_i — часть строки w_i,
  ортогональная U. Её норма — одно заранее посчитанное число на строку, а ‖r‖ дёшево считается из x.

Что меряем на одном MoE-слое (granite-3.1-3b-a800m: 40 экспертов, выбираются 8):
  1. сколько энергии состояния остаётся в остатке r при m главных направлениях;
  2. как часто ДОКАЗУЕМО совпадает выбор 8 экспертов (граница выбранных выше границы невыбранных);
  3. строгие интервалы на выход экспертов (через gate/up → SiLU → down, тоже с главными направлениями)
     и во сколько раз граница шире реальной ошибки;
  4. какую долю байтов слоя пришлось бы читать (компактная часть + полная подгрузка там, где доказать не удалось);
  5. меняется ли итоговый токен, если подставить компактный выход слоя (эмпирически, не доказательство).

python -m exact.probe --layers 2 16 30 --m 32 64 128 256 512
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODEL = os.path.join(DATA, "granite-moe")
OUT = os.path.join(ROOT, "runs", "exact_probe")

SILU_XMIN = -1.2784645427610738          # точка минимума SiLU
SILU_YMIN = SILU_XMIN / (1 + math.exp(-SILU_XMIN))


# ---------------------------------------------------------------- данные
def texts(tok, n_seq, seq, seed):
    """Окна текста: Википедия RU/EN (валидационная часть), код Python, задачи экзамена."""
    import random
    from tkan.bench import tasks as T
    rng = random.Random(seed)
    srcs = {}
    for name, path in [("wiki_ru", "wiki_ru.txt"), ("wiki_en", "wiki_en.txt"), ("code", "py_stdlib.txt")]:
        raw = open(os.path.join(DATA, path), "rb").read()[-2_000_000:]
        srcs[name] = raw.decode("utf-8", errors="ignore")
    out = []
    names = ["wiki_ru", "wiki_en", "code", "tasks"]
    for i in range(n_seq):
        name = names[i % 4]
        if name == "tasks":
            ts = T.test_set(n_per_cell=1, seed=seed + i)
            rng.shuffle(ts)
            txt = "".join(t.text for t in ts[:12])
        else:
            s = srcs[name]
            j = rng.randrange(0, len(s) - 20000)
            txt = s[j:j + 20000]
        ids = tok(txt, add_special_tokens=False)["input_ids"][:seq]
        out.append((name, ids))
    return out


@torch.no_grad()
def collect(model, seqs, layers):
    """Прогон модели: входы MoE-блоков нужных слоёв и итоговые логиты (запас победителя)."""
    caps = {L: [] for L in layers}
    hooks = []
    for L in layers:
        moe = model.model.layers[L].block_sparse_moe
        hooks.append(moe.register_forward_pre_hook(lambda m, inp, L=L: caps[L].append(inp[0][0].float().clone())))
    margins, top1 = [], []
    for _, ids in seqs:
        logits = model(torch.tensor([ids])).logits[0].float()
        t2 = logits.topk(2, dim=-1).values
        margins.append(t2[:, 0] - t2[:, 1])
        top1.append(logits.argmax(-1))
    for h in hooks:
        h.remove()
    return {L: torch.cat(v) for L, v in caps.items()}, torch.cat(margins), torch.cat(top1)


# ---------------------------------------------------------------- интервальная арифметика
def silu_interval(lo, hi):
    a, b = F.silu(lo), F.silu(hi)
    low = torch.minimum(a, b)
    inside = (lo <= SILU_XMIN) & (hi >= SILU_XMIN)
    low = torch.where(inside, torch.full_like(low, SILU_YMIN), low)
    return low, torch.maximum(a, b)


def mul_interval(a_lo, a_hi, b_lo, b_hi):
    c = torch.stack([a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi])
    return c.min(0).values, c.max(0).values


def basis(X, m):
    """Главные направления (без центрирования: нужна линейная часть W x целиком)."""
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    return Vt[:m].T.contiguous()                     # (d, m)


def perp_norms(W, U):
    """‖P⊥ w_i‖ для каждой строки W: часть строки, ортогональная пространству U."""
    proj = (W @ U) @ U.T
    return (W - proj).norm(dim=-1)


# ---------------------------------------------------------------- компактное представление слоя
class Compact:
    """То, что движок держал бы на GPU: W·U для маршрутизатора и gate/up, D·V для down, и числа для границ."""

    def __init__(self, moe, Xc, m, m2, top_k):
        self.top_k = top_k
        Wr = moe.router.weight.float()
        GU = moe.experts.gate_up_proj.float()
        D = moe.experts.down_proj.float()
        self.E, twoI, self.d = GU.shape
        self.I = twoI // 2
        self.U = basis(Xc, m)
        self.WrU, self.Wr_perp = Wr @ self.U, perp_norms(Wr, self.U)
        top_c = (Xc @ Wr.T).topk(top_k, dim=-1).indices
        self.GUU, self.GU_perp, self.V, self.DV, self.DVperp_2, self.D_2 = [], [], [], [], [], []
        for e in range(self.E):
            g, u = F.linear(Xc[(top_c == e).any(-1)], GU[e]).chunk(2, -1)
            H = F.silu(g) * u
            Ve = basis(H, min(m2, H.shape[0], self.I))
            self.GUU.append(GU[e] @ self.U)
            self.GU_perp.append(perp_norms(GU[e], self.U))
            self.V.append(Ve)
            self.DV.append(D[e] @ Ve)
            self.DVperp_2.append(torch.linalg.matrix_norm(D[e] - (D[e] @ Ve) @ Ve.T, 2))   # ‖D P⊥_V‖₂
            self.D_2.append(torch.linalg.matrix_norm(D[e], 2))
        self.full_bytes = Wr.numel() + top_k * (GU[0].numel() + D[0].numel())
        m2_eff = self.V[0].shape[1]
        self.compact_bytes = Wr.shape[0] * m + top_k * (twoI * m + self.d * m2_eff + self.I * m2_eff)

    def __call__(self, x):
        """Центр компактного вычисления + строгий радиус ‖y − ŷ‖₂ + доказан ли выбор экспертов."""
        a = x @ self.U
        rn = (x - a @ self.U.T).norm(dim=-1)
        s_hat = a @ self.WrU.T
        b = rn[:, None] * self.Wr_perp[None, :]
        tv, ti = s_hat.topk(self.top_k, dim=-1)
        sel = torch.zeros_like(s_hat, dtype=torch.bool).scatter_(1, ti, True)
        lo_sel = torch.where(sel, s_hat - b, torch.full_like(s_hat, float("inf"))).min(-1).values
        hi_rest = torch.where(~sel, s_hat + b, torch.full_like(s_hat, -float("inf"))).max(-1).values
        cert = lo_sel > hi_rest
        # интервалы весов смешивания: softmax по выбранным, логиты в [ŝ − b, ŝ + b]
        bs = torch.gather(b, 1, ti)
        lo, hi = tv - bs, tv + bs
        w_hat = torch.softmax(tv, -1)
        e_lo, e_hi = lo.exp(), hi.exp()
        w_lo = e_lo / (e_lo + (e_hi.sum(-1, keepdim=True) - e_hi))
        w_hi = e_hi / (e_hi + (e_lo.sum(-1, keepdim=True) - e_lo))
        dw = torch.maximum(w_hi - w_hat, w_hat - w_lo)
        y = torch.zeros_like(x)
        rad = torch.zeros(x.shape[0])
        for k in range(self.top_k):
            for e in ti[:, k].unique().tolist():
                t = (ti[:, k] == e).nonzero().flatten()
                z = a[t] @ self.GUU[e].T
                zr = rn[t, None] * self.GU_perp[e][None, :]
                (g_lo, u_lo), (g_hi, u_hi) = (z - zr).chunk(2, -1), (z + zr).chunk(2, -1)
                s_lo, s_hi = silu_interval(g_lo, g_hi)
                h_lo, h_hi = mul_interval(s_lo, s_hi, u_lo, u_hi)
                h_c, h_r = (h_lo + h_hi) / 2, (h_hi - h_lo) / 2
                c = h_c @ self.V[e]
                q = h_c - c @ self.V[e].T
                ye = c @ self.DV[e].T
                re = q.norm(dim=-1) * self.DVperp_2[e] + h_r.norm(dim=-1) * self.D_2[e]
                y[t] += w_hat[t, k, None] * ye
                rad[t] += dw[t, k] * (ye.norm(dim=-1) + re) + w_hi[t, k] * re
        return y, rad, cert, rn


def _exact(moe, x):
    Wr = moe.router.weight.float()
    GU = moe.experts.gate_up_proj.float()
    D = moe.experts.down_proj.float()
    tv, ti = (x @ Wr.T).topk(moe.router.top_k, dim=-1)
    w = torch.softmax(tv, -1)
    y = torch.zeros_like(x)
    for k in range(ti.shape[1]):
        for e in ti[:, k].unique().tolist():
            t = (ti[:, k] == e).nonzero().flatten()
            g, u = F.linear(x[t], GU[e]).chunk(2, -1)
            y[t] += w[t, k, None] * F.linear(F.silu(g) * u, D[e])
    return y, ti


@torch.no_grad()
def analyze_layer(moe, Xc, Xt, m, m2, top_k):
    C = Compact(moe, Xc, m, m2, top_k)
    y_hat, rad, cert, rn = C(Xt)
    y_ex, ti_ex = _exact(moe, Xt)
    err = (y_ex - y_hat).norm(dim=-1)
    yn = y_ex.norm(dim=-1).clamp_min(1e-9)
    ok = cert
    if ok.any():
        assert (err[ok] <= rad[ok] * (1 + 1e-3) + 1e-4).all(), "строгая граница нарушена — ошибка в выводе"
    resid = rn ** 2 / Xt.norm(dim=-1).clamp_min(1e-9) ** 2
    # доля байтов: компактная часть всегда + полные 8 экспертов там, где маршрут не доказан
    read = C.compact_bytes + (~cert).float().mean().item() * C.full_bytes
    return C, {
        "m": m, "m2": C.V[0].shape[1],
        "resid_energy_mean": float(resid.mean()), "resid_energy_p90": float(resid.quantile(0.9)),
        "route_certified": float(cert.float().mean()),
        "out_rel_err_mean": float((err / yn)[ok].mean()) if ok.any() else None,
        "out_rel_bound_mean": float((rad / yn)[ok].mean()) if ok.any() else None,
        "bound_looseness_median": float((rad / err.clamp_min(1e-9))[ok].median()) if ok.any() else None,
        "compact_frac_of_layer_bytes": C.compact_bytes / C.full_bytes,
        "read_frac_with_fallback": read / C.full_bytes,
    }


@torch.no_grad()
def downstream_flip(model, seqs, L, C, top1_ref):
    """Эмпирика «движка без последующих доказательств»: где маршрут доказан — компактный центр,
    где нет — точный расчёт. Меняется ли итоговый токен?"""
    moe = model.model.layers[L].block_sparse_moe

    def hook(mod, inp, out):
        x = inp[0][0].float()
        y_hat, _, cert, _ = C(x)
        y_ex, _ = _exact(mod, x)
        y = torch.where(cert[:, None], y_hat, y_ex)
        return y.to(out.dtype)[None]

    h = moe.register_forward_hook(hook)
    tops = [model(torch.tensor([ids])).logits[0].float().argmax(-1) for _, ids in seqs]
    h.remove()
    return float((torch.cat(tops) != top1_ref).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 16, 30])
    ap.add_argument("--m", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--m2", type=int, default=128)
    ap.add_argument("--n_seq", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    top_k = model.config.num_experts_per_tok

    t0 = time.time()
    calib = texts(tok, args.n_seq, args.seq, seed=1)
    test = texts(tok, args.n_seq, args.seq, seed=2)
    Xc, _, _ = collect(model, calib, args.layers)
    Xt, margins, top1 = collect(model, test, args.layers)
    print(f"собрано: калибровка {Xc[args.layers[0]].shape[0]} токенов, тест {Xt[args.layers[0]].shape[0]} "
          f"({time.time() - t0:.0f} с)", flush=True)
    m_q = margins.quantile(torch.tensor([0.1, 0.25, 0.5]))
    print(f"запас победителя в итоговых логитах: 10% токенов < {m_q[0]:.2f}, 25% < {m_q[1]:.2f}, медиана {m_q[2]:.2f}",
          flush=True)

    results = []
    for L in args.layers:
        moe = model.model.layers[L].block_sparse_moe
        # шумовой порог: точный расчёт слоя в fp32 вместо bf16 модели — сколько токенов меняется от одной лишь точности
        noise = downstream_flip(model, test, L, lambda x: (_exact(moe, x)[0], None,
                                                            torch.zeros(x.shape[0], dtype=torch.bool), None), top1)
        print(json.dumps({"layer": L, "noise_floor_flip_fp32_exact": noise}), flush=True)
        results.append({"layer": L, "noise_floor_flip_fp32_exact": noise})
        for m in args.m:
            C, res = analyze_layer(moe, Xc[L], Xt[L], m, args.m2, top_k)
            res.update({"layer": L, "final_token_flip": downstream_flip(model, test, L, C, top1)})
            results.append(res)
            print(json.dumps(res, ensure_ascii=False), flush=True)
    json.dump({"results": results, "margin_quantiles": m_q.tolist(), "minutes": (time.time() - t0) / 60},
              open(os.path.join(OUT, "results.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
