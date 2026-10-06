"""Эксперимент 2 «точного движка»: читать биты весов постепенно и останавливаться,
когда доказано, что выход операции округлится в то же значение BF16, что и у эталона.

Гипотеза (пользователь): вычислить y = W x по части битов весов, получить строгий интервал [L, U],
содержащий результат эталонного ядра; если round_bf16(L) == round_bf16(U), выход доказан побитово —
следующий слой получает тот же тензор, ошибки между слоями не накапливаются.

Формат весов BF16: знак (1) + порядок (8) + мантисса (7). Читаем по «плоскостям»:
  сначала знак и порядок (9 бит — без них не построить границу), затем мантиссу по одному биту.
Если прочитаны старшие k бит мантиссы, истинный вес = w_k + знак·δ, 0 ≤ δ ≤ D(k, порядок) (усечение к нулю).
Интервал вклада Σ знак_i·δ_i·x_i точный покоординатно: A⁺x⁺ + A⁻x⁻ сверху и A⁺x⁻ + A⁻x⁺ снизу.
Эталонное ядро накапливает в FP32 в неизвестном порядке: добавляем строгую границу ошибки
накопления (n−1)·2⁻²⁴·Σ|w x| (любой порядок суммирования). Отдельно считаем «идеальный» случай без неё.

Меряем на gate/up экспертов одного MoE-слоя granite-3.1-3b-a800m:
  1. доля выходов, доказанных при k = 0..7 прочитанных битах мантиссы;
  2. сколько бит на вес пришлось бы читать при адаптивном дочитывании (и экономия против 16);
  3. совпадают ли доказанные значения побитово с выходом настоящего BF16-ядра PyTorch;
  4. время проверки против самого умножения.

python -m exact.bitplanes --layer 16 --experts 6
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .probe import MODEL, OUT, collect, texts

U32 = 2.0 ** -24


def bf16_fields(w):
    b = w.view(torch.int16).to(torch.int32) & 0xFFFF
    return (b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F


def truncated(sign, exp, man, k):
    """Вес по знаку, порядку и старшим k битам мантиссы + максимальный недочитанный остаток D (по модулю)."""
    keep = man & (0x7F ^ ((1 << (7 - k)) - 1)) if k < 7 else man
    e = exp.double()
    normal = exp > 0
    scale = torch.where(normal, torch.exp2(e - 127), torch.full_like(e, 2.0 ** -126))
    frac = keep.double() / 128 + normal.double()
    val = torch.where(sign.bool(), -1.0, 1.0) * scale * frac
    rest = ((1 << (7 - k)) - 1) / 128.0
    return val, scale * rest


def round_bf16(v):
    """Точное округление float64 → BF16 (к ближайшему, при равенстве — к чётному), без двойного округления."""
    m, e = np.frexp(v)                      # v = m·2^e, 0.5 ≤ |m| < 1
    r = np.rint(m * 256.0)                  # 8 значащих бит BF16; rint — половина к чётному
    return np.ldexp(r / 256.0, e)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--n_seq", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    L = args.layer
    X, _, _ = collect(model, texts(tok, args.n_seq, args.seq, 2), [L])
    X = X[L].to(torch.bfloat16)                          # вход ровно такой, какой видит ядро (BF16)
    moe = model.model.layers[L].block_sparse_moe
    Wr = moe.router.weight
    ti = (X.float() @ Wr.float().T).topk(moe.router.top_k, -1).indices
    GU = moe.experts.gate_up_proj                        # (E, 2I, d) BF16

    stats = {k: [0, 0] for k in range(8)}                # k → [доказано, всего] (строгий FP32-учёт)
    ideal = {k: [0, 0] for k in range(8)}                # без учёта ошибки накопления
    need_bits, need_bits_ideal, mism, checked = [], [], 0, 0
    t_check = t_mm = 0.0
    experts = torch.bincount(ti.flatten(), minlength=GU.shape[0]).argsort(descending=True)[:args.experts]
    for e in experts.tolist():
        rows = (ti == e).any(-1)
        x = X[rows]
        if x.shape[0] == 0:
            continue
        W = GU[e]                                        # (2I, d)
        t0 = time.time()
        ref = F.linear(x, W)                             # настоящее BF16-ядро PyTorch (эталон)
        t_mm += time.time() - t0
        ref64 = ref.double().numpy()
        xd = x.double()
        xp, xn = xd.clamp_min(0), xd.clamp_max(0)
        sign, exp, man = bf16_fields(W)
        first_k = np.full(ref64.shape, 8)                # минимальный k, при котором выход доказан (8 = не доказан)
        first_k_ideal = np.full(ref64.shape, 8)
        t0 = time.time()
        for k in range(8):
            wk, D = truncated(sign, exp, man, k)
            A = torch.where(sign.bool(), -D, D)          # вклад остатка: знак·δ, δ ∈ [0, D]
            Ap, An = A.clamp_min(0), A.clamp_max(0)
            yk = xd @ wk.T
            hi = yk + xp @ Ap.T + xn @ An.T
            lo = yk + xn @ Ap.T + xp @ An.T
            absum = xd.abs() @ (wk.abs() + D).T          # Σ|w||x| сверху
            acc = (x.shape[1] - 1) * U32 * absum         # строгая граница ошибки FP32-накопления
            for (lo_, hi_, store, fk) in ((lo - acc, hi + acc, stats, first_k), (lo, hi, ideal, first_k_ideal)):
                rl, rh = round_bf16(lo_.numpy()), round_bf16(hi_.numpy())
                ok = rl == rh
                store[k][0] += int(ok.sum())
                store[k][1] += ok.size
                fk[(fk == 8) & ok] = k
                if store is stats:
                    # проверка корректности: доказанное значение обязано совпасть с настоящим BF16-ядром
                    mism += int((rl[ok] != ref64[ok]).sum())
                    checked += int(ok.sum())
        t_check += time.time() - t0
        need_bits += (9 + np.where(first_k == 8, 7, first_k)).flatten().tolist()
        need_bits_ideal += (9 + np.where(first_k_ideal == 8, 7, first_k_ideal)).flatten().tolist()

    res = {
        "layer": L, "outputs": stats[0][1],
        "certified_by_k_strict": {k: stats[k][0] / stats[k][1] for k in stats},
        "certified_by_k_ideal": {k: ideal[k][0] / ideal[k][1] for k in ideal},
        "mean_bits_per_weight_strict": float(np.mean(need_bits)),
        "mean_bits_per_weight_ideal": float(np.mean(need_bits_ideal)),
        "bitwise_mismatch_of_certified": mism, "certified_checked": checked,
        "check_time_over_matmul": t_check / max(t_mm, 1e-9),
    }
    json.dump(res, open(os.path.join(OUT, "bitplanes.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
