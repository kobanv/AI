"""Пункт 1 предложения: есть ли у экспертов общие конструкции W_e = Σ a_ej B_j + R_e, и что стоит точное хранение.

python -m exact.expert_structure --layer 16
"""
import argparse
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .probe import MODEL


def entropy_bits(x):
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    moe = model.model.layers[args.layer].block_sparse_moe
    GU = moe.experts.gate_up_proj            # (E, 2I, d) bf16
    E = GU.shape[0]

    # 1. точное хранение bf16: энтропия знака+порядка и мантиссы (как в DFloat11)
    bits = GU.view(torch.int16).numpy().astype(np.uint16)
    hi, lo = bits >> 8, bits & 0xFF                     # старший байт ≈ знак+порядок, младший ≈ мантисса
    h_hi, h_lo = entropy_bits(hi), entropy_bits(lo)
    print(f"bf16: старший байт (знак+порядок) {h_hi:.2f} бит из 8, младший (мантисса) {h_lo:.2f} из 8 "
          f"→ точно без потерь ≈ {h_hi + h_lo:.2f} бит/вес (экономия {100 * (1 - (h_hi + h_lo) / 16):.0f}%)")

    # 2. общие компоненты экспертов без перестановок: SVD по экспертам
    W = GU.float().reshape(E, -1)
    W = W - W.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(W) ** 2
    expl = sv.cumsum(0) / sv.sum()
    print("доля, объяснённая k общими компонентами (без перестановок): " +
          ", ".join(f"k={k}: {100 * expl[k - 1]:.1f}%" for k in (1, 4, 8, 16, 32)))

    # 3. с лучшей перестановкой нейронов: насколько строки эксперта A похожи на строки эксперта B
    I = GU.shape[1] // 2
    A, B = GU[0, :I].float(), GU[1, :I].float()          # gate-строки двух экспертов (нейроны)
    An, Bn = torch.nn.functional.normalize(A, dim=1), torch.nn.functional.normalize(B, dim=1)
    C = (An @ Bn.T).numpy()
    r, c = linear_sum_assignment(-np.abs(C))
    matched = np.abs(C[r, c])
    rand = np.abs(C).mean()
    print(f"нейроны эксперта 0 vs 1 после лучшей перестановки: |корреляция| медиана {np.median(matched):.3f} "
          f"(90% < {np.quantile(matched, 0.9):.3f}); случайная пара строк: {rand:.3f}; "
          f"для случайных векторов размерности {A.shape[1]} ожидается ~{math.sqrt(2 / math.pi / A.shape[1]):.3f}")
    # сколько энергии строк B объясняет лучшая сопоставленная строка A (с подбором множителя)
    resid = 1 - matched ** 2
    print(f"остаток после вычитания сопоставленного нейрона: {100 * resid.mean():.1f}% энергии")


if __name__ == "__main__":
    main()
