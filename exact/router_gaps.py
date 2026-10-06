"""Почему выбор экспертов не доказывается: запас между 8-м и 9-м экспертом против реальной ошибки и строгой границы.

python -m exact.router_gaps --layer 16
"""
import argparse

import torch

from .probe import MODEL, basis, collect, perp_norms, texts


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(4)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    L = args.layer
    Xc, _, _ = collect(model, texts(tok, 16, 256, 1), [L])
    Xt, _, _ = collect(model, texts(tok, 16, 256, 2), [L])
    Xc, Xt = Xc[L], Xt[L]
    Wr = model.model.layers[L].block_sparse_moe.router.weight.float()
    k = model.config.num_experts_per_tok
    s = Xt @ Wr.T
    srt = s.sort(-1, descending=True).values
    gap = srt[:, k - 1] - srt[:, k]
    print(f"слой {L}: запас 8-й vs 9-й эксперт: медиана {gap.median():.3f}, 10% < {gap.quantile(0.1):.3f}; "
          f"разброс оценок маршрутизатора (σ) {s.std():.2f}")
    for m in (256, 512, 1024, 1400):
        U = basis(Xc, m)
        a = Xt @ U
        r = Xt - a @ U.T
        s_hat = a @ (Wr @ U).T
        err = (s - s_hat).abs()
        bound = r.norm(dim=-1)[:, None] * perp_norms(Wr, U)[None, :]
        changed = (s_hat.topk(k, -1).indices.sort(-1).values != s.topk(k, -1).indices.sort(-1).values).any(-1)
        print(f"  m={m:4d}: реальная ошибка оценки медиана {err.median():.3f}, строгая граница медиана {bound.median():.3f}; "
              f"набор экспертов РЕАЛЬНО меняется у {100 * changed.float().mean():.1f}% токенов")


if __name__ == "__main__":
    main()
