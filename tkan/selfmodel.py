"""Самомодель: «знаю ли я ответ?» — выученная из собственного опыта, а не вписанная.

Модель решает обучающие (не тестовые) задачи, проверяет себя исполнителем/сверкой
и запоминает, когда была права. По этому опыту обучается маленький предсказатель
P(верно | признаки). Признаки — только внутренние сигналы самой модели:
  - уверенность в ответе (совместная вероятность, минимальная вероятность байта);
  - согласие с собой при разной глубине мысли (ответ при r и при 2r совпал?);
  - длина ответа.
Домен задачи и правильный ответ предсказателю не показываются.

python -m tkan.selfmodel --run A_learned --r 6 --n 600
"""
import argparse
import itertools
import json
import math
import os

import torch

from .bench import exam as E
from .bench import tasks as T
from .train import RUNS


def features(model, prompt, max_new, r):
    a, pa = model.generate(prompt.encode("utf-8"), max_new=max_new, r=r)
    b, pb = model.generate(prompt.encode("utf-8"), max_new=max_new, r=2 * r)
    ta = a.decode("utf-8", errors="replace").split("\n\n")[0].strip()
    tb = b.decode("utf-8", errors="replace").split("\n\n")[0].strip()
    f = [math.log(max(pa, 1e-12)) / 10, math.log(max(pb, 1e-12)) / 10, float(ta == tb), len(ta) / 50, 1.0]
    return a.decode("utf-8", errors="replace"), f


class SelfModel:
    def __init__(self, w=None):
        self.w = torch.tensor(w) if w is not None else None

    def fit(self, X, y, steps=2000):
        X, y = torch.tensor(X), torch.tensor(y, dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=steps)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w, y) + 1e-3 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach()
        return self

    def __call__(self, f):
        return float(torch.sigmoid(torch.tensor(f) @ self.w))


class SelfAwareAgent:
    """Агент, который отвечает и сам оценивает, прав ли он."""

    def __init__(self, model, r, selfmodel):
        self.model, self.r, self.sm = model, r, selfmodel

    def answer(self, prompt, max_new):
        out, f = features(self.model, prompt, max_new, self.r)
        return out, self.sm(f)


def experience(model, r, n, seed=777):
    """Опыт на обучающих задачах: что модель ответила и была ли права."""
    X, y = [], []
    for t in itertools.islice(T.train_stream(seed), n):
        out, f = features(model, t.prompt, E.max_new_for(t), r)
        X.append(f)
        y.append(float(E.is_correct(t, out)))
    return X, y


def main():
    from .evaluate import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--r", type=int, default=6)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--test_n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, _ = load(args.run)

    X, y = experience(model, args.r, args.n)
    sm = SelfModel().fit(X, y)
    print(f"опыт: {len(y)} задач, правильно {100 * sum(y) / len(y):.1f}%; веса самомодели {sm.w.tolist()}")

    tasks = T.test_set(n_per_cell=args.test_n)
    rows = E.run(SelfAwareAgent(model, args.r, sm), tasks)
    acc, cal = E.summarize(rows)
    res ={"r": args.r, "train_experience": len(y), "train_acc": sum(y) / len(y), "weights": sm.w.tolist(),
           "test_acc": acc[("ALL", "both", "all")], "ece_selfmodel": cal}
    # «знает, что не знает»: точность среди ответов, в которых самомодель уверена > 0.5
    conf_rows = [r_ for r_ in rows if r_["conf"] > 0.5]
    res["acc_when_confident"] = sum(r_["ok"] for r_ in conf_rows) / max(1, len(conf_rows))
    res["share_confident"] = len(conf_rows) / len(rows)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(RUNS, args.run, "selfmodel.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
