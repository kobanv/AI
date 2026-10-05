"""Цикл саморазвития «Ткани»: сама проверяет себя, сама выбирает, что учить, сама добывает опыт.

В этом цикле нет ни одного ответа учителя. Есть только:
  Мир    — ставит задачи и говорит «верно/неверно» (исполнение кода, сверка результата);
  Агент  — модель с руками (Python) и памятью.

Один раунд:
  1. Самопроверка: карта навыков (домен × уровень), доля решённых задач в каждом.
  2. Выбор фокуса: «зона ближайшего развития» (c·(1−c)), прогресс прошлого раунда,
     любопытство к следующему уровню после освоенного. Освоенное и безнадёжное — реже.
  3. Практика: по каждой задаче несколько попыток (жадно + с температурой: разные пути мысли).
     В опыт попадают только решения, прошедшие проверку мира.
  4. Сон: дообучение на собственном проверенном опыте + повтор старого опыта и немного живого языка.
  5. Дневник: что выбрала, почему, сколько получилось, как изменилась карта навыков.

python -m tkan.selfimprove --run R_recur --rounds 8
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch

from .agent import ToolAgent
from .bench import exam as E
from .bench import tasks as T
from .data import Mixture
from .evaluate import load
from .train import RUNS

SKILLS = [(d, l) for d in T.DOMAINS for l in T.ALL_LEVELS]


class World:
    """Мир ставит задачи и проверяет решения. Экзаменационные задачи для практики закрыты."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.closed = {t.key for t in T.test_set(n_per_cell=10)}

    def problem(self, domain, level):
        """Новая открытая задача или None, если все задачи навыка ушли на экзамен (тогда практики нет)."""
        for _ in range(3000):
            key, versions, check, tests, tool = T.GENERATORS[domain](self.rng, level)
            if T.is_test_key(key) or key in self.closed:
                continue
            lang = self.rng.choice(T.LANGS)
            q, a = versions[lang]
            return T.Task(domain, lang, level, q, a, key, check, tests, tool)
        return None

    def open_skills(self):
        return [s for s in SKILLS if self.problem(*s) is not None]

    @staticmethod
    def check(task, out):
        return E.is_correct(task, out)


def trajectory(task, out):
    """Собственное решение агента как учебный текст (ответ мира не подставляется)."""
    return task.prompt + out.split("\n\n")[0] + "\n\n"


def assess(model, world, r, k):
    agent = ToolAgent(model, r)
    comp, tool_use = {}, {}
    for d, l in world.skills:
        ok = used = 0
        for _ in range(k):
            t = world.problem(d, l)
            out, _ = agent.answer(t.prompt, E.max_new_for(t))
            ok += world.check(t, out)
            used += "<py>" in out
        comp[(d, l)] = ok / k
        tool_use[(d, l)] = used / k
    return comp, tool_use


def choose_focus(comp, prev, n, rng):
    pri, why = {}, {}
    for d, l in comp:
        c = comp[(d, l)]
        lp = max(0.0, c - prev.get((d, l), c)) if prev else 0.0
        zone = c * (1 - c)
        below = comp.get((d, l - 1), 1.0) if l > 1 else 1.0
        curious = 0.15 if c < 0.1 and below >= 0.5 else 0.0
        pri[(d, l)] = zone + 2 * lp + curious + 0.01
        why[(d, l)] = (f"почти получается ({c:.0%})" if zone >= 0.16 else
                       f"растёт (+{lp:.0%})" if lp > 0.05 else
                       "следующий шаг после освоенного" if curious else
                       f"поддержание ({c:.0%})")
    chosen, pool = [], dict(pri)
    for _ in range(min(n, len(pool))):
        keys = list(pool)
        s = rng.choices(keys, weights=[pool[k] for k in keys])[0]
        chosen.append(s)
        pool.pop(s)
    return chosen, why


def practice(model, world, skills, r, problems, attempts, temperature):
    data, stats = [], {}
    greedy = ToolAgent(model, r)
    explorer = ToolAgent(model, r, temperature=temperature)
    for s in skills:
        solved = tried = 0
        for _ in range(problems):
            t = world.problem(*s)
            seen = set()
            got = False
            for i in range(attempts):
                agent = greedy if i == 0 else explorer
                out, _ = agent.answer(t.prompt, E.max_new_for(t))
                tried += 1
                if world.check(t, out):
                    traj = trajectory(t, out)
                    if traj not in seen:
                        seen.add(traj)
                        data.append(traj)
                    got = True
            solved += got
        stats[s] = {"solved": solved / problems, "attempts": tried}
    return data, stats


def sleep(model, opt, fresh, replay, mix, steps, bsz, seq, rng):
    """Консолидация: собственный проверенный опыт + повтор старого + немного живого языка."""
    model.train()
    losses = []
    for _ in range(steps):
        rows = []
        for _ in range(bsz):
            u = rng.random()
            if u < 0.1:
                rows.append(mix.sample_text(rng.choice(["wiki_ru", "wiki_en"]))[: seq + 1])
                continue
            pool = fresh if (u < 0.75 or not replay) else replay
            buf = bytearray()
            while len(buf) < seq + 1:
                buf += rng.choice(pool).encode("utf-8")
            rows.append(np.frombuffer(bytes(buf[: seq + 1]), dtype=np.uint8))
        x = torch.from_numpy(np.stack(rows).astype(np.int64)).to(model.head.weight.device)
        ce, rl, _ = model.loss(x)
        loss = ce + model.cfg.ratio_loss_w * rl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(ce.item())
    model.eval()
    return float(np.mean(losses))


def exam(model, r, n):
    rows = E.run(ToolAgent(model, r), T.test_set(n_per_cell=n))
    acc, cal = E.summarize(rows)
    return rows, acc


def comp_table(comp):
    lines = ["| домен | " + " | ".join(f"ур.{l}" for l in T.ALL_LEVELS) + " |",
             "|---|" + "---|" * len(T.ALL_LEVELS)]
    for d in T.DOMAINS:
        lines.append(f"| {d} | " + " | ".join(f"{100 * comp[(d, l)]:.0f}%" if (d, l) in comp else "—"
                                             for l in T.ALL_LEVELS) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--assess_k", type=int, default=8)
    ap.add_argument("--focus", type=int, default=8)
    ap.add_argument("--problems", type=int, default=16)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--sleep_steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--exam_n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model, ck = load(args.run)
    out_dir = os.path.join(RUNS, args.run + "_self")
    os.makedirs(out_dir, exist_ok=True)
    diary = open(os.path.join(out_dir, "diary.jsonl"), "w")
    world = World(args.seed + 1)
    world.skills = world.open_skills()
    closed = [f"{d}/{l}" for d, l in SKILLS if (d, l) not in world.skills]
    if closed:
        print("навыки без открытых задач (все ушли на экзамен):", ", ".join(closed), flush=True)
    mix = Mixture(256, seed=args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    t0 = time.time()
    rows0, acc0 = exam(model, args.r, args.exam_n)
    print(f"экзамен ДО: ru {acc0[('ALL', 'ru', 'all')]:.1%} en {acc0[('ALL', 'en', 'all')]:.1%}", flush=True)
    open(os.path.join(out_dir, "exam_before.md"), "w").write(E.table(rows0, f"{args.run} до саморазвития"))

    replay, prev = [], None
    history = []
    for rnd in range(1, args.rounds + 1):
        comp, tool_use = assess(model, world, args.r, args.assess_k)
        focus, why = choose_focus(comp, prev, args.focus, rng)
        fresh, stats = practice(model, world, focus, args.r, args.problems, args.attempts, args.temperature)
        loss = sleep(model, opt, fresh, replay, mix, args.sleep_steps, 32, 256, rng) if fresh else float("nan")
        replay = (replay + fresh)[-20000:]
        mean_c = sum(comp.values()) / len(comp)
        rec = {"round": rnd, "min": round((time.time() - t0) / 60, 1), "mean_competence": mean_c,
               "competence": {f"{d}/{l}": c for (d, l), c in comp.items()},
               "tool_use": {f"{d}/{l}": u for (d, l), u in tool_use.items() if u > 0},
               "focus": [{"skill": f"{d}/{l}", "why": why[(d, l)], "solved": stats[(d, l)]["solved"]} for d, l in focus],
               "new_experience": len(fresh), "replay": len(replay), "sleep_loss": loss}
        history.append(rec)
        diary.write(json.dumps(rec, ensure_ascii=False) + "\n")
        diary.flush()
        print(f"раунд {rnd} ({rec['min']} мин): средняя компетентность {mean_c:.1%}; "
              f"фокус: {', '.join(f['skill'] + ' — ' + f['why'] for f in rec['focus'])}; "
              f"новый опыт: {len(fresh)}", flush=True)
        prev = comp

    comp, _ = assess(model, world, args.r, args.assess_k)
    rows1, acc1 = exam(model, args.r, args.exam_n)
    print(f"экзамен ПОСЛЕ: ru {acc1[('ALL', 'ru', 'all')]:.1%} en {acc1[('ALL', 'en', 'all')]:.1%}", flush=True)
    open(os.path.join(out_dir, "exam_after.md"), "w").write(E.table(rows1, f"{args.run} после саморазвития"))
    summary = {"before": {k: acc0[("ALL", k, "all")] for k in T.LANGS},
               "after": {k: acc1[("ALL", k, "all")] for k in T.LANGS},
               "competence_start": history[0]["mean_competence"] if history else None,
               "competence_end": sum(comp.values()) / len(comp),
               "final_map": comp_table(comp), "minutes": (time.time() - t0) / 60}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), ensure_ascii=False, indent=1)
    torch.save({"cfg": ck["cfg"], "model": model.state_dict(), "step": ck["step"], "bytes": ck["bytes"]},
               os.path.join(out_dir, "model.pt"))
    print(summary["final_map"], flush=True)


if __name__ == "__main__":
    main()
