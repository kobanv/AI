"""Прогон экзамена: генерация ответов, проверка (в т.ч. исполнением кода), отчёт и калибровка.

Модель должна иметь метод
    answer(prompt: str, max_new: int) -> (text: str, confidence: float)
где confidence ∈ [0, 1] — собственная оценка модели, что ответ верен.
"""
import json
import subprocess
import sys
from collections import defaultdict

from . import tasks as T

RUNNER = r"""
import json, sys
src = sys.stdin.read()
d = json.loads(src)
ns = {}
try:
    exec(d["code"], ns)
    f = ns["f"]
    ok = all(f(x) == y for x, y in d["tests"])
except BaseException:
    ok = False
print("OK" if ok else "FAIL")
"""

HE_RUNNER = r"""
import json, sys
d = json.loads(sys.stdin.read())
ns = {}
try:
    exec(d["code"] + "\n" + d["test"] + "\ncheck(" + d["entry"] + ")", ns)
    print("OK")
except BaseException:
    print("FAIL")
"""


HE_STOPS = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#")


def run_python(runner, payload, timeout=3.0):
    try:
        r = subprocess.run([sys.executable, "-I", "-c", runner], input=json.dumps(payload),
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip().endswith("OK")
    except subprocess.TimeoutExpired:
        return False


def extract(task, out):
    """Вытаскивает ответ из сгенерированного текста (до пустой строки)."""
    if task.check == "humaneval":
        for stop in HE_STOPS:
            out = out.split(stop)[0]
        return out.rstrip()
    out = out.split("\n\n")[0]
    if "</out>" in out:                       # ответ после работы инструмента
        out = out.rsplit("</out>", 1)[1]
    if task.check == "code":
        return out.strip("\n").lstrip(" ")
    return out.strip().split("\n")[0].strip()


def is_correct(task, out):
    ans = extract(task, out)
    if task.check == "exact":
        return ans.rstrip(".").strip() == task.answer
    if task.check == "number":
        n = T.first_number(ans) if task.domain not in ("gsm8k", "mgsm") else T.last_number(out)
        try:
            return n is not None and float(n) == float(task.answer)
        except ValueError:
            return False
    if task.check == "code":
        code = ans if ans.lstrip().startswith("def ") else "def f(x):\n" + ans
        return run_python(RUNNER, {"code": code, "tests": task.tests})
    if task.check == "humaneval":
        test, entry = task.tests
        return run_python(HE_RUNNER, {"code": task.question + ans, "test": test, "entry": entry})
    raise ValueError(task.check)


def max_new_for(task):
    return {"code": 120, "humaneval": 400, "gsm8k": 300, "mgsm": 300}.get(
        task.check if task.check in ("code", "humaneval") else task.domain, 24)


def ece(records, bins=10):
    """Expected calibration error: насколько уверенность модели совпадает с её точностью."""
    if not records:
        return float("nan")
    buckets = defaultdict(list)
    for conf, ok in records:
        buckets[min(int(conf * bins), bins - 1)].append((conf, ok))
    total = len(records)
    err = 0.0
    for b in buckets.values():
        mc = sum(c for c, _ in b) / len(b)
        acc = sum(o for _, o in b) / len(b)
        err += len(b) / total * abs(mc - acc)
    return err


def run(model, task_list, log_path=None, verbose=False):
    rows = []
    for t in task_list:
        out, conf = model.answer(t.prompt, max_new_for(t))
        ok = is_correct(t, out)
        rows.append({"domain": t.domain, "lang": t.lang, "level": t.level, "ok": bool(ok),
                     "conf": float(conf), "prompt": t.prompt, "out": extract(t, out), "ref": t.answer})
        if verbose:
            print(("✔" if ok else "✘"), t.domain, t.lang, t.level, repr(extract(t, out))[:60])
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def summarize(rows):
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        for k in [(r["domain"], r["lang"], r["level"]), (r["domain"], r["lang"], "all"),
                  ("ALL", r["lang"], "all"), ("ALL", "both", "all")]:
            cell[k][0] += r["ok"]
            cell[k][1] += 1
    acc = {k: v[0] / v[1] for k, v in cell.items()}
    cal = ece([(r["conf"], r["ok"]) for r in rows])
    return acc, cal


def table(rows, title=""):
    acc, cal = summarize(rows)
    domains = sorted({r["domain"] for r in rows}, key=lambda d: (T.DOMAINS + (d,)).index(d))
    levels = sorted({r["level"] for r in rows})
    lines = [f"### {title}" if title else "", "",
             "| домен | язык | " + " | ".join(f"ур.{l}" for l in levels) + " | всего |",
             "|---|---|" + "---|" * (len(levels) + 1)]
    for d in domains:
        for lang in T.LANGS:
            if (d, lang, "all") not in acc:
                continue
            cells = [f"{100 * acc[(d, lang, l)]:.0f}%" if (d, lang, l) in acc else "—" for l in levels]
            lines.append(f"| {d} | {lang} | " + " | ".join(cells) + f" | {100 * acc[(d, lang, 'all')]:.0f}% |")
    for lang in T.LANGS:
        if ("ALL", lang, "all") in acc:
            lines.append(f"| **ИТОГО** | {lang} | " + " | ".join("" for _ in levels) +
                         f" | **{100 * acc[('ALL', lang, 'all')]:.1f}%** |")
    lines.append("")
    lines.append(f"Калибровка (ECE, меньше — лучше): {cal:.3f}")
    return "\n".join(lines)
