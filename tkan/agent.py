"""Агентный цикл «Ткани»: думать → (при желании) написать программу → увидеть результат → ответить.

Модель сама решает, звать ли инструмент: если она пишет <py>...</py>, программа
исполняется в изолированном процессе, а её вывод возвращается в контекст как <out>...</out>.
"""
import subprocess
import sys

SANDBOX = r"""
import sys, io, contextlib
code = sys.stdin.read()
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(code, {"__builtins__": __builtins__})
    print(buf.getvalue().strip()[:200])
except BaseException as e:
    print("ошибка: " + type(e).__name__)
"""


def run_python(code, timeout=2.0):
    try:
        r = subprocess.run([sys.executable, "-I", "-c", SANDBOX], input=code, capture_output=True,
                           text=True, timeout=timeout)
        return r.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ошибка: превышено время"


class ToolAgent:
    def __init__(self, model, r, max_calls=3):
        self.model, self.r, self.max_calls = model, r, max_calls
        self.calls = 0

    def answer(self, prompt, max_new):
        ctx, out, conf = prompt.encode("utf-8"), b"", 1.0
        for _ in range(self.max_calls + 1):
            piece, c = self.model.generate(ctx + out, max_new=max_new + 200, r=self.r, stop=(b"</py>", b"\n\n"))
            out += piece
            conf *= c
            if out.endswith(b"</py>") and b"<py>" in out:
                code = out.rsplit(b"<py>", 1)[1][:-len(b"</py>")].decode("utf-8", errors="replace")
                self.calls += 1
                out += f"<out>{run_python(code)}</out>".encode("utf-8")
                continue
            break
        return out.decode("utf-8", errors="replace"), conf
