"""Сколько стоит повторить обучение GLM-5.x с нуля, и сколько стоит «встать на плечи гигантов».

python tools/glm_gap.py
"""
YEAR = 365 * 24 * 3600

active = 40e9          # активных параметров на токен (MoE)
tokens = 28.5e12       # токенов предобучения (технический отчёт GLM-5)
pretrain = 6 * active * tokens
post = 0.3 * pretrain  # грубо: дообучение, RL и т.п. (неизвестно точно)
total = pretrain + post

devices = {
    "RTX 5070 Ti (~50 TFLOPS реально)": 50e12,
    "наш облачный CPU (4 ядра, ~0.1 TFLOPS)": 0.1e12,
    "1 × H100 (~400 TFLOPS реально)": 400e12,
    "1000 × H100": 1000 * 400e12,
}
print(f"Обучение GLM-5.x: ≈ {pretrain:.1e} FLOP предобучение + ≈ {post:.1e} дообучение ≈ {total:.1e} FLOP\n")
for name, flops in devices.items():
    t = total / flops
    print(f"  {name:42s} {t / YEAR:12,.1f} лет" if t > YEAR else f"  {name:42s} {t / 86400:12,.1f} дней")
h100_hours = total / 400e12 / 3600
print(f"\n  ≈ {h100_hours:,.0f} часов H100 ≈ ${h100_hours * 2 / 1e6:,.1f} млн при $2/час")

print("\nА если не учить с нуля, а взять открытые знания:")
cases = {
    "пересадка тела «Ткани-Т» вокруг Qwen3-0.6B (сделано)": None,
    "дистилляция рассуждений GLM-5.3 в ядро 14B (1 млрд токенов)": 6 * 14e9 * 1e9,
    "то же в ядро 8B (1 млрд токенов)": 6 * 8e9 * 1e9,
    "генерация 1 млрд токенов учителем GLM-5.3 через API": None,
}
for name, f in cases.items():
    if f is None:
        continue
    d = f / 50e12 / 86400
    print(f"  {name:62s} ≈ {d:5.0f} дней на 5070 Ti  (≈ {f / 400e12 / 3600:4.0f} ч на H100)")
print("  пересадка тела «Ткани-Т» вокруг Qwen3-0.6B (сделано)            ≈ 2 часа на 4 ядрах CPU")
