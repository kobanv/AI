"""Проверки корректности «Ткани»: причинность и работа обоих режимов нарезки."""
import torch

from tkan.model import Config, Tkan


def _check_causal(chunking):
    torch.manual_seed(0)
    m = Tkan(Config(d_byte=64, d_main=64, n_heads=2, chunking=chunking)).eval()
    x = torch.randint(0, 256, (2, 96))
    y = x.clone()
    t = 50
    y[:, t + 1:] = torch.randint(0, 256, (2, 96 - t - 1))
    with torch.no_grad():
        la, _, _ = m(x, r=3)
        lb, _, _ = m(y, r=3)
    # логиты для позиций <= t не должны зависеть от байтов после t
    assert torch.allclose(la[:, :t + 1], lb[:, :t + 1], atol=1e-5), chunking


def test_causal_learned():
    _check_causal("learned")


def test_causal_fixed():
    _check_causal("fixed")


def test_train_step_and_generate():
    torch.manual_seed(0)
    m = Tkan(Config(d_byte=64, d_main=64, n_heads=2))
    x = torch.randint(0, 256, (2, 65))
    ce, rl, b = m.loss(x)
    (ce + 0.03 * rl).backward()
    assert all(p.grad is not None for n, p in m.named_parameters() if not n.startswith("wq") and not n.startswith("wk"))
    assert m.wq.weight.grad is not None  # роутер получает градиент
    out, conf = m.generate("Вопрос: 2+2?\nОтвет:".encode(), max_new=5)
    assert len(out) <= 5 and 0 <= conf <= 1


def test_ema_scan_matches_loop():
    from tkan.model import ema_scan
    torch.manual_seed(0)
    z, P = torch.randn(3, 40, 8), torch.rand(3, 40)
    P[:, 0] = 1.0
    prev, outs = torch.zeros(3, 8), []
    for j in range(40):
        prev = P[:, j:j + 1] * z[:, j] + (1 - P[:, j:j + 1]) * prev
        outs.append(prev)
    assert torch.allclose(ema_scan(z, P), torch.stack(outs, 1), atol=1e-4)
