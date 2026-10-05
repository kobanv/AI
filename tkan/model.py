"""Ткань-0: байты → самообучаемая нарезка → рекуррентное ядро → байты.

Ни словаря, ни токенайзера, ни заданных понятий. Задана только анатомия:
  1. Сенсорный слой (байтовый энкодер) читает сырые байты UTF-8.
  2. Роутер сам решает, где начинается новый смысловой кусок (чанк):
     граница там, где соседние представления резко различаются (идея H-Net).
     Целевую степень сжатия держит ratio-loss, но *где* резать, модель решает сама.
  3. Кора — рекуррентное ядро над чанками: прелюдия → общий блок × r → кода.
     Число циклов r не зашито: при обучении оно случайное, при работе его можно
     увеличивать, чтобы «подумать дольше» над трудной задачей.
  4. Сглаживание и разжатие возвращают мысль ядра к байтам,
     байтовый декодер выговаривает следующий байт.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_byte: int = 192          # ширина байтового уровня
    d_main: int = 320          # ширина ядра
    n_heads: int = 5
    enc_layers: int = 2
    dec_layers: int = 2
    prelude_layers: int = 1
    core_layers: int = 2       # общий (рекуррентный) блок
    coda_layers: int = 1
    conv_kernel: int = 12
    chunking: str = "learned"  # learned | fixed
    target_ratio: float = 4.0  # байт на чанк (в среднем)
    ratio_loss_w: float = 0.03
    r_train_max: int = 6       # при обучении r ~ U{r_train_min..r_train_max}
    r_train_min: int = 1
    s0_noise: float = 0.0      # случайное начальное состояние мысли (как в Huginn): ядро учится сходиться
    bptt: int = 2              # градиент течёт через последние bptt циклов
    max_chunks: int = 1024
    byte_attn: int = 0         # слоёв внимания по байтам в декодере («точный взгляд» для копирования)
    byte_window: int = 256     # окно этого внимания (байт)
    inject: str = "concat"     # как мысль смешивается со входом: concat (адаптер) | add (без обхода)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class ConvBlock(nn.Module):
    """Локальный причинный смеситель для байтов: depthwise-свёртка + SwiGLU."""

    def __init__(self, d, kernel):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.k = kernel
        self.dw = nn.Conv1d(d, d, kernel, groups=d)
        self.pw = nn.Linear(d, d, bias=False)
        self.mlp = SwiGLU(d, 2 * d)

    def forward(self, x):
        h = self.n1(x).transpose(1, 2)
        h = F.pad(h, (self.k - 1, 0))
        x = x + self.pw(self.dw(h).transpose(1, 2))
        return x + self.mlp(self.n2(x))


def rope(x, pos):
    # x: (B, H, N, Dh)
    dh = x.size(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class AttnBlock(nn.Module):
    def __init__(self, d, n_heads, window=None):
        super().__init__()
        self.h = n_heads
        self.window = window
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.mlp = SwiGLU(d, int(8 * d / 3))

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        pos = torch.arange(N, device=x.device)
        q, k = rope(q, pos), rope(k, pos)
        if self.window and N > self.window:
            i = torch.arange(N, device=x.device)
            mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))


def ema_scan(z, P):
    """Параллельное сглаживание z̄_j = P_j z_j + (1 - P_j) z̄_{j-1} (без цикла по чанкам).

    z̄_j = Σ_{i<=j} P_i z_i Π_{k=i+1..j} (1 - P_k) = Σ_i exp(S_j - S_i) P_i z_i, где S = cumsum(log(1 - P)).
    """
    S = torch.log((1 - P).clamp_min(1e-6)).cumsum(1)                  # (B, N)
    diff = S[:, :, None] - S[:, None, :]                               # (B, N, N): S_j - S_i
    mask = torch.ones(P.size(1), P.size(1), dtype=torch.bool, device=P.device).tril()
    W = torch.exp(diff.masked_fill(~mask, float("-inf"))) * P[:, None, :]
    return W @ z


class Tkan(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        db, dm = cfg.d_byte, cfg.d_main
        self.embed = nn.Embedding(256, db)
        self.encoder = nn.ModuleList([ConvBlock(db, cfg.conv_kernel) for _ in range(cfg.enc_layers)])
        # роутер: косинусная «неожиданность» между соседними байтами
        self.wq = nn.Linear(db, db, bias=False)
        self.wk = nn.Linear(db, db, bias=False)
        with torch.no_grad():
            self.wq.weight.copy_(torch.eye(db))
            self.wk.weight.copy_(torch.eye(db))
        self.down = nn.Linear(db, dm, bias=False)
        self.prelude = nn.ModuleList([AttnBlock(dm, cfg.n_heads) for _ in range(cfg.prelude_layers)])
        self.adapter = nn.Linear(2 * dm, dm, bias=False)
        self.core = nn.ModuleList([AttnBlock(dm, cfg.n_heads) for _ in range(cfg.core_layers)])
        self.core_norm = RMSNorm(dm)
        self.coda = nn.ModuleList([AttnBlock(dm, cfg.n_heads) for _ in range(cfg.coda_layers)])
        self.up = nn.Linear(dm, db, bias=False)
        self.byte_attn = nn.ModuleList([AttnBlock(db, max(1, db // 64), window=cfg.byte_window)
                                        for _ in range(cfg.byte_attn)])
        self.decoder = nn.ModuleList([ConvBlock(db, cfg.conv_kernel) for _ in range(cfg.dec_layers)])
        self.out_norm = RMSNorm(db)
        self.head = nn.Linear(db, 256, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and m.weight.shape[0] != m.weight.shape[1]:
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------ нарезка
    def boundaries(self, h):
        B, T, _ = h.shape
        if self.cfg.chunking == "fixed":
            b = (torch.arange(T, device=h.device) % int(self.cfg.target_ratio) == 0).expand(B, T)
            return b, torch.ones(B, T, device=h.device)
        q = F.normalize(self.wq(h[:, 1:]), dim=-1)
        k = F.normalize(self.wk(h[:, :-1]), dim=-1)
        p = (0.5 * (1 - (q * k).sum(-1))).clamp(0, 1)
        p = torch.cat([torch.ones(B, 1, device=h.device), p], 1)
        return p >= 0.5, p

    def ratio_loss(self, b, p):
        n = self.cfg.target_ratio
        f, g = b.float().mean(), p.mean()
        return n / (n - 1) * ((n - 1) * f * g + (1 - f) * (1 - g))

    # ------------------------------------------------------------ кора
    def think(self, x, r, bptt):
        for blk in self.prelude:
            x = blk(x)
        e = x
        s = torch.randn_like(e) * self.cfg.s0_noise if self.cfg.s0_noise else torch.zeros_like(e)
        for i in range(r):
            grad = torch.is_grad_enabled() and i >= r - bptt
            with torch.set_grad_enabled(grad):
                if self.cfg.inject == "add":
                    s = s + e if i else e      # мысль нельзя обойти: она всегда в потоке
                else:
                    s = self.adapter(torch.cat([self.core_norm(s), e], -1))
                for blk in self.core:
                    s = blk(s)
        x = self.core_norm(s)
        for blk in self.coda:
            x = blk(x)
        return x

    # ------------------------------------------------------------ прямой проход
    def forward(self, idx, r=None, return_stats=False):
        cfg = self.cfg
        B, T = idx.shape
        h = self.embed(idx)
        for blk in self.encoder:
            h = blk(h)

        b, p = self.boundaries(h)
        cidx = b.long().cumsum(1) - 1                        # номер чанка для каждого байта
        n_chunks = int(b.sum(1).max())
        bi, ti = b.nonzero(as_tuple=True)
        ci = cidx[bi, ti]
        z = h.new_zeros(B, n_chunks, cfg.d_main)
        z = z.index_put((bi, ci), self.down(h[bi, ti]))
        P = h.new_zeros(B, n_chunks).index_put((bi, ci), p[bi, ti])

        if r is None:
            r = int(torch.randint(cfg.r_train_min, cfg.r_train_max + 1, (1,))) if self.training else cfg.r_train_max
        z = self.think(z, r, cfg.bptt if self.training else r)

        # сглаживание (EMA по чанкам): даёт роутеру градиент, куда сдвинуть границы
        if cfg.chunking == "learned":
            z = ema_scan(z, P)

        zu = z[torch.arange(B, device=idx.device)[:, None], cidx]   # разжатие обратно к байтам
        if cfg.chunking == "learned":
            c = torch.where(b, p, 1 - p)
            zu = zu * (c + (1 - c).detach()).unsqueeze(-1)            # STE: вперёд 1, назад c
        x = h + self.up(zu)
        for blk in self.byte_attn:      # точный взгляд назад по байтам: копирование чисел, имён, кода
            x = blk(x)
        for blk in self.decoder:
            x = blk(x)
        logits = self.head(self.out_norm(x))
        if not return_stats:
            return logits, b, p
        return logits, b, p, {"bytes_per_chunk": T * B / max(1, int(b.sum())), "r": r}

    def loss(self, idx, r=None):
        logits, b, p = self(idx[:, :-1], r=r)
        ce = F.cross_entropy(logits.reshape(-1, 256), idx[:, 1:].reshape(-1))
        rl = self.ratio_loss(b, p) if self.cfg.chunking == "learned" else torch.zeros(())
        return ce, rl, b

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------ генерация
    @torch.no_grad()
    def generate(self, prompt: bytes, max_new=32, r=None, stop=(b"\n\n",), temperature=0.0, gen=None):
        """Генерация байт за байтом (жадная или с температурой). Возвращает (байты, вероятность ответа)."""
        stop = (stop,) if isinstance(stop, bytes) else stop
        self.eval()
        seq = list(prompt)
        out, logp = [], 0.0
        dev = self.head.weight.device
        for _ in range(max_new):
            x = torch.tensor([seq[-2048:]], dtype=torch.long, device=dev)
            logits, _, _ = self(x, r=r)
            lp = F.log_softmax(logits[0, -1], -1)
            if temperature > 0:
                nxt = int(torch.multinomial(F.softmax(logits[0, -1] / temperature, -1), 1, generator=gen))
            else:
                nxt = int(lp.argmax())
            logp += float(lp[nxt])
            seq.append(nxt)
            out.append(nxt)
            if any(bytes(out[-len(st):]) == st for st in stop):
                break
        return bytes(out), math.exp(logp)
