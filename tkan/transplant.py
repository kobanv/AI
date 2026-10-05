"""Ткань-Т: пересадка знаний открытой модели в байтовое тело «Ткани».

Тело (новое, обучаемое):
  глаз   — байтовый энкодер: сырые байты → векторы-«понятия» на концах кусков;
  голова — предсказатель границ кусков (видит 1 байт вперёд, только при чтении);
  голос  — байтовый декодер: мысль ядра → байты следующего куска + знак конца куска.
Ядро (пересаженное): слои трансформера учителя Qwen3-0.6B (Apache-2.0), на этапе 1 заморожены.

Токенайзер учителя нужен только как учебное пособие на этапе 1 (где учитель резал текст);
в работе «Ткань-Т» читает и пишет голые байты.

Этап 1а: глаз учится восстанавливать входной вектор ядра по байтам куска и контексту.
Этап 1б: голос учится выговаривать следующий кусок по мысли ядра.
Этап 2 (GPU): разморозка, свободная нарезка, рекурсия по глубине, память.

python -m tkan.transplant eye   --minutes 90
python -m tkan.transplant voice --minutes 60
python -m tkan.transplant eval
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Mixture
from .model import ConvBlock, RMSNorm
from .train import RUNS

TEACHER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "qwen3-0.6b")
OUT = os.path.join(RUNS, "T_transplant")
END = 256  # символ «конец куска» в голосе
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK = 24


# ---------------------------------------------------------------- учитель
def bytes_to_unicode():
    """Таблица байтового BPE (как в GPT-2/Qwen): байт → печатный символ."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Teacher:
    def __init__(self, path=TEACHER):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(DEV).eval()
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.E = self.lm.get_input_embeddings().weight          # (V, 1024)
        self.d = self.E.shape[1]
        dec = {c: b for b, c in bytes_to_unicode().items()}
        self.tok_bytes = {}
        self._dec = dec

    def token_bytes(self, i):
        b = self.tok_bytes.get(i)
        if b is None:
            s = self.tok.convert_ids_to_tokens(i)
            b = bytes(self._dec[c] for c in s) if all(c in self._dec for c in s) else s.encode("utf-8")
            self.tok_bytes[i] = b
        return b

    def split(self, raw: bytes, max_tokens=None):
        """Где бы учитель разрезал текст: (ids, байтовые концы кусков) — только если склейка точна."""
        text = raw.decode("utf-8", errors="ignore")
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if max_tokens:
            ids = ids[:max_tokens]
        pieces = [self.token_bytes(i) for i in ids]
        if any(len(p) > MAX_CHUNK for p in pieces):
            return None
        joined = b"".join(pieces)
        if not text.encode("utf-8").startswith(joined):
            return None
        ends, pos = [], 0
        for p in pieces:
            pos += len(p)
            ends.append(pos - 1)
        return ids, joined, ends

    @torch.no_grad()
    def core(self, inputs_embeds):
        """Мысль ядра: скрытые состояния после последнего слоя и финальной нормы."""
        out = self.lm.model(inputs_embeds=inputs_embeds)
        return out.last_hidden_state

    def logits(self, h):
        return self.lm.lm_head(h)


# ---------------------------------------------------------------- тело
class Eye(nn.Module):
    """Байты → векторы для ядра. Причинный: вектор куска зависит только от байтов до его конца."""

    def __init__(self, d_out, d=512, layers=4, kernel=16, n_hash=6, buckets=1 << 15):
        super().__init__()
        self.byte = nn.Embedding(256, d)
        # хеш-эмбеддинги n-грамм (n = 2..n_hash+1): дешёвая «память формы» кусков
        self.ngram = nn.ModuleList([nn.Embedding(buckets, d, sparse=True) for _ in range(n_hash)])
        self.buckets = buckets
        self.blocks = nn.ModuleList([ConvBlock(d, kernel) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.out = nn.Linear(d, d_out, bias=False)
        self.bound = nn.Linear(2 * d, 1)

    def hashes(self, x):
        hs, h = [], torch.zeros_like(x)
        P, M = 257, 2147483647
        for n in range(1, len(self.ngram) + 2):
            shifted = F.pad(x, (n - 1, 0), value=0)[:, : x.size(1)] if n > 1 else x
            h = (h * P + shifted + 1) % M
            if n >= 2:
                hs.append(h % self.buckets)
        return hs

    def forward(self, x):
        e = self.byte(x)
        for emb, hh in zip(self.ngram, self.hashes(x)):
            e = e + emb(hh)
        for blk in self.blocks:
            e = blk(e)
        return self.norm(e)

    def boundary_logits(self, h):
        nxt = F.pad(h[:, 1:], (0, 0, 0, 1))
        return self.bound(torch.cat([h, nxt], -1)).squeeze(-1)


class Voice(nn.Module):
    """Мысль ядра (вектор) → байты следующего куска и символ конца. Маленький причинный трансформер."""

    def __init__(self, d_in, d=384, layers=3, heads=6):
        super().__init__()
        self.inp = nn.Linear(d_in, d, bias=False)
        self.byte = nn.Embedding(257, d)
        self.pos = nn.Embedding(MAX_CHUNK + 2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tr = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 257)

    def forward(self, h, prev):
        """h: (N, d_in) мысль; prev: (N, L) уже сказанные байты куска (256 = паддинг-заглушка). → (N, L+1, 257)."""
        N, L = prev.shape
        x = torch.cat([self.inp(h)[:, None], self.byte(prev)], 1)
        x = x + self.pos(torch.arange(L + 1, device=h.device))[None]
        mask = torch.triu(torch.full((L + 1, L + 1), float("-inf"), device=h.device), 1)
        return self.head(self.norm(self.tr(x, mask=mask, is_causal=True)))


# ---------------------------------------------------------------- данные
def samples(mix, teacher, seq=256, max_tokens=96):
    while True:
        name = mix.rng.choices(list(mix.weights), weights=list(mix.weights.values()))[0]
        raw = bytes(mix.sample_tasks() if name == "tasks" else mix.sample_text(name))
        if name != "tasks":                      # начинаем с границы слова/строки, а не с середины
            cut = max(raw.find(b" "), raw.find(b"\n"))
            raw = raw[cut + 1:] if cut >= 0 else raw
        s = teacher.split(raw[:seq], max_tokens)
        if s and len(s[0]) >= 8:
            yield s


def batch(gen, bsz):
    rows = [next(gen) for _ in range(bsz)]
    T = max(len(r[1]) for r in rows)
    x = torch.zeros(bsz, T, dtype=torch.long)
    is_end = torch.zeros(bsz, T)
    for i, (_, b, ends) in enumerate(rows):
        x[i, : len(b)] = torch.tensor(list(b))
        is_end[i, ends] = 1
    return rows, x, is_end


# ---------------------------------------------------------------- этап 1а: глаз
def train_eye(args):
    torch.set_num_threads(args.threads)
    teacher = Teacher()
    eye = Eye(teacher.d).to(DEV)
    mix = Mixture(512, seed=11)
    gen = samples(mix, teacher)
    dense = [p for n, p in eye.named_parameters() if not n.startswith("ngram")]
    opt = torch.optim.AdamW(dense, lr=args.lr, weight_decay=0.0)
    sopt = torch.optim.SparseAdam(list(eye.ngram.parameters()), lr=args.lr)
    E = teacher.E
    os.makedirs(OUT, exist_ok=True)
    t0, step, log = time.time(), 0, open(os.path.join(OUT, "eye_log.jsonl"), "w")
    while time.time() - t0 < args.minutes * 60:
        rows, x, is_end = batch(gen, args.bsz)
        x, is_end = x.to(DEV), is_end.to(DEV)
        h = eye(x)
        bi, ti = is_end.nonzero(as_tuple=True)
        ids = torch.tensor([i for r in rows for i in r[0]], device=DEV)
        pred = eye.out(h[bi, ti])
        tgt = E[ids]
        rel = ((pred - tgt) ** 2).sum(-1) / (tgt ** 2).sum(-1).clamp_min(1e-6)
        cos = F.cosine_similarity(pred, tgt, dim=-1)
        valid = torch.zeros_like(is_end)
        for i, r in enumerate(rows):
            valid[i, : len(r[1])] = 1
        bl = F.binary_cross_entropy_with_logits(eye.boundary_logits(h), is_end, weight=valid)
        loss = rel.mean() + (1 - cos).mean() + 0.2 * bl
        opt.zero_grad(set_to_none=True)
        sopt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense, 1.0)
        frac = (time.time() - t0) / (args.minutes * 60)
        for o in (opt, sopt):
            for g in o.param_groups:
                g["lr"] = args.lr * min(1, (step + 1) / 100) * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))
        opt.step()
        sopt.step()
        step += 1
        if step % 50 == 0:
            with torch.no_grad():
                # узнаёт ли глаз «понятие» ядра по байтам: ближайший сосед среди всех 152k векторов
                k = min(256, pred.size(0))
                nn_ids = (F.normalize(pred[:k], dim=-1) @ F.normalize(E, dim=-1).T).argmax(-1)
                top1 = float((nn_ids == ids[:k]).float().mean())
                bacc = float((((eye.boundary_logits(h) > 0).float() == is_end).float() * valid).sum() / valid.sum())
            rec = {"step": step, "min": round((time.time() - t0) / 60, 1), "rel_err": float(rel.mean()),
                   "cos": float(cos.mean()), "top1": top1, "boundary_acc": bacc}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    torch.save(eye.state_dict(), os.path.join(OUT, "eye.pt"))


# ---------------------------------------------------------------- этап 1б: голос
def chunk_targets(pieces):
    """Байты кусков → (prev, target) для учителя-форсинга голоса.

    Вход голоса: [мысль, p_0, ..., p_{n-1}]; выход в позиции k предсказывает (p + [END])[k].
    """
    L = max(len(p) for p in pieces)
    prev = torch.full((len(pieces), L), END, dtype=torch.long)
    tgt = torch.full((len(pieces), L + 1), -100, dtype=torch.long)
    for i, p in enumerate(pieces):
        if p:
            prev[i, : len(p)] = torch.tensor(list(p))
        tgt[i, : len(p) + 1] = torch.tensor(list(p) + [END])
    return prev, tgt


def train_voice(args):
    torch.set_num_threads(args.threads)
    teacher = Teacher()
    voice = Voice(teacher.d).to(DEV)
    mix = Mixture(512, seed=12)
    gen = samples(mix, teacher)
    opt = torch.optim.AdamW(voice.parameters(), lr=args.lr, weight_decay=0.0)
    t0, step, log = time.time(), 0, open(os.path.join(OUT, "voice_log.jsonl"), "w")
    while time.time() - t0 < args.minutes * 60:
        rows = [next(gen) for _ in range(args.bsz)]
        H, pieces = [], []
        for ids, _, _ in rows:          # мысль ядра в позиции j предсказывает кусок j+1
            h = teacher.core(teacher.E[torch.tensor(ids, device=DEV)][None])[0]
            H.append(h[:-1])
            pieces += [teacher.token_bytes(i) for i in ids[1:]]
        H = torch.cat(H)
        prev, tgt = chunk_targets(pieces)
        prev, tgt = prev.to(DEV), tgt.to(DEV)
        logits = voice(H, prev)
        loss = F.cross_entropy(logits.reshape(-1, 257), tgt.reshape(-1), ignore_index=-100)
        loss_sum = float(loss) * int((tgt != -100).sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(voice.parameters(), 1.0)
        frac = (time.time() - t0) / (args.minutes * 60)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1, (step + 1) / 50) * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))
        opt.step()
        step += 1
        if step % 20 == 0:
            n_bytes = int((tgt != -100).sum()) - len(pieces)
            rec = {"step": step, "min": round((time.time() - t0) / 60, 1),
                   "bits_per_byte": loss_sum / max(1, n_bytes) / math.log(2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    torch.save(voice.state_dict(), os.path.join(OUT, "voice.pt"))


# ---------------------------------------------------------------- собранный организм
class TkanT:
    """Голые байты на входе и выходе; внутри — пересаженное ядро."""

    def __init__(self, teacher, eye, voice):
        self.t, self.eye, self.voice = teacher, eye.eval(), voice.eval()

    @torch.no_grad()
    def read(self, raw: bytes):
        """Глаз читает байты, голова сама ставит границы кусков, ядро думает."""
        x = torch.tensor([list(raw)], device=DEV)
        h = self.eye(x)
        ends = (self.eye.boundary_logits(h)[0] > 0).nonzero().flatten().tolist()
        if not ends or ends[-1] != len(raw) - 1:
            ends.append(len(raw) - 1)
        vec = self.eye.out(h[0, ends])
        return vec, ends

    @torch.no_grad()
    def speak_chunk(self, thought):
        out = []
        for _ in range(MAX_CHUNK):
            prev = torch.tensor([out], dtype=torch.long, device=DEV).reshape(1, len(out))
            logits = self.voice(thought[None], prev)[0, -1]
            b = int(logits.argmax())
            if b == END:
                break
            out.append(b)
        return bytes(out)

    @torch.no_grad()
    def generate(self, prompt: bytes, max_new=64, stop=b"\n\n"):
        out = b""
        vec, _ = self.read(prompt)
        for _ in range(64):
            h = self.t.core(vec[None])[0, -1]
            piece = self.speak_chunk(h) or b" "
            out += piece
            if stop in out or len(out) >= max_new:
                break
            # глаз дочитывает новый кусок (причинно: его вектор зависит от всего прочитанного)
            full = prompt + out
            hh = self.eye(torch.tensor([list(full)], device=DEV))
            vec = torch.cat([vec, self.eye.out(hh[0, -1:])], 0)
        return out.split(stop)[0] + (stop if stop in out else b""), 1.0

    def answer(self, prompt, max_new):
        out, c = self.generate(prompt.encode("utf-8"), max_new=max_new)
        return out.decode("utf-8", errors="replace"), c


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(args.threads)
    teacher = Teacher()
    eye, voice = Eye(teacher.d).to(DEV), Voice(teacher.d).to(DEV)
    eye.load_state_dict(torch.load(os.path.join(OUT, "eye.pt"), map_location=DEV))
    voice.load_state_dict(torch.load(os.path.join(OUT, "voice.pt"), map_location=DEV))
    eye.eval()
    voice.eval()
    mix = Mixture(512, seed=999)
    rep = {}
    # 1) понимает ли ядро то, что прочитал глаз: loss следующего куска с глазами учителя vs с нашими
    for name in ("wiki_ru", "wiki_en", "code"):
        tot_t, tot_e, tot_v, n_bytes = 0.0, 0.0, 0.0, 0
        rng = random.Random(5)
        for _ in range(12):
            raw = bytes(mix.sample_text(name, val=True))
            cut = max(raw.find(b" "), raw.find(b"\n"))
            s = teacher.split(raw[cut + 1:][:256], 96)
            if not s:
                continue
            ids, b, ends = s
            ids_t = torch.tensor(ids, device=DEV)
            lt = teacher.logits(teacher.core(teacher.E[ids_t][None]))[0, :-1]
            h = eye(torch.tensor([list(b)], device=DEV))
            le = teacher.logits(teacher.core(eye.out(h[0, ends])[None]))[0, :-1]
            tot_t += float(F.cross_entropy(lt, ids_t[1:], reduction="sum"))
            tot_e += float(F.cross_entropy(le, ids_t[1:], reduction="sum"))
            # байтовая речь целиком: глаз → ядро → голос
            He = teacher.core(eye.out(h[0, ends])[None])[0, :-1]
            prev, tgt = chunk_targets([teacher.token_bytes(i) for i in ids[1:]])
            prev, tgt = prev.to(DEV), tgt.to(DEV)
            lv = voice(He, prev)
            tot_v += float(F.cross_entropy(lv.reshape(-1, 257), tgt.reshape(-1), ignore_index=-100, reduction="sum"))
            n_bytes += len(b) - len(teacher.token_bytes(ids[0]))
        rep[name] = {"teacher_bpb": tot_t / n_bytes / math.log(2), "eye_core_bpb": tot_e / n_bytes / math.log(2),
                     "organism_bpb": tot_v / n_bytes / math.log(2)}
        print(name, rep[name], flush=True)
    # 2) свободная речь голыми байтами
    organism = TkanT(teacher, eye, voice)
    rep["samples"] = {}
    for p in ["Вопрос: Какая столица России?\nОтвет:", "Question: What is the capital of France?\nAnswer:",
              "Вопрос: Сколько будет 12 + 7?\nОтвет:", "Москва — это", "Python is a programming language that"]:
        out, _ = organism.generate(p.encode("utf-8"), max_new=80)
        rep["samples"][p] = out.decode("utf-8", errors="replace")
        print("----", p, "→", rep["samples"][p], flush=True)
    json.dump(rep, open(os.path.join(OUT, "eval.json"), "w"), ensure_ascii=False, indent=1)


@torch.no_grad()
def exam(args):
    """Экзамен «Ткани-Т» с теми же 3 примерами, что у эталона Qwen: то же ядро, но глаза и голос — байтовые."""
    from .baselines import FewShot
    from .bench import exam as E
    from .bench import tasks as T
    torch.set_num_threads(args.threads)
    teacher = Teacher()
    eye, voice = Eye(teacher.d).to(DEV), Voice(teacher.d).to(DEV)
    eye.load_state_dict(torch.load(os.path.join(OUT, "eye.pt"), map_location=DEV))
    voice.load_state_dict(torch.load(os.path.join(OUT, "voice.pt"), map_location=DEV))
    organism = TkanT(teacher, eye, voice)
    shots = FewShot()

    class Agent:
        def answer(self, prompt, max_new):
            return organism.answer(shots.prefix(prompt), max_new)

    rows, t0 = [], time.time()
    for t in T.test_set(n_per_cell=args.n):
        shots.current = t.domain
        rows += E.run(Agent(), [t])
    with open(os.path.join(OUT, "exam.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = E.table(rows, f"Ткань-Т (байты ↔ ядро Qwen3-0.6B, 3-shot), {time.time() - t0:.0f} с")
    open(os.path.join(OUT, "exam.md"), "w").write(md)
    print(md, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eye", "voice", "eval", "exam"])
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    {"eye": train_eye, "voice": train_voice, "eval": evaluate, "exam": exam}[args.stage](args)


if __name__ == "__main__":
    main()
