"""Гиппокамп «Ткани»: растущая память и выход в интернет.

Память — append-only хранилище всего прочитанного (ткань нарастает при чтении, без обучения).
Вспоминание — поиск по символьным n-граммам (одинаково для русского, английского и кода):
никаких словарей и заданных понятий, только форма текста.
Интернет — поиск и чтение статей Википедии (ru/en) через открытый API.

Протокол для агента:
  <search>запрос</search>  → <doc>...</doc>   (интернет; прочитанное сразу ложится в память)
  <recall>запрос</recall>  → <doc>...</doc>   (вспомнить из своей памяти)
"""
import json
import math
import os
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

UA = {"User-Agent": "TkanResearchAgent/0.1 (https://github.com/kobanv/AI; research prototype)"}


def grams(text, n=4):
    t = re.sub(r"\s+", " ", text.lower())
    return [t[i:i + n] for i in range(max(1, len(t) - n + 1))]


class Memory:
    def __init__(self, path=None):
        self.path = path
        self.docs = []                     # (источник, текст)
        self.index = defaultdict(set)      # n-грамма → номера документов
        self.df = Counter()
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    self._add(d["source"], d["text"])

    def _add(self, source, text):
        i = len(self.docs)
        self.docs.append((source, text))
        for g in set(grams(text)):
            self.index[g].add(i)
            self.df[g] += 1
        return i

    def write(self, source, text):
        """Запомнить прочитанное (по абзацам, чтобы вспоминать точнее)."""
        n = 0
        for para in re.split(r"\n\s*\n|\n", text):
            para = para.strip()
            if len(para) >= 40:
                self._add(source, para)
                n += 1
                if self.path:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"source": source, "text": para}, ensure_ascii=False) + "\n")
        return n

    def recall(self, query, k=3):
        if not self.docs:
            return []
        q = Counter(grams(query))
        N = len(self.docs)
        score = defaultdict(float)
        for g, c in q.items():
            ids = self.index.get(g)
            if not ids:
                continue
            idf = math.log(1 + N / (1 + self.df[g]))
            for i in ids:
                score[i] += c * idf
        best = sorted(score, key=lambda i: -score[i] / math.sqrt(len(self.docs[i][1]) + 50))[:k]
        return [self.docs[i] for i in best]

    def __len__(self):
        return len(self.docs)


def _get(url, timeout=10, retries=4):
    """GET с вежливыми повторами: Wikimedia ограничивает частоту запросов (429), уважаем Retry-After."""
    import time
    import urllib.error
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 500, 502, 503) or i == retries - 1:
                raise
            wait = e.headers.get("Retry-After")
            time.sleep(min(30, float(wait)) if wait and wait.isdigit() else 3 * 2 ** i)


def detect_lang(text):
    cyr = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in text)
    return "ru" if cyr > len(text) * 0.2 else "en"


def wiki_search(query, lang=None, n=2, chars=1500):
    """Найти и прочитать статьи Википедии. Возвращает [(заголовок, текст)]."""
    lang = lang or detect_lang(query)
    base = f"https://{lang}.wikipedia.org/w/api.php?"
    try:
        res = _get(base + urllib.parse.urlencode(
            {"action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": n, "prop": "extracts",
             "explaintext": 1, "exintro": 1, "exlimit": n, "format": "json"}))
        pages = sorted(res.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        out = [(p.get("title", ""), (p.get("extract") or "")[:chars]) for p in pages]
    except Exception:
        # запасной путь: REST API (другие лимиты)
        hits = _get(f"https://{lang}.wikipedia.org/w/rest.php/v1/search/page?" +
                    urllib.parse.urlencode({"q": query, "limit": n}))
        out = []
        for h in hits.get("pages", []):
            summ = _get(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/" +
                        urllib.parse.quote(h["key"]))
            out.append((summ.get("title", h["title"]), (summ.get("extract") or "")[:chars]))
    return out


class Hippocampus:
    """Память + интернет. Всё, что агент прочитал в сети, остаётся в памяти навсегда."""

    def __init__(self, path=None):
        self.mem = Memory(path)

    def search(self, query):
        try:
            res = wiki_search(query)
        except Exception as e:                          # сеть недоступна — честно говорим об этом
            return f"ошибка сети: {type(e).__name__}"
        for title, text in res:
            self.mem.write(f"wiki:{title}", text)
        return "\n".join(f"{t}: {x[:600]}" for t, x in res) or "ничего не найдено"

    def recall(self, query):
        hits = self.mem.recall(query, k=2)
        return "\n".join(t[:600] for _, t in hits) or "не помню"
