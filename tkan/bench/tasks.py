"""Экзамен «Ткани»: реальные задачи на русском и английском.

Каждая задача существует в двух языковых версиях с одинаковым содержанием.
Разбиение на обучение и тест детерминировано по содержанию (crc32 ключа без языка).
Тестовые задачи никогда не попадают в обучающий поток, а уровень 5 вообще
не используется в обучении: он проверяет обобщение на более трудное.

Домены:
  math  — арифметика
  word  — текстовые задачи
  order — логика упорядочивания (транзитивность)
  seq   — продолжение числового ряда
  code  — написать функцию на Python (проверяется исполнением на тестах)
  trace — что выведет программа (истина получена исполнением)
"""
import gzip
import json
import os
import random
import re
import zlib
from dataclasses import dataclass, field

LANGS = ("ru", "en")
DOMAINS = ("math", "word", "order", "seq", "code", "trace")
TRAIN_LEVELS = (1, 2, 3, 4)
ALL_LEVELS = (1, 2, 3, 4, 5)
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

Q = {"ru": "Вопрос", "en": "Question"}
A = {"ru": "Ответ", "en": "Answer"}


@dataclass
class Task:
    domain: str
    lang: str
    level: int
    question: str
    answer: str
    key: str
    check: str = "exact"          # exact | number | code | humaneval
    tests: list = field(default_factory=list)

    @property
    def prompt(self):
        return f"{Q[self.lang]}: {self.question}\n{A[self.lang]}:"

    @property
    def text(self):
        """Полный пример для обучения: вопрос, ответ, пустая строка-разделитель."""
        return f"{self.prompt} {self.answer}\n\n"


def is_test_key(key):
    return zlib.crc32(key.encode()) % 5 == 0


# ---------------------------------------------------------------- грамматика
def ru_plural(n, forms):
    n = abs(n)
    if 11 <= n % 100 <= 14:
        return forms[2]
    if n % 10 == 1:
        return forms[0]
    if 2 <= n % 10 <= 4:
        return forms[1]
    return forms[2]


def en_plural(n, forms):
    return forms[0] if n == 1 else forms[1]


# имя (им.), имя (род.), род; параллельное английское имя
PEOPLE = [
    ("Маша", "Маши", "f", "Masha"), ("Аня", "Ани", "f", "Anna"),
    ("Оля", "Оли", "f", "Olga"), ("Катя", "Кати", "f", "Kate"),
    ("Петя", "Пети", "m", "Peter"), ("Ваня", "Вани", "m", "John"),
    ("Дима", "Димы", "m", "Dima"), ("Саша", "Саши", "m", "Alex"),
]
OBJECTS = [
    (("яблоко", "яблока", "яблок"), ("apple", "apples")),
    (("книга", "книги", "книг"), ("book", "books")),
    (("конфета", "конфеты", "конфет"), ("candy", "candies")),
    (("марка", "марки", "марок"), ("stamp", "stamps")),
    (("карандаш", "карандаша", "карандашей"), ("pencil", "pencils")),
    (("мяч", "мяча", "мячей"), ("ball", "balls")),
]
BOX_RU = ("коробка", "коробки", "коробок")
BOX_EN = ("box", "boxes")
VERB_RU = {"buy": ("купил", "купила"), "give": ("отдал", "отдала"), "find": ("нашёл", "нашла")}
PRON_RU = {"m": "Он", "f": "Она"}
PRON_EN = {"m": "He", "f": "She"}


def ru_verb(v, g):
    return VERB_RU[v][0 if g == "m" else 1]


# ---------------------------------------------------------------- генераторы
# Каждый генератор: (rng, level) -> (key, {lang: (question, answer)}, check, tests)

def gen_math(rng, level):
    if level == 1:
        a, b, op = rng.randint(0, 9), rng.randint(0, 9), "+"
    elif level == 2:
        a, b, op = rng.randint(10, 99), rng.randint(10, 99), rng.choice("+-")
    elif level == 3:
        a, b, op = rng.randint(100, 999), rng.randint(100, 999), rng.choice("+-")
    elif level == 4:
        a, b, op = rng.randint(10, 99), rng.randint(2, 9), "*"
    else:
        a, b, op = rng.randint(10, 99), rng.randint(10, 99), "*"
    if op == "-" and b > a:
        a, b = b, a
    res = {"+": a + b, "-": a - b, "*": a * b}[op]
    sym = {"+": "+", "-": "-", "*": "*"}[op]
    q = {"ru": f"Сколько будет {a} {sym} {b}?", "en": f"What is {a} {sym} {b}?"}
    key = f"math|{level}|{a}{op}{b}"
    return key, {l: (q[l], str(res)) for l in LANGS}, "number", []


def gen_word(rng, level):
    pi = rng.randrange(len(PEOPLE))
    nom, gen, g, en_name = PEOPLE[pi]
    oi = rng.randrange(len(OBJECTS))
    ru_o, en_o = OBJECTS[oi]
    P_ru, P_en = PRON_RU[g], PRON_EN[g]

    def ro(n):
        return f"{n} {ru_plural(n, ru_o)}"

    def eo(n):
        return f"{n} {en_plural(n, en_o)}"

    if level == 1:
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        res = a + b
        ru = (f"У {gen} было {ro(a)}. {P_ru} {ru_verb('buy', g)} ещё {ro(b)}. "
              f"Сколько {ru_o[2]} стало у {gen}?")
        en = (f"{en_name} had {eo(a)}. {P_en} bought {eo(b)} more. "
              f"How many {en_o[1]} does {en_name} have now?")
        params = (a, b)
    elif level == 2:
        a = rng.randint(5, 40)
        b = rng.randint(1, a)
        res = a - b
        ru = (f"У {gen} было {ro(a)}. {P_ru} {ru_verb('give', g)} другу {ro(b)}. "
              f"Сколько {ru_o[2]} осталось у {gen}?")
        en = (f"{en_name} had {eo(a)}. {P_en} gave {eo(b)} to a friend. "
              f"How many {en_o[1]} does {en_name} have left?")
        params = (a, b)
    elif level == 3:
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        c = rng.randint(1, a + b)
        res = a + b - c
        ru = (f"У {gen} было {ro(a)}. {P_ru} {ru_verb('buy', g)} ещё {ro(b)}, "
              f"а потом {ru_verb('give', g)} {ro(c)}. Сколько {ru_o[2]} стало у {gen}?")
        en = (f"{en_name} had {eo(a)}. {P_en} bought {eo(b)} more and then gave away {eo(c)}. "
              f"How many {en_o[1]} does {en_name} have now?")
        params = (a, b, c)
    elif level == 4:
        n, k = rng.randint(2, 9), rng.randint(2, 9)
        res = n * k
        ru = (f"У {gen} {n} {ru_plural(n, BOX_RU)}, в каждой по {k} {ru_plural(k, ru_o)}. "
              f"Сколько всего {ru_o[2]} у {gen}?")
        en = (f"{en_name} has {n} {en_plural(n, BOX_EN)} with {eo(k)} in each box. "
              f"How many {en_o[1]} does {en_name} have in total?")
        params = (n, k)
    else:
        n, k = rng.randint(2, 9), rng.randint(2, 9)
        b = rng.randint(1, 20)
        c = rng.randint(1, n * k + b)
        res = n * k + b - c
        ru = (f"У {gen} {n} {ru_plural(n, BOX_RU)}, в каждой по {k} {ru_plural(k, ru_o)}. "
              f"{P_ru} {ru_verb('buy', g)} ещё {ro(b)} и {ru_verb('give', g)} {ro(c)}. "
              f"Сколько {ru_o[2]} стало у {gen}?")
        en = (f"{en_name} has {n} {en_plural(n, BOX_EN)} with {eo(k)} in each box. "
              f"{P_en} bought {eo(b)} more and gave away {eo(c)}. "
              f"How many {en_o[1]} does {en_name} have now?")
        params = (n, k, b, c)
    key = f"word|{level}|{pi}|{oi}|{params}"
    return key, {"ru": (ru, str(res)), "en": (en, str(res))}, "number", []


ORDER_NAMES = [("Аня", "Anna"), ("Боря", "Boris"), ("Витя", "Victor"), ("Галя", "Galina"),
               ("Дима", "Dmitry"), ("Женя", "Eugene"), ("Зоя", "Zoe"), ("Игорь", "Igor")]
# (больше, меньше, самый-больший, самый-меньший)
ORDER_REL = [
    (("выше", "ниже", "самый высокий", "самый низкий"), ("taller", "shorter", "the tallest", "the shortest")),
    (("старше", "младше", "самый старший", "самый младший"), ("older", "younger", "the oldest", "the youngest")),
    (("быстрее", "медленнее", "самый быстрый", "самый медленный"), ("faster", "slower", "the fastest", "the slowest")),
]


def gen_order(rng, level):
    n = level + 1
    idx = rng.sample(range(len(ORDER_NAMES)), n)   # idx[0] > idx[1] > ... по отношению
    ri = rng.randrange(len(ORDER_REL))
    ru_r, en_r = ORDER_REL[ri]
    facts = []
    for i in range(n - 1):
        hi, lo = idx[i], idx[i + 1]
        flip = rng.random() < 0.5
        facts.append((hi, lo, flip))
    rng.shuffle(facts)
    ask_max = rng.random() < 0.5
    ans = idx[0] if ask_max else idx[-1]
    ru_f, en_f = [], []
    for hi, lo, flip in facts:
        if flip:
            ru_f.append(f"{ORDER_NAMES[lo][0]} {ru_r[1]}, чем {ORDER_NAMES[hi][0]}.")
            en_f.append(f"{ORDER_NAMES[lo][1]} is {en_r[1]} than {ORDER_NAMES[hi][1]}.")
        else:
            ru_f.append(f"{ORDER_NAMES[hi][0]} {ru_r[0]}, чем {ORDER_NAMES[lo][0]}.")
            en_f.append(f"{ORDER_NAMES[hi][1]} is {en_r[0]} than {ORDER_NAMES[lo][1]}.")
    ru = " ".join(ru_f) + f" Кто {ru_r[2] if ask_max else ru_r[3]}?"
    en = " ".join(en_f) + f" Who is {en_r[2] if ask_max else en_r[3]}?"
    key = f"order|{level}|{idx}|{ri}|{facts}|{ask_max}"
    return key, {"ru": (ru, ORDER_NAMES[ans][0]), "en": (en, ORDER_NAMES[ans][1])}, "exact", []


def gen_seq(rng, level):
    if level == 1:
        a, d = rng.randint(1, 10), rng.randint(1, 5)
        s = [a + d * i for i in range(5)]
    elif level == 2:
        a, d = rng.randint(1, 50), rng.randint(2, 15)
        s = [a + d * i for i in range(5)]
    elif level == 3:
        a, r = rng.randint(1, 9), rng.randint(2, 4)
        s = [a * r ** i for i in range(5)]
    elif level == 4:
        a, d = rng.randint(1, 10), rng.randint(1, 4)
        s = [a]
        for i in range(4):
            s.append(s[-1] + d + i)          # разности растут на 1
    else:
        a, b = rng.randint(1, 5), rng.randint(1, 5)
        s = [a, b]
        for _ in range(4):
            s.append(s[-1] + s[-2])          # как Фибоначчи
        s = s[:6]
    shown, nxt = s[:-1], s[-1]
    body = ", ".join(map(str, shown))
    q = {"ru": f"Продолжи ряд: {body}, ...", "en": f"Continue the sequence: {body}, ..."}
    key = f"seq|{level}|{s}"
    return key, {l: (q[l], str(nxt)) for l in LANGS}, "number", []


def _ints(rng, lo=-20, hi=20):
    return [rng.randint(lo, hi) for _ in range(6)]


def _lists(rng):
    return [[rng.randint(-9, 9) for _ in range(rng.randint(1, 6))] for _ in range(6)]


def _strs(rng):
    alpha = "abcde"
    return ["".join(rng.choice(alpha) for _ in range(rng.randint(0, 7))) for _ in range(6)]


# (уровень, ru-описание, en-описание, тело функции, генератор входов)
CODE_TEMPLATES = [
    (1, "возвращает x, умноженное на {k}", "returns x multiplied by {k}", "return x * {k}", _ints),
    (1, "возвращает x плюс {k}", "returns x plus {k}", "return x + {k}", _ints),
    (1, "возвращает x минус {k}", "returns x minus {k}", "return x - {k}", _ints),
    (1, "возвращает квадрат x", "returns the square of x", "return x * x", _ints),
    (2, "возвращает сумму элементов списка x", "returns the sum of the elements of list x", "return sum(x)", _lists),
    (2, "возвращает наибольший элемент списка x", "returns the largest element of list x", "return max(x)", _lists),
    (2, "возвращает наименьший элемент списка x", "returns the smallest element of list x", "return min(x)", _lists),
    (2, "возвращает длину строки x", "returns the length of string x", "return len(x)", _strs),
    (2, "возвращает строку x задом наперёд", "returns the string x reversed", "return x[::-1]", _strs),
    (2, "возвращает первые {k} элементов списка x", "returns the first {k} elements of list x", "return x[:{k}]", _lists),
    (2, "возвращает строку x, повторённую {k} раз", "returns the string x repeated {k} times", "return x * {k}", _strs),
    (2, "возвращает список x, отсортированный по возрастанию", "returns list x sorted in ascending order",
     "return sorted(x)", _lists),
    (3, "возвращает True, если x делится на {k}, иначе False", "returns True if x is divisible by {k}, otherwise False",
     "return x % {k} == 0", _ints),
    (3, "возвращает количество чётных чисел в списке x", "returns the number of even numbers in list x",
     "return len([v for v in x if v % 2 == 0])", _lists),
    (3, "возвращает список чисел из x, которые больше {k}", "returns a list of the numbers in x that are greater than {k}",
     "return [v for v in x if v > {k}]", _lists),
    (3, "возвращает большее из чисел x и {k}", "returns the larger of x and {k}", "return max(x, {k})", _ints),
    (3, "возвращает модуль числа x", "returns the absolute value of x", "return abs(x)", _ints),
    (4, "возвращает сумму квадратов элементов списка x", "returns the sum of the squares of the elements of list x",
     "return sum(v * v for v in x)", _lists),
    (4, "возвращает сумму чётных чисел списка x", "returns the sum of the even numbers in list x",
     "return sum(v for v in x if v % 2 == 0)", _lists),
    (4, "возвращает список, в котором каждый элемент x умножен на {k}",
     "returns a list where every element of x is multiplied by {k}", "return [v * {k} for v in x]", _lists),
    (4, "возвращает, сколько раз буква '{c}' встречается в строке x",
     "returns how many times the letter '{c}' occurs in string x", "return x.count('{c}')", _strs),
    (4, "возвращает сумму элементов списка x, умноженную на {k}",
     "returns the sum of the elements of list x multiplied by {k}", "return sum(x) * {k}", _lists),
    (5, "возвращает сумму чисел от 1 до x", "returns the sum of the numbers from 1 to x",
     "return sum(range(1, x + 1))", lambda r: [r.randint(0, 30) for _ in range(6)]),
    (5, "возвращает произведение элементов списка x", "returns the product of the elements of list x",
     "p = 1\n    for v in x:\n        p = p * v\n    return p", _lists),
    (5, "возвращает сумму чисел от 1 до x, делящихся на {k}", "returns the sum of the numbers from 1 to x that are divisible by {k}",
     "return sum(v for v in range(1, x + 1) if v % {k} == 0)", lambda r: [r.randint(0, 30) for _ in range(6)]),
    (5, "возвращает количество гласных букв a и e в строке x", "returns the number of vowels a and e in string x",
     "return x.count('a') + x.count('e')", _strs),
]


def gen_code(rng, level):
    cands = [i for i, t in enumerate(CODE_TEMPLATES) if t[0] == level]
    ti = rng.choice(cands)
    _, ru_d, en_d, body, inputs = CODE_TEMPLATES[ti]
    k, c = rng.randint(2, 9), rng.choice("abcde")
    ru_d, en_d, body = (s.replace("{k}", str(k)).replace("{c}", c) for s in (ru_d, en_d, body))
    code = f"def f(x):\n    {body}"
    ns = {}
    exec(code, ns)
    tests = []
    for x in inputs(rng):
        tests.append([x, ns["f"](x)])
    ru = f"Напиши на Python функцию f(x), которая {ru_d}."
    en = f"Write a Python function f(x) that {en_d}."
    key = f"code|{level}|{ti}|{k if '{k}' in CODE_TEMPLATES[ti][3] else ''}|{c if '{c}' in CODE_TEMPLATES[ti][3] else ''}"
    return key, {"ru": (ru, code), "en": (en, code)}, "code", tests


def gen_trace(rng, level):
    a, b, c, d = (rng.randint(1, 9) for _ in range(4))
    if level == 1:
        prog = f"x = {a}\nprint(x + {b})"
    elif level == 2:
        prog = f"x = {a}\ny = x * {b} - {c}\nprint(y)"
    elif level == 3:
        prog = f"x = {a}\nif x > {b}:\n    x = x - {c}\nelse:\n    x = x + {d}\nprint(x)"
    elif level == 4:
        prog = f"s = {a}\nfor i in range({b}):\n    s = s + {c}\nprint(s)"
    else:
        prog = f"s = 0\nfor i in range({a + 2}):\n    if i % 2 == 0:\n        s = s + i * {b}\nprint(s)"
    out = []
    exec(prog, {"print": lambda v: out.append(str(v))})
    q = {"ru": f"Что выведет программа?\n{prog}", "en": f"What does this program print?\n{prog}"}
    key = f"trace|{level}|{prog}"
    return key, {l: (q[l], out[0]) for l in LANGS}, "number", []


GENERATORS = {"math": gen_math, "word": gen_word, "order": gen_order,
              "seq": gen_seq, "code": gen_code, "trace": gen_trace}


def make(domain, lang, level, rng):
    key, versions, check, tests = GENERATORS[domain](rng, level)
    q, a = versions[lang]
    return Task(domain, lang, level, q, a, key, check, tests)


def train_stream(seed, levels=TRAIN_LEVELS, domains=DOMAINS):
    """Бесконечный поток обучающих задач (тестовые ключи и уровень 5 исключены)."""
    rng = random.Random(seed)
    while True:
        domain = rng.choice(domains)
        level = rng.choice(levels)
        key, versions, check, tests = GENERATORS[domain](rng, level)
        if is_test_key(key):
            continue
        lang = rng.choice(LANGS)
        q, a = versions[lang]
        yield Task(domain, lang, level, q, a, key, check, tests)


def test_set(n_per_cell=20, seed=12345, domains=DOMAINS, levels=ALL_LEVELS):
    """Фиксированный тест: n задач на каждую ячейку домен×уровень, обе языковые версии."""
    rng = random.Random(seed)
    tasks = []
    for domain in domains:
        for level in levels:
            seen, tries = set(), 0
            while len(seen) < n_per_cell and tries < 20000:
                tries += 1
                key, versions, check, tests = GENERATORS[domain](rng, level)
                if key in seen or not (is_test_key(key) or level == 5):
                    continue
                seen.add(key)
                for lang in LANGS:
                    q, a = versions[lang]
                    tasks.append(Task(domain, lang, level, q, a, key, check, tests))
    return tasks


# ---------------------------------------------------------------- реальные бенчмарки
def load_gsm8k(n=None):
    out = []
    with open(os.path.join(DATA, "gsm8k_test.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            ans = d["answer"].split("####")[-1].strip().replace(",", "")
            out.append(Task("gsm8k", "en", 0, d["question"], ans, f"gsm8k|{i}", "number"))
    return out[:n]


def load_mgsm(lang, n=None):
    out = []
    with open(os.path.join(DATA, f"mgsm_{lang}.tsv"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            q, a = line.rstrip("\n").split("\t")
            out.append(Task("mgsm", lang, 0, q, a.replace(",", ""), f"mgsm|{i}", "number"))
    return out[:n]


def load_humaneval(n=None):
    out = []
    with gzip.open(os.path.join(DATA, "humaneval.jsonl.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = Task("humaneval", "en", 0, d["prompt"], d["canonical_solution"], d["task_id"], "humaneval",
                     [d["test"], d["entry_point"]])
            out.append(t)
    return out[:n]


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def first_number(s):
    m = NUM_RE.search(s.replace(",", ""))
    return m.group(0) if m else None


def last_number(s):
    m = NUM_RE.findall(s.replace(",", ""))
    return m[-1] if m else None
