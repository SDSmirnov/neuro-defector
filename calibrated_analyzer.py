"""
Калибровщик и анализатор перплексии с абсолютными порогами.

Два режима:

  calibrate — прогнать два каталога с эталонами (человеческие тексты
  и ИИ-генерации), посчитать распределения log_std_nll и log_mean_nll
  по чанкам, обучить логистическую регрессию, сохранить в JSON.

  analyze — анализировать новый текст, используя сохранённую
  калибровку. Для каждого чанка считает log-likelihood ratio и
  выдаёт балл «человечности» от -1 (явно ИИ) до +1 (явно человек).

Использование:

  # Шаг 1: разложить эталоны
  corpora/
    human/
      platonov_chevengur.txt
      dovlatov_compromise.txt
      ...
    ai/
      chatgpt_story_1.txt
      claude_story_1.txt
      ...

  # Шаг 2: откалибровать (один раз)
  python calibrated_analyzer.py calibrate corpora/human corpora/ai
  python calibrated_analyzer.py calibrate corpora/human corpora/ai --name fiction-ru-2026-01

  # Шаг 3: анализировать любой текст с этой калибровкой
  python calibrated_analyzer.py analyze AK.txt
  python calibrated_analyzer.py analyze AK.txt --verbose

Семантические фичи (опционально, требуется sentence-transformers и numpy):

  Если установлены sentence-transformers и numpy, при калибровке
  дополнительно строятся индексы предложений двух корпусов и
  считаются три семантические фичи:

    sem_tail_ratio   — kNN-расстояние до человеческого корпуса минус
                       до ИИ-корпуса; семантический аналог log_mean_nll.
    sem_self_repeat  — повторяемость идей внутри чанка (max self-cosine).
    cliche_proximity — близость предложений к встроенному списку клише.

  Эмбеддинги корпусов сохраняются в calibration_embeds.npz рядом с JSON.
  Если зависимости не установлены, всё продолжает работать на 7 базовых
  фичах — старые калибровки тоже остаются совместимыми.

  Установка:
    pip install sentence-transformers numpy

Свой список клише (опционально):

  python calibrated_analyzer.py calibrate corpora/human corpora/ai \\
      --cliches my_cliches.json
  python calibrated_analyzer.py analyze AK.txt --cliches my_cliches.json

  Формат JSON: {"bad_style_examples": ["взгляд стал ледяным", "вдруг", ...]}
  или просто список строк на верхнем уровне.

  Одиночные слова в списке («вдруг», «казалось») идут в lexical_marker_rate
  как частотный признак.  Многословные фразы — в семантический индекс
  и сравниваются с предложениями текста по косинусу.

  Список сохраняется в calibration.json — analyze автоматически
  применит ту же разметку, если --cliches не задан повторно.
"""

import sys
import re
import math
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-270m" # "sberbank-ai/rugpt3small_based_on_gpt2" # "ai-forever/mGPT" #    #
TARGET_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 800       # поднято с 600 — на коротких чанках PPL слишком шумная
MAX_CHUNK_SIZE = 1400
CALIBRATION_FILE = "calibration.json"
CALIBRATION_EMBEDS_FILE = "calibration_embeds.npz"

# Малая константа для логарифмов
EPS = 1e-6

# ─────────────────────── семантический энкодер ───────────────────────

# Sentence-encoder для семантических фич. Lazy-loading: подгружается
# только при первом обращении.  Если sentence-transformers/numpy не
# установлены, скрипт продолжает работать на базовых фичах.
SEM_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEM_KNN_K = 5  # сколько ближайших соседей усреднять для устойчивости

_SEM_CACHE = {"available": None, "model": None, "np": None, "error": None}


def _try_import_semantic():
    """Пытается импортировать numpy и sentence-transformers.
    Кэширует результат — повторные вызовы бесплатны."""
    if _SEM_CACHE["available"] is not None:
        return _SEM_CACHE["available"]
    try:
        import numpy as np  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        _SEM_CACHE["np"] = np
        _SEM_CACHE["available"] = True
    except ImportError as e:
        _SEM_CACHE["available"] = False
        _SEM_CACHE["error"] = str(e)
    return _SEM_CACHE["available"]


def _get_sem_model():
    """Возвращает загруженную sentence-transformer модель.
    Загружает при первом обращении.  Возвращает None, если модуль
    недоступен или модель не удалось скачать."""
    if not _try_import_semantic():
        return None
    if _SEM_CACHE["model"] is not None:
        return _SEM_CACHE["model"]
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Загружаю semantic encoder {SEM_MODEL_NAME}…")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(SEM_MODEL_NAME, device=device)
        _SEM_CACHE["model"] = model
        return model
    except Exception as e:
        _SEM_CACHE["error"] = f"Не удалось загрузить semantic encoder: {e}"
        _SEM_CACHE["available"] = False
        return None


def encode_sentences(sentences: list[str]):
    """Возвращает np.ndarray (n, dim) нормированных эмбеддингов
    или None, если энкодер недоступен."""
    if not sentences:
        return None
    model = _get_sem_model()
    if model is None:
        return None
    np = _SEM_CACHE["np"]
    embs = model.encode(sentences, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    return embs.astype(np.float32)


# ─────────────────────── дополнительные фичи ───────────────────────

# Порог для tail_ratio. Подобран под rugpt3small: токены с nll > 8 —
# это «удивившие» модель (имена, редкая лексика, неологизмы).
TAIL_NLL_THRESHOLD = 8.0


def feature_tail_ratio(token_nll: list[float],
                         threshold: float = TAIL_NLL_THRESHOLD) -> float:
    """Доля «удививших» модель токенов. У живого текста выше."""
    if not token_nll:
        return 0.0
    return sum(1 for x in token_nll if x > threshold) / len(token_nll)


def feature_cv_nll(token_nll: list[float]) -> float:
    """Коэффициент вариации NLL — нормированный std.
    Лучше отделяет «ровный ИИ» от «рваного человека», чем std отдельно."""
    if not token_nll:
        return 0.0
    n = len(token_nll)
    mean = sum(token_nll) / n
    if mean < 1e-9:
        return 0.0
    var = sum((x - mean) ** 2 for x in token_nll) / n
    std = math.sqrt(var)
    return std / mean


def feature_repeat_3gram_ratio(text: str) -> float:
    """Доля повторяющихся словесных триграмм. ИИ повторяет
    конструкции чаще: «он посмотрел на неё», «она тихо вздохнула»
    и подобные."""
    words = re.findall(r'[а-яёa-z0-9]+', text.lower())
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    counts = {}
    for tg in trigrams:
        counts[tg] = counts.get(tg, 0) + 1
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(trigrams)


def feature_cv_sent_len(text: str) -> float:
    """Коэффициент вариации длин предложений (в словах).
    Высокий = живой ритм с короткими и длинными вперемешку."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', text.strip())
                 if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lens = [len(s.split()) for s in sentences]
    n = len(lens)
    mean = sum(lens) / n
    if mean < 1e-9:
        return 0.0
    var = sum((x - mean) ** 2 for x in lens) / n
    std = math.sqrt(var)
    return std / mean


def feature_punct_entropy(text: str) -> float:
    """Энтропия распределения знаков пунктуации в тексте.
    У живого текста — выше, у ИИ — обычно беднее (меньше разных знаков)."""
    # Считаем основные знаки. Простая кириллица + латиница.
    punct_chars = ['.', ',', '—', '–', '-', '!', '?', '…', ':', ';',
                   '(', ')', '«', '»', '"', "'"]
    counts = {}
    for ch in text:
        if ch in punct_chars:
            counts[ch] = counts.get(ch, 0) + 1
    total = sum(counts.values())
    if total < 2:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p)
    return entropy


# ─────────────────────── семантические клише ───────────────────────

# Список ИИ-маркеров делится на две категории:
#
#   1) Лексические маркеры — одиночные слова («вдруг», «казалось»,
#      «безусловно»). Они НЕ годятся для семантического сравнения
#      (вектор одного слова слишком общий и даст ложные срабатывания);
#      их ловим частотным счётчиком — высокая плотность таких слов
#      сама по себе диагностический сигнал.
#
#   2) Семантические клише — фразы из 2+ слов («в горле пересохло»,
#      «взгляд стал ледяным»). Эмбеддятся, сравниваются с предложениями
#      текста по косинусу.
#
# Порог CLICHE_SIM_HIT — выше него считаем, что предложение
# семантически рифмуется с клише.  Подобран эмпирически для
# paraphrase-multilingual-MiniLM-L12-v2; при смене энкодера может
# потребоваться корректировка.
CLICHE_SIM_HIT = 0.62

# Минимальная длина предложения для семантического матчинга.
# Эмбеддинги MiniLM на 1–3-словных репликах шумные — почти всегда ложные срабатывания.
MIN_CLICHE_SENT_WORDS = 5
MIN_CLICHE_SENT_CHARS = 25

# Максимум подсвечиваемых клише в одном чанке (остальные учитываются в счёте, но не выделяются).
MAX_CLICHE_HIGHLIGHTS = 5

# Высокая плотность слов-маркеров считается диагностическим сигналом.
# Например, 3 «вдруг» на 200 слов = 1.5% — это уже подозрительно много.
LEXICAL_MARKER_RATE_HIT = 0.008  # 0.8% от слов

# Экспрессивные глаголы речи — отдельная редакторская диагностика.
# В единственном числе могут быть приёмом; в большой концентрации — маркер мелодрамы/шаблона.
EXPRESSIVE_SPEECH_VERBS = re.compile(
    r'\b(прохрипел[аи]?|прорычал[аи]?|прошипел[аи]?|выдохнул[аи]?|процедил[аи]?|'
    r'рявкнул[аи]?|пробормотал[аи]?|прошептал[аи]?|пробурчал[аи]?|'
    r'воскликнул[аи]?|выпалил[аи]?|отчеканил[аи]?|прошелестел[аи]?|'
    r'промычал[аи]?|прошепнул[аи]?|прорыдал[аи]?|проворчал[аи]?|'
    r'пробасил[аи]?|прогудел[аи]?|прокаркал[аи]?|замямлил[аи]?|засипел[аи]?)\b',
    re.IGNORECASE,
)


# Дефолтный встроенный список — fallback на случай, если внешний JSON
# не передан. Намеренно короткий: для серьёзного анализа лучше передать
# свой список через --cliches PATH.
DEFAULT_CLICHES = [
    "его сердце сжалось от боли",
    "по спине пробежал холодок",
    "слёзы навернулись на глаза",
    "к горлу подступил комок",
    "тишина была оглушительной",
    "время словно остановилось",
    "их взгляды встретились",
    "он сделал глубокий вдох",
    "она крепко сжала кулаки",
    "лунный свет заливал комнату",
    "вдруг", "внезапно", "безусловно", "конечно",
]


def _normalize_cliche_phrase(raw: str) -> str:
    """Убирает скобочные пометки опционального субъекта.
    «(взгляд) похож на лезвие» → «взгляд похож на лезвие»
    «(голос) ровный, как сталь» → «голос ровный, как сталь»

    Обоснование: в исходном списке скобки помечают подразумеваемый
    субъект, но для эмбеддинга нужно естественное предложение
    без служебной разметки."""
    # Просто снимаем парные скобки целиком
    cleaned = re.sub(r'[()]', '', raw).strip()
    # Сворачиваем двойные пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def load_cliches_from_json(path: str) -> tuple[list[str], list[str]]:
    """Загружает список клише из JSON и разделяет его на:
      - лексические маркеры (одиночные слова)
      - семантические клише (2+ слова, нормализованные)

    Поддерживаемые форматы JSON:
      - {"bad_style_examples": [...]}
      - {"cliches": [...]}
      - просто [...] на верхнем уровне
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Берём первый список, который найдём
        for key in ("bad_style_examples", "cliches", "phrases", "items"):
            if key in data and isinstance(data[key], list):
                raw_list = data[key]
                break
        else:
            # Fallback: первое же значение, если оно список
            for v in data.values():
                if isinstance(v, list):
                    raw_list = v
                    break
            else:
                raise ValueError(f"В {path} не нашёл списка клише")
    elif isinstance(data, list):
        raw_list = data
    else:
        raise ValueError(f"Неожиданный формат {path}")

    lexical = []
    semantic = []
    seen_lex = set()
    seen_sem = set()
    for raw in raw_list:
        if not isinstance(raw, str):
            continue
        normalized = _normalize_cliche_phrase(raw)
        if not normalized:
            continue
        words = normalized.split()
        if len(words) == 1:
            w = words[0].lower()
            if w not in seen_lex:
                lexical.append(w)
                seen_lex.add(w)
        else:
            if normalized not in seen_sem:
                semantic.append(normalized)
                seen_sem.add(normalized)
    return lexical, semantic


# Глобальные списки клише — могут быть переопределены через --cliches
# при запуске. Дефолт — из DEFAULT_CLICHES.
LEXICAL_MARKERS: list[str] = []
SEMANTIC_CLICHES: list[str] = []


def _init_cliches_from_default():
    """Инициализирует LEXICAL_MARKERS и SEMANTIC_CLICHES из дефолтного
    списка, если они ещё не были загружены."""
    global LEXICAL_MARKERS, SEMANTIC_CLICHES
    if LEXICAL_MARKERS or SEMANTIC_CLICHES:
        return
    for raw in DEFAULT_CLICHES:
        normalized = _normalize_cliche_phrase(raw)
        if not normalized:
            continue
        words = normalized.split()
        if len(words) == 1:
            LEXICAL_MARKERS.append(words[0].lower())
        else:
            SEMANTIC_CLICHES.append(normalized)


def set_cliches(lexical: list[str], semantic: list[str]):
    """Заменяет глобальные списки и сбрасывает кэш эмбеддингов."""
    global LEXICAL_MARKERS, SEMANTIC_CLICHES
    LEXICAL_MARKERS = lexical
    SEMANTIC_CLICHES = semantic
    _CLICHE_EMBEDS_CACHE["embs"] = None
    _CLICHE_EMBEDS_CACHE["cluster_ids"] = None


# Ленивый кэш эмбеддингов для семантических клише — считаются один раз.
_CLICHE_EMBEDS_CACHE: dict = {"embs": None, "cluster_ids": None}


def _cluster_cliche_embs(embs, threshold: float = 0.85) -> list[int]:
    """Greedy union-find кластеризация клише по косинусной близости.
    Возвращает список cluster_id для каждого клише (< threshold → собственный кластер)."""
    n = len(embs)
    np = _SEM_CACHE["np"]
    if np is None or n == 0:
        return list(range(n))
    sim = embs @ embs.T
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= threshold:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[max(pi, pj)] = min(pi, pj)

    return [find(i) for i in range(n)]


def _get_cliche_embeddings():
    """Возвращает np.ndarray (n_cliches, dim) с нормированными
    эмбеддингами семантических клише или None, если энкодер недоступен."""
    if _CLICHE_EMBEDS_CACHE["embs"] is not None:
        return _CLICHE_EMBEDS_CACHE["embs"]
    _init_cliches_from_default()
    if not SEMANTIC_CLICHES:
        return None
    embs = encode_sentences(SEMANTIC_CLICHES)
    if embs is not None:
        _CLICHE_EMBEDS_CACHE["embs"] = embs
        _CLICHE_EMBEDS_CACHE["cluster_ids"] = _cluster_cliche_embs(embs)
    return embs


def feature_lexical_marker_rate(text: str) -> tuple[float, list[tuple[int, int, str]]]:
    """Доля слов-маркеров от общего числа слов в тексте + позиции
    каждого употребления (для подсветки).

    Возвращает (rate, occurrences), где occurrences — список
    (start, end, marker_word).
    """
    _init_cliches_from_default()
    if not LEXICAL_MARKERS:
        return 0.0, []

    # Все слова текста с offset'ами
    word_matches = list(re.finditer(r'[А-Яа-яЁёA-Za-z]+', text))
    total_words = len(word_matches)
    if total_words == 0:
        return 0.0, []

    marker_set = set(LEXICAL_MARKERS)
    occurrences = []
    for m in word_matches:
        if m.group().lower() in marker_set:
            occurrences.append((m.start(), m.end(), m.group()))

    rate = len(occurrences) / total_words
    return rate, occurrences


def _split_sentences_for_sem(text: str) -> list[tuple[str, int, int]]:
    """Разбивает текст на предложения, возвращая тройки
    (предложение, start_offset, end_offset).  Offsets нужны для
    подсветки в diagnose_chunk."""
    out = []
    cursor = 0
    for m in re.finditer(r'[^.!?…]+[.!?…]+', text):
        s, e = m.start(), m.end()
        sent = text[s:e].strip()
        if sent and len(sent) > 5:
            out.append((sent, s, e))
        cursor = e
    # Хвост без терминатора
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail and len(tail) > 5:
            out.append((tail, cursor, len(text)))
    return out


def feature_sem_tail_ratio(chunk_sent_embs, calib_embeds: dict | None) -> float:
    """Семантический аналог tail_ratio.  Для каждого предложения чанка
    считает среднее косинусное расстояние до K ближайших соседей в
    человеческом корпусе минус то же для ИИ-корпуса.  Усредняется по
    предложениям.  Положительное значение → текст семантически ближе
    к человеческому корпусу.

    Эмбеддинги нормированы, поэтому расстояние = 1 - cosine_similarity.

    Возвращает 0.0, если индексы недоступны."""
    if chunk_sent_embs is None or calib_embeds is None:
        return 0.0
    H = calib_embeds.get("human")
    A = calib_embeds.get("ai")
    if H is None or A is None or len(H) == 0 or len(A) == 0:
        return 0.0
    np = _SEM_CACHE["np"]
    if np is None:
        return 0.0

    # косинусные близости (эмбеддинги нормированы)
    sim_h = chunk_sent_embs @ H.T  # (n_sent, n_human)
    sim_a = chunk_sent_embs @ A.T

    k = min(SEM_KNN_K, sim_h.shape[1], sim_a.shape[1])
    # топ-k ближайших — наибольшие cos similarity = наименьшие расстояния
    top_h = np.partition(sim_h, -k, axis=1)[:, -k:].mean(axis=1)
    top_a = np.partition(sim_a, -k, axis=1)[:, -k:].mean(axis=1)

    # Чем выше близость к человеческому корпусу относительно ИИ —
    # тем «человечнее» предложение.
    score = (top_h - top_a).mean()
    return float(score)


def _compute_sem_tail_per_sent(chunk_sent_embs,
                                calib_embeds: dict) -> list[float]:
    """Возвращает per-sentence sem_tail scores (top_h - top_a) для подсветки
    конкретных предложений. Положительное → ближе к человеку, отрицательное → к ИИ."""
    H = calib_embeds.get("human")
    A = calib_embeds.get("ai")
    if H is None or A is None or len(H) == 0 or len(A) == 0:
        return []
    np = _SEM_CACHE["np"]
    if np is None:
        return []
    sim_h = chunk_sent_embs @ H.T
    sim_a = chunk_sent_embs @ A.T
    k = min(SEM_KNN_K, sim_h.shape[1], sim_a.shape[1])
    top_h = np.partition(sim_h, -k, axis=1)[:, -k:].mean(axis=1)
    top_a = np.partition(sim_a, -k, axis=1)[:, -k:].mean(axis=1)
    return (top_h - top_a).tolist()


def feature_sem_self_repeat(chunk_sent_embs) -> float:
    """Повторяемость идей внутри чанка.  Для каждого предложения
    находит максимальную косинусную близость к остальным предложениям
    того же чанка и усредняет.  Высокое значение = чанк крутится
    вокруг одной мысли, переформулированной разными словами."""
    if chunk_sent_embs is None or len(chunk_sent_embs) < 3:
        return 0.0
    np = _SEM_CACHE["np"]
    if np is None:
        return 0.0

    sim = chunk_sent_embs @ chunk_sent_embs.T
    # Зануляем диагональ (самосходство = 1)
    np.fill_diagonal(sim, -1.0)
    max_sim_per_sent = sim.max(axis=1)
    return float(max_sim_per_sent.mean())


def feature_cliche_proximity(chunk_sent_embs) -> float:
    """Близость предложений чанка к каноническому списку клише.
    Для каждого предложения — максимум косинуса по всем клише.
    Усредняется по предложениям.  Высокое значение = в чанке много
    фраз, семантически рифмующихся с типовыми ИИ-формулировками."""
    if chunk_sent_embs is None or len(chunk_sent_embs) == 0:
        return 0.0
    cliche_embs = _get_cliche_embeddings()
    if cliche_embs is None:
        return 0.0
    np = _SEM_CACHE["np"]
    sim = chunk_sent_embs @ cliche_embs.T  # (n_sent, n_cliches)
    max_per_sent = sim.max(axis=1)
    return float(max_per_sent.mean())


def find_cliche_matches(sentences_with_offsets: list[tuple[str, int, int]],
                         chunk_sent_embs,
                         threshold: float = CLICHE_SIM_HIT) -> list[dict]:
    """Возвращает список найденных клише-сходств:
      [{"sentence": ..., "cliche": ..., "similarity": 0.71,
        "start": 0, "end": 42}, ...]
    Используется в diagnose_chunk для подсветки конкретных фраз.

    Короткие предложения (< MIN_CLICHE_SENT_WORDS слов или < MIN_CLICHE_SENT_CHARS
    символов) пропускаются — их эмбеддинги слишком шумные.
    Результат дедуплицируется по кластерам близких клише."""
    if chunk_sent_embs is None or not sentences_with_offsets:
        return []
    cliche_embs = _get_cliche_embeddings()
    if cliche_embs is None:
        return []
    np = _SEM_CACHE["np"]

    sim = chunk_sent_embs @ cliche_embs.T  # (n_sent, n_cliches)
    matches = []
    for i, (sent, s_off, e_off) in enumerate(sentences_with_offsets):
        if i >= sim.shape[0]:
            break
        # Пропускаем слишком короткие предложения — их эмбеддинг ненадёжен
        if len(sent) < MIN_CLICHE_SENT_CHARS or len(sent.split()) < MIN_CLICHE_SENT_WORDS:
            continue
        best_j = int(np.argmax(sim[i]))
        best_sim = float(sim[i, best_j])
        if best_sim >= threshold:
            matches.append({
                "sentence": sent,
                "cliche": SEMANTIC_CLICHES[best_j],
                "cliche_idx": best_j,
                "similarity": best_sim,
                "start": s_off,
                "end": e_off,
            })
    # Сортируем по убыванию близости — самые «клишированные» сверху
    matches.sort(key=lambda m: -m["similarity"])

    # Дедупликация по кластеру клише: если два предложения попали в один
    # кластер близких клише, оставляем только то, что ближе.
    cluster_ids = _CLICHE_EMBEDS_CACHE.get("cluster_ids")
    if cluster_ids:
        seen_clusters: set[int] = set()
        deduped = []
        for m in matches:
            cid = cluster_ids[m["cliche_idx"]]
            if cid not in seen_clusters:
                seen_clusters.add(cid)
                deduped.append(m)
        matches = deduped

    return matches


# ─────────────────────── общая инфраструктура ───────────────────────

def split_into_chunks(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks, current = [], ""
    for sent in sentences:
        if not sent.strip():
            continue
        if current and len(current) + len(sent) > MAX_CHUNK_SIZE:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent) if current else sent
            if len(current) >= TARGET_CHUNK_SIZE:
                chunks.append(current.strip())
                current = ""
    if current.strip():
        if chunks and len(current) < MIN_CHUNK_SIZE:
            chunks[-1] = chunks[-1] + " " + current.strip()
        else:
            chunks.append(current.strip())
    return chunks


def compute_chunk_stats(text: str, model, tokenizer, device,
                         max_length: int = 2048,
                         keep_tokens: bool = False,
                         calib_embeds: dict | None = None,
                         use_semantic: bool = True) -> dict | None:
    """
    Считает статистики по чанку.

    keep_tokens=True — дополнительно возвращает сырые token_nll и
    декодированные токены для диагностики.  В режиме калибровки этого
    не нужно (только финальные статистики), при анализе включается.

    calib_embeds — dict с ключами "human", "ai" (np.ndarray эмбеддингов
    калибровочных предложений).  Если задан, считается sem_tail_ratio.

    use_semantic — если False, семантические фичи пропускаются (используется
    при первом проходе калибровки до построения индекса).
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    n_tokens = input_ids.size(1)

    if n_tokens < 2:
        return None

    # Предупреждение, если упёрлись в лимит — статистика будет по обрезку
    truncated = n_tokens >= max_length

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
    token_nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_nll = token_nll.squeeze(0).cpu().numpy()

    mean_nll = float(token_nll.mean())
    std_nll = float(token_nll.std())

    # Список nll для расчёта производных признаков
    token_nll_list = token_nll.tolist()

    # ─── Дополнительные признаки ───
    tail_ratio = feature_tail_ratio(token_nll_list)
    cv_nll = feature_cv_nll(token_nll_list)
    repeat_3gram = feature_repeat_3gram_ratio(text)
    cv_sent_len = feature_cv_sent_len(text)
    punct_entropy = feature_punct_entropy(text)

    # ─── Лексические маркеры (одиночные слова из списка клише) ───
    # Эта фича не требует энкодера — просто частотный счётчик.
    lex_rate, lex_occurrences = feature_lexical_marker_rate(text)

    # ─── Семантические фичи (если энкодер доступен) ───
    sem_tail_ratio = 0.0
    sem_self_repeat = 0.0
    cliche_proximity = 0.0
    sent_offsets = []  # для подсветки в diagnose_chunk
    chunk_sent_embs = None
    sem_tail_per_sent: list[float] = []  # per-sentence scores для fine-grained подсветки

    if use_semantic and _try_import_semantic():
        sentences_with_offsets = _split_sentences_for_sem(text)
        if len(sentences_with_offsets) >= 2:
            sentences_only = [s[0] for s in sentences_with_offsets]
            chunk_sent_embs = encode_sentences(sentences_only)
            if chunk_sent_embs is not None:
                sem_tail_ratio = feature_sem_tail_ratio(chunk_sent_embs, calib_embeds)
                sem_self_repeat = feature_sem_self_repeat(chunk_sent_embs)
                cliche_proximity = feature_cliche_proximity(chunk_sent_embs)
                sent_offsets = sentences_with_offsets
                if keep_tokens and calib_embeds is not None:
                    sem_tail_per_sent = _compute_sem_tail_per_sent(chunk_sent_embs, calib_embeds)

    result = {
        "mean_nll": mean_nll,
        "std_nll": std_nll,
        "log_mean_nll": math.log(max(mean_nll, EPS)),
        "log_std_nll": math.log(max(std_nll, EPS)),
        "tail_ratio": tail_ratio,
        "cv_nll": cv_nll,
        "repeat_3gram": repeat_3gram,
        "cv_sent_len": cv_sent_len,
        "punct_entropy": punct_entropy,
        "lexical_marker_rate": lex_rate,
        "sem_tail_ratio": sem_tail_ratio,
        "sem_self_repeat": sem_self_repeat,
        "cliche_proximity": cliche_proximity,
        "n_tokens": int(token_nll.shape[0]),
        "truncated": truncated,
    }

    if keep_tokens:
        # Декодируем каждый токен (кроме первого, для которого нет nll)
        # для последующего сопоставления с позициями в тексте.
        token_ids = input_ids.squeeze(0).cpu().numpy()[1:]  # сдвиг как у nll
        token_strs = [tokenizer.decode([int(tid)]) for tid in token_ids]
        result["token_nll"] = token_nll.tolist()
        result["token_strs"] = token_strs

        # Получаем позиции токенов в исходном тексте — для подсветки.
        # Используем отдельный вызов с return_offsets_mapping=True.
        try:
            enc_off = tokenizer(text, return_tensors=None, truncation=True,
                                max_length=max_length,
                                return_offsets_mapping=True)
            offsets = enc_off.get("offset_mapping", [])
            # Сдвигаем на 1, как и token_nll/token_strs
            if len(offsets) >= 2:
                result["token_offsets"] = list(offsets[1:])
            else:
                result["token_offsets"] = []
        except (TypeError, NotImplementedError):
            # Не все токенизаторы поддерживают offset_mapping
            # (rugpt3small использует медленный токенизатор без offsets).
            # Сделаем приближённое сопоставление через декодирование.
            result["token_offsets"] = _approximate_offsets(text, token_strs)

        # Для диагностики клише: сохраним предложения с offset'ами
        # и их эмбеддинги (если посчитались).
        result["sent_offsets"] = sent_offsets
        result["sent_embs"] = chunk_sent_embs  # np.ndarray или None
        result["lex_occurrences"] = lex_occurrences  # для подсветки слов-маркеров
        result["sem_tail_per_sent"] = sem_tail_per_sent  # per-sentence sem_tail

    return result


def _approximate_offsets(text: str, token_strs: list[str]) -> list[tuple[int, int]]:
    """Сопоставляет декодированные токены с позициями в исходном тексте
    приближённо: ищет каждый токен последовательно от текущей позиции.
    Не идеально для BPE с пробелами, но для подсветки достаточно."""
    offsets = []
    cursor = 0
    for tok in token_strs:
        if not tok:
            offsets.append((cursor, cursor))
            continue
        # Очищаем токен от ведущих/висящих пробелов для поиска
        clean = tok.strip()
        if not clean:
            offsets.append((cursor, cursor))
            continue
        idx = text.find(clean, cursor)
        if idx == -1:
            # Не нашли — ставим заглушку
            offsets.append((cursor, cursor))
        else:
            offsets.append((idx, idx + len(clean)))
            cursor = idx + len(clean)
    return offsets


def load_model():
    print(f"Загружаю модель {MODEL_NAME}…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        #use_safetensors=False,
    ).to(device)
    model.eval()
    print(f"Модель загружена, устройство: {device}\n")
    return model, tokenizer, device


# ─────────────────────── статистика без numpy/scipy ───────────────────────

def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _mad(xs: list[float]) -> float:
    med = _median(xs)
    return _median([abs(x - med) for x in xs])


def percentiles(values: list[float], qs: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    out = {}
    for q in qs:
        idx = q * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        out[f"p{int(q*100)}"] = s[lo] * (1 - frac) + s[hi] * frac
    return out


def _gaussian_loglik(x: float, center: float, scale: float) -> float:
    if scale < 1e-9:
        scale = 1e-9
    return -0.5 * ((x - center) / scale) ** 2 - math.log(scale)


# ─────────────────────── логистическая регрессия ───────────────────────

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(features: list[list[float]], labels: list[int],
                 lr: float = 0.1, n_iter: int = 2000,
                 l2: float = 0.01) -> tuple[list[float], float]:
    """
    Минимальный градиентный спуск для бинарной логрегрессии.
    label=1 → человек, label=0 → ИИ.

    Возвращает (weights, bias). Положительный score → человек.

    Стандартизация фич делается снаружи; здесь предполагаем, что
    фичи уже центрированы и отнормированы.
    """
    n_features = len(features[0])
    w = [0.0] * n_features
    b = 0.0
    n = len(features)

    for it in range(n_iter):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        loss = 0.0

        for x, y in zip(features, labels):
            z = b + sum(wi * xi for wi, xi in zip(w, x))
            p = _sigmoid(z)
            err = p - y
            grad_b += err
            for j in range(n_features):
                grad_w[j] += err * x[j]
            # для логирования
            if y == 1:
                loss += -math.log(max(p, 1e-12))
            else:
                loss += -math.log(max(1 - p, 1e-12))

        # средние градиенты + L2
        for j in range(n_features):
            grad_w[j] = grad_w[j] / n + l2 * w[j]
            w[j] -= lr * grad_w[j]
        b -= lr * (grad_b / n)

        if (it + 1) % 500 == 0:
            print(f"  iter {it+1}: loss={loss/n:.4f}")

    return w, b


def standardize_params(values_per_feature: list[list[float]]) -> tuple[list[float], list[float]]:
    """Считает (mean, std) для каждой фичи по всему калибровочному набору."""
    means, stds = [], []
    for col in values_per_feature:
        m = sum(col) / len(col)
        v = sum((x - m) ** 2 for x in col) / len(col)
        s = math.sqrt(v) if v > 1e-12 else 1.0
        means.append(m)
        stds.append(s)
    return means, stds


def standardize(x: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(xi - m) / s for xi, m, s in zip(x, means, stds)]


# ─────────────────────── ROC-AUC без sklearn ───────────────────────

def roc_auc(scores: list[float], labels: list[int]) -> float:
    """AUC через подсчёт правильно упорядоченных пар."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")

    n_correct = 0
    n_ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                n_correct += 1
            elif p == n:
                n_ties += 1
    return (n_correct + 0.5 * n_ties) / (len(pos) * len(neg))


# ─────────────────────── калибровка ───────────────────────

def gather_chunk_stats(folder: Path, model, tokenizer, device) -> tuple[list[dict], list[str]]:
    """Прогоняет все .txt файлы в папке, возвращает плоский список
    статистик и параллельный список имён файлов (для leave-one-file-out).

    На этом этапе семантические фичи sem_tail_ratio/sem_self_repeat/
    cliche_proximity считаются предварительно (без индексов корпусов
    для sem_tail_ratio) — итоговое значение sem_tail_ratio
    пересчитывается в calibrate(), когда индексы построены."""
    files = sorted(folder.glob("*.txt"))
    if not files:
        print(f"⚠️  В папке {folder} нет .txt файлов")
        return [], []

    all_stats, all_files = [], []
    n_truncated = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)
        n_kept = 0
        for chunk in chunks:
            if len(chunk) < MIN_CHUNK_SIZE:
                continue
            # keep_tokens=True нужен, чтобы получить sent_embs/sent_offsets;
            # они занимают память, но это калибровочный путь и текста немного.
            stats = compute_chunk_stats(chunk, model, tokenizer, device,
                                          keep_tokens=True)
            if stats is None:
                continue
            if stats["truncated"]:
                n_truncated += 1
            # Сохраним сам чанк — пригодится при перерасчёте sem_tail_ratio
            stats["_chunk_text"] = chunk
            all_stats.append(stats)
            all_files.append(f.name)
            n_kept += 1
        print(f"  {f.name}: {n_kept}/{len(chunks)} чанков использовано")
    if n_truncated:
        print(f"  ⚠️  {n_truncated} чанков были обрезаны по max_length токенизатора")
    print(f"  Итого: {len(all_stats)} чанков из {len(files)} файлов\n")
    return all_stats, all_files


def _build_corpus_index(all_stats: list[dict]):
    """Из набора per-chunk статистик собирает плоскую матрицу
    эмбеддингов всех предложений + параллельный массив id чанков
    (нужен для leave-self-out при расчёте sem_tail_ratio внутри
    собственного класса).  Возвращает (np.ndarray, np.ndarray) или None."""
    if not _try_import_semantic():
        return None
    np = _SEM_CACHE["np"]
    chunks_embs = []
    chunks_owner = []
    for chunk_id, s in enumerate(all_stats):
        embs = s.get("sent_embs")
        if embs is None or len(embs) == 0:
            continue
        chunks_embs.append(embs)
        chunks_owner.append(np.full(len(embs), chunk_id, dtype=np.int32))
    if not chunks_embs:
        return None
    embs_matrix = np.concatenate(chunks_embs, axis=0)
    owner_array = np.concatenate(chunks_owner, axis=0)
    return embs_matrix, owner_array


def _sem_tail_ratio_with_leave_self_out(chunk_embs, chunk_id: int,
                                          own_index, opp_index) -> float:
    """sem_tail_ratio с исключением предложений того же чанка из
    собственного индекса.  own_index/opp_index — кортежи
    (embs_matrix, owner_array).  Используется только в калибровке."""
    if chunk_embs is None or own_index is None or opp_index is None:
        return 0.0
    np = _SEM_CACHE["np"]

    own_embs, own_owner = own_index
    opp_embs, _ = opp_index

    # косинусные близости (нормированные эмбеддинги)
    sim_own = chunk_embs @ own_embs.T  # (n_sent, n_own)
    sim_opp = chunk_embs @ opp_embs.T

    # Маскируем предложения того же чанка в собственном индексе
    same_chunk_mask = (own_owner == chunk_id)
    if same_chunk_mask.any():
        sim_own[:, same_chunk_mask] = -1.0

    k = min(SEM_KNN_K, sim_own.shape[1], sim_opp.shape[1])
    if k < 1:
        return 0.0

    top_own = np.partition(sim_own, -k, axis=1)[:, -k:].mean(axis=1)
    top_opp = np.partition(sim_opp, -k, axis=1)[:, -k:].mean(axis=1)
    return float((top_own - top_opp).mean())


def calibrate(human_dir: str, ai_dir: str, name: str = "default"):
    model, tokenizer, device = load_model()

    # Проверяем доступность семантического энкодера сразу — чтобы
    # пользователь увидел сообщение в начале, а не в середине прогона.
    sem_available = _try_import_semantic()
    if sem_available:
        # Прогреваем энкодер: лучше упасть/скачать сейчас, до тяжёлой
        # работы по NLL, чтобы не терять час впустую.
        sem_model = _get_sem_model()
        sem_available = sem_model is not None
    if not sem_available:
        print("⚠️  Семантический энкодер недоступен "
              f"({_SEM_CACHE.get('error', 'sentence-transformers не установлен')}).")
        print("    Калибровка пройдёт на 7 базовых фичах. Чтобы включить")
        print("    sem_tail_ratio / sem_self_repeat / cliche_proximity:")
        print("        pip install sentence-transformers numpy\n")

    print("─── Человеческий корпус ───")
    human_stats, human_files = gather_chunk_stats(Path(human_dir), model, tokenizer, device)
    print("─── ИИ корпус ───")
    ai_stats, ai_files = gather_chunk_stats(Path(ai_dir), model, tokenizer, device)

    if len(human_stats) < 10 or len(ai_stats) < 10:
        print("Недостаточно данных для калибровки (нужно минимум 10 чанков на класс).")
        sys.exit(1)

    # ─── Семантические индексы и пересчёт sem_tail_ratio ───
    human_index = None
    ai_index = None
    if sem_available:
        print("─── Построение семантических индексов ───")
        human_index = _build_corpus_index(human_stats)
        ai_index = _build_corpus_index(ai_stats)
        if human_index is None or ai_index is None:
            print("  ⚠️  Не удалось построить индексы (возможно, в чанках нет "
                  "пригодных предложений). Семантические фичи отключены.\n")
            sem_available = False
        else:
            n_h_sent = len(human_index[0])
            n_a_sent = len(ai_index[0])
            print(f"  Человеческий индекс: {n_h_sent} предложений, "
                  f"размерность {human_index[0].shape[1]}")
            print(f"  ИИ индекс: {n_a_sent} предложений\n")

            print("─── Пересчёт sem_tail_ratio с leave-self-out ───")
            for chunk_id, s in enumerate(human_stats):
                s["sem_tail_ratio"] = _sem_tail_ratio_with_leave_self_out(
                    s.get("sent_embs"), chunk_id, human_index, ai_index)
            for chunk_id, s in enumerate(ai_stats):
                # Для ИИ-чанка «свой» индекс — ИИ, «чужой» — человеческий.
                # Знак фичи остаётся «> 0 ⇔ ближе к человеку», поэтому
                # для ИИ-чанков мы считаем (top_human - top_ai_self_excluded).
                # Для этого вызываем функцию «наоборот»: own=ai_index,
                # opp=human_index, и инвертируем знак.
                s_val = _sem_tail_ratio_with_leave_self_out(
                    s.get("sent_embs"), chunk_id, ai_index, human_index)
                s["sem_tail_ratio"] = -s_val
            print("  Готово.\n")

    # ─── Подготовка фич ───
    # Базовые признаки (не требуют эмбеддера):
    base_features = [
        "log_std_nll", "log_mean_nll",
        "tail_ratio", "cv_nll",
        "repeat_3gram", "cv_sent_len", "punct_entropy",
        "lexical_marker_rate",
    ]
    # Семантические — только если энкодер доступен
    sem_features = (
        ["sem_tail_ratio", "sem_self_repeat", "cliche_proximity"]
        if sem_available else []
    )
    feature_names = base_features + sem_features

    def feats(s):
        return [s[name] for name in feature_names]

    X_human = [feats(s) for s in human_stats]
    X_ai = [feats(s) for s in ai_stats]
    X_all = X_human + X_ai
    y_all = [1] * len(X_human) + [0] * len(X_ai)

    # Стандартизация по объединённому набору
    cols = list(zip(*X_all))
    means, stds = standardize_params([list(c) for c in cols])
    X_std = [standardize(x, means, stds) for x in X_all]

    # ─── Логрегрессия ───
    # При большем числе фич чуть увеличиваем L2 для устойчивости —
    # часть фич может быть скоррелирована, и без регуляризации веса
    # начинают «бороться».
    print("─── Обучение логистической регрессии ───")
    weights, bias = fit_logistic(X_std, y_all, l2=0.05)
    weights_str = ", ".join(f"{n}={w:+.3f}" for n, w in zip(feature_names, weights))
    print(f"  Веса: {weights_str}")
    print(f"  Bias: {bias:+.3f}\n")

    # ─── Гауссианы по log-фичам для отображения LLR на чанке ───
    # Используем медиану/MAD по логарифмированным значениям — это
    # эквивалентно лог-нормальной модели исходной величины.
    def fit_gauss(values):
        c = _median(values)
        s = 1.4826 * _mad(values) + 1e-3
        return c, s

    log_std_nll_h = [s["log_std_nll"] for s in human_stats]
    log_std_nll_a = [s["log_std_nll"] for s in ai_stats]
    log_mean_nll_h = [s["log_mean_nll"] for s in human_stats]
    log_mean_nll_a = [s["log_mean_nll"] for s in ai_stats]

    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    # Гарантированно инициализируем дефолтный список, если внешний не задан,
    # чтобы записать что-то вменяемое в калибровку.
    _init_cliches_from_default()

    calib = {
        "name": name,
        "model": MODEL_NAME,
        "feature_names": feature_names,
        "semantic_enabled": sem_available,
        "semantic_model": SEM_MODEL_NAME if sem_available else None,
        "embeds_file": CALIBRATION_EMBEDS_FILE if sem_available else None,
        # Сохраняем сам список клише, чтобы analyze применял ту же
        # лексическую разметку, с которой калибровка обучалась.
        "lexical_markers": list(LEXICAL_MARKERS),
        "semantic_cliches": list(SEMANTIC_CLICHES),
        "standardize_means": means,
        "standardize_stds": stds,
        "logreg_weights": weights,
        "logreg_bias": bias,
        "human": {
            "n_chunks": len(human_stats),
            "n_files": len(set(human_files)),
            "log_std_nll": fit_gauss(log_std_nll_h),
            "log_mean_nll": fit_gauss(log_mean_nll_h),
            "mean_nll_percentiles": percentiles([s["mean_nll"] for s in human_stats], qs),
            "std_nll_percentiles": percentiles([s["std_nll"] for s in human_stats], qs),
            "raw_log_std_nll": log_std_nll_h,
            "raw_log_mean_nll": log_mean_nll_h,
            "raw_mean_nll": [s["mean_nll"] for s in human_stats],
            "raw_std_nll": [s["std_nll"] for s in human_stats],
        },
        "ai": {
            "n_chunks": len(ai_stats),
            "n_files": len(set(ai_files)),
            "log_std_nll": fit_gauss(log_std_nll_a),
            "log_mean_nll": fit_gauss(log_mean_nll_a),
            "mean_nll_percentiles": percentiles([s["mean_nll"] for s in ai_stats], qs),
            "std_nll_percentiles": percentiles([s["std_nll"] for s in ai_stats], qs),
            "raw_log_std_nll": log_std_nll_a,
            "raw_log_mean_nll": log_mean_nll_a,
            "raw_mean_nll": [s["mean_nll"] for s in ai_stats],
            "raw_std_nll": [s["std_nll"] for s in ai_stats],
        },
    }

    # ─── Перцентили в исходном (не лог) пространстве — для красивой сводки ───
    print("─── Сводка калибровки (исходные единицы) ───")
    print(f"{'метрика':<20} {'квантиль':<8} {'человек':<10} {'ИИ':<10}  Δ")
    print("─" * 70)
    for metric in ("std_nll_percentiles", "mean_nll_percentiles"):
        for q in ("p10", "p25", "p50", "p75", "p90"):
            h = calib["human"][metric][q]
            a = calib["ai"][metric][q]
            print(f"{metric.replace('_percentiles',''):<20} {q:<8} {h:<10.3f} {a:<10.3f}  {a-h:+.3f}")
        print()

    # ─── Метрики разделимости на тренировочной выборке (in-sample) ───
    train_scores = []
    for x in X_std:
        z = bias + sum(w * xi for w, xi in zip(weights, x))
        train_scores.append(z)
    auc_in = roc_auc(train_scores, y_all)
    print(f"In-sample AUC (логрегрессия на всём корпусе): {auc_in:.3f}")

    # ─── Leave-one-file-out: честная оценка обобщающей способности ───
    print("─── Leave-one-file-out CV ───")
    files_list = human_files + ai_files
    unique_files = sorted(set(files_list))

    if len(unique_files) >= 4:
        loo_scores, loo_labels = [], []
        for held_out in unique_files:
            train_X, train_y, test_X, test_y = [], [], [], []
            for x, y, f in zip(X_all, y_all, files_list):
                if f == held_out:
                    test_X.append(x)
                    test_y.append(y)
                else:
                    train_X.append(x)
                    train_y.append(y)

            if not train_X or not test_X:
                continue

            # Стандартизация по train fold
            cols_tr = list(zip(*train_X))
            ms, ss = standardize_params([list(c) for c in cols_tr])
            Xtr = [standardize(x, ms, ss) for x in train_X]
            Xte = [standardize(x, ms, ss) for x in test_X]

            w_fold, b_fold = fit_logistic(Xtr, train_y, n_iter=500, l2=0.05)
            for x, y in zip(Xte, test_y):
                z = b_fold + sum(w * xi for w, xi in zip(w_fold, x))
                loo_scores.append(z)
                loo_labels.append(y)

        auc_loo = roc_auc(loo_scores, loo_labels)
        print(f"  AUC (leave-one-file-out): {auc_loo:.3f}")
        if auc_in - auc_loo > 0.1:
            print("  ⚠️  Большая разница между in-sample и LOO — возможен оверфит "
                  "под конкретные файлы или неоднородный корпус")
        calib["auc_in_sample"] = auc_in
        calib["auc_loo"] = auc_loo
    else:
        print(f"  Пропускаю LOO: всего {len(unique_files)} уникальных файлов "
              f"(нужно минимум 4)\n")

    Path(CALIBRATION_FILE).write_text(
        json.dumps(calib, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nКалибровка '{name}' сохранена в {CALIBRATION_FILE}")

    # ─── Сохраняем индексы эмбеддингов отдельно (бинарно) ───
    if sem_available and human_index is not None and ai_index is not None:
        np = _SEM_CACHE["np"]
        np.savez_compressed(
            CALIBRATION_EMBEDS_FILE,
            human_embs=human_index[0],
            human_owner=human_index[1],
            ai_embs=ai_index[0],
            ai_owner=ai_index[1],
        )
        size_mb = Path(CALIBRATION_EMBEDS_FILE).stat().st_size / (1024 * 1024)
        print(f"Эмбеддинги корпусов сохранены в {CALIBRATION_EMBEDS_FILE} "
              f"({size_mb:.1f} MB)")


# ─────────────────────── анализ ───────────────────────

# Пороги для диагностики чанка. Подобраны под rugpt3small;
# при смене модели имеет смысл откалибровать заново.
DIAG_LOW_NLL = 1.5      # «модель почти уверена в этом токене»
DIAG_HIGH_NLL = 8.0     # «модель сильно удивлена»
DIAG_SMOOTH_RUN = 6     # длина гладкой серии, начиная с которой подсвечиваем
DIAG_SENT_STD_LOW = 5.0 # стандартное отклонение длин предложений (в словах) ниже которого ритм считается монотонным
DIAG_TTR_LOW = 0.55     # лексическое разнообразие ниже которого считаем бедным


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?…])\s+', text.strip()) if s.strip()]


def _word_tokens(text: str) -> list[str]:
    """Слова без пунктуации, в нижнем регистре, для TTR."""
    return re.findall(r'[а-яёa-z0-9]+', text.lower())


def find_smooth_runs(token_nll: list[float], token_strs: list[str],
                      threshold: float = DIAG_LOW_NLL,
                      min_len: int = DIAG_SMOOTH_RUN) -> list[dict]:
    """Находит серии подряд идущих токенов с nll < threshold.
    Это куски, которые модель предсказывала почти наверняка —
    штампы, устойчивые сцепки, заученные конструкции."""
    runs = []
    i = 0
    n = len(token_nll)
    while i < n:
        if token_nll[i] < threshold:
            j = i
            while j < n and token_nll[j] < threshold:
                j += 1
            if j - i >= min_len:
                phrase = "".join(token_strs[i:j]).strip()
                runs.append({
                    "start_token": i,
                    "length": j - i,
                    "mean_nll": sum(token_nll[i:j]) / (j - i),
                    "phrase": phrase,
                })
            i = j
        else:
            i += 1
    # Сортируем по длине серии, длинные сверху
    runs.sort(key=lambda r: -r["length"])
    return runs


def find_surprising_tokens(token_nll: list[float], token_strs: list[str],
                             threshold: float = DIAG_HIGH_NLL,
                             top_n: int = 5) -> list[dict]:
    """Самые неожиданные для модели токены — маркер живой авторской
    лексики, имён собственных, неологизмов."""
    indexed = [(i, nll, token_strs[i]) for i, nll in enumerate(token_nll)
               if nll > threshold]
    indexed.sort(key=lambda x: -x[1])
    return [{"position": i, "nll": nll, "token": tok.strip()}
            for i, nll, tok in indexed[:top_n] if tok.strip()]


def diagnose_chunk(stats: dict, chunk_text: str, calib: dict,
                    text_context: dict | None = None) -> list[dict]:
    """
    Возвращает список структурированных диагностик: что не так,
    почему это плохо, какой совет автору.

    Каждая диагностика — словарь:
      {
        "id": "smooth_rhythm",       # для группировки
        "category": "стилистика",    # стилистика | лексика | структура
        "severity": "сильно" | "умеренно" | "слабо",
        "what": "Слишком ровный ритм…",       # описание проблемы
        "why": "это маркер причёсанного…",    # почему это сигнал
        "advice": "Вставьте одно длинное…",   # что делать
        "highlights": [(start, end), …],      # позиции для подсветки в чанке
      }

    text_context — словарь со статистиками всех чанков текста, для
    относительных сравнений. Может быть None при разовом анализе.
    """
    diags = []

    h_lstd_med = calib["human"]["log_std_nll"][0]
    a_lstd_med = calib["ai"]["log_std_nll"][0]
    h_lnll_med = calib["human"]["log_mean_nll"][0]
    a_lnll_med = calib["ai"]["log_mean_nll"][0]

    # ─── 1. Ровный ритм предсказуемости (стилистика) ───
    if stats["log_std_nll"] < (h_lstd_med + a_lstd_med) / 2:
        h_std = math.exp(h_lstd_med)
        diff = h_std - stats["std_nll"]
        if diff > 0.15:
            severity = "сильно" if diff > 0.4 else ("умеренно" if diff > 0.25 else "слабо")
            # Сравнение с остальным текстом
            ctx_note = ""
            if text_context and text_context.get("std_nll_text_median") is not None:
                text_med = text_context["std_nll_text_median"]
                if stats["std_nll"] < text_med - 0.2:
                    ctx_note = (f" Это и относительно остального вашего текста "
                                f"низковато — у вас обычно std_nll ≈ {text_med:.2f}.")
                elif abs(stats["std_nll"] - text_med) < 0.1:
                    ctx_note = (" Впрочем, у вас весь текст идёт примерно с такими "
                                "же значениями — возможно, это особенность вашего стиля, "
                                "а не локальный сбой.")
            diags.append({
                "id": "smooth_rhythm",
                "category": "стилистика",
                "severity": severity,
                "what": (f"Ровный ритм предсказуемости (std_nll={stats['std_nll']:.2f} "
                          f"при типичной для человека {h_std:.2f})."),
                "why": ("Предложения и обороты идут однородно по сложности — "
                         "признак сглаженного, «причёсанного» текста." + ctx_note),
                "advice": ("Сломайте однородность: вставьте одно неожиданно длинное "
                            "описание (15+ слов с придаточными), а рядом — резко короткое "
                            "(1-3 слова, реплика или междометие). Контраст важнее средней."),
                "highlights": [],
            })

    # ─── 2. Высокая предсказуемость лексики (лексика) ───
    if stats["log_mean_nll"] < h_lnll_med - 0.1:
        h_nll = math.exp(h_lnll_med)
        diff = h_nll - stats["mean_nll"]
        if diff > 0.2:
            severity = "сильно" if diff > 0.7 else ("умеренно" if diff > 0.4 else "слабо")
            ctx_note = ""
            if text_context and text_context.get("mean_nll_text_median") is not None:
                text_med = text_context["mean_nll_text_median"]
                if stats["mean_nll"] < text_med - 0.3:
                    ctx_note = (f" По вашему тексту в среднем mean_nll ≈ {text_med:.2f} — "
                                f"этот фрагмент заметно проще обычного.")
            diags.append({
                "id": "low_perplexity",
                "category": "лексика",
                "severity": severity,
                "what": (f"Высокая общая предсказуемость лексики "
                          f"(mean_nll={stats['mean_nll']:.2f} при норме {h_nll:.2f})."),
                "why": ("Слова и обороты в основном частотные, мало неожиданных "
                         "формулировок. Текст читается «как ожидаешь»." + ctx_note),
                "advice": ("Добавьте конкретики: имена с отчествами, бытовые детали, "
                            "локальные топонимы, неожиданные сравнения. Замените "
                            "общие слова («дорогое», «красивое», «странное») на "
                            "точные."),
                "highlights": [],
            })

    # ─── 3. Гладкие серии — конкретные штампы (лексика) ───
    if "token_nll" in stats and "token_strs" in stats:
        runs = find_smooth_runs(stats["token_nll"], stats["token_strs"])
        if runs:
            top_runs = runs[:3]
            phrases_text = "; ".join(
                f'«{r["phrase"]}» ({r["length"]} токенов)'
                for r in top_runs
            )
            n = len(runs)
            if n == 1:
                series_word = "одна «гладкая» серия — отрезок"
            elif 2 <= n <= 4:
                series_word = f"{n} «гладких» серии — отрезки"
            else:
                series_word = f"{n} «гладких» серий — отрезков"
            severity = "сильно" if any(r["length"] >= 10 for r in runs) else "умеренно"

            # Собираем диапазоны для подсветки
            highlights = []
            offsets = stats.get("token_offsets") or []
            for r in runs:
                start_tok = r["start_token"]
                end_tok = start_tok + r["length"] - 1
                if start_tok < len(offsets) and end_tok < len(offsets):
                    s_off = offsets[start_tok]
                    e_off = offsets[end_tok]
                    if s_off and e_off and s_off[0] < e_off[1]:
                        highlights.append((s_off[0], e_off[1]))

            diags.append({
                "id": "smooth_runs",
                "category": "лексика",
                "severity": severity,
                "what": (f"Найдена(ы) {series_word}, где модель уверенно предсказывала "
                          f"каждое следующее слово. Самые длинные: {phrases_text}."),
                "why": ("Это маркер устойчивых сцепок и штампов: «бархатным голосом», "
                         "«пожал плечами Х», «слегка усмехнулся» и подобных. "
                         "Эти фразы модель видела в обучающей выборке десятки тысяч раз."),
                "advice": ("Замените подсвеченные сцепки на конкретные физические "
                            "детали: вместо «пожал плечами Иван» — «Иван поставил "
                            "бокал, стекло звякнуло» или просто реплика без атрибуции. "
                            "Достаточно изменить 2-3 штампа в этом фрагменте."),
                "highlights": highlights,
            })

        # ─── 4. Мало неожиданных токенов (лексика) ───
        n_high = sum(1 for x in stats["token_nll"] if x > DIAG_HIGH_NLL)
        n_total = len(stats["token_nll"])
        rate = n_high / n_total if n_total else 0
        if rate < 0.04:
            severity = "сильно" if rate < 0.02 else "умеренно"
            diags.append({
                "id": "few_surprises",
                "category": "лексика",
                "severity": severity,
                "what": (f"Мало неожиданных слов: {n_high} из {n_total} токенов "
                          f"({rate:.1%})."),
                "why": ("В живом тексте модель чаще «удивляется» — натыкается на "
                         "имена собственные, редкие слова, авторские обороты. "
                         "Здесь почти ничего такого не встретилось."),
                "advice": ("Введите конкретные имена (с фамилиями), географические "
                            "названия, профессиональные термины, диалектизмы или "
                            "выраженные авторские слова. Даже одно-два таких слова "
                            "на чанк сильно меняют профиль."),
                "highlights": [],
            })

    # ─── 5. Монотонный ритм длин предложений (стилистика) ───
    sentences = _split_sentences(chunk_text)
    if len(sentences) >= 5:
        sent_lens = [len(s.split()) for s in sentences]
        mean_len = sum(sent_lens) / len(sent_lens)
        var = sum((x - mean_len) ** 2 for x in sent_lens) / len(sent_lens)
        sent_std = math.sqrt(var)
        if sent_std < DIAG_SENT_STD_LOW:
            severity = "сильно" if sent_std < 3.0 else "умеренно"
            diags.append({
                "id": "monotone_rhythm",
                "category": "стилистика",
                "severity": severity,
                "what": (f"Монотонный ритм длин предложений (отклонение {sent_std:.1f} "
                          f"слов при средней длине {mean_len:.1f})."),
                "why": ("В живом тексте предложения сильнее различаются: короткие "
                         "реплики или ударные фразы перемежаются с длинными описаниями."),
                "advice": ("Найдите 1-2 места, где можно объединить два соседних "
                            "предложения через причастный/деепричастный оборот, и одно "
                            "место, где можно «отрубить» короткую односложную фразу. "
                            "Это сразу ломает ровную линию."),
                "highlights": [],
            })

    # ─── 6. Бедный лексикон (структура) ───
    words = _word_tokens(chunk_text)
    if len(words) >= 50:
        ttr = len(set(words)) / len(words)
        if ttr < DIAG_TTR_LOW:
            severity = "сильно" if ttr < 0.45 else "умеренно"
            # Различаем «системные окна» и шаблонные описания
            has_numbers = sum(1 for w in words if w.isdigit()) > 5
            advice_text = (
                "Это часто служебный фрагмент (системные окна, перечисление, "
                "технические данные). Если так — пометьте этот кусок как «не "
                "художественный текст», его правка может быть избыточной. "
                "Если же это художественная сцена — введите больше разных слов "
                "вместо повторов местоимений и общих глаголов."
                if has_numbers else
                "Замените повторяющиеся местоимения и служебные глаголы на "
                "конкретные действия. Вместо «он сказал, он подумал, он сделал» — "
                "разные формы: «бросил», «промычал», «двинулся к двери»."
            )
            diags.append({
                "id": "low_diversity",
                "category": "структура",
                "severity": severity,
                "what": (f"Невысокое лексическое разнообразие (TTR={ttr:.2f}). "
                          f"Многие слова повторяются."),
                "why": ("Часто признак шаблонных описаний — боевых, технических, "
                         "или «системных окон» в ЛитРПГ."),
                "advice": advice_text,
                "highlights": [],
            })

    # ─── 7. Семантические клише (если эмбеддинги доступны) ───
    sent_embs = stats.get("sent_embs")
    sent_offsets = stats.get("sent_offsets") or []
    if sent_embs is not None and sent_offsets:
        matches = find_cliche_matches(sent_offsets, sent_embs)
        if matches:
            top = matches[:3]
            # severity по самому близкому совпадению
            best_sim = top[0]["similarity"]
            severity = ("сильно" if best_sim >= 0.78 else
                        "умеренно" if best_sim >= 0.70 else "слабо")
            n = len(matches)
            if n == 1:
                what_lead = "Найдена 1 фраза, рифмующаяся с жанровым штампом"
            elif 2 <= n <= 4:
                what_lead = f"Найдено {n} фразы, рифмующихся с жанровыми штампами"
            else:
                what_lead = f"Найдено {n} фраз, рифмующихся с жанровыми штампами"

            details_lines = []
            for m in top:
                # Усекаем для читаемости
                sent_short = m["sentence"]
                if len(sent_short) > 80:
                    sent_short = sent_short[:77] + "…"
                details_lines.append(
                    f'  «{sent_short}» ↔ «{m["cliche"]}» (sim={m["similarity"]:.2f})'
                )

            # Ограничиваем подсветку топ-N — иначе чанк превращается в кашу
            highlights = [(m["start"], m["end"]) for m in matches[:MAX_CLICHE_HIGHLIGHTS]]

            diags.append({
                "id": "cliche_match",
                "category": "семантика",
                "severity": severity,
                "what": (f"{what_lead}.\n" + "\n".join(details_lines)),
                "why": ("Фразы рифмуются с распространёнными образами жанровой прозы. "
                         "Это не обязательно означает ИИ-генерацию — такие образы "
                         "встречаются у многих авторов. Редактору стоит решить: "
                         "работает ли этот образ в данном контексте или воспроизводит "
                         "шаблон без необходимости."),
                "advice": ("Перепишите эти места через конкретное действие или "
                            "деталь, а не через прямое называние эмоции. Вместо "
                            "«его сердце сжалось» — что именно он сделал в этот "
                            "момент: уронил ложку, посмотрел в окно, сказал что-то "
                            "невпопад. Эмоция должна вычитываться из поведения."),
                "highlights": highlights,
            })

    # ─── 8. Семантическая повторяемость (если эмбеддинги доступны) ───
    sem_self_repeat = stats.get("sem_self_repeat", 0.0)
    if sent_embs is not None and sem_self_repeat > 0.75:
        severity = ("сильно" if sem_self_repeat > 0.85 else
                    "умеренно" if sem_self_repeat > 0.80 else "слабо")
        diags.append({
            "id": "sem_self_repeat",
            "category": "семантика",
            "severity": severity,
            "what": (f"Чанк семантически повторяется (self-similarity "
                      f"={sem_self_repeat:.2f})."),
            "why": ("Несколько предложений в этом фрагменте выражают одну и ту же "
                     "мысль разными словами. Это ИИ-привычка: переформулировать "
                     "только что сказанное, чтобы заполнить объём."),
            "advice": ("Найдите 2-3 предложения, которые повторяют один тезис, "
                        "и оставьте одно — самое сильное. Или замените повторы на "
                        "продвижение действия / новую деталь."),
            "highlights": [],
        })

    # ─── 9. Лексические маркеры (одиночные слова из списка клише) ───
    lex_rate = stats.get("lexical_marker_rate", 0.0)
    lex_occurrences = stats.get("lex_occurrences") or []
    if lex_rate >= LEXICAL_MARKER_RATE_HIT and lex_occurrences:
        severity = ("сильно" if lex_rate >= 0.020 else
                    "умеренно" if lex_rate >= 0.013 else "слабо")
        # Группируем по слову, чтобы показать частоту каждого
        word_counts = {}
        for s_off, e_off, word in lex_occurrences:
            key = word.lower()
            word_counts[key] = word_counts.get(key, 0) + 1
        # Топ-5 самых частых
        top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:5]
        words_summary = ", ".join(
            f"«{w}»×{c}" if c > 1 else f"«{w}»"
            for w, c in top_words
        )
        diags.append({
            "id": "lexical_markers",
            "category": "лексика",
            "severity": severity,
            "what": (f"Высокая плотность слов-маркеров "
                      f"({len(lex_occurrences)} употреблений, {lex_rate:.1%} "
                      f"от слов): {words_summary}."),
            "why": ("Эти слова — типичные «костыли» нейросетевой прозы: "
                     "«вдруг», «казалось», «безусловно», «конечно». В живом "
                     "тексте они встречаются, но не на каждом шагу. "
                     "Высокая плотность сигнализирует о служебном, "
                     "«объясняющем» письме."),
            "advice": ("Уберите 2/3 этих слов — большинство из них «воздух», "
                        "не несущий смысла. «Вдруг он понял» = «он понял». "
                        "«Казалось, она устала» = «она устала». Если действие "
                        "действительно внезапное — пусть это будет видно из "
                        "ритма, а не из слова «вдруг»."),
            "highlights": [(s, e) for s, e, _ in lex_occurrences],
        })

    # ─── 10. Per-sentence sem_tail (item 6) ───
    # Подсвечивает конкретные предложения внутри чанка, которые семантически
    # ближе к ИИ-корпусу, чем к человеческому — даже если нет конкретного клише.
    sem_tail_per_sent = stats.get("sem_tail_per_sent") or []
    if (sem_tail_per_sent and sent_offsets
            and len(sem_tail_per_sent) == len(sent_offsets)
            and stats.get("sem_tail_ratio", 0.0) < -0.02):
        indexed_scores = sorted(range(len(sem_tail_per_sent)),
                                key=lambda i: sem_tail_per_sent[i])
        worst_idxs = [i for i in indexed_scores[:3] if sem_tail_per_sent[i] < 0.0]
        if worst_idxs:
            worst_sents = [sent_offsets[i] for i in worst_idxs]
            details = "; ".join(
                f'«{s[:55]}{"…" if len(s) > 55 else ""}»'
                for s, _, _ in worst_sents
            )
            sem_tr = stats.get("sem_tail_ratio", 0.0)
            severity = "умеренно" if sem_tr < -0.04 else "слабо"
            diags.append({
                "id": "sem_tail_sentences",
                "category": "семантика",
                "severity": severity,
                "what": (f"Предложения, семантически наиболее близкие к ИИ-корпусу "
                          f"({len(worst_idxs)} из {len(sent_offsets)}): {details}."),
                "why": ("В семантическом пространстве эти предложения ближе к "
                         "типичным ИИ-текстам, чем к человеческим — даже не "
                         "попадая под конкретные клише."),
                "advice": ("Перепишите подсвеченные предложения: добавьте "
                            "конкретную деталь, нестандартный угол зрения или "
                            "авторскую лексику, чтобы сдвинуть их из «среднего» "
                            "семантического пространства."),
                "highlights": [(s_off, e_off) for _, s_off, e_off in worst_sents],
            })

    # ─── 11. Экспрессивные глаголы речи (item 3) ───
    ev_matches = list(EXPRESSIVE_SPEECH_VERBS.finditer(chunk_text))
    if len(ev_matches) >= 3:
        n_ev = len(ev_matches)
        n_sent_local = max(len(sentences), 1)
        density = n_ev / n_sent_local
        severity = "сильно" if n_ev >= 5 or density >= 0.4 else "умеренно"
        verbs_seen = sorted({m.group().lower() for m in ev_matches})
        verbs_str = ", ".join(f"«{v}»" for v in verbs_seen[:6])
        diags.append({
            "id": "expressive_verbs",
            "category": "стилистика",
            "severity": severity,
            "what": (f"Высокая плотность экспрессивных глаголов речи "
                      f"({n_ev} вхождений): {verbs_str}."),
            "why": ("Глаголы «прорычал», «выдохнул», «процедил» и подобные "
                     "создают яркость в единичных случаях, но в большой "
                     "концентрации дают мелодраматический или пародийный эффект. "
                     "Это распространённая черта нейросетевой прозы."),
            "advice": ("Замените часть на нейтральные «сказал», «спросил», «ответил» "
                        "или уберите атрибуцию диалога вовсе — читатель сам поймёт, "
                        "кто говорит. Экспрессивный глагол работает только как "
                        "исключение из правила."),
            "highlights": [(m.start(), m.end()) for m in ev_matches],
        })

    # Сортируем по severity
    severity_order = {"сильно": 0, "умеренно": 1, "слабо": 2}
    diags.sort(key=lambda d: severity_order.get(d["severity"], 3))
    return diags


def compute_text_context(all_stats: list[dict]) -> dict:
    """Считает медианы метрик по всему тексту — для относительных
    сравнений в diagnose_chunk."""
    if not all_stats:
        return {}
    return {
        "std_nll_text_median": _median([s["std_nll"] for s in all_stats]),
        "mean_nll_text_median": _median([s["mean_nll"] for s in all_stats]),
        "z_text_median": _median([s["z"] for s in all_stats]),
        "z_text_mad": _mad([s["z"] for s in all_stats]),
    }


def highlight_text(chunk_text: str, ranges: list[tuple[int, int]],
                    use_color: bool = True) -> str:
    """Возвращает текст чанка с подсветкой указанных диапазонов символов.
    В терминале — ANSI inverse, в plain — квадратные скобки.

    Перекрывающиеся/соседние диапазоны объединяются."""
    if not ranges:
        return chunk_text

    # Слияние перекрывающихся
    sorted_ranges = sorted([(s, e) for s, e in ranges if 0 <= s < e <= len(chunk_text)])
    if not sorted_ranges:
        return chunk_text

    merged = [sorted_ranges[0]]
    for s, e in sorted_ranges[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    if use_color:
        # ANSI inverse + bold yellow
        OPEN = "\033[1;33;7m"
        CLOSE = "\033[0m"
    else:
        OPEN, CLOSE = "[[", "]]"

    out = []
    cursor = 0
    for s, e in merged:
        out.append(chunk_text[cursor:s])
        out.append(OPEN)
        out.append(chunk_text[s:e])
        out.append(CLOSE)
        cursor = e
    out.append(chunk_text[cursor:])
    return "".join(out)


def summarize_diagnostics(per_chunk_diags: list[list[dict]]) -> list[str]:
    """По всем диагностикам всех подозрительных чанков делает сводку
    по типам проблем — что встречается чаще всего."""
    counter = {}
    titles = {}
    for diags in per_chunk_diags:
        seen_in_chunk = set()
        for d in diags:
            if d["id"] in seen_in_chunk:
                continue  # один тип проблемы из чанка считаем один раз
            seen_in_chunk.add(d["id"])
            counter[d["id"]] = counter.get(d["id"], 0) + 1
            # Короткое название для сводки
            short = {
                "smooth_rhythm": "Ровный ритм предсказуемости",
                "low_perplexity": "Высокая предсказуемость лексики",
                "smooth_runs": "Гладкие серии (штампы и сцепки)",
                "few_surprises": "Мало неожиданных слов",
                "monotone_rhythm": "Монотонные длины предложений",
                "low_diversity": "Бедный лексикон / повторы",
                "cliche_match": "Жанровые штампы (семантические)",
                "sem_self_repeat": "Повтор одной мысли разными словами",
                "lexical_markers": "Слова-маркеры («вдруг», «казалось»…)",
                "sem_tail_sentences": "Предложения близко к ИИ-пространству",
                "expressive_verbs": "Экспрессивные глаголы речи",
            }
            titles[d["id"]] = short.get(d["id"], d["id"])

    if not counter:
        return []

    total = len(per_chunk_diags)
    items = sorted(counter.items(), key=lambda x: -x[1])
    lines = []
    for diag_id, count in items:
        pct = count / total
        lines.append(f"{titles[diag_id]}: {count} из {total} чанков ({pct:.0%})")
    return lines


def humanness_score_logreg(stats: dict, calib: dict) -> tuple[float, float, dict]:
    """
    Считает z = bias + w·x в стандартизованном пространстве.
    Положительный → человек.

    Список фич определяется полем `feature_names` в калибровке —
    это даёт обратную совместимость со старыми (двухфичными) калибровками.
    """
    means = calib["standardize_means"]
    stds = calib["standardize_stds"]
    weights = calib["logreg_weights"]
    bias = calib["logreg_bias"]
    feature_names = calib.get("feature_names", ["log_std_nll", "log_mean_nll"])

    raw_x = [stats[name] for name in feature_names]
    x_std = standardize(raw_x, means, stds)
    z = bias + sum(w * xi for w, xi in zip(weights, x_std))

    # Информационные LLR по двум основным осям (для печати таблицы) —
    # они есть в любой калибровке, и старой и новой.
    h_lstd_c, h_lstd_s = calib["human"]["log_std_nll"]
    a_lstd_c, a_lstd_s = calib["ai"]["log_std_nll"]
    h_lnll_c, h_lnll_s = calib["human"]["log_mean_nll"]
    a_lnll_c, a_lnll_s = calib["ai"]["log_mean_nll"]

    llr_std = (_gaussian_loglik(stats["log_std_nll"], h_lstd_c, h_lstd_s)
               - _gaussian_loglik(stats["log_std_nll"], a_lstd_c, a_lstd_s))
    llr_nll = (_gaussian_loglik(stats["log_mean_nll"], h_lnll_c, h_lnll_s)
               - _gaussian_loglik(stats["log_mean_nll"], a_lnll_c, a_lnll_s))

    # Покомпонентный вклад фич в z (для возможной диагностики)
    contributions = {
        name: w * xi
        for name, w, xi in zip(feature_names, weights, x_std)
    }

    return z, z, {
        "llr_std": llr_std,
        "llr_nll": llr_nll,
        "z": z,
        "feature_contributions": contributions,
    }


def label_for_score(score: float) -> str:
    if score < -0.5:
        return "🚨🚨 ИИ-профиль"
    elif score < -0.2:
        return "🚨 склоняется к ИИ"
    elif score < 0.2:
        return "≈ нейтрально"
    elif score < 0.5:
        return "✓ склоняется к человеку"
    else:
        return "✓✓ человеческий профиль"


def check_ood(stats_list: list[dict], calib: dict) -> list[str]:
    """Проверка out-of-distribution: попадает ли анализируемый текст в
    диапазон калибровочного человеческого корпуса по mean_nll."""
    warnings = []
    p95_human = calib["human"]["mean_nll_percentiles"]["p95"]
    p5_human = calib["human"]["mean_nll_percentiles"]["p5"]

    high = sum(1 for s in stats_list if s["mean_nll"] > p95_human)
    low = sum(1 for s in stats_list if s["mean_nll"] < p5_human)
    n = len(stats_list)

    if high / n > 0.3:
        warnings.append(
            f"⚠️  {high}/{n} чанков ({high/n:.0%}) имеют mean_nll выше p95 "
            f"человеческого корпуса ({p95_human:.2f}). Текст может быть "
            f"вне домена калибровки — возможны ложные срабатывания."
        )
    if low / n > 0.3:
        warnings.append(
            f"⚠️  {low}/{n} чанков ({low/n:.0%}) имеют mean_nll ниже p5 "
            f"человеческого корпуса ({p5_human:.2f}). Текст может быть "
            f"вне домена калибровки или содержать заученные модели фрагменты."
        )
    return warnings


def print_histogram(values: list[float], bins: int = 20, width: int = 40,
                     label: str = "") -> None:
    """ASCII-гистограмма для verbose-режима."""
    if not values:
        return
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / (hi - lo) * bins), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    print(f"  {label} (n={len(values)}, range={lo:.2f}…{hi:.2f}):")
    for i, c in enumerate(counts):
        bar = "█" * int(c / max_count * width)
        edge = lo + (i + 0.5) * (hi - lo) / bins
        print(f"    {edge:>+6.2f} │{bar} {c}")


def _load_calibration_embeds(calib: dict) -> dict | None:
    """Загружает .npz с эмбеддингами корпусов, если калибровка их
    использовала.  Возвращает {"human": ndarray, "ai": ndarray} или None."""
    if not calib.get("semantic_enabled"):
        return None
    if not _try_import_semantic():
        print("⚠️  Калибровка использует семантические фичи, но "
              "sentence-transformers/numpy не установлены. "
              "Семантические фичи будут заполнены нулями — "
              "результат может быть менее точным.")
        return None
    embeds_path = calib.get("embeds_file") or CALIBRATION_EMBEDS_FILE
    if not Path(embeds_path).exists():
        print(f"⚠️  Файл эмбеддингов {embeds_path} не найден. "
              "Семантические фичи будут заполнены нулями.")
        return None
    np = _SEM_CACHE["np"]
    data = np.load(embeds_path)
    return {
        "human": data["human_embs"],
        "ai": data["ai_embs"],
    }


def generate_html_report(text_path: str, all_stats: list[dict], calib: dict,
                          flagged: list[dict], per_chunk_diagnostics: list[list[dict]],
                          summary: list[str], output_path: str) -> None:
    """Генерирует самодостаточный HTML-отчёт об анализе текста."""
    import html as _html
    from datetime import datetime

    filename = Path(text_path).name
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    mean_z = sum(s["z"] for s in all_stats) / len(all_stats)
    aggregate_score = math.tanh(mean_z / 2.0)

    def score_color(score: float) -> str:
        if score < -0.5: return "#e94560"
        if score < -0.2: return "#f5a623"
        if score < 0.2:  return "#888888"
        return "#3d9970"

    def score_css_class(score: float) -> str:
        if score < -0.5: return "s-ai2"
        if score < -0.2: return "s-ai1"
        if score < 0.2:  return "s-neu"
        return "s-hum"

    def highlight_html(text: str, ranges: list[tuple[int, int]]) -> str:
        if not ranges:
            return _html.escape(text)
        sorted_ranges = sorted((s, e) for s, e in ranges if 0 <= s < e <= len(text))
        merged: list[tuple[int, int]] = []
        for s, e in sorted_ranges:
            if merged and s <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        parts: list[str] = []
        cursor = 0
        for s, e in merged:
            parts.append(_html.escape(text[cursor:s]))
            parts.append(f'<mark>{_html.escape(text[s:e])}</mark>')
            cursor = e
        parts.append(_html.escape(text[cursor:]))
        return "".join(parts)

    # ── Веса фич ──
    feature_names = calib.get("feature_names", [])
    weights = calib.get("logreg_weights", [])
    weight_rows = ""
    if feature_names and weights:
        for w, name in sorted(zip(weights, feature_names), key=lambda x: -abs(x[0])):
            bar_w = min(int(abs(w) * 60), 120)
            bar_color = "#3d9970" if w > 0 else "#e94560"
            direction = "человек" if w > 0 else "ИИ"
            weight_rows += (
                f"<tr><td class='fn'>{_html.escape(name)}</td>"
                f"<td class='fv'>{w:+.3f}</td>"
                f"<td class='fd'>{direction}</td>"
                f"<td><div style='width:{bar_w}px;height:10px;"
                f"background:{bar_color};border-radius:2px'></div></td></tr>\n"
            )

    # ── Таблица чанков ──
    chunk_rows = ""
    for s in all_stats:
        cls = score_css_class(s["score"])
        chunk_rows += (
            f"<tr class='{cls}'>"
            f"<td>{s['index']}</td><td>{len(s['chunk'])}</td>"
            f"<td>{s['mean_nll']:.2f}</td><td>{s['std_nll']:.2f}</td>"
            f"<td>{s['z']:+.2f}</td><td>{s['score']:+.2f}</td>"
            f"<td>{_html.escape(label_for_score(s['score']))}</td></tr>\n"
        )

    # ── Подозрительные чанки ──
    chunks_html = ""
    sev_colors = {"сильно": "#e94560", "умеренно": "#f5a623", "слабо": "#7a7a9a"}
    for s, diags in zip(flagged, per_chunk_diagnostics):
        confidence = (1 - math.exp(-abs(s["z"]))) * 100
        cls = score_css_class(s["score"])

        all_hl: list[tuple[int, int]] = []
        for d in diags:
            all_hl.extend(d.get("highlights", []))

        preview = s["chunk"][:1200]
        preview_hl = [(a, min(b, 1200)) for a, b in all_hl if a < 1200]
        text_body = highlight_html(preview, preview_hl)
        if len(s["chunk"]) > 1200:
            text_body += "<span class='ell'>…</span>"

        diag_html = ""
        for d in diags:
            sc = sev_colors.get(d["severity"], "#888")
            what_esc = _html.escape(d["what"]).replace("\n", "<br>")
            diag_html += (
                f"<div class='diag' style='border-left:3px solid {sc}'>"
                f"<div class='dw'>[{_html.escape(d['severity'])}] {what_esc}</div>"
                f"<div class='di'>Почему: {_html.escape(d['why'])}</div>"
                f"<div class='da'>Совет: {_html.escape(d['advice'])}</div>"
                f"</div>\n"
            )

        chunks_html += (
            f"<div class='cd {cls}'>"
            f"<div class='ch'>Чанк #{s['index']} &mdash; score {s['score']:+.2f}"
            f" | уверенность: {confidence:.0f}%</div>"
            f"{diag_html or '<p>Конкретных локальных маркеров не выявлено.</p>'}"
            f"<pre class='ct'>{text_body}</pre>"
            f"</div>\n"
        )

    # ── Сводка ──
    summary_html = (
        "<ul>" + "".join(f"<li>{_html.escape(l)}</li>" for l in summary) + "</ul>"
        if summary else "<p>Нет данных.</p>"
    )

    cal_name = _html.escape(calib.get("name", "default"))
    agg_color = score_color(aggregate_score)
    auc_str = f"{calib['auc_loo']:.3f}" if "auc_loo" in calib else "N/A"
    sem_status = "ВКЛ" if calib.get("semantic_enabled") else "ВЫКЛ"

    css = """
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', sans-serif; max-width: 1100px; margin: 0 auto;
       padding: 24px; background: #12121f; color: #d0d0e0; line-height: 1.5; }
h1 { color: #e94560; margin-bottom: 4px; }
h2 { color: #a0c4ff; border-bottom: 1px solid #2a2a4a; padding-bottom: 6px; margin-top: 32px; }
.meta { color: #888; font-size: .85em; margin-bottom: 24px; }
.sb { background: #1e1e3a; border-radius: 8px; padding: 16px 24px; margin-bottom: 24px;
      display: flex; gap: 32px; flex-wrap: wrap; }
.si { text-align: center; }
.sv { font-size: 2em; font-weight: bold; }
.sl { font-size: .8em; color: #888; }
table { width: 100%; border-collapse: collapse; font-size: .9em; }
th { background: #1e1e3a; color: #a0c4ff; padding: 8px; text-align: left; }
td { padding: 6px 8px; border-bottom: 1px solid #1a1a30; }
.s-ai2 { background: rgba(233,69,96,.25); }
.s-ai1 { background: rgba(245,166,35,.15); }
.s-neu { background: rgba(100,100,120,.1); }
.s-hum { background: rgba(61,153,112,.1); }
.fn { font-family: monospace; }
.fv { font-family: monospace; text-align: right; padding-right: 12px; }
.fd { color: #888; font-size: .85em; }
.cd { border-radius: 8px; padding: 16px; margin-bottom: 20px; border: 1px solid #2a2a4a; }
.ch { font-weight: bold; font-size: 1.05em; margin-bottom: 12px; }
.diag { margin: 8px 0; padding: 8px 12px; background: rgba(0,0,0,.2); border-radius: 4px; }
.dw { font-weight: 600; margin-bottom: 4px; }
.di, .da { font-size: .88em; color: #aaa; margin: 3px 0; }
.ct { background: #0d0d1a; border-radius: 6px; padding: 14px; margin-top: 12px;
      white-space: pre-wrap; font-size: .88em; line-height: 1.7; overflow-x: auto; }
mark { background: rgba(233,69,96,.35); border-radius: 2px; padding: 0 1px;
       color: inherit; text-decoration: underline rgba(233,69,96,.7); }
.ell { color: #555; }
ul { padding-left: 20px; }
li { margin: 4px 0; }
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>NeuroDefector: {_html.escape(filename)}</title>
<style>{css}</style>
</head>
<body>
<h1>NeuroDefector Report</h1>
<div class="meta">{_html.escape(filename)} &middot; {date_str} &middot; Калибровка: &laquo;{cal_name}&raquo;</div>

<div class="sb">
  <div class="si">
    <div class="sv" style="color:{agg_color}">{aggregate_score:+.2f}</div>
    <div class="sl">Агрегированный балл</div>
  </div>
  <div class="si">
    <div class="sv" style="font-size:1.2em;color:{agg_color}">{_html.escape(label_for_score(aggregate_score))}</div>
    <div class="sl">Вердикт</div>
  </div>
  <div class="si">
    <div class="sv">{len(flagged)}/{len(all_stats)}</div>
    <div class="sl">Подозрительных чанков</div>
  </div>
  <div class="si">
    <div class="sv">{auc_str}</div>
    <div class="sl">AUC (LOO)</div>
  </div>
  <div class="si">
    <div class="sv" style="font-size:1em">{sem_status}</div>
    <div class="sl">Семантика</div>
  </div>
</div>

<h2>Веса фич</h2>
<table>
<tr><th>Фича</th><th>Вес</th><th>Направление</th><th>Величина</th></tr>
{weight_rows}
</table>

<h2>Чанки</h2>
<table>
<tr><th>#</th><th>симв</th><th>mean_nll</th><th>std_nll</th><th>z</th><th>score</th><th>метка</th></tr>
{chunk_rows}
</table>

<h2>Подозрительные чанки ({len(flagged)})</h2>
{chunks_html or '<p>Подозрительных чанков нет.</p>'}

<h2>Сводка</h2>
{summary_html}

<p style="color:#444;font-size:.8em;margin-top:40px">Создано NeuroDefector &middot; {date_str}</p>
</body>
</html>"""

    Path(output_path).write_text(html_doc, encoding="utf-8")
    print(f"\nHTML-отчёт сохранён: {output_path}")


def analyze(text_path: str, verbose: bool = False, html_path: str | None = None):
    if not Path(CALIBRATION_FILE).exists():
        print(f"Нет файла {CALIBRATION_FILE}. Сначала запустите 'calibrate'.")
        sys.exit(1)

    calib = json.loads(Path(CALIBRATION_FILE).read_text(encoding="utf-8"))
    print(f"Калибровка: '{calib.get('name', 'default')}', "
          f"{calib['human']['n_chunks']} человеческих ({calib['human']['n_files']} файлов), "
          f"{calib['ai']['n_chunks']} ИИ ({calib['ai']['n_files']} файлов)")
    if "auc_loo" in calib:
        print(f"AUC (leave-one-file-out): {calib['auc_loo']:.3f}")
    if calib.get("semantic_enabled"):
        print(f"Семантические фичи: ВКЛ ({calib.get('semantic_model', '?')})")
    else:
        print("Семантические фичи: ВЫКЛ")

    # Веса фич — печатаем для интерпретируемости (item 4)
    feature_names = calib.get("feature_names", [])
    weights = calib.get("logreg_weights", [])
    if feature_names and weights:
        print("\nТоп фичи (вес > 0 → человек, < 0 → ИИ):")
        paired = sorted(zip(weights, feature_names), key=lambda x: -abs(x[0]))
        for w, name in paired:
            bar = ("+" * min(int(abs(w) * 6), 12) if w > 0
                   else "-" * min(int(abs(w) * 6), 12))
            direction = "→ человек" if w > 0 else "→ ИИ    "
            print(f"  {name:<24} {w:+.3f}  {direction}  {bar}")

    # Если в калибровке сохранены списки клише, и пользователь не
    # переопределил их через --cliches — используем те же списки, что
    # были при калибровке.  Это важно: lexical_marker_rate и
    # cliche_proximity должны считаться по тому же словарю, иначе
    # стандартизация дрейфует.
    if not LEXICAL_MARKERS and not SEMANTIC_CLICHES:
        # Глобальные списки пусты => --cliches не передавался
        cal_lex = calib.get("lexical_markers")
        cal_sem = calib.get("semantic_cliches")
        if cal_lex is not None or cal_sem is not None:
            set_cliches(cal_lex or [], cal_sem or [])
            print(f"\nКлише восстановлены из калибровки: "
                  f"{len(cal_lex or [])} лексических, "
                  f"{len(cal_sem or [])} семантических")
    print()

    model, tokenizer, device = load_model()

    # Прогреваем семантический энкодер заранее, если калибровка его требует
    calib_embeds = _load_calibration_embeds(calib)
    use_semantic = calib_embeds is not None

    text = Path(text_path).read_text(encoding="utf-8")
    chunks = split_into_chunks(text)
    print(f"Текст разбит на {len(chunks)} чанков\n")

    all_stats = []
    n_skipped_short = 0
    for i, chunk in enumerate(chunks):
        if len(chunk) < MIN_CHUNK_SIZE:
            n_skipped_short += 1
            continue
        stats = compute_chunk_stats(chunk, model, tokenizer, device,
                                     keep_tokens=True,
                                     calib_embeds=calib_embeds,
                                     use_semantic=use_semantic)
        if stats is None:
            continue
        z, llr, details = humanness_score_logreg(stats, calib)
        stats["index"] = i
        stats["chunk"] = chunk
        stats["z"] = z
        stats["llr"] = llr
        stats["score"] = math.tanh(z / 2.0)
        stats.update(details)
        all_stats.append(stats)

    if not all_stats:
        print("Не осталось чанков для анализа после фильтрации по длине.")
        sys.exit(1)

    if n_skipped_short:
        print(f"Пропущено коротких чанков (<{MIN_CHUNK_SIZE} симв): {n_skipped_short}\n")

    # ─── OOD warnings ───
    for w in check_ood(all_stats, calib):
        print(w)
    print()

    # ─── Таблица по чанкам ───
    print("=" * 100)
    print(f"{'#':>3} {'симв':>5} {'mean_nll':>8} {'std_nll':>7}  "
          f"{'llr_std':>7} {'llr_nll':>7}  {'z':>6}  {'score':>6}  метка")
    print("=" * 100)
    for s in all_stats:
        print(f"{s['index']:>3} {len(s['chunk']):>5} {s['mean_nll']:>8.2f} {s['std_nll']:>7.2f}  "
              f"{s['llr_std']:>+7.2f} {s['llr_nll']:>+7.2f}  "
              f"{s['z']:>+6.2f}  {s['score']:>+6.2f}  {label_for_score(s['score'])}")

    # ─── Агрегация в LLR-пространстве ───
    # Сначала среднее z по чанкам (или сумма — выбираю среднее как
    # устойчивую оценку среднего log-likelihood ratio на чанк),
    # потом tanh — а не среднее tanh'ов по чанкам.
    mean_z = sum(s["z"] for s in all_stats) / len(all_stats)
    aggregate_score = math.tanh(mean_z / 2.0)

    # Доля подозрительных чанков — отдельная характеристика
    flagged = [s for s in all_stats if s["score"] < -0.2]
    very_flagged = [s for s in all_stats if s["score"] < -0.5]

    print("\n" + "─" * 60)
    print(f"Средний z по тексту: {mean_z:+.2f}")
    print(f"Агрегированный балл (tanh от среднего z): {aggregate_score:+.2f}  "
          f"→  {label_for_score(aggregate_score)}")
    print(f"Чанков со склоном к ИИ (score < -0.2): {len(flagged)}/{len(all_stats)} "
          f"({len(flagged)/len(all_stats):.0%})")
    print(f"Чанков с явным ИИ-профилем (score < -0.5): {len(very_flagged)}/{len(all_stats)} "
          f"({len(very_flagged)/len(all_stats):.0%})")
    print("─" * 60)
    print("Шкала: -1 = чистый ИИ, 0 = неразличимо, +1 = чистый человек")

    # ─── Verbose: гистограммы ───
    if verbose:
        print("\n" + "═" * 60)
        print("Распределение log_std_nll:")
        print_histogram(calib["human"]["raw_log_std_nll"], label="калибровка человек")
        print_histogram(calib["ai"]["raw_log_std_nll"], label="калибровка ИИ")
        print_histogram([s["log_std_nll"] for s in all_stats], label="анализируемый текст")
        print("\nРаспределение log_mean_nll:")
        print_histogram(calib["human"]["raw_log_mean_nll"], label="калибровка человек")
        print_histogram(calib["ai"]["raw_log_mean_nll"], label="калибровка ИИ")
        print_histogram([s["log_mean_nll"] for s in all_stats], label="анализируемый текст")

    # ─── Подробно — самые подозрительные ───
    per_chunk_diagnostics: list[list[dict]] = []  # для итоговой сводки и HTML
    if flagged:
        # Контекст текста — для относительных сравнений в диагностике
        text_ctx = compute_text_context(all_stats)

        # Определяем, есть ли цветной терминал
        use_color = sys.stdout.isatty()

        print(f"\nПодозрительные чанки ({len(flagged)}):")

        for s in flagged:
            print(f"\n{'═' * 72}")

            # Уверенность как функция от |z|
            confidence = (1 - math.exp(-abs(s["z"]))) * 100
            print(f"Чанк #{s['index']} — score {s['score']:+.2f} | "
                  f"уверенность ИИ-сигнала: {confidence:.0f}%")

            # Сравнение с остальным текстом
            z_med = text_ctx.get("z_text_median", 0)
            z_mad = text_ctx.get("z_text_mad", 0) or 0.01
            sigma_off = (s["z"] - z_med) / (1.4826 * z_mad) if z_mad else 0
            if sigma_off < -1.0:
                rel_note = f"({sigma_off:+.1f}σ от медианы вашего текста — выбивается)"
            elif sigma_off > -0.3:
                rel_note = f"({sigma_off:+.1f}σ от медианы — близко к остальному тексту)"
            else:
                rel_note = f"({sigma_off:+.1f}σ от медианы)"
            print(f"Положение в вашем тексте: {rel_note}")
            print(f"{'═' * 72}")

            # Диагностика
            diagnostics = diagnose_chunk(s, s["chunk"], calib, text_context=text_ctx)
            per_chunk_diagnostics.append(diagnostics)

            if not diagnostics:
                print("Конкретных локальных маркеров не выявлено — общий "
                      "профиль сдвинут к ИИ-зоне, но без ярких признаков.")
            else:
                # Группируем по категориям
                by_cat = {}
                for d in diagnostics:
                    by_cat.setdefault(d["category"], []).append(d)

                for cat, items in by_cat.items():
                    print(f"\n— {cat.upper()}:")
                    for d in items:
                        sev_marker = {"сильно": "▰▰▰", "умеренно": "▰▰▱",
                                       "слабо": "▰▱▱"}.get(d["severity"], "▱▱▱")
                        print(f"  {sev_marker} [{d['severity']}] {d['what']}")
                        print(f"      Почему: {d['why']}")
                        print(f"      Совет: {d['advice']}")

            # Собираем все highlights в один диапазон
            all_highlights = []
            for d in diagnostics:
                all_highlights.extend(d.get("highlights", []))

            # Текст чанка с подсветкой
            preview_len = 700
            preview = s["chunk"][:preview_len]
            # Корректируем диапазоны под обрезку
            preview_highlights = [(a, min(b, preview_len)) for a, b in all_highlights
                                   if a < preview_len]
            highlighted = highlight_text(preview, preview_highlights, use_color)
            if len(s["chunk"]) > preview_len:
                highlighted += "…"
            print(f"\nТекст:\n{highlighted}")

        # ─── Сводка по типам проблем во всём тексте ───
        summary = summarize_diagnostics(per_chunk_diagnostics)
        if summary and len(flagged) > 1:
            print(f"\n{'═' * 72}")
            print("СВОДКА: какие проблемы встречаются чаще всего")
            print(f"{'═' * 72}")
            for line in summary:
                print(f"  • {line}")
            print(
                "\nЕсли одна проблема доминирует в нескольких чанках — это "
                "системная особенность текста, и часто одно действие "
                "(например, разнообразить атрибуции диалога) чинит сразу "
                "много флагов."
            )
    else:
        summary = []
        print("\nПодозрительных чанков нет.")

    if html_path:
        generate_html_report(text_path, all_stats, calib, flagged,
                             per_chunk_diagnostics, summary, html_path)


# ─────────────────────── CLI ───────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Калибровщик и анализатор перплексии",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Откалибровать на двух корпусах")
    p_cal.add_argument("human_dir", help="Папка с человеческими .txt")
    p_cal.add_argument("ai_dir", help="Папка с ИИ-генерированными .txt")
    p_cal.add_argument("--name", default="default",
                       help="Имя калибровки (для версионирования)")
    p_cal.add_argument("--cliches", default=None,
                       help="Путь к JSON-списку клише (см. формат в начале файла). "
                            "Если не задан, используется встроенный мини-список.")

    p_an = sub.add_parser("analyze", help="Анализировать текст")
    p_an.add_argument("text_path", help="Путь к .txt для анализа")
    p_an.add_argument("--verbose", "-v", action="store_true",
                       help="Показать гистограммы распределений")
    p_an.add_argument("--html", metavar="PATH", nargs="?", const="report.html",
                       help="Сохранить HTML-отчёт (по умолчанию report.html)")
    p_an.add_argument("--cliches", default=None,
                       help="Путь к JSON-списку клише.  ВАЖНО: при анализе "
                            "лучше использовать тот же список, что и при "
                            "калибровке — иначе lexical_marker_rate будет "
                            "несовместима со стандартизацией.")

    args = parser.parse_args()

    # Загружаем клише, если указан внешний JSON
    if args.cliches:
        try:
            lex, sem = load_cliches_from_json(args.cliches)
            set_cliches(lex, sem)
            print(f"Загружено клише из {args.cliches}: "
                  f"{len(lex)} лексических маркеров, "
                  f"{len(sem)} семантических фраз\n")
        except Exception as e:
            print(f"⚠️  Не удалось загрузить клише из {args.cliches}: {e}")
            print("    Использую встроенный список.\n")

    if args.cmd == "calibrate":
        calibrate(args.human_dir, args.ai_dir, args.name)
    elif args.cmd == "analyze":
        analyze(args.text_path, verbose=args.verbose,
                html_path=getattr(args, "html", None))


if __name__ == "__main__":
    main()
