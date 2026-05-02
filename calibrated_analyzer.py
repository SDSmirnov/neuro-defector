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
"""

import sys
import re
import math
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-270m" # "sberbank-ai/rugpt3small_based_on_gpt2"
TARGET_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 800       # поднято с 600 — на коротких чанках PPL слишком шумная
MAX_CHUNK_SIZE = 1400
CALIBRATION_FILE = "calibration.json"

# Малая константа для логарифмов
EPS = 1e-6


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
                         keep_tokens: bool = False) -> dict | None:
    """
    Считает статистики по чанку.

    keep_tokens=True — дополнительно возвращает сырые token_nll и
    декодированные токены для диагностики.  В режиме калибровки этого
    не нужно (только финальные статистики), при анализе включается.
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
    статистик и параллельный список имён файлов (для leave-one-file-out)."""
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
            stats = compute_chunk_stats(chunk, model, tokenizer, device)
            if stats is None:
                continue
            if stats["truncated"]:
                n_truncated += 1
            all_stats.append(stats)
            all_files.append(f.name)
            n_kept += 1
        print(f"  {f.name}: {n_kept}/{len(chunks)} чанков использовано")
    if n_truncated:
        print(f"  ⚠️  {n_truncated} чанков были обрезаны по max_length токенизатора")
    print(f"  Итого: {len(all_stats)} чанков из {len(files)} файлов\n")
    return all_stats, all_files


def calibrate(human_dir: str, ai_dir: str, name: str = "default"):
    model, tokenizer, device = load_model()

    print("─── Человеческий корпус ───")
    human_stats, human_files = gather_chunk_stats(Path(human_dir), model, tokenizer, device)
    print("─── ИИ корпус ───")
    ai_stats, ai_files = gather_chunk_stats(Path(ai_dir), model, tokenizer, device)

    if len(human_stats) < 10 or len(ai_stats) < 10:
        print("Недостаточно данных для калибровки (нужно минимум 10 чанков на класс).")
        sys.exit(1)

    # ─── Подготовка фич ───
    # Используем семь признаков:
    #   - log_std_nll, log_mean_nll — основа (центр и разброс перплексии)
    #   - tail_ratio — доля «удивительных» токенов
    #   - cv_nll — нормированный разброс
    #   - repeat_3gram — повторяемость словесных триграмм
    #   - cv_sent_len — вариативность длин предложений
    #   - punct_entropy — разнообразие пунктуации
    feature_names = [
        "log_std_nll", "log_mean_nll",
        "tail_ratio", "cv_nll",
        "repeat_3gram", "cv_sent_len", "punct_entropy",
    ]

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

    calib = {
        "name": name,
        "model": MODEL_NAME,
        "feature_names": feature_names,
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


def analyze(text_path: str, verbose: bool = False):
    if not Path(CALIBRATION_FILE).exists():
        print(f"Нет файла {CALIBRATION_FILE}. Сначала запустите 'calibrate'.")
        sys.exit(1)

    calib = json.loads(Path(CALIBRATION_FILE).read_text(encoding="utf-8"))
    print(f"Калибровка: '{calib.get('name', 'default')}', "
          f"{calib['human']['n_chunks']} человеческих ({calib['human']['n_files']} файлов), "
          f"{calib['ai']['n_chunks']} ИИ ({calib['ai']['n_files']} файлов)")
    if "auc_loo" in calib:
        print(f"AUC (leave-one-file-out): {calib['auc_loo']:.3f}")
    print()

    model, tokenizer, device = load_model()

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
                                     keep_tokens=True)
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
    if flagged:
        # Контекст текста — для относительных сравнений в диагностике
        text_ctx = compute_text_context(all_stats)

        # Определяем, есть ли цветной терминал
        use_color = sys.stdout.isatty()

        print(f"\nПодозрительные чанки ({len(flagged)}):")
        per_chunk_diagnostics = []  # для итоговой сводки

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
        print("\nПодозрительных чанков нет.")


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

    p_an = sub.add_parser("analyze", help="Анализировать текст")
    p_an.add_argument("text_path", help="Путь к .txt для анализа")
    p_an.add_argument("--verbose", "-v", action="store_true",
                       help="Показать гистограммы распределений")

    args = parser.parse_args()

    if args.cmd == "calibrate":
        calibrate(args.human_dir, args.ai_dir, args.name)
    elif args.cmd == "analyze":
        analyze(args.text_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
