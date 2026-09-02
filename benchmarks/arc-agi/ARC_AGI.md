# ARC-AGI Benchmark Specification & Evaluation Protocol

Документ описывает методологию, процедуры, метрики и практическое руководство по запуску бенчмарка **ARC-AGI** (Abstraction and Reasoning Corpus) для локальных моделей (Gemma 4 26B MoE) с использованием движка **Native Swift MLX (`gemm`)**.

---

## 1. Обзор датасета ARC-AGI-1

Корпус в репозиторий не вендорится — он чужой и весит 5 МБ в 800 файлах.
Положите его сами:

```bash
git clone --depth 1 https://github.com/fchollet/ARC-AGI /tmp/arc-agi
mkdir -p benchmarks/arc-agi/dataset
cp -R /tmp/arc-agi/data/training benchmarks/arc-agi/dataset/training
cp -R /tmp/arc-agi/data/evaluation benchmarks/arc-agi/dataset/evaluation
```

Локальная база задач ARC-AGI-1 после этого лежит по адресу:
`benchmarks/arc-agi/dataset/` (относительно корня репозитория)

Общий объем составляет **800 задач**:
- **`dataset/training/`** — **400 задач** (Обучающий набор)
- **`dataset/evaluation/`** — **400 задач** (Официальный тестовый/оценочный набор)

Каждая задача представляет собой JSON-файл следующей структуры:
```json
{
  "train": [
    {"input": [[...]], "output": [[...]]},
    {"input": [[...]], "output": [[...]]}
  ],
  "test": [
    {"input": [[...]], "output": [[...]]}
  ]
}
```

---

## 2. Официальный регламент и правила проведения бенчмаркинга

Бенчмарк `arc_agi_benchmark.py` строго выполняет спецификацию соревнований **ARC Prize / ARC-AGI**:

1. **Количество попыток (Pass@2)**:
   - На каждую задачу модели даётся **2 независимые попытки** (Attempt 1 и Attempt 2).
   - Задача считается **решённой (PASSED)**, если хотя бы одна из двух попыток дает 100% совпадение матрицы ответа с эталоном.
2. **Метрики прохождения**:
   - **Pass@1**: Доля задач, решённых строго с первой попытки.
   - **Pass@2 (Official Score)**: Официальный итоговый балл соревнований ARC-AGI.
3. **Бюджет рассуждений (Reasoning Budget)**:
   - По умолчанию для Gemma 4 MoE установлен бюджет в **16 000 токенов** (`--max-tokens 16000`).
   - Модель формирует цепочку рассуждений в специальном канале `<|channel|>thought`.
4. **Режимы генерации (`--mode`)**:
   - **`--mode hybrid` (По умолчанию / Рекомендуемый)**: **Program-of-Thought (PoT)**. Модель генерирует алгоритм на Python `def transform(grid)`, который исполняется в изолированном песочнице Python REPL с тестовой сеткой. Это устраняет галлюцинации нейросети при вычислении больших матриц (14x14, 20x20).
   - **`--mode direct`**: Модель генерирует 2D JSON-массив ответа напрямую токенами.

---

## 2.1. Архитектура гибридного режима (Hybrid Program-of-Thought)

```
[ Промпт задачи ]  -->  [ Gemma 4 MoE + <|channel|>thought ]  -->  [ Извлечение Python-кода ]  -->  [ Python Sandbox REPL ]  -->  [ Точная матрица ]
  Примеры IN/OUT          Логический анализ и алгоритм               `def transform(grid)`              `transform(test_input)`           100% точность
```

В гибридном режиме `arc_agi_benchmark.py`:
1. Модели подаётся системная инструкция сформировать функцию `transform(grid: list[list[int]]) -> list[list[int]]`.
2. Код извлекается из блока ` ```python ... ``` ` в ответе или в цепочке рассуждений.
3. Код безопасно исполняется через `exec()` в изоляции, на вход подаётся матрица `test_input`.
4. Итоговая вычисленная матрица сравнивается с эталоном. В случае ошибки выполнения делается автоматический откат (fallback) к прямому JSON-парсингу.

---

## 2.2. Объектно-Ориентированный DSL и Строгая Типизация (Typed OOP DSL)

Для исключения падений из-за ошибок индексации массивов ($IndexError$), путаницы координат $(x, y)$ vs $(r, c)$ и повышения точности рассуждений модель обучается выражать трансформации через высокоуровневый **Typed OOP DSL**:

```python
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Set

class Color(IntEnum):
    BLACK = 0; BLUE = 1; RED = 2; GREEN = 3; YELLOW = 4
    GREY = 5; MAGENTA = 6; ORANGE = 7; CYAN = 8; MAROON = 9

@dataclass(frozen=True)
class Point:
    r: int
    c: int

@dataclass
class GridObject:
    pixels: Set[Tuple[Point, int]]
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]: ...
    def move(self, dr: int, dc: int) -> 'GridObject': ...
    def change_color(self, new_color: int) -> 'GridObject': ...

class Grid:
    def extract_objects(self, connectivity: int = 4) -> List[GridObject]: ...
    def filter_by_color(self, color: int) -> List[GridObject]: ...
    def to_matrix(self) -> List[List[int]]: ...
```

### Выигрыш от использования ООП DSL:
1. **Семантический сдвиг:** Переход от низкоуровневой индексации пикселей к операциям с объектами («найти фигуры», «изменить цвет», «переместить»).
2. **Снижение ошибок падения (IndexError):** Поиск компонент связности и границы объектив скрыты внутри методов классов `Grid` и `GridObject`.
3. **Экономия токенов рассуждения:** Лаконичный код (5–15 строк) вместо 50+ строк вложенных циклов $i, j, k$.

---

## 3. Измеряемые метрики (Measured Metrics)

Для каждой задачи в процессе выполнения собирается полный комплекс диагностических и вычислительных метрик:

| Категория | Метрика | Описание |
| :--- | :--- | :--- |
| **Время (Latency)** | `elapsed_sec` | Полное время выполнения попытки (стенка / wall-clock time в сек). |
| | `prefill_time` | Время обработки вводного промпта (Prefill / Prompt Evaluation). |
| | `decode_time` | Время генерации токенов (Decode / Generation time). |
| **Токены (Tokens)** | `reasoning_len` | Количество символов / токенов в канале рассуждений `<|channel|>thought`. |
| | `content_len` | Количество символов / токенов в итоговом канале ответа. |
| | `tokens_per_sec` | Скорость генерации токенов (например, **75.8 tok/s** на Swift MLX). |
| **Память (RAM / Unified)** | `peak_memory_gb` | Пиковое потребление единой памяти (Unified Memory). |
| | `kv_cache_mb` | Объём динамического KV-кэша (рассчитывается как ~120.88 КБ/токен). |
| **Точность сетки (Grid Accuracy)** | `input_shape` | Размеры входной матрицы `N x M`. |
| | `output_shape` | Размеры целевой матрицы `K x L`. |
| | `parsed` | Успешность извлечения 2D JSON массива из вывода. |
| | `correct` | Полное совпадение матрицы с эталоном. |
| | `error_diff` | Количество ошибочных строк / ячеек при расхождении. |
| **Результативность (Score)** | `passed_attempt` | Номер успешной попытки (1 или 2, либо null при неудаче). |

---

## 4. Инструкция по вызову CLI бенчмарка (`arc_agi_benchmark.py`)

### 4.1. Прогон против ВСЕГО пула задач (800 задач)

Запуск полного тестирования по обучающей и оценочной выборкам с бюджетом **16 000 токенов**:

```bash
python3 arc_agi_benchmark.py --split all --engine swift_mlx --max-tokens 16000 --attempts 2 --resume
```

### 4.2. Прогон против конкретной выборки (`training` или `evaluation`)

```bash
# 400 задач из обучающего набора
python3 arc_agi_benchmark.py --split training --engine swift_mlx --max-tokens 16000 --resume

# 400 задач из официального набора оценки (Evaluation)
python3 arc_agi_benchmark.py --split evaluation --engine swift_mlx --max-tokens 16000 --resume
```

### 4.3. Прогон ограниченного числа задач (`--limit`) или со смещением (`--start-index`)

```bash
# Первые 20 задач
python3 arc_agi_benchmark.py --split training --limit 20 --engine swift_mlx

# Задачи с 50 по 100
python3 arc_agi_benchmark.py --split evaluation --start-index 50 --limit 50 --engine swift_mlx
```

### 4.4. Прогон конкретных задач по ID

```bash
# Запуск конкретных задач 007bbfb7 и 00d62c1b
python3 arc_agi_benchmark.py --tasks 007bbfb7 00d62c1b --engine swift_mlx --max-tokens 16000
```

### 4.5. Запуск с изменённым бюджетом рассуждений и температурой

```bash
# Увеличенный бюджет 24 000 токенов (Максимум Mac RAM) и температура 0.8
python3 arc_agi_benchmark.py --split training --limit 10 --engine swift_mlx --max-tokens 24000 --temperature 0.8
```

---

## 5. Формат отчётов и результатов

После и во время выполнения бенчмарка автоматически создаются два файла:

1. **`arc_benchmark_results_swift_mlx_*.json`**: Исходный JSON со всеми метриками по каждой попытке каждого задания.
2. **`arc_benchmark_results_swift_mlx_*.md`**: Markdown-отчёт с итоговыми таблицами Pass@1, Pass@2 и пошаговой разбивкой по задачам.
