# Архитектурный и Бизнес-План Модернизации GemmaServer (BA & Architecture)

Этот документ описывает декомпозицию эпиков продукта с точки зрения бизнес-ценности (Business Value), архитектурных компонентов (Architecture) и итеративного цикла разработки (TDD -> Implementation -> Integration -> Profiling -> Debug -> Commit).

---

## Эпик 1: Профилирование Контекстного Окна и Бенчмаркинг 🚀 (Наивысший приоритет)
**Бизнес-ценность:** Предоставить пользователям (и агенту) понимание лимитов контекста на их оборудовании, чтобы предотвратить деградацию производительности (падение TPS) и падения из-за Out-Of-Memory.
**Архитектура:** Модуль `PerformanceBenchmark`, подсистема сбора MLX-метрик (`MLXInferenceEngine`), профилировщик Unified Memory.

- [x] **1.1. Swift Benchmark:** Target для замера производительности оркестратора.
- [x] **1.2. Метрики MLX:** Сбор TTFT (Time To First Token) и TPS (Tokens Per Second).
- [x] **1.3. Мониторинг памяти:** Логирование Unified Memory.
- [x] **1.4. Context Window Degradation Benchmark:**
  - *TDD:* Написать тест, создающий моковый контекст от 1k до 128k и вызывающий замер.
  - *Имплементация:* Скрипт/класс автоматизированного замера деградации TPS/TTFT при поэтапном заполнении контекста.
  - *Профилирование:* Запись результатов в `context_latency.json`.
  - *Ожидаемый результат:* Построен график/json "золотой середины" производительности.
- [ ] **1.5. Dynamic Token Budgeting:**
  - *TDD:* Тест на расчет бюджетирования исходя из свободной RAM.
  - *Имплементация:* Динамический расчет доступного контекста на основе `os_proc_available_memory()`.

---

## Эпик 2: Внедрение TurboQuant (Google Research) для сжатия K/V Cache 🧠⚡
**Бизнес-ценность:** Экономия до 80% RAM (сжатие K/V кэша до 5.7x). Это позволит запускать агента с огромным контекстным окном (128k+ токенов) на базовых Mac (16GB/32GB RAM).
**Архитектура:** Кастомные операции в MLX (Swift), интеграция алгоритмов ротации `PolarQuant` и 1-битного слоя коррекции `QJL`.

- [ ] **2.1. PolarQuant Implementation:**
  - *TDD:* Тест на корректность вращения матриц (совпадение с Python-референсом).
  - *Имплементация:* Написание фазы PolarQuant на Swift MLX для ротации векторов K/V.
- [ ] **2.2. QJL (Quantized Johnson-Lindenstrauss) Layer:**
  - *TDD:* Юнит-тест на несмещенную оценку скалярного произведения (distortion rate < 1e-3).
  - *Имплементация:* 1-битный слой коррекции ошибок (residuals).
- [ ] **2.3. MLX KV-Cache Integration:**
  - *Интеграция:* Подмена стандартного K/V кэша в `MLXInferenceEngine` на `TurboQuantMSE / TurboQuantProd`.
  - *Дебаг:* Интеграционные тесты генерации с длинным контекстом.
- [ ] **2.4. Memory & Quality Benchmark:**
  - *Профилирование:* Замер Memory Footprint и сравнение качества (cosine similarity) с FP16 кэшем. Вывод в консоль.

---

## Эпик 3: Интеллектуальный CLI Агент (Agentic Execution) 🤖
**Бизнес-ценность:** Превращение сервера в автономного CLI-ассистента (уровня Gemini CLI / Claude Code), способного самостоятельно читать код, применять патчи и запускать тесты.
**Архитектура:** Модули `CLI/ChatCommand`, подсистема `Tools` (ReAct Loop), `SQLite` для управления контекстом проектов.

### 3.1. Управление Сессиями и Контекстом
- [ ] **3.1.1. Workspace/Project Binding:** Изолированные сессии SQLite по `workingDir`.
- [ ] **3.1.2. System & Context Instructions:** Поддержка `.gemini.md` и `.cursorrules`.
- [ ] **3.1.3. Smart Context Compaction:** Суммаризация старых блоков истории при исчерпании Token Budget.
- [ ] **3.1.4. Seamless Model Switching:** Горячая смена модели `/model` без потери истории.

### 3.2. Управление Документами и Кодом (Document Tools)
- [ ] **3.2.1. Создание файлов (`write_file`):** Генерация новых файлов.
- [ ] **3.2.2. Поиск и анализ (`glob` / `grep_search`):** Семантический поиск по кодовой базе.
- [ ] **3.2.3. Чтение контекста (`read_file`):** Чтение с поддержкой `start_line` и `end_line`.
- [ ] **3.2.4. Продвинутое редактирование (`replace` / Diff):** Точечная замена строк и поддержка diff-патчей.

### 3.3. Автономность и ReAct Loop
- [ ] **3.3.1. Tool Calling Loop (ReAct):** Имплементация цикла `Thought -> Action -> Observation`.
- [ ] **3.3.2. Safe Shell Execution:** Безопасное выполнение shell-команд с песочницей/интерактивным подтверждением.
- [ ] **3.3.3. MCP Client Integration:** Возможность GemmaServer подключаться к внешним БД и инструментам.

---

## Эпик 4: UX Терминала и Интерактивность 🎨
**Бизнес-ценность:** Гладкий, красивый и понятный пользовательский опыт (как у лучших CLI утилит).
**Архитектура:** Расширение `CLI` парсерами Markdown и компонентами ANSI/Terminal UI.

- [ ] **4.1. Rich Markdown & Syntax Highlighting:** Рендеринг таблиц, списков, подсветка синтаксиса в потоковом режиме.
- [ ] **4.2. Interactive Approvals & Inputs:** UI-компоненты (Yes/No, Select) для деструктивных действий.
- [ ] **4.3. Global Alias & Zero-Config:** Вызов `gemma chat` "из коробки" с автоподбором модели.
- [ ] **4.4. Installation Script:** `brew tap` / скрипт `curl | bash`.

---

## Эпик 5 & 6: Рефакторинг и Auth 🛠️🔐 (Завершено)
- [x] **5.1 - 5.3:** Рефакторинг MCPServer, Pattern Matching в HTTP, Clean DTOs.
- [x] **6.1 - 6.3:** SwiftCrypto, External Config, Hummingbird JWT Middleware.

## Эпик 7: Расширенное тестирование 🧪
**Бизнес-ценность:** Гарантия отказоустойчивости при автономной работе.
- [x] **7.1 - 7.3:** Fuzz Testing, Concurrency, UI/CLI тесты.
- [ ] **7.4. Agentic Behavior Tests:** Мокирование файловой системы и shell для проверки Tool-calling цикла.
- [ ] **7.5. Tool Safety Tests:** Проверка блокировки деструктивных команд (rm -rf).

---

## 📦 Инструкция (Default Usage)
```bash
# Интеллектуальный CLI агент с поддержкой Tool Calling
gemma chat

# Сборка и установка
swift build -c release
sudo cp .build/release/GemmaServer /usr/local/bin/gemma
```