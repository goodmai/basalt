# Архитектурные паттерны и назначение модулей
## 1) Цель документа
Этот документ фиксирует:
- какие архитектурные паттерны используются в проекте;
- какие модули за что отвечают;
- какие зависимости между модулями допустимы;
- как безопасно расширять систему.

## 2) Контекст системы
Сервис реализует локальный LLM inference с двумя транспортами:
- MCP (stdio, JSON-RPC) для IDE/агентов;
- REST (HTTP) для A2A/интеграций.

Оба транспорта используют единый оркестратор запросов.

## 3) Ключевые паттерны
### 3.1 Actor-based orchestrator (Single Source of Truth)
**Где:** `Sources/BasaltCore/ModelOrchestratorActor.swift`
**Суть:** единая actor-точка входа для генерации, сериализация запросов (FIFO), защита от race conditions.
**Зачем:** безопасная конкурентность при одновременных вызовах из MCP и REST.

### 3.2 Ports & Adapters (Hexagonal style)
**Где:** `ModelOrchestratorActor` зависит от `InferenceEngine` (порт), `MLXInferenceEngine` выступает адаптером.
**Суть:** бизнес-логика не зависит напрямую от конкретного MLX backend.
**Зачем:** упрощает тестирование и замену движка инференса.

### 3.3 Dual-transport facade
**Где:** `Sources/BasaltServer/REST/*`, `Sources/BasaltServer/MCP/*`, `Sources/BasaltCLI/ServeCommand.swift`
**Суть:** два внешних интерфейса поверх одного ядра доменной логики.
**Зачем:** единое поведение модели при разных способах доступа.

### 3.4 Middleware pipeline (REST)
**Где:** `Sources/BasaltServer/REST/RESTServer.swift`, `Sources/BasaltServer/REST/Middleware/*`
**Суть:** composable цепочка middleware (логирование, JWT, rate limit).
**Зачем:** отделение кросс-срезовых concerns от контроллеров.

### 3.5 Command pattern (CLI)
**Где:** `Sources/BasaltCLI/*Command.swift`
**Суть:** каждая CLI-команда инкапсулирует сценарий запуска/управления.
**Зачем:** расширяемый CLI с независимыми subcommands.

## 4) Карта модулей (SwiftPM-таргеты)
### 4.1 `BasaltCore` — `Sources/BasaltCore`
- DTO, `BasaltError`, `ServerConfig` (`Config/`), `ModelProfile` (`Chat/`), логгер, телеметрия.
- Профилирование и оценка: `SystemProfiler`, `MemoryEvaluator`, `TokenBudgetCalculator`, `FitScorer`, `ModelFitAnalyzer`.
- `ModelOrchestratorActor` и порт `InferenceEngine`; HF-кэш и клиент хаба (`HuggingFaceHub.swift`).

### 4.2 `BasaltInference` — `Sources/BasaltInference`
- `MLXInferenceEngine` (адаптер порта `InferenceEngine`), `ContextDegradationProfiler`, `RepetitionGuard`.

### 4.3 `BasaltCloud` — `Sources/BasaltCloud`
- `CloudAPIClient`, `OpenRouterClient`, `ModelRouter`, `CostTracker`.

### 4.4 `BasaltServer` — `Sources/BasaltServer`
- `REST/`: HTTP API на Hummingbird, контроллеры, middleware (API key, rate limit).
- `MCP/`: MCP сервер по stdio (JSON-RPC), tools поверх оркестратора.

### 4.5 `BasaltUI` — `Sources/BasaltUI`
- Терминальный UI (Markdown, спиннер, прогресс-бар, diff, таблицы). Без SwiftUI/AppKit.

### 4.5a `BasaltApp` — `Sources/BasaltApp`
- SwiftUI-клиент REST (чат, ввод API-ключа). В CLI не линкуется; точка запуска — #55.

### 4.6 `BasaltCLI` — `Sources/BasaltCLI`
- Подкоманды (`serve`, `chat`, `models`, `fit`, `cloud`, `onboard`) и корневая команда (`App/`).

### 4.7 `Basalt` (исполняемый) — `Sources/BasaltBin`
- Тонкий entrypoint; Homebrew ставит его как `basalt`.

### 4.8 `PerformanceBenchmark` — `Sources/PerformanceBenchmark`
- Отдельный инструмент нагрузочного/перформанс-тестирования.

## 5) Матрица зависимостей (правила)
Граф таргетов (ацикличный, задан в `Package.swift`):
- `BasaltInference`, `BasaltCloud`, `BasaltServer`, `BasaltUI`, `BasaltApp` → `BasaltCore`
- `BasaltCLI` → все библиотеки выше, кроме `BasaltApp`; `Basalt` → `BasaltCLI`
- `PerformanceBenchmark` → `BasaltCore`, `BasaltInference`

Ограничения:
- Ни один таргет, кроме `BasaltApp`, не импортирует SwiftUI/SwiftUICore/AppKit/Cocoa (проверяет `ModuleBoundaryTests`; таргеты берутся из `Sources/`). AppKit всё же попадает в бинарь транзитивно — через Splash и swift-huggingface.
- `BasaltCore` не зависит от ArgumentParser: `LogLevel` получает `ExpressibleByArgument` в `BasaltCLI`.
- `BasaltServer` не зависит от `BasaltInference`, `BasaltCloud` и `BasaltUI`; движок приходит через порт `InferenceEngine`.
- Новые межмодульные символы открываются уровнем `package`, а не `public` (уже публичный API остался `public`).
- Кросс-срезовые политики (auth, rate limit, logging) — через middleware/обёртки, а не в доменной логике.

## 6) Поток обработки запроса
1. Клиент отправляет запрос через REST или MCP.
2. Транспортный слой нормализует вход в доменный `GenerationRequest`.
3. `ModelOrchestratorActor` валидирует/ограничивает параметры и вызывает `InferenceEngine`.
4. Ответ маппится обратно в формат транспорта (HTTP JSON или MCP result).

## 7) Шаблон описания нового модуля
При добавлении нового модуля заполняйте карточку:
- **Имя модуля:**
- **Путь:**
- **Ответственность (1–2 предложения):**
- **Входы/выходы:**
- **Зависимости:**
- **Недопустимые зависимости:**
- **Точки расширения:**
- **Риски и ограничения:**

## 8) Шаблон описания нового паттерна
- **Название паттерна:**
- **Проблема, которую решает:**
- **Где применён (файлы):**
- **Почему выбран:**
- **Компромиссы:**
- **Альтернативы:**

## 9) Checklist при изменении архитектуры
- Обновлён ли этот документ?
- Не нарушены ли направления зависимостей?
- Сохранена ли единая точка оркестрации в `Core`?
- Покрыты ли изменения тестами (unit/integration)?

## 10) Назначение директорий верхнего уровня (итерация 2)
### 10.1 Основные директории репозитория
- `Sources` — production-код (CLI, Core, REST, MCP, benchmark и runtime-модули).
- `Tests` — unit/integration тесты.
- `scripts` — операционные утилиты проекта (архивация, аудит, версии, очистка и т.д.).
- `docs` — проектная документация и архитектурные материалы.
- `.github` — CI/CD, шаблоны и GitHub-автоматизация.
- `Formula` — Homebrew-формула/пакетирование (если используется в релизном процессе).

### 10.2 Служебные директории (локальная инфраструктура)
- `.git` — метаданные git (не часть исходного контента проекта).
- `.build` — артефакты `swift build` и checkouts зависимостей.
- `.swiftpm` — локальные данные SwiftPM.
- `.archive` — локальный архив рабочих материалов/логов/бенчмарков.
- `.claude` — локальные артефакты AI-инструментов.

## 11) Временные файлы и директории для очистки перед публикацией
### 11.1 Удалять всегда (runtime/build artifacts)
- Директории: `.build/`, `.swiftpm/`, `logs/`, `docs-output/`, `docs-static/`, `.docc/`.
- Файлы: `.DS_Store`, `*.log`, `*.tmp`, `*.swp`, `*~`, `.coverage`, `*.profdata`.
- Генерируемые файлы: `default.metallib`, `server_*.log`, `benchmark_*.json`, `res_*.json`, `context_latency.json`.
- Локальные runtime-БД: `*.db`, `*.sqlite`, `*.sqlite3` (например `auth.sqlite3`).

### 11.2 Удалять опционально (локальные рабочие артефакты)
- `.archive/` (если архив не должен входить в публичный репозиторий).
- `.claude/`, `.gemini.md` (локальные AI/assistant артефакты).
- `test_*.swift` в корне (временные ad-hoc тесты, не входящие в `Tests/`).

## 12) Скрипт очистки перед GitHub
- Скрипт: `scripts/cleanup.swift --mode=release` (там же `--mode=daily` — ротация logs/images/reports, и `--mode=all`)
- Назначение: удалить build/runtime мусор и временные локальные файлы перед публикацией.
- Режимы:
  - `--dry-run` (по умолчанию): только показать, что будет удалено.
  - `--apply`: реально удалить найденные артефакты.
  - `--include-local`: дополнительно удалить опциональные локальные артефакты (`.archive`, `.claude`, `.gemini.md` и т.п.).

### 12.1 Рекомендуемый порядок запуска перед публикацией
1. Просмотр кандидатов на удаление:
   - `./scripts/cleanup.swift --mode=release`
2. Очистка build/runtime артефактов:
   - `./scripts/cleanup.swift --mode=release --apply`
3. Опционально (если нужен максимально чистый публичный репозиторий):
   - `./scripts/cleanup.swift --mode=release --apply --include-local`
