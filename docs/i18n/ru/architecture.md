# Архитектура

> Languages: [English](../../architecture.md) · **Русский** · [中文](../zh/architecture.md) · [Español](../es/architecture.md) · [Français](../fr/architecture.md) · [日本語](../ja/architecture.md) · [Deutsch](../de/architecture.md) · [한국어](../ko/architecture.md) · [Bahasa Indonesia](../id/architecture.md)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

Два транспорта используют один и тот же экземпляр актора — **MCP stdio** для интеграции с IDE (Claude Desktop, Cursor) и **REST HTTP** для агентных рабочих процессов и бэкендов Claude Code.

## Структура проекта

```
Sources/                    — one SwiftPM target per directory; dependencies point down only
  BasaltCore/               — DTOs, errors, config, logger, profilers, calculators, HF cache, orchestrator
    Chat/                   — ModelProfile: everything that differs between model families
    Config/                 — ServerConfig value type
  BasaltInference/          — MLX inference engine, ContextDegradationProfiler, RepetitionGuard  (→ Core)
  BasaltCloud/              — OpenRouter / cloud model fallback (CostTracker, ModelRouter)    (→ Core)
  BasaltServer/             — REST (Hummingbird 2.x, Host/Origin/Content-Type guard, API key and rate-limit middleware) and MCP JSON-RPC stdio server (→ Core)
  BasaltUI/                 — Terminal UI (Markdown, spinner, progress bar, diff, table)       (→ Core)
  BasaltApp/                — SwiftUI REST chat client, not linked into the CLI (launch: #55)  (→ Core)
  BasaltCLI/                — Subcommands and the CLI root command (App/)                     (→ all of the above)
  BasaltBin/                — Thin executable wrapper (product `Basalt`, installed as `basalt`)
  PerformanceBenchmark/     — Standalone benchmark CLI target

Tests/
  BasaltTests/              — Unit + integration tests
    CLI/                    — Command parsing, Cloud, Fit, PromptContextBuilder
    Cloud/                  — CostTracker, ModelRouter, OpenRouterClient
    UI/                     — Clipboard, Diff, Markdown, ProgressBar, Spinner, Table, TerminalUI
    MockInferenceEngine.swift — Controllable mock for orchestrator tests

basalt-claude                       — Executable Swift launcher (build + serve + claude)
scripts/                    — Build and maintenance Swift scripts
docs/                       — Extended documentation
```

Импортировать SwiftUI/AppKit разрешено только модулю `BasaltApp`; все таргеты, с которыми линкуется CLI, остаются headless (это проверяется тестом `ModuleBoundaryTests`).
Символы, впервые разделяемые между таргетами, используют уровень доступа `package` (существующий публичный API остаётся `public`).

---

## Архитектурные заметки

**Динамическое квантование, MACaQ и MoE** — `MLXInferenceEngine` разбирает `config.json` (включая `macaq_quantization`, `flatquant_quantization`, `lirquant_quantization`), чтобы направлять плотные (Dense) и разреженные (Sparse) MoE-чекпоинты на оптимизированные Metal-ядра (`SwitchGLU`, однопроходный `MACaQSwitchLinear` WHT + `MLX.gatherQuantizedMM` `#628`, 2-битная решётка `E8P`, `QTIP`, 3D-внимание для видео `STA` `#609`).

**SDK, управление и многоуровневая память (`v0.12`)** — `BasaltFoundationModels` предоставляет мост Swift SDK (`#606`); `VideoRunnerCore` поэтапно выполняет генерацию видео в 24 ГБ UMA (`#611`); `TieredKVHierarchyCore` управляет уровнями KV-кэша L0/L1/L2 в унифицированной памяти (`#627`); `SemanticCacheActor`, `HITLApprovalActor` и `SymbolicKnowledgeGraph` обеспечивают корпоративную маршрутизацию и подтверждения Human-in-the-Loop (`#625`, `#626`).

**Изоляция акторов** — `ModelOrchestratorActor` является актором Swift 6. Все вызовы инференса сериализуются (FIFO) без явных блокировок. И MCP, и REST разделяют один экземпляр актора.

**Горячая замена моделей** — `switchModel(to:)` разрешает ID репозитория HuggingFace из локального кэша, выгружает текущую модель (`container = nil` + `MLX.GPU.clearCache()`), затем загружает новые веса. Актор гарантирует, что выполняющиеся запросы завершатся до начала замены.

**Защита от таймаутов** — каждый вызов `generate` и `generateStream` обёрнут в 5-минутный таймаут с кооперативной отменой Task, предотвращая бессрочную блокировку сервера при зависании модели.

**Потоковая передача** — тело ответа `AsyncStream<ByteBuffer>` в Hummingbird 2 используется для обоих форматов SSE: OpenAI (`data: {...}`) и Anthropic (`event: content_block_delta`).

**Удаление блоков размышлений** — `generateStream` запускает конечный автомат для подавления блоков рассуждений `<think>...</think>` перед отправкой токенов клиентам.

**Трансляция ID** — `ModelsController` преобразует идентификаторы репозиториев HuggingFace (`mlx-community/Qwen3.5-4B-4bit`) в формат `claude-local/mlx-community--Qwen3.5-4B-4bit`, требуемый фильтром обнаружения моделей в Claude Code. В API-запросах принимаются оба формата.

---

## Скрипты обслуживания

| Скрипт | Назначение |
|---|---|
| `scripts/build_metal.swift` | Скомпилировать Metal-ядра GPU (пропускается, если актуально) |
| `scripts/cleanup.swift` | `--mode=daily` ротирует логи/изображения/отчёты, `--mode=release` удаляет артефакты сборки и runtime перед публикацией, `--mode=all` выполняет оба режима; по умолчанию пробный прогон (dry run), если не указан `--apply` |
| `scripts/archive.sh` | Архивировать логи/отчёты/скриншоты |

## Зависимости

Всё оформлено как пакет SwiftPM; в runtime нет ни CocoaPods, ни Node, ни Python. Точные ревизии зафиксированы в `Package.resolved`.

| Пакет | Назначение |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | слой массивов и нейросетей (NN) на Metal |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | архитектуры и загрузка моделей — `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | токенизаторы и шаблоны чата |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | HTTP-сервер |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | CLI |
| [Rainbow](https://github.com/onevcat/Rainbow) | цвета и оформление в терминале |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | рендеринг Markdown и подсветка синтаксиса в терминале |

`MLXVLM` обязателен: чекпоинты Gemma 4 и Ornith объявляют `*ForConditionalGeneration`, и эти архитектуры регистрируются именно там, а не в `MLXLLM`.

Обновления приходят в виде PR от Dependabot — еженедельно для SwiftPM и ежемесячно для GitHub Actions. mlx-swift развивается быстро, и поддержка новой архитектуры моделей часто появляется уже в следующем релизе, чем и обусловлена еженедельная периодичность.

---

## Участие в разработке

Issues и pull requests приветствуются. Перед открытием PR:

```bash
swift build          # must compile
swift test           # must stay green
```

Никогда не коммитьте учётные данные, токены или файлы состояния `*.sqlite3` — `.gitignore` блокирует стандартные случаи, а рабочий процесс релиза (release workflow) запускает сканирование дерева на секреты.
