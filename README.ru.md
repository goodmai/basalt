# Gemm

[English](README.md) · [Русский](README.ru.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%2015%2B%20·%20Apple%20Silicon-black.svg)](#требования)
[![Swift 6](https://img.shields.io/badge/Swift-6-black.svg)](Package.swift)

Локальный сервер инференса LLM для Apple Silicon. Запускает Gemma 4, Qwen 3, Ornith 1.5 и другие MLX-совместимые модели целиком на устройстве (Metal GPU). Без аутентификации и без обращений в облако — рассчитан на локальную разработку и агентные сценарии.

```
┌──────────────────────────────────────────────────────┐
│                       Gemm                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (актор, FIFO)      Metal GPU  │
│   WebSocket ──┘                                      │
└──────────────────────────────────────────────────────┘
```

Два транспорта работают через один экземпляр актора — **MCP stdio** для интеграции с IDE (Claude Desktop, Cursor) и **REST HTTP** для агентных сценариев и в роли бэкенда Claude Code.

---

## Требования

| | |
|---|---|
| **macOS** | 15+ (Sequoia) |
| **Xcode / Swift** | 16+ / Swift 6 |
| **Железо** | Apple Silicon M1–M4, Unified Memory |
| **Диск** | 2–30 ГБ в зависимости от модели |

---

## Быстрый старт

### Установка через Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/gemm
```

Формула собирает из исходников — нужен Xcode 16+, первая сборка занимает несколько минут.

### Сборка из исходников

```bash
git clone https://github.com/goodmai/basalt
cd basalt

# Сборка
swift build -c release

# Компиляция Metal-ядер MLX. Нужна один раз на каждый checkout: SwiftPM-сборка
# mlx-swift не содержит metallib, и без него первый же вызов инференса падает с
# "Failed to load the default metallib". Скрипт кладёт библиотеку рядом с
# бинарём, поэтому он работает из любой директории.
./scripts/build_metal.swift

# Интерактивный чат
.build/release/gemm chat --model mlx-community/Qwen3.5-4B-4bit

# REST-сервер на :8080 (совместим с OpenAI и Anthropic)
.build/release/gemm serve --model mlx-community/Qwen3.5-4B-4bit --rest

# MCP stdio сервер (для Claude Desktop / Cursor)
.build/release/gemm serve --model mlx-community/gemma-4-e4b-it-4bit --mcp
```

### Лаунчер одной командой: `./Gemma`

В репозитории лежит самодостаточный launcher: он собирает сервер (если нужно), ждёт готовности модели и открывает Claude Code — все переменные окружения живут только внутри этой сессии:

```bash
chmod +x ./Gemma

./Gemma                                             # Qwen 4B, порт 8080
./Gemma --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./Gemma --port 8081                                 # свой порт
./Gemma -- --model haiku                            # передать --model haiku в claude
```

`Gemma` жёстко ставит `ANTHROPIC_API_KEY=local`, а не наследует переменную, — так настоящий ключ никогда не уходит на локальный сервер и не попадает в payload `--settings`. Другие терминалы не затрагиваются.

---

## Проверенные модели

Модели скачиваются с HuggingFace и кэшируются в `~/.cache/huggingface/hub/`.

```bash
gemm models download mlx-community/Qwen3.5-4B-4bit
gemm models download ornith-ai/Ornith-1.5-9B-MLX-4bit
```

| Модель | Параметры | RAM | Статус на Mac с 24 ГБ |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 ГБ | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 ГБ | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 ГБ | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 ГБ | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 ГБ | ✅ ~37 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 5.8 ГБ | ✅ ~45 TPS (reasoning — см. заметку) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 ГБ | ✅ на 24 ГБ не проверялась |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 ГБ | ✅ ~12 TPS (abliterated) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 ГБ | ✅ ~14 TPS (Dynamic Quant) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 ГБ | ✅ ~8 TPS (NVFP4) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (base BF16) | 27B | ~54 ГБ | ❌ нужно 64 ГБ+ |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 ГБ | ❌ мусор на выходе |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 ГБ | ❌ мусор на выходе |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 ГБ | ❌ OOM (нужно 32 ГБ+) |

> **Про Mac с 24 ГБ:**
> - Неквантованные 27B/35B (~54 ГБ весов) не влезают в физическую память. Берите MLX 4-bit или Dynamic Quant — они укладываются в ~15 ГБ.
> - Модели тяжелее 10 ГБ зажимают KV-кэш на длинных диалогах. `TokenBudgetCalculator` сам считает и ограничивает бюджет контекста по свободной памяти.

> **Про reasoning-модели (Ornith 1.5, Qwen3.x):** шаблон чата открывает `<think>` на каждом ходу. На задачах, где модель уходит в длинные размышления, весь бюджет токенов может уйти в них — ответ не успеет начаться. Лечится флагом `--reasoning-effort none`.

---

## REST API

Базовый URL: `http://127.0.0.1:8080` — **аутентификация не требуется**.

### Управление моделями

```bash
# Список локально закэшированных моделей (формат OpenAI)
curl http://127.0.0.1:8080/v1/models

# Текущая модель и готовность
curl http://127.0.0.1:8080/v1/models/current

# Горячая замена модели на лету (блокируется до загрузки)
curl -s http://127.0.0.1:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-31b-it-4bit"}'
```

`GET /v1/models` возвращает ID с префиксом `claude-local/` — иначе автоподбор моделей в Claude Code их не подхватит. Оригинальный repo ID с HuggingFace лежит в поле `display_name`.

> Обращайтесь по `127.0.0.1`, а не по `localhost`: сервер слушает только IPv4, а `localhost` во многих рантаймах сначала резолвится в `::1`.

### Сырая генерация

```bash
curl -s http://127.0.0.1:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Объясни квантовую запутанность.", "maxTokens": 256}'
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `prompt` | string | обязательное | Входной текст |
| `maxTokens` | int | 8192 | Максимум токенов генерации |
| `temperature` | float | 0.7 | Температура сэмплирования (0–2) |
| `topP` | float | 0.9 | Nucleus sampling |

### Совместимость с OpenAI

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemm",
    "messages": [{"role": "user", "content": "Привет"}],
    "stream": true
  }'
```

Поддерживаются потоковый SSE, системные промпты и многоходовые диалоги. Если в поле `"model"` передать HuggingFace ID, произойдёт горячая замена модели.

### Совместимость с Anthropic

```bash
curl -s http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemm",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Привет"}]
  }'
```

Реализована полная последовательность SSE-событий Anthropic (`message_start`, `content_block_start`, `content_block_delta`, `message_stop`). Принимается и строка, и массив блоков в `content`.

### WebSocket

```js
const ws = new WebSocket("ws://127.0.0.1:8080/ws/generate");
ws.send(JSON.stringify({ prompt: "Привет", maxTokens: 512 }));
ws.onmessage = e => console.log(JSON.parse(e.data));
```

### Swagger UI

Интерактивная документация — `http://127.0.0.1:8080/swagger`.

---

## Интеграция с Claude Code

### Вариант A — переменные окружения (на сессию терминала)

```bash
# Запускаем Gemm
.build/release/gemm serve --model mlx-community/Qwen3.5-4B-4bit --rest

# В другом терминале — переменные видит только этот процесс claude
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` уходит как `Authorization: Bearer local` (а не `x-api-key`), поэтому настоящий `ANTHROPIC_API_KEY` не задействуется.

### Вариант B — функция в `~/.zshrc`

```bash
function gemm-claude() {
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080          \
  ANTHROPIC_AUTH_TOKEN=local                        \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
  ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
  ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
  claude "$@"
}

gemm-claude                   # алиас sonnet → Qwen 4B
gemm-claude --model haiku     # алиас haiku → Gemma 4B (самая быстрая)
gemm-claude --model opus      # алиас opus → Gemma 31B (самая сильная)
```

### Вариант C — лаунчер `./Gemma`

Поднимает сервер и Claude Code одной командой (см. «Быстрый старт»).

### Обнаружение моделей

Claude Code (v2.1.126+) при старте дёргает `GET /v1/models` и добавляет модели в пикер `/model` — но только если ID начинается с `claude` или `anthropic`. Gemm отдаёт их в форме `claude-local/<hf-id>`, поэтому всё подхватывается само.

---

## Интеграция с OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) — терминальный агент для кода. Gemm отдаёт нативно совместимый OpenAI API (`/v1/chat/completions`, `/v1/models`), так что подключается напрямую, без облака и ключей.

### 1. Запустить Gemm

```bash
# Крупная модель (abliterated / MoE)
.build/release/gemm serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Или быстрая кодовая
.build/release/gemm serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Подключить OpenCode

Переменными окружения:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=gemm \
opencode
```

Или конфигом (`~/.config/opencode/config.json`):

```json
{
  "provider": "openai",
  "base_url": "http://127.0.0.1:8080/v1",
  "api_key": "local",
  "model": "gemm",
  "temperature": 0.7,
  "max_tokens": 16384
}
```

---

## Скачивание моделей

Gemm умеет искать, показывать и скачивать модели любого автора на Hugging Face:

```bash
# Список моделей по автору
gemm models list --author ornith-ai
gemm models list --author Ex0bit
gemm models list --author mlx-community

# Поиск с фильтром
gemm models list --author Ex0bit --search PRISM

# Скачать конкретную модель
gemm models download ornith-ai/Ornith-1.5-9B-MLX-4bit
gemm models download AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit

# Интерактивный выбор по автору
gemm models download --author Ex0bit
```

---

## MCP (Claude Desktop / Cursor)

Добавьте в конфиг MCP (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gemm": {
      "command": "/path/to/gemm",
      "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--mcp"]
    }
  }
}
```

Доступные MCP-инструменты:

| Инструмент | Описание |
|---|---|
| `gemma_generate` | Генерация текста — `prompt`, `maxTokens`, `temperature`, `topP` |
| `gemma_status` | Готовность сервера, версия, текущая модель |
| `playwright_screenshot` | Скриншот страницы через Playwright |
| `gemma_add_knowledge` | Подмешать свой контекст в сессию |

---

## Структура проекта

```
Sources/
  Gem/                      — библиотека GemCore (вся логика, импортируется тестами)
    App/                    — точка входа: корневая CLI-команда и роутинг
    CLI/                    — подкоманды: chat, serve, models, fit, cloud, onboard
    Chat/                   — ModelProfile: всё, что отличается между семействами моделей
    Cloud/                  — облачный фолбэк OpenRouter (CostTracker, ModelRouter)
    Config/                 — ServerConfig
    Core/                   — движок инференса, оркестратор, DTO, ошибки, утилиты
    MCP/                    — MCP JSON-RPC 2.0 поверх stdio
    REST/                   — HTTP-сервер на Hummingbird 2.x
    UI/                     — терминальный UI: Markdown, спиннер, прогресс-бар, diff, таблицы
  GemBin/                   — тонкая обёртка-исполняемый файл
  PerformanceBenchmark/     — отдельный CLI для бенчмарков

Tests/GemTests/             — юнит- и интеграционные тесты
Formula/gemm.rb             — формула Homebrew
Gemma                       — лаунчер (сборка + сервер + claude)
scripts/                    — скрипты сборки и обслуживания
docs/                       — расширенная документация
```

---

## Заметки по архитектуре

**Слой чата** — всё, что различается между семействами моделей, живёт за протоколом `ModelProfile` (`Sources/Gem/Chat/ModelProfile.swift`): переменные шаблона чата, маркеры thinking, дефолтные лимиты. Профиль выбирается по `model_type` из `config.json` модели. Добавить семейство — это одна новая конформность и одна строка в реестре.

**Динамическая квантизация и MoE** — `MLXInferenceEngine` читает `config.json`, определяет 2/3/4/8-битную квантизацию и переключается между Dense и Sparse MoE (оптимизированные ядра `SwitchGLU`).

**Изоляция через актор** — `ModelOrchestratorActor` это Swift 6 актор. Все вызовы инференса сериализуются (FIFO) без явных блокировок. MCP и REST делят один экземпляр.

**Горячая замена модели** — `switchModel(to:)` резолвит repo ID в локальном кэше, выгружает текущую модель (`container = nil` + `MLX.GPU.clearCache()`) и грузит новые веса. Актор дожидается завершения запросов в полёте.

**Защита по таймауту** — каждый `generate` и `generateStream` обёрнут в пятиминутный таймаут с кооперативной отменой Task, чтобы зависшая модель не блокировала сервер.

**Стриминг** — тело ответа Hummingbird 2 в виде `AsyncStream<ByteBuffer>` используется для обоих форматов SSE: OpenAI (`data: {...}`) и Anthropic (`event: content_block_delta`).

**Вырезание think-блоков** — в потоковом пути `generateStream` работает конечный автомат, который подавляет `<think>…</think>` до отправки клиенту. В непотоковом пути (`"stream": false`) фильтра нет — размышления приходят внутри `content`.

**Трансляция ID** — `ModelsController` переводит между HuggingFace ID (`mlx-community/Qwen3.5-4B-4bit`) и формой `claude-local/mlx-community--Qwen3.5-4B-4bit`, которую требует фильтр обнаружения моделей в Claude Code. В запросах принимаются обе формы.

---

## Как поучаствовать

Issues и pull request'ы приветствуются. Перед PR:

```bash
swift build          # должно собираться
swift test           # должно оставаться зелёным
```

Никогда не коммитьте креды, токены и состояние `*.sqlite3` — `.gitignore` закрывает обычных подозреваемых, а релизный workflow прогоняет скан секретов по дереву.

---

## Лицензия

[MIT](LICENSE) © 2026 goodmai

Веса моделей этой лицензией **не** покрываются — у каждой модели на HuggingFace свои условия.
