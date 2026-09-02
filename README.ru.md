# Gemm

[English](README.md) · [Русский](README.ru.md)

[![Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

[![Platform](https://img.shields.io/badge/platform-macOS%2015%2B%20·%20Apple%20Silicon-black.svg)](#требования)
[![Swift 6](https://img.shields.io/badge/Swift-6-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fgemm-black.svg)](#установка-через-homebrew)
[![Ladder](https://img.shields.io/badge/лестница-6%20задач-black.svg)](#бенчмарки)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](#бенчмарки)

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
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` — в Xcode 26 компилятор Metal ставится отдельно, без него ядра MLX не собрать |
| **Железо** | Apple Silicon M1–M4, Unified Memory |
| **Диск** | 2–30 ГБ в зависимости от модели |

---

## Быстрый старт

### Установка через Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/gemm
```

URL в `tap` указывается явно, потому что репозиторий называется не
`homebrew-basalt`. Бутылки нет: формула собирает из исходников, поэтому нужен
Xcode 16+ и Metal toolchain — в Xcode 26 он качается отдельно:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Формула проверяет его наличие до сборки и останавливается с этой командой, а не
заваливает вас ошибками компиляции по каждому ядру.

Сборка делает две вещи, и важны обе:

1. `swift build -c release` — сам сервер.
2. `scripts/build_metal.swift` — компилирует Metal-ядра MLX и кладёт
   `mlx.metallib` рядом с бинарём в `libexec`. SwiftPM-сборка mlx-swift не
   содержит metallib, а запасной путь поиска у MLX считается от рабочей
   директории — без этого шага `gemm` работает только из папки, где лежит
   библиотека. Этот шаг и занимает основное время установки.

```bash
brew install --HEAD goodmai/basalt/gemm   # собрать main, а не последний тег
brew upgrade goodmai/basalt/gemm
brew uninstall gemm && brew untap goodmai/basalt
```

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

## Команды

`gemm` без подкоманды запускает `chat`.

| Команда | Что делает |
|---|---|
| `gemm onboard` | мастер первого запуска: подбирает модель под машину и качает её |
| `gemm fit` | читает железо и ранжирует каталог моделей под него |
| `gemm chat --model <id>` | интерактивный чат в терминале |
| `gemm serve --model <id> --rest` | REST-сервер на :8080 (совместим с OpenAI и Anthropic) |
| `gemm serve --model <id> --mcp` | MCP stdio сервер для Claude Desktop / Cursor |
| `gemm models list --author <org>` | посмотреть модели автора на HuggingFace |
| `gemm models download <repo-id>` | скачать в общий кэш HF |
| `gemm models info <repo-id>` | размер, квантизация, окно контекста |
| `gemm models cache` | что лежит на диске и сколько занимает |
| `gemm models check` | проверить, что модель скачана полностью и грузится |
| `gemm cloud configure` | облачный фолбэк OpenRouter для того, что не влезает локально |
| `gemm cloud cost` | сколько уже потрачено на облако |

Внутри `gemm chat`:

| Ввод | Действие |
|---|---|
| `/clear` | очистить диалог и экран |
| `/color`, `/theme` | переключить цветовую тему терминала |
| `exit`, `quit` | выйти (без слеша) |

Флаги `serve`, которые стоит знать:

| Флаг | Зачем нужен |
|---|---|
| `--reasoning-effort none` | не дать reasoning-модели потратить весь бюджет внутри `<think>` |
| `--reasoning-effort xhigh\|medium\|low` | бюджет размышлений для семейства Qwen |
| `--quant 4bit` | выбрать подпапку с квантизацией в репозиториях, где их несколько |
| `--kv-bits 4\|8` | квантовать KV-кэш — покупает контекст на машине с тесной памятью |
| `--min-p`, `--top-k`, `--seed` | сэмплирование; `--seed` делает негреди-прогон воспроизводимым |
| `--max-tokens` | потолок генерации по умолчанию (2048–128000) |
| `--dry-run` | проверить, влезает ли модель в память, и выйти без загрузки |
| `--port`, `--host` | по умолчанию 127.0.0.1:8080 |

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

## Бенчмарки

Два стенда, оба работают против запущенного сервера и дают числа, которые можно
воспроизвести у себя.

**Лестница до Луны** — `benchmarks/ladder/run.py`. Пять ступеней, каждая строго
тяжелее предыдущей, финал — полная миссия Земля → Луна. Оценивается не то, как
ответ выглядит: код модели исполняется на входах, которых не было в промпте,
поэтому правдоподобный, но неверный ответ всё равно падает.

| Ступень | Задача | Что реально проверяется |
|---|---|---|
| 1 | Биквадрат, вещественные корни | замкнутая форма плюс краевые случаи: пусто, кратный, ноль |
| 2 | Биквадрат, комплексные корни | работа с ветвями — все четыре корня, с кратностью |
| 3 | Коэффициенты Фурье | численное интегрирование: пила, меандр, косинус |
| 4 | Напор насоса | единицы (мм против м, л/с против м³/с) + Colebrook без замкнутой формы |
| 5 | **Миссия Земля → Луна** | скорость истечения из `sqrt(2ηQ)`, Циолковский по ступеням, задача четырёх тел, торможение у Луны, мягкая посадка — все константы даны в промпте, из справочника брать нечего |

Верхняя ступень оценивается по шести инвариантам, а не по одному ответу: модель,
которая верно взяла энергетику, но не довела интегрирование траектории, всё
равно получает число. Её промпт лежит в `benchmarks/ladder/lunar_prompt.md`,
эталонные значения в `lunar_task.md` модели не показываются.

`--warmup` добавляет перед лестницей три дешёвых дымовых теста (алгебра,
Фибоначчи, перевод), которые обязана брать любая instruct-модель. В счёт
лестницы они не идут.

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`, официальные правила
pass@2. Корпус не вендорится: сначала положите его в
`benchmarks/arc-agi/dataset/` (см. `benchmarks/arc-agi/ARC_AGI.md`).

### Результаты

Замеры на Mac с 24 ГБ unified memory. `thinking` — это блок `<think>` из шаблона
чата: у reasoning-моделей включён по умолчанию, отключается флагом
`--reasoning-effort none`.

| Модель | Разминка | Лестница | Верхняя ступень | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking выкл) | 3/3 | 1/4 ² | не запускалась | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking вкл) | 2/3 | 0/4 ² | не запускалась | не запускался | ~43 |

Ornith 9B по ступеням, thinking выключен:

| Ступень | Итог | Детали |
|---|---|---|
| 1 · биквадрат вещественный | ✅ | корни верные, включая кратный и пустой случай |
| 2 · биквадрат комплексный | ❌ | для `x⁴ + 1` вернула 2 корня вместо 4 |
| 3 · Фурье | ❌ | `a0 = 10.86` для `f(x) = x` вместо 0 — неверно считает интеграл, формула правильная |
| 4 · напор насоса | ❌ | `v = 0.0298` м/с вместо 1.79 — не перевёл мм в метры |
| 5 · лунная миссия | — | не запускалась |

Разминка, thinking выключен: алгебра ✅ (163 tok, 4 с), Фибоначчи ✅ (757 tok,
16 с), перевод ✅ (303 tok, 7 с — немецкий идиоматичный, французский пересказан,
а не переведён).

² Ступени 1 и 2 замерялись до того, как лестницу разделили на отдельные
промпты: один промпт просил обе функции, вещественная половина оказалась верной,
комплексная — нет. Ступень 5 на этой модели ещё не запускалась.

¹ Колонка ARC-AGI — неполный прогон: три задачи из training (`007bbfb7`,
`00d62c1b`, `017c7c7b`), ни одна не решена по pass@2, 330 с суммарно. Оставшиеся
две не запускались, так что это дымовой тест, а не оценка.

Сами валидаторы тоже проверяются — без модели и GPU:

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 проверок в обе стороны
```

С **включённым** thinking все три тяжёлые задачи и перевод закончились
`finish_reason: length` — бюджет ушёл в блок `<think>`, до ответа дело не дошло.
Это отказ по бюджету, а не потолок способностей, поэтому в таблице два режима
разведены.

### Как воспроизвести

```bash
gemm serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

Добавить модель в таблицу — это один прогон каждого стенда. PR с новыми
строками приветствуются.

---

## Зависимости

Всё — пакеты SwiftPM; ни CocoaPods, ни Node, ни Python в рантайме.
`Package.resolved` фиксирует точные ревизии.

| Пакет | Для чего |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | слой массивов и NN на Metal |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | архитектуры моделей и загрузка — `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | токенизаторы и шаблоны чата |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | HTTP-сервер |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | CLI |
| [Rainbow](https://github.com/onevcat/Rainbow), [console-kit](https://github.com/vapor/console-kit) | цвет и вёрстка в терминале |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | рендер Markdown и подсветка синтаксиса |

`MLXVLM` здесь не опционален: Gemma 4 и чекпойнты Ornith объявляют
`*ForConditionalGeneration`, а эти архитектуры зарегистрированы именно там, а не
в `MLXLLM`.

Обновления приходят PR-ами от Dependabot — еженедельно для SwiftPM, ежемесячно
для GitHub Actions. mlx-swift развивается быстро, и нужная архитектура модели
часто оказывается в одном релизе от нас — отсюда недельный ритм.

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
