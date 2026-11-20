# Интеграции

> Languages: [English](../../integrations.md) · **Русский** · [中文](../zh/integrations.md) · [Español](../es/integrations.md) · [Français](../fr/integrations.md) · [日本語](../ja/integrations.md) · [Deutsch](../de/integrations.md)

## Использование в ваших инструментах

basalt говорит сразу на трёх клиентских протоколах, так что выбирайте тот, которым уже пользуется ваш инструмент.

| Клиент | Эндпоинт | Настройка |
|---|---|---|
| Любой клиент OpenAI (Python/JS SDK, Continue, Cline, aider) | `http://localhost:8080/v1` | базовый URL + любой API-ключ на loopback |
| OpenCode | `/v1/chat/completions` | см. [Интеграция с OpenCode](#интеграция-с-opencode) |
| Claude Code в терминале | `/v1/messages` (Anthropic) | см. [Интеграция с Claude Code](#интеграция-с-claude-code) |
| Claude Desktop, Cursor | MCP stdio | см. [Интеграция по MCP](#интеграция-по-mcp-claude-desktop--cursor) |

Запустите один сервер для всех сразу с нужными дополнениями:

```bash
basalt serve --rest --model Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed --mtp \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rerank-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

Официальный Python SDK от OpenAI работает без изменений (проверено с `openai` 3.19.2):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(model="basalt", messages=[{"role": "user", "content": "Hello"}])
vectors = client.embeddings.create(model="basalt", input=["first text", "second text"])
```

Ранжирование (rerank) использует формат Cohere/Jina:

```bash
curl -s http://localhost:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 1}'
```

## Интеграция с Claude Code

### Вариант A — переменные окружения (на один сеанс терминала)

```bash
# Start Basalt
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` передаётся как `Authorization: Bearer local` (а не `x-api-key`), поэтому ваш настоящий `ANTHROPIC_API_KEY` остаётся нетронутым.

### Вариант B — функция оболочки (добавьте в `~/.zshrc`)

```bash
function claude-local() {
  ANTHROPIC_BASE_URL=http://localhost:8080          \
  ANTHROPIC_AUTH_TOKEN=local                        \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
  ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
  ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
  claude "$@"
}

claude-local                   # sonnet alias → Qwen 4B
claude-local --model haiku     # haiku alias → Gemma 4B (fastest)
claude-local --model opus      # opus alias → Gemma 31B (strongest)
```

### Вариант C — лаунчер `./basalt-claude`

Запускает сервер и Claude Code одной командой (см. [описание лаунчера в руководстве по установке](install.md#one-command-launcher-basalt-claude)).

### Вызов инструментов (Tool calling)

Инструменты Claude Code работают через `/v1/messages`: переданные им `tools` попадают в шаблон чата модели, вызовы модели возвращаются как блоки `tool_use` (в потоковом режиме как `content_block_start` / `input_json_delta` / `content_block_stop`, `stop_reason: "tool_use"`), а блоки `tool_use` / `tool_result` из истории рендерятся в промпт, чтобы модель видела, что вернули её инструменты. `/v1/chat/completions` делает то же самое в формате OpenAI (`tools`, `message.tool_calls` / `delta.tool_calls`, `finish_reason: "tool_calls"`, сообщения с ролью `tool`), как уже делал `/v1/responses`. Результаты инструментов в истории сопоставляются со своими вызовами по `tool_use_id` / `tool_call_id`, так что параллельные вызовы сохраняют свои имена; изображения в результате инструмента отображаются как `[image omitted]`. `tool_choice: none` оставляет инструменты в промпте и не пропускает ни одного вызова; указанный по имени инструмент оставляет все инструменты в промпте и пропускает только этот (указание имени инструмента, отсутствующего в `tools`, возвращает 400); `any` / `required` не принуждаются и работают как `auto` (см. #42); `parallel_tool_calls: false` / `disable_parallel_tool_use: true` завершают генерацию после первого вызова. Входные данные инструмента, не являющиеся JSON-объектом, отправляются как `{}` с исходными аргументами в видимом примечании. Вызов, который модель испортила сверх возможности восстановления, или вызов инструмента, который не предлагался, приходит в тексте как видимое примечание `[tool call … rejected: …]`. Устаревшие поля Chat Completions `functions` / `function_call` игнорируются. Проверено с Claude Code 2.1 на `mlx-community/Qwen3-4B-4bit`: просмотр каталога через `Bash`, а также чтение и последующее редактирование файла через `Read` / `Edit`.

### Обнаружение моделей

Claude Code (v2.1.126+) запрашивает `GET /v1/models` при запуске и добавляет возвращённые модели в меню `/model` — но только если ID начинается с `claude` или `anthropic`. Basalt возвращает ID в форме `claude-local/<hf-id>`, чтобы обнаружение работало автоматически.

---

## Интеграция с OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) — нативный терминальный ИИ-агент для программирования. Basalt предоставляет нативный OpenAI-совместимый API (`/v1/chat/completions`, `/v1/models`) и Anthropic-совместимый API, подключающийся к OpenCode напрямую без облачных зависимостей и API-ключей.

### 1. Запуск сервера Basalt

Запустите Basalt с нужной вам локальной моделью (например, Huihui Qwen3.8 Abliterated, Ex0bit MYTHOS 26B или Qwen3.5 4B):

```bash
# High capability (Abliterated / MoE)
"$(swift build -c release --show-bin-path)/Basalt" serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Подключение OpenCode

#### Вариант A — переменные окружения в терминале

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### Вариант B — файл конфигурации OpenCode (`~/.config/opencode/config.json` или `opencode.json`)

```json
{
  "provider": "openai",
  "base_url": "http://localhost:8080/v1",
  "api_key": "local",
  "model": "basalt",
  "temperature": 0.7,
  "max_tokens": 16384
}
```

#### Вариант C — псевдоним оболочки (добавьте в `~/.zshrc`)

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

## Интеграция по MCP (Claude Desktop / Cursor)

Добавьте в ваш конфиг MCP (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "basalt": {
      "command": "/path/to/basalt",
      "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--mcp"]
    }
  }
}
```

Доступные MCP-инструменты:

| Инструмент | Описание |
|---|---|
| `gemma_generate` | Генерация текста — `prompt`, `maxTokens`, `temperature`, `topP`, `minP`, `dynamicTemperature`, `images` (см. [Ввод изображений](rest-api.md#ввод-изображений)) |
| `gemma_status` | Возвращает готовность сервера, версию и текущую модель |
| `status` | Состояние движка, загруженная модель, снимок памяти и телеметрия — опционально `modelId` для проверки |
| `check_hardware` | Ресурсы Apple Silicon: чип, объединённая память (unified memory), GPU, диск |
| `inspect_model` | `modelId` (ID репозитория или локальный путь) — проверка шардов весов и определение реальной битности весов MLX/Safetensors |
| `switch_model` | Переключение на другую модель из локального кэша — `modelId` |
| `gemma_add_knowledge` | Сохранить урок в библиотеку опыта Harness (`~/.basalt/harness-library`) |
| `harness_validate` | Проверить конфиг/обвязку Claude Code / opencode / AGENTS.md в рабочем каталоге или `BASALT_HARNESS_ROOTS` |
| `harness_optimize` | Один проход Harness² propose–probe–compose на загруженной модели (также `basalt harness optimize`) |
| `harness_library_query` | Поиск по библиотеке опыта |
| `model_describe`, `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run` | Model X-ray (только чтение, фоновый приоритет): анатомия слоёв, список параметров и ограниченная статистика тензоров, префиксное дерево prompt-кэша, один прогон трассировки промпта (#328) |
| `intervene`, `weights_delta`, `overlays_list`, `overlay_undo`, `experiment_compare` | Эксперименты в песочнице, только с `serve --research-tools`: steering и абляция голов/экспертов в виде обратимых оверлеев, применяемых только к исследовательским вызовам; дельты весов меняют живую модель, поэтому пока активна хотя бы одна дельта, каждый ответ помечается как модифицированный и переиспользование prompt-кэша отключается; см. [research-tools.md](research-tools.md) (#328) |
