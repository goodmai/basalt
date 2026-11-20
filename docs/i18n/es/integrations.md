# Integraciones

> Languages: [English](../../integrations.md) · [Русский](../ru/integrations.md) · [中文](../zh/integrations.md) · **Español** · [Français](../fr/integrations.md) · [日本語](../ja/integrations.md) · [Deutsch](../de/integrations.md)

## Úsalo desde tus herramientas

basalt habla tres protocolos de cliente a la vez, así que elige el que ya use tu herramienta.

| Cliente | Endpoint | Configuración |
|---|---|---|
| Cualquier cliente OpenAI (SDK Python/JS, Continue, Cline, aider) | `http://localhost:8080/v1` | URL base + cualquier clave API en loopback |
| OpenCode | `/v1/chat/completions` | ver [Integración con OpenCode](#integración-con-opencode) |
| Claude Code en la terminal | `/v1/messages` (Anthropic) | ver [Integración con Claude Code](#integración-con-claude-code) |
| Claude Desktop, Cursor | MCP stdio | ver [Integración con MCP](#integración-con-mcp-claude-desktop--cursor) |

Inicia un solo servidor para todos ellos, con los extras que quieras:

```bash
basalt serve --rest --model Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed --mtp \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rerank-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

El SDK oficial de OpenAI en Python funciona sin cambios (comprobado con `openai` 3.19.2):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(model="basalt", messages=[{"role": "user", "content": "Hello"}])
vectors = client.embeddings.create(model="basalt", input=["first text", "second text"])
```

Rerank usa el formato de Cohere/Jina:

```bash
curl -s http://localhost:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 1}'
```

## Integración con Claude Code

### Opción A — variables de entorno (por sesión de terminal)

```bash
# Start Basalt
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` se envía como `Authorization: Bearer local` (no `x-api-key`), de modo que tu `ANTHROPIC_API_KEY` real nunca se toca.

### Opción B — función de shell (añadir a `~/.zshrc`)

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

### Opción C — lanzador `./basalt-claude`

Inicia el servidor y Claude Code en un solo comando (ver [el lanzador en Instalación](install.md#one-command-launcher-basalt-claude)).

### Llamadas a herramientas

Las herramientas de Claude Code funcionan en `/v1/messages`: las `tools` que envía llegan a la plantilla de chat del modelo, las llamadas del modelo regresan como bloques `tool_use` (emitidos en stream como `content_block_start` / `input_json_delta` / `content_block_stop`, `stop_reason: "tool_use"`), y los bloques `tool_use` / `tool_result` del historial se renderizan en el prompt, de modo que el modelo ve lo que devolvieron sus herramientas. `/v1/chat/completions` hace lo mismo en formato OpenAI (`tools`, `message.tool_calls` / `delta.tool_calls`, `finish_reason: "tool_calls"`, mensajes con rol `tool`), como ya hacía `/v1/responses`. Los resultados de herramientas en el historial se emparejan con sus llamadas mediante `tool_use_id` / `tool_call_id`, así que las llamadas en paralelo conservan sus nombres; las imágenes en un resultado de herramienta se muestran como `[image omitted]`. `tool_choice: none` mantiene las herramientas en el prompt y no deja pasar ninguna llamada; una herramienta nombrada mantiene todas las herramientas en el prompt y solo deja pasar esa (nombrar una herramienta que no está en `tools` devuelve 400); `any` / `required` no se imponen y funcionan como `auto` (ver #42); `parallel_tool_calls: false` / `disable_parallel_tool_use: true` terminan la generación tras la primera llamada. Una entrada de herramienta que no sea un objeto JSON sale como `{}` con los argumentos originales en una nota visible. Una llamada que el modelo malforme de forma irreparable, o una llamada a una herramienta no ofrecida, llega como una nota visible `[tool call … rejected: …]` en el texto. Los campos heredados `functions` / `function_call` de Chat Completions se ignoran. Comprobado con Claude Code 2.1 sobre `mlx-community/Qwen3-4B-4bit`: listar un directorio con `Bash`, y leer un archivo y luego editarlo con `Read` / `Edit`.

### Descubrimiento de modelos

Claude Code (v2.1.126+) consulta `GET /v1/models` al arrancar y añade los modelos devueltos al selector `/model` — pero solo si el ID empieza por `claude` o `anthropic`. Basalt devuelve IDs en formato `claude-local/<hf-id>` para que el descubrimiento funcione automáticamente.

---

## Integración con OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) es un agente de programación con IA nativo de terminal. Basalt ofrece una API nativa compatible con OpenAI (`/v1/chat/completions`, `/v1/models`) y una API compatible con Anthropic que conectan directamente con OpenCode sin dependencias en la nube ni claves API.

### 1. Iniciar el servidor Basalt

Inicia Basalt con el modelo local que prefieras (p. ej. Huihui Qwen3.8 Abliterated, Ex0bit MYTHOS 26B o Qwen3.5 4B):

```bash
# High capability (Abliterated / MoE)
"$(swift build -c release --show-bin-path)/Basalt" serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Conectar OpenCode

#### Opción A — Variables de entorno en terminal

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### Opción B — Archivo de configuración de OpenCode (`~/.config/opencode/config.json` o `opencode.json`)

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

#### Opción C — Alias de shell (añadir a `~/.zshrc`)

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

## Integración con MCP (Claude Desktop / Cursor)

Añádelo a tu configuración MCP (`~/.config/claude/claude_desktop_config.json`):

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

Herramientas MCP disponibles:

| Herramienta | Descripción |
|---|---|
| `gemma_generate` | Genera texto — `prompt`, `maxTokens`, `temperature`, `topP`, `minP`, `dynamicTemperature`, `images` (ver [Entrada de imágenes](rest-api.md#entrada-de-imágenes)) |
| `gemma_status` | Devuelve el estado de disponibilidad del servidor, la versión y el modelo actual |
| `status` | Salud del motor, modelo cargado, instantánea de memoria y telemetría — `modelId` opcional para comprobar |
| `check_hardware` | Recursos de Apple Silicon: chip, memoria unificada, GPU, disco |
| `inspect_model` | `modelId` (ID de repo o ruta local) — verifica los fragmentos de pesos y detecta el ancho de bits real de pesos MLX/Safetensors |
| `switch_model` | Cambia a otro modelo de la caché local — `modelId` |
| `gemma_add_knowledge` | Guarda una lección en la Biblioteca de Experiencia del Harness (`~/.basalt/harness-library`) |
| `harness_validate` | Valida un harness de Claude Code / opencode / AGENTS.md bajo el directorio de trabajo o `BASALT_HARNESS_ROOTS` |
| `harness_optimize` | Una pasada propose–probe–compose de Harness² sobre el modelo cargado (también `basalt harness optimize`) |
| `harness_library_query` | Busca en la Biblioteca de Experiencia |
| `model_describe`, `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run` | Rayos X del modelo (solo lectura, prioridad en segundo plano): anatomía de capas, lista de parámetros y estadísticas acotadas de tensores, el árbol de prompt-cache, una ejecución de captura de un prompt (#328) |
| `intervene`, `weights_delta`, `overlays_list`, `overlay_undo`, `experiment_compare` | Experimentos en sandbox, solo con `serve --research-tools`: steering y ablación de cabezas/expertos como superposiciones reversibles aplicadas únicamente a llamadas de investigación; los deltas de pesos modifican el modelo en vivo, de modo que mientras haya uno activo cada respuesta queda marcada como modificada y la reutilización de prompt-cache se desactiva; ver [research-tools.md](research-tools.md) (#328) |
