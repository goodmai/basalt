# Integrations

## Use it from your tools

basalt speaks three client protocols at once, so pick whatever your tool already uses.

| Client | Endpoint | Setup |
|---|---|---|
| Any OpenAI client (Python/JS SDK, Continue, Cline, aider) | `http://localhost:8080/v1` | base URL + any API key on loopback |
| OpenCode | `/v1/chat/completions` | see [OpenCode Integration](#opencode-integration) |
| Claude Code in the terminal | `/v1/messages` (Anthropic) | see [Claude Code Integration](#claude-code-integration) |
| Claude Desktop, Cursor | MCP stdio | see [MCP Integration](#mcp-integration-claude-desktop--cursor) |

Start one server for all of them, with the extras you want:

```bash
basalt serve --rest --model Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed --mtp \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rerank-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

The official OpenAI Python SDK works unchanged (checked with `openai` 3.19.2):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(model="basalt", messages=[{"role": "user", "content": "Hello"}])
vectors = client.embeddings.create(model="basalt", input=["first text", "second text"])
```

Rerank uses the Cohere/Jina shape:

```bash
curl -s http://localhost:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 1}'
```

## Claude Code Integration

### Option A — environment variables (per terminal session)

```bash
# Start Basalt
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` sends as `Authorization: Bearer local` (not `x-api-key`), so your real `ANTHROPIC_API_KEY` is never touched.

### Option B — shell function (add to `~/.zshrc`)

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

### Option C — `./basalt-claude` launcher

Starts the server and Claude Code in one command (see [the launcher in Install](install.md#one-command-launcher-basalt-claude)).

### Tool calling

Claude Code's tools work on `/v1/messages`: the `tools` it sends reach the model's chat template, the model's calls come back as `tool_use` blocks (streamed as `content_block_start` / `input_json_delta` / `content_block_stop`, `stop_reason: "tool_use"`), and the `tool_use` / `tool_result` blocks of the history are rendered into the prompt, so the model sees what its tools returned. `/v1/chat/completions` does the same in the OpenAI form (`tools`, `message.tool_calls` / `delta.tool_calls`, `finish_reason: "tool_calls"`, role `tool` messages), as `/v1/responses` already did. Tool results in the history are matched to their calls by `tool_use_id` / `tool_call_id`, so parallel calls keep their names; images in a tool result show as `[image omitted]`. `tool_choice: none` keeps the tools in the prompt and lets no call through; a named tool keeps every tool in the prompt and lets only that one through (naming a tool that is not in `tools` is a 400); `any` / `required` are not enforced and work as `auto` (see #42); `parallel_tool_calls: false` / `disable_parallel_tool_use: true` end the generation after the first call. Tool input that is not a JSON object goes out as `{}` with the original arguments in a visible note. A call the model botches beyond repair, or a call to a tool that was not offered, arrives as a visible `[tool call … rejected: …]` note in the text. The legacy Chat Completions `functions` / `function_call` fields are ignored. Checked with Claude Code 2.1 on `mlx-community/Qwen3-4B-4bit`: list a directory with `Bash`, and read a file then edit it with `Read` / `Edit`.

### Model discovery

Claude Code (v2.1.126+) queries `GET /v1/models` at startup and adds returned models to the `/model` picker — but only if the ID starts with `claude` or `anthropic`. Basalt returns IDs in `claude-local/<hf-id>` form so discovery works automatically.

---

## OpenCode Integration

[OpenCode](https://github.com/opencode-ai/opencode) is a terminal-native AI coding agent. Basalt provides a native OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) and Anthropic-compatible API that connects directly to OpenCode without cloud dependencies or API keys.

### 1. Start Basalt Server

Start Basalt with your desired local model (e.g. Huihui Qwen3.8 Abliterated, Ex0bit MYTHOS 26B, or Qwen3.5 4B):

```bash
# High capability (Abliterated / MoE)
"$(swift build -c release --show-bin-path)/Basalt" serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Connect OpenCode

#### Option A — Terminal Environment Variables

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### Option B — OpenCode Configuration File (`~/.config/opencode/config.json` or `opencode.json`)

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

#### Option C — Shell Alias (add to `~/.zshrc`)

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

## MCP Integration (Claude Desktop / Cursor)

Add to your MCP config (`~/.config/claude/claude_desktop_config.json`):

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

Available MCP tools:

| Tool | Description |
|---|---|
| `gemma_generate` | Generate text — `prompt`, `maxTokens`, `temperature`, `topP`, `minP`, `dynamicTemperature`, `images` (see [Image input](rest-api.md#image-input)) |
| `gemma_status` | Returns server readiness, version, and current model |
| `status` | Engine health, loaded model, memory snapshot and telemetry — optional `modelId` to check |
| `check_hardware` | Apple Silicon resources: chip, unified memory, GPU, disk |
| `inspect_model` | `modelId` (repo ID or local path) — verify weight shards and detect the real bit width of MLX/Safetensors weights |
| `switch_model` | Switch to another model from the local cache — `modelId` |
| `gemma_add_knowledge` | Store a lesson in the Harness Experience Library (`~/.basalt/harness-library`) |
| `harness_validate` | Validate a Claude Code / opencode / AGENTS.md harness under the working directory or `BASALT_HARNESS_ROOTS` |
| `harness_optimize` | One Harness² propose–probe–compose pass on the loaded model (also `basalt harness optimize`) |
| `harness_library_query` | Search the Experience Library |
| `model_describe`, `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run` | Model X-ray (read-only, background priority): layer anatomy, parameter list and bounded tensor statistics, the prompt-cache tree, one capture run of a prompt (#328) |
| `intervene`, `weights_delta`, `overlays_list`, `overlay_undo`, `experiment_compare` | Sandboxed experiments, only with `serve --research-tools`: steering and head/expert ablation as undoable overlays applied to research calls only; weight deltas change the live model, so while one is active every response is marked as modified and prompt-cache reuse is off; see [research-tools.md](research-tools.md) (#328) |
