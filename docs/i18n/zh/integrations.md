# 集成指南

> Languages: [English](../../integrations.md) · [Русский](../ru/integrations.md) · **中文** · [Español](../es/integrations.md) · [Français](../fr/integrations.md) · [日本語](../ja/integrations.md) · [Deutsch](../de/integrations.md) · [한국어](../ko/integrations.md) · [Bahasa Indonesia](../id/integrations.md)

## 在现有工具中使用

basalt 可同时兼容三种客户端协议，直接使用你现有工具所支持的协议即可。

| 客户端 | 端点 | 配置方式 |
|---|---|---|
| 任意 OpenAI 客户端（Python/JS SDK、Continue、Cline、aider） | `http://localhost:8080/v1` | 配置 Base URL，在回环地址上填写任意 API Key 即可 |
| OpenCode | `/v1/chat/completions` | 参见 [OpenCode 集成](#opencode-集成) |
| 终端中的 Claude Code | `/v1/messages`（Anthropic） | 参见 [Claude Code 集成](#claude-code-集成) |
| Claude Desktop、Cursor | MCP stdio | 参见 [MCP 集成](#mcp-集成claude-desktop--cursor) |

只需启动一个服务端实例，即可同时满足上述所有客户端及附加能力：

```bash
basalt serve --rest --model Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed --mtp \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rerank-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

官方 OpenAI Python SDK 无需任何修改即可直接使用（已在 `openai` 3.19.2 上验证）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(model="basalt", messages=[{"role": "user", "content": "Hello"}])
vectors = client.embeddings.create(model="basalt", input=["first text", "second text"])
```

Rerank 重排序接口采用 Cohere/Jina 请求格式：

```bash
curl -s http://localhost:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 1}'
```

## Claude Code 集成

### 方案 A —— 环境变量（按终端会话生效）

```bash
# Start Basalt
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` 会以 `Authorization: Bearer local`（而非 `x-api-key`）的形式发送，因此不会触及或覆盖你真实的 `ANTHROPIC_API_KEY`。

### 方案 B —— Shell 函数（添加到 `~/.zshrc`）

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

### 方案 C —— `./basalt-claude` 启动器

一条命令同时启动服务端与 Claude Code（参见[安装指南中的启动器说明](install.md#one-command-launcher-basalt-claude)）。

### 工具调用（Tool calling）

Claude Code 的工具在 `/v1/messages` 上开箱即用：客户端发送的 `tools` 会传入模型的对话模板，模型发起的调用会以 `tool_use` 块形式返回（流式传输为 `content_block_start` / `input_json_delta` / `content_block_stop`，结束原因为 `stop_reason: "tool_use"`），历史记录中的 `tool_use` / `tool_result` 块也会被完整渲染回提示词中，使模型能看到工具的返回结果。`/v1/chat/completions` 同样以 OpenAI 格式提供完整支持（`tools`、`message.tool_calls` / `delta.tool_calls`、`finish_reason: "tool_calls"`、`tool` 角色消息），与 `/v1/responses` 一致。历史中的工具结果通过 `tool_use_id` / `tool_call_id` 与对应的调用关联，因此并行工具调用也能准确保留各自的名称；工具结果中的图像会显示为 `[image omitted]`。`tool_choice: none` 会在提示词中保留工具定义但禁止模型发起任何调用；指定具体工具名会在提示词中保留所有工具但仅允许调用该工具（若指定的工具不在 `tools` 列表中则返回 400）；`any` / `required` 暂不作强制约束，行为等同于 `auto`（参见 #42）；`parallel_tool_calls: false` / `disable_parallel_tool_use: true` 会在第一次工具调用后立即结束本轮生成。若工具参数不是合法 JSON 对象，则会以 `{}` 输出并在可见备注中附带原始参数内容。若模型生成的调用格式损坏到无法修复，或试图调用未提供的工具，文本中会输出可见的 `[tool call … rejected: …]` 提示。旧版 Chat Completions 的 `functions` / `function_call` 字段会被忽略。已在 Claude Code 2.1 配合 `mlx-community/Qwen3-4B-4bit` 上实测验证：使用 `Bash` 列出目录，并用 `Read` / `Edit` 读取并修改文件。

### 模型发现

Claude Code（v2.1.126+）在启动时会请求 `GET /v1/models`，并将返回的模型添加到 `/model` 选择器中——但前提是模型 ID 必须以 `claude` 或 `anthropic` 开头。Basalt 会以 `claude-local/<hf-id>` 的格式返回模型 ID，从而自动适配模型发现。

---

## OpenCode 集成

[OpenCode](https://github.com/opencode-ai/opencode) 是一款终端原生的 AI 编程智能体。Basalt 提供原生兼容 OpenAI（`/v1/chat/completions`、`/v1/models`）与 Anthropic 的 API，可直接接入 OpenCode，无需云端依赖或 API 密钥。

### 1. 启动 Basalt 服务

使用你期望的本地模型（例如 Huihui Qwen3.8 Abliterated、Ex0bit MYTHOS 26B 或 Qwen3.5 4B）启动 Basalt：

```bash
# High capability (Abliterated / MoE)
"$(swift build -c release --show-bin-path)/Basalt" serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. 连接 OpenCode

#### 方案 A —— 终端环境变量

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### 方案 B —— OpenCode 配置文件（`~/.config/opencode/config.json` 或 `opencode.json`）

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

#### 方案 C —— Shell 别名（添加到 `~/.zshrc`）

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

## MCP 集成（Claude Desktop / Cursor）

在你的 MCP 配置文件（`~/.config/claude/claude_desktop_config.json`）中添加以下配置：

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

可用的 MCP 工具列表：

| 工具 | 说明 |
|---|---|
| `gemma_generate` | 文本生成 —— 支持 `prompt`、`maxTokens`、`temperature`、`topP`、`minP`、`dynamicTemperature`、`images`（参见[图像输入](rest-api.md#图像输入)） |
| `gemma_status` | 返回服务就绪状态、版本号与当前已加载的模型 |
| `status` | 引擎健康状态、已加载模型、内存快照与遥测数据 —— 可传入可选参数 `modelId` 进行检查 |
| `check_hardware` | 查询 Apple Silicon 硬件资源：芯片型号、统一内存、GPU 与磁盘空间 |
| `inspect_model` | 传入 `modelId`（仓库 ID 或本地路径）—— 校验权重分片并检测 MLX/Safetensors 权重的真实位宽 |
| `switch_model` | 切换到本地缓存中的另一个模型 —— 需指定 `modelId` |
| `gemma_add_knowledge` | 向 Harness 经验库（`~/.basalt/harness-library`）存入一条经验记录 |
| `harness_validate` | 校验当前工作目录或 `BASALT_HARNESS_ROOTS` 下的 Claude Code / opencode / AGENTS.md harness 配置 |
| `harness_optimize` | 在当前已加载模型上执行一轮 Harness² propose–probe–compose 优化流程（等同于 `basalt harness optimize`） |
| `harness_library_query` | 检索 Harness 经验库 |
| `model_describe`, `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run` | 模型 X-ray（只读，后台优先级）：查看层结构解剖、参数列表与有界张量统计信息、提示词缓存树，以及单次提示词捕获追踪运行（#328） |
| `intervene`, `weights_delta`, `overlays_list`, `overlay_undo`, `experiment_compare` | 沙盒实验工具，仅在指定 `serve --research-tools` 时可用：作为可撤销的覆盖层（overlay）对研究调用施加激活引导（steering）和注意力头/专家消融；权重增量（weight deltas）会修改线上模型，因此生效期间所有响应都会被标记为已修改且关闭提示词缓存复用；详见 [research-tools.md](research-tools.md)（#328） |
