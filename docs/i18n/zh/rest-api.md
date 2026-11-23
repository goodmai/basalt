# REST API

> Languages: [English](../../rest-api.md) · [Русский](../ru/rest-api.md) · **中文** · [Español](../es/rest-api.md) · [Français](../fr/rest-api.md) · [日本語](../ja/rest-api.md) · [Deutsch](../de/rest-api.md) · [한국어](../ko/rest-api.md) · [Bahasa Indonesia](../id/rest-api.md)

基础地址：`http://127.0.0.1:8080` —— **无需身份验证**。

## 安全性

在回环地址上，任何本地进程都无需密钥即可调用 API，但浏览器中的网页不可直接访问：

- **Host 校验。** 绑定到回环地址时，服务端仅响应 `Host` 为 `localhost`、`127.0.0.1` 或 `[::1]`（任意端口）的请求，其余请求一律返回 403；这可以防止 DNS 重绑定攻击（攻击者域名重新解析到 127.0.0.1 后通过页面读取本地响应）。绑定到特定 IP 地址时，该地址同样被接受。绑定到 `0.0.0.0`/`::` 时接受任何 `Host`，但此时强制要求提供 API 密钥。
- **Origin 校验。** 带有 `Origin` 请求头的请求将被拒绝（403），除非它属于服务端自身的回环 Origin（`http(s)://localhost|127.0.0.1|[::1]` 加上服务端监听端口；省略端口时 http 默认 80、https 默认 443），或已通过 `--allow-origin` / `allowed_origins` 列入白名单（格式为 `scheme://host[:port]`，不含路径、末尾斜杠或通配符 `*`；默认端口自动匹配，因此 `https://app.example:443` 等同于 `https://app.example`）。其他本地端口不受信任：否则开发服务器（如 `localhost:3000`）或 Jupyter 上的网页就能擅自加载模型并执行生成，因此这类前端页面需要配置 `--allow-origin http://localhost:3000`。不发送 `Origin` 的客户端（curl、各类 SDK、Claude Code、Codex）不受影响。
- **Content-Type 校验。** 所有 `POST` 请求都必须携带 `Content-Type: application/json`（允许附带 `charset` 参数），否则返回 415。使用 `curl -d` 但未指定 `-H "Content-Type: application/json"` 时会发送表单类型而被拒绝。
- **模型 ID 校验。** REST 接口中的 `model` 参数值均视为 Hugging Face 仓库 ID：`/v1/models/load` 仅接受 `org/name` 格式；在生成请求的 `model` 字段中，路径形式的值（以 `/`、`.` 或 `~` 开头，或包含 `..`，包括隐藏在 `claude-local/..--x` 中的情况；前后空白符会被忽略）将返回 400；而其他非仓库 ID 的值（如 `basalt`、`qwen/qwen3-8b-instruct:free`）会被视为别名：直接由当前已加载的模型响应，不会触发切换模型。本地路径仅对 `basalt serve --model-path` 和 MCP 保持可用。模型加载失败时仍保留当前已加载的模型，且错误响应体中绝不泄露文件系统路径（路径信息仅记录在服务端日志中）。

## 模型管理

```bash
# List all locally cached models (OpenAI-compatible format)
curl http://localhost:8080/v1/models

# Currently loaded model + readiness status
curl http://localhost:8080/v1/models/current

# Hot-swap to a different model at runtime (blocks until loaded)
curl -s http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-31b-it-4bit"}'
```

`GET /v1/models` 返回的模型 ID 带有 `claude-local/` 前缀，以便 Claude Code 的自动模型选择器能将其加入列表。`display_name` 字段则保留原始的 HuggingFace 仓库 ID。

## 原生文本生成

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement.", "maxTokens": 256}'
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | string | 必填 | 输入文本 |
| `maxTokens` | int | 8192 | 最多生成的 token 数量 |
| `temperature` | float | 模型默认值 | 采样温度（0–2）；未设置时取模型 `generation_config.json` 中的值，否则默认 0.7 |
| `topP` | float | 模型默认值 | 核采样阈值（0–1）；未设置时取 `generation_config.json`，否则默认 0.9 |
| `topK` | int | 模型默认值 | 仅保留概率最高的 k 个 token；未设置时依次取 `--top-k`、`generation_config.json`，否则关闭 |
| `presencePenalty` | float | 关闭 | 取值 −2…2，对最近生成的 1024 个 token 中出现过的每个 token 施加减法惩罚（不对提示词部分惩罚） |
| `repetitionPenalty` | float | `--repetition-penalty` | 取值 (0, 2]，对近期出现过的 token 施加乘法惩罚 |
| `thinkingBudget` | int | 关闭 | 在服务端强制结束思考之前允许输出的推理 token 数（Qwen 模型会注入 `"Considering the limited time by the user…" + </think>`），以确保最终回答能容纳在 `maxTokens` 内；该值会自动下调以留出过渡文本和 `min(512, maxTokens/4)` 的回答预留空间；若扣除预留空间后 `maxTokens` 剩余不足 64 个 token 则返回 400；当关闭思考、模型无推理协议或非 Qwen 模型的协议未声明过渡文本时该字段被忽略 |
| `minP` | float | 服务端默认值 | Min-P（0–1）：丢弃概率低于 `minP × max p` 的 token。覆盖 `--min-p` 和模型配置；设为 `0` 可关闭 |
| `dynamicTemperature` | bool | false | 基于熵的动态温度（EDT）：逐 token 按 `T = T0·0.8^(1.5/H) + 0.5·R` 计算，并截断至 0.05–2（温度更低的请求会以自身温度作为下限；`top_p`/`min_p` 作用于原始分布），其中 `H` 为下一 token 分布的熵，`R ∈ [0,1]` 为重复指数。对高置信度 token 接近贪心采样，出现循环重复时提高温度。当 `temperature: 0` 时无效 |
| `speculativeLookup` | bool | `--speculative-lookup` | 为当前请求启用提示词查找投机解码（Prompt-lookup speculative decoding）：从提示词或已生成输出中的 n-gram 匹配位置复制最多 16 个 token，在单次前向传播中验证；草稿被拒绝后会指数退避暂停起草 1、2、4… 直至 64 个 token。配合 `serve --speculative-recycling` 时，若无 n-gram 匹配，则从模型自身连续两次记录的 argmax 后继中起草最多 6 个 token（Token Recycling，#38），并具有独立的暂停计数器。支持纯注意力模型、稠密循环混合模型（Qwen3.5/Ornith、Nemotron-H：草稿被拒时恢复循环状态并重放已接受的 token）以及滑动窗口模型（Gemma 4：每轮验证均暂存，被拒的草稿绝不会覆写窗口环形缓存），前提是未启用 `--kv-bits` 且为纯文本输入；带专家路由的循环混合模型（LFM2.5 MoE）会忽略此选项。在 Gemma 4 26B-A4B（MoE）上，lookup 在代码编辑任务中提速 1.97×，而在散文与推理任务中慢 3–6% |

`/v1/chat/completions`、`/v1/responses` 和 `/v1/messages` 均支持同义字段 `min_p` 与 `dynamic_temperature`；MCP 工具 `gemma_generate` 支持 `minP` 与 `dynamicTemperature`。OpenAI 兼容端点支持 `top_k`、`presence_penalty`、`repetition_penalty` 和 `thinking_budget`；`/v1/messages` 支持 `top_k` 以及将 `thinking: {"type": "enabled", "budget_tokens": N}` 作为思考预算（要求 `1 ≤ N < max_tokens`，否则返回 400）。上述三个端点均支持 `speculative_lookup`（传入非布尔值返回 400）；MCP `gemma_generate` 支持 `speculativeLookup`（传入非布尔值返回 -32602）。用量统计中会返回 `completion_tokens_details.reasoning_tokens`（chat 端点；流式需配合 `stream_options.include_usage`）以及 `output_tokens_details.reasoning_tokens`（Responses 端点）：当模型未进行推理时为 `0`，强制注入的结束过渡文本计入推理 token。

## 兼容 OpenAI 协议

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

支持 SSE 流式输出、系统提示词与多轮对话。在 `"model"` 字段中传入 HuggingFace 模型 ID 会自动触发热切换。

推理模型的思考过程会同时以 `message.reasoning`（流式为 `delta.reasoning`，即 OpenRouter / Ollama 使用的字段名）以及 `reasoning_content`（vLLM、DeepSeek、llama.cpp 与 LM Studio 客户端读取的字段名）返回相同文本。在多轮历史中回传的助手消息可携带其中任意一个字段；思考内容本身不会被重新放回提示词中。

## 兼容 Anthropic 协议

```bash
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

完整实现 Anthropic SSE 事件序列（`message_start`、`content_block_start`、`content_block_delta`、`message_stop`），同时支持纯字符串和内容块数组两种消息格式。

`thinking` 块中的 `signature` 字段为空：basalt 没有用于签名的私钥。这些块可以在 basalt 内正常往返传递，但真实的 Anthropic API 会拒绝包含未签名思考块的对话历史，因此若将同一历史转发给官方 API，请先剔除这些块。

**受限输出（Constrained output）。** `response_format` 可在推理块结束之后，使用 XGrammar 对输出格式进行约束：
- `{"type": "json_object"}` 或 `{"type": "json_schema", "json_schema": {"schema": …}}`；
- `{"type": "regex", "regex": "[A-Z]{3}-\\d{4}"}`；
- `{"type": "grammar", "grammar": "root ::= …"}`，采用以 `root` 为根规则的 XGrammar EBNF/GBNF 方言；
- `{"type": "structural_tag", "format": …}`，XGrammar 的结构化标签 JSON（`sequence`、`const_string`、`regex`、`json_schema`、`tag` 等）。

也支持 vLLM 的 `guided_regex` 与 `guided_grammar` 请求字段（每次仅限使用其中一个）。MCP `generate` 工具支持 `regex` 或 `grammar` 参数。若规则源码超过 64 KB 或 XGrammar 无法编译，将返回包含具体格式名称的 400 错误。

**预填充续写（Prefill）。** 在 `/v1/messages`、`/v1/chat/completions` 和 `/v1/responses` 上，若对话以助手消息结尾，模型会直接续写该消息，而不是开启新的一轮回答（与 Anthropic API、vLLM 的 `continue_final_message` 及 llama.cpp 一致）。此前的对话轮次会在关闭思考的状态下渲染至生成提示符，随后紧接未闭合的助手文本，响应中仅包含模型在其后继续生成的内容：例如预填充 `{"name":`，回复即为 ` "Whiskers", "age": 3}`。当同时指定 `response_format` 或强制工具调用时，预填充内容将作为 JSON 文档的开头（若 schema 不允许以此开头则返回 400），强制工具调用的参数中也会包含该前缀。若预填充内容打开了思考块（`<think>…`），则会保留所请求的思考模式并作为推理过程继续生成。末尾连续的多条助手消息会合并为一段预填充；空消息会被丢弃；带有工具调用的助手消息仍视为已结束的轮次。包含图像的提示词不支持预填充续写（返回 400）。预填充文本中的控制 token（如 `<|im_end|>`、`<|im_start|>` 等）会与对话其余部分一样被无害化转义，因此预填充无法伪造对话轮次；唯有开启与闭合推理的思考标记会保留其控制语义，而在本身已由模板自动打开 `<think>` 的模型（Ornith、Qwen3.5+）上，预填充文本开头的 `<think>` 会被自动去重而非重复输出。启用 `--cloud-fallback` 时，带有预填充的请求始终留在本地处理，因为并非所有云端服务商都支持续写末尾助手消息。在 Nemotron Nano v2 上，关闭思考是在系统消息中添加 `/no_think` 行实现的，因此带预填充的请求与同一历史上开启思考的请求会渲染出不同的系统消息，两者仅能共享系统消息之前的提示词缓存。

## 图像输入

任何视觉模型（Gemma 4、Qwen3.5-VL 等——即所有 MLXVLM 可加载的模型）均可在 `/v1/chat/completions`、`/v1/responses`、`/v1/messages` 以及 MCP `generate` 工具中接收图像输入：

- **OpenAI：** 内容部分使用 `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`（`/v1/responses` 使用 `{"type": "input_image", "image_url": "data:…"}`）。
- **Anthropic：** 内容块使用 `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "…"}}`。
- **MCP：** 使用 `images: [{"data": "<base64>", "mime": "image/png"}]`，或传入通过 `serve --image-dir <dir>`（可重复指定）授权目录内的本地文件路径 `[{"path": "/abs/file.png"}]`。服务端会在解析符号链接后校验路径；若未指定 `--image-dir`，所有路径输入均会被拒绝。

支持 PNG、JPEG、WebP 与 HEIC 格式，每次请求最多 8 张图像，单张编码字节上限 20 MB，单次请求所有图像总计上限 32 MB；超出任一限制均返回明确指出对应限制的 400 错误。远程 URL（`image_url` 中的 `http(s)://` 或 `source.type: "url"`）会直接返回 400 拒绝——basalt 不会代为抓取远程图片；请改用 `data:` URI 传入。向非视觉模型发送图像会返回 400（`<model> does not take images`），绝不会静默丢弃，且包含图像的请求绝不会转交至 `--cloud-fallback`。提示词缓存会以每张图像的 SHA-256 摘要作为键：相同图像按相同顺序出现时可复用缓存前缀，图像不同则不复用。会话日志仅记录摘要与字节数，绝不记录图像原始字节。工具返回结果中包含的图像目前仍向模型展示为 `[image omitted]`。

## SSE 流式输出（原生 API）

```bash
curl -N http://localhost:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "maxTokens": 512}'
```

每条事件以 `data: {...}` 单行推送；流结束时发送 `data: [DONE]`。若流式传输中途发生错误，将发送 `event: error` 及对应的 `data: {"error": ...}`。

## 遥测与 Prometheus

- **`GET /api/v1/stats`** 以 JSON 格式返回当前遥测快照。
  - 其中包含 `windows` 字段：分别统计过去 1 分钟、5 分钟与 1 小时内的请求数、token 数、p50/p95 首字延迟（TTFT）与吞吐量（tok/s）、提示词缓存命中率以及投机草稿接受率。
  - 同时包含 `decisions` 字段：按决策模型统计请求数、回答数、拒答数、升级云端数、云端费用、延迟以及各置信度十分位上的回答数。
- 启用 `--experimental-expert-streaming` 时，还会包含 `experts` 字段：记录流式专家缓存的命中次数、未命中次数、淘汰次数与读取字节数，以及自模型加载以来各 MoE 层的专家路由统计（总次数、去重专家数、最常使用的 8 个专家及其占比）。
- 启用 `--telemetry-confidence-every N` 时，每隔 N 次生成会记录逐 token 的最高概率、熵以及前两名概率差（margin）。显式要求记录的请求始终会被采集。随后 `confidence` 会统计最低最高概率、平均概率、最小 margin 在各个十分位上的分布，以及平均熵在各 0.5-nat 区间内的分布。由于每个被测 token 都会触发一次 GPU 同步，默认值 0 仅统计显式请求的记录。
- **`GET /api/v1/metrics`** 以 OpenMetrics 文本格式返回相同指标，供 Prometheus 或 Grafana 采集。
  - 指标名称均以 `basalt_` 开头。时间窗口（window）与分位数（quantile）作为标签提供，决策模型同样作为标签。
  - 与其他端点一样受服务端 API 密钥保护：请在采集器中配置 Bearer 令牌。

```yaml
scrape_configs:
  - job_name: basalt
    metrics_path: /api/v1/metrics
    authorization: { credentials_file: /etc/prometheus/basalt.token }
    static_configs: [{ targets: ["127.0.0.1:8080"] }]
```

遥测仅记录计数、耗时与直方图：绝不包含任何提示词或回答文本（#450）。

## 多模态、转换器、X-Ray 与治理端点（`v0.12`）

- **Video Studio（`/v1/videos/generations`、`/v1/videos/{id}`）** —— 分阶段 24 GB UMA 视频生成（`LTX-2.5 AV-DiT`、`Kandinsky 6.0`、`LingBot-Video MoE`、`Wan 2.2`），返回经 `AVAssetWriter` 封装的 `.mp4`（`#608`–`#611`）。
- **音频与音乐（`/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/music/generations`）** —— 集成 `ZONOS3-Mini`、`Kokoro-82M`、`LFM2.5-Audio` Mimi/SNAC 双工编解码器、`SmartTurn`/`Sortformer`、`MossFormer2`、`Soprano-80M`、`Qwen3ForcedAligner`、`ACE-Step 1.5`（`#604`, `#605`）。
- **图像生成与编辑（`/v1/images/generations`、`/v1/images/edits`）** —— 支持 `FLUX.3`、`Z-Image-Turbo`、`Iris-3B`、`SANA-1.5`、`SeedVR2` 超分辨率放大及 RGBA 透明通道（`#542`, `#605`）。
- **通用转换器（`/v1/models/convert`、`/v1/models/convert/jobs/{id}`）** —— 异步任意格式互转量化（`MACaQ`、`FlatQuant`、`LIRQuant`、`SINQ`、`QTIP`、`E8P`、`PV-Tuning`、`QAT STE` + `QuantizationAuditCore`，`#546`, `#623`）。
- **Token 归因与幻觉风险（`POST /v1/xray/tokens`）** —— 提供逐 token 对数概率、top-k 熵、层/头/MoE 归因及幻觉风险信号（`#613`, `#614`）。
- **HITL 治理（`/v1/mcp/approvals`）** —— 针对高影响 MCP 工具调用的人机协同审批队列，配合 `HITLGovernanceCore` 与 `SymbolicKnowledgeGraph` 使用（`#626`）。

## Swagger UI

访问 `http://localhost:8080/swagger` 查看交互式 API 文档。
