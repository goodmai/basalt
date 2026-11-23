# basalt compose

> Languages: [English](../../compose.md) · [Русский](../ru/compose.md) · **中文** · [Español](../es/compose.md) · [Français](../fr/compose.md) · [日本語](../ja/compose.md) · [Deutsch](../de/compose.md) · [한국어](../ko/compose.md) · [Bahasa Indonesia](../id/compose.md)

`basalt compose` 用于启动并托管由单个 YAML 文件描述的模型系统。该文件列出了本地和云端模型以及它们各自在内存中的驻留方式，同时也描述了在模型之间路由请求的流水线（pipeline）。每条流水线依次执行感知（perceive：OCR、语音）、决策（decide：System One）和回答（answer：对话模型）。这是一种面向模型的混合架构编排方式，风格类似于针对模型的 docker-compose 文件。

```bash
basalt compose plan -f basalt.yaml        # validate and show the compute plan on this Mac
basalt compose up -f basalt.yaml          # serve it; run it again with a changed file to apply the diff
basalt compose ps                         # what is loaded, when on-demand models unload, cloud spend
basalt compose down <name>                # stop it
basalt compose up document-qa             # legacy bundled files: document-qa, voice-agent, zonos-voice
basalt compose up 24gb-voice-zonos        # 84 hardware-tiered profiles: <tier>-<archetype>
```

`compose up` 会使用该 YAML 文件展开生成的参数启动 `basalt serve`，并在启动时打印这些参数。它提供相同的 REST 和 MCP 接口。`--` 之后的参数会透传给 serve，例如 `basalt compose up -f basalt.yaml -- --port 8090`。

服务器会先绑定端口，然后再加载模型。在加载过程中，`compose ps` 会将模型显示为 `loading`，`GET /v1/health/ready` 返回 503，且运行记录此时已创建，因此再次执行 `compose up` 时会找到已运行的系统而不是启动新实例。模型加载完成后，每个模型都会执行一次启动探针（startup probe）：
- 对话模型和每个本地回答模型各生成一个 token；
- 每个本地决策模型回答一个问题；
- 按需加载（on-demand）模型会检查其文件完整性而不直接加载。

探针检测失败的模型会在 `compose ps` 中显示为 `failed` 并附带原因。在通过 `compose up` 替换该模型之前，`/v1/health/ready` 将一直返回 503。通过 `compose up` 加载或重新加载的模型也会执行相同的探针检测。使用 `--mcp` 时，模型仍像以前一样在服务器启动前加载。

## 配置文件

```yaml
version: 1
name: support-desk            # the system's name, used by ps, down and up-as-diff
memory: auto                  # resident | sequential | auto
budget: { usd_per_day: 2 }    # optional cap on all cloud spend

models:
  chat:     { source: mlx-community/gemma-4-12B-it-qat-4bit }
  big:      { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, idle_unload: 10m }
  ocr:      { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 2m }
  asr:      { source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX, keep: sequential }
  jev:      { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
  frontier: { source: "cloud:openrouter/anthropic/claude-haiku-5.5", api_key_env: OPENROUTER_API_KEY, budget: { usd_per_day: 1 } }

pipeline:                     # the default pipeline: /v1/pipeline and MCP basalt_pipeline
  perceive:
    documents: [ocr, apple-vision]     # the first available reads pdf and image attachments
    audio: [asr, speech-analyzer]
  decide:
    cascade: [jev, chat]               # up to four stages, cloud and local in any order, the last one local
    escalate_below: 0.6
    abstain_below: 0.75                # questions below this confidence go to the answer model
  answer:
    routes:                            # the first matching rule wins
      - when: { decide: { route: code }, cost_left: true }
        model: frontier
        fallback: chat
      - when: { prompt_tokens: { over: 8000 } }
        model: big
      - model: chat                    # the last rule has no when and names a local model

pipelines:                    # more pipelines, served at /v1/pipelines/<name>
  triage:
    decide: { cascade: [chat] }
    answer: chat
```

### 模型
- **`source`：** 可以是 Hugging Face 仓库、本地路径（`./`、`/` 或 `~`）、`cloud:openrouter/<model>`，或用于 Google 决策模型（如 `gemini-decision-1`）的 `cloud:gemini/<model>`（仅限 decide 级联，密钥通常位于 `GEMINI_API_KEY`）。
- **`keep`** 适用于本地模型：
  - `resident` 为默认值，表示模型常驻内存；
  - `on-demand` 在首个需要该模型的请求到来时加载，并在超过 `idle_unload` 后卸载。时长格式如 `30s`、`5m` 或 `1h`，默认为 120 s；
  - `sequential` 仅在感知器（perceiver）执行该阶段时加载它。在默认流水线中，如果感知器与对话模型无法同时装入内存，它还会在感知器运行前后临时卸载对话模型。
- **云端模型：**
  - 使用 `api_key_env` 指定存放密钥的环境变量名。直接在文件中明文写入密钥会被拒绝，`${…}` 插值语法同样会被拒绝。
  - `budget: { usd_per_day }` 用于限制单个模型的每日支出上限。
  - `data: no-perceived` 为默认设置：从文档或音频中读取到的文本绝不会发送给云端模型。
  - `data: media` 仅适用于 `cloud:gemini` 决策模型，会将流水线附带的原始文件本身（PDF、图片、转为 WAV 的音频）及其感知文本一并发送给该模型用于决策。请仅对允许离开当前 Mac 的文档使用此选项。

### 内存与规划
`compose plan` 会累加每个本地模型的常驻字节数。按需加载（on-demand）模型也会按常驻内存计入，因为它随时可能被加载。

- `memory: resident` 要求总内存占用能放入当前可用工作集（working set）。
- `memory: sequential` 仅要求能容纳最大单个阶段外加 2 GB。
- `memory: auto` 在能装下时优先选择 `resident`。

任何无法装入内存的系统都会在加载任何模型之前被拒绝，并给出具体内存数值。未下载的模型会报告为大小未知。

### 决策（Decide）
- **顺序：** `cascade` 按顺序列出决策模型。每个阶段回答其置信度达到要求（至少为 `escalate_below`）的问题，并将其余问题交给下一阶段。
- **任意顺序的阶段：** 最多支持四个阶段，云端和本地模型可按任意顺序组合。例如，先用一个小型本地裁判模型，再用云端模型，最后用一个更大的本地模型：`[laya, jev, chat]`。
  - 最后一个阶段必须是本地模型。
  - 每个本地阶段都作为决策引擎持有：可以是对话模型（以 `self` 读出模式运行），也可以是专门的决策模型（如 Laya Core AI 捆绑包或 OpenJev 捆绑包）。专用的本地决策模型可以设置 `keep: on-demand`（和 `idle_unload`），使其在首个到达该阶段的决策请求时加载，并在空闲后卸载。
  - 每条流水线各自指定其阶段。被多条流水线引用的决策模型只会加载一次。
  - 决策模型只负责决策；只有对话模型既能回答又能决策。
- **校准：** 读出（readout）决策模型会从其目录中的 `basalt_readout.json`（`{"temperature": …, "noul_temperature": …, "noul_bias": …}`）读取温度参数（如果存在）。否则使用对应模型名称的已知默认值，若无已知默认值则不做校准。#448 计划针对 Gemini Decision 1 拟合这些校准文件。
- **体积计算：** `compose plan` 会将 OpenJev head 捆绑包的大小计算为其基础模型加上适配器，因为 LoRA 是按存储形式直接未合并（unmerged）应用到基础模型上的（包括 4-bit）。`serve` 自身的内存规划也是如此。
  - 在 OpenJev-0.6B 上的实测数据：100 个已标注状态，300 个问题，M4 Pro，commit 066da0a。参考基准为合并后的 float32 路径，该路径与 PyTorch 的误差在 1e-4 以内（#406）。

    | 基础模型（LoRA 未合并） | 准确率 | 与 float32 首选答案一致 | 活跃内存 | 单状态中位耗时 |
    |---|---|---|---|---|
    | merged float32 (reference) | 176/300 | — | 2,273 MB | 185 ms |
    | bf16, as the bundle names it (the default) | 175/300 | 288/300 | 1,141 MB | 181 ms |
    | 8-bit | 176/300 | 279/300 | 608 MB | 184 ms |
    | 4-bit | 149/300 | 215/300 | 324 MB | 185 ms |

  - 4-bit 基础模型会让这个 0.6B 模型损失 9 个百分点，而 8-bit 在仅占四分之一内存的情况下保持了准确率。`serve` 和 `compose` 会加载捆绑包中指定的基础模型，因此量化版基础模型只能直接通过 `JevEvaluator` 使用。
  - 可通过 `BASALT_E2E=1 BASALT_E2E_JEV_MODEL=<bundle> BASALT_E2E_JEV_BASE=<bf16 base> BASALT_E2E_JEV_BASE_8BIT=… BASALT_E2E_JEV_BASE_4BIT=… swift test --filter JevRealModelTests/unmergedArms` 复现该测试。
- **在 24 GB 内存上与 Gemma 4 12B 并行运行 9B 裁判模型：** 使用 `openjev/OpenJev-Flash-9B-MLX`（8-bit，9.5 GB）和 `openjev/OpenJev-Flash-9B-MLX-4bit`（5.9 GB）进行实测，由 Gemma 4 12B（`gemma-4-12B-it-qat-4bit`）负责回答升级（escalation）上来的问题。测试集包含 60 个已标注状态、180 个决策，在 M4 Pro 24 GB、commit 8294bf7 上测得。
  - 请使用 4-bit 裁判模型。它与 Gemma 并存时在 24 GB 内存中仅占用 15.3 GB 活跃内存（峰值 15.9 GB）。
  - 8-bit 裁判模型与对话模型及一个 0.6B head 共同驻留时大约占用 21 GB。它单独执行决策时表现良好（无弃权时 92.8%，耗时 1.85 s），但一旦 Gemma 开始解码升级上来的问题，KV cache 就会将 Mac 推入内存交换（swap）。原本两分钟的测试分支随后运行超过 24 分钟并被手动中止。

    | 与 Gemma 4 12B 并存的决策器（4-bit 裁判） | 准确率 | 转交 LLM 数量 | 节省 LLM token | 决策 p50 耗时 |
    |---|---|---|---|---|
    | LLM only | 97.8% | 180 | — | — |
    | OpenJev-Flash-9B 4-bit, no abstain | 96.7% | 0 | 100% | 2.0 s |
    | OpenJev-Flash-9B 4-bit, abstain below 0.6 | 98.9% | 20 | 88.7% | 2.2 s |
    | OpenJev-Flash-9B 4-bit, abstain below 0.75 | 98.9% | 41 | 76.7% | 2.4 s |
    | OpenJev-Flash-9B 4-bit, abstain below 0.9 | 97.8% | 74 | 58.5% | 2.5 s |
    | readout of Gemma itself (`self`), abstain below 0.75 | 99.4% | 1 | 99.4% | 1.9 s |
    | OpenJev-0.6B head (8-bit judge run), abstain below 0.75 | 82.8% | 86 | 48% | 0.26 s |

  - 在 Gemma 解码长回答的同时执行决策，会使解码速度从 21 降至 3 tok/s（−86%）。总挂钟时间为 341 s，而依次顺序执行只需 202 s。因此，GPU 决策器在两次生成之间会保持互斥门控。
  - 可通过 `BASALT_E2E=1 BASALT_E2E_BUDGET_CHAT_MODEL=<chat> BASALT_E2E_BUDGET_FLASH=<judge dir> swift test --filter SystemOneBudgetBenchmarkTests` 复现该测试。
- **云端接收的内容：** 带有文档或音频的请求会跳过云端阶段，仅按顺序流经本地阶段。
- **阈值：** 置信度低于 `abstain_below` 或前两个选项差距小于 `min_margin` 的问题会转交回答模型处理。
- **认知门控（Epistemic gate，version 2）：** `epistemic` 在解码任何回答 token 之前对 `(support, refutation, needs_tool)` 进行分类：支持 `enabled`、`support_threshold`、`refutation_threshold`、`tool_threshold`、`on_no_evidence`（`abstain` 或 `escalate`）、`on_conflict`（`report` 或 `escalate`）以及可选的 `abstain_message`。

### 回答路由（Answer routes）
- **匹配规则：** 当规则的 `when` 中所有条件均满足时触发匹配。可用条件包括：
  - `modality`：`documents`、`audio` 或 `text`；
  - `decide`：来自 System One 的问题 ID 到标签的映射；
  - `prompt_tokens: { over: N }`：按每 token 4 字节估算；
  - `escalated`：decide 阶段发生升级，或某个问题被转交给回答模型；
  - `cost_left`：该规则对应的模型仍有剩余预算。
- **尝试顺序：** 优先尝试规则的 `model`，其次是 `fallback`，最后是流水线的默认本地模型。
- **跳过机制：** 当输入为感知文本、预算耗尽、密钥被拒或触发速率限制时，云端模型会被跳过。每次跳过都会记录在执行回执中。

### 语音合成（Speak，`#493`, `#525`）
流水线可声明一个可选的 `speak` 阶段来合成音频（`Kokoro-82M`、`Qwen3-TTS`、`Fun-CosyVoice3`、`Chatterbox`、`Zyphra/ZONOS3-Mini-hf`），并支持确定性的 GPU → 神经网络引擎（Neural Engine）→ 系统语音回退链路：

```yaml
pipeline:
  answer: chat
  speak:
    chain: [tts_zonos, tts_ane, apple-speech]
    voice: Savannah
    language: en_us
    reference_audio: ./speaker.wav
    on_gpu_busy: fallback_ane
    sentence_streaming: true
    format: wav
    exaggeration: 0.5
    max_ttfa_ms: 80
```

- **`speak`：** 支持单个名称（`speak: apple-speech`）、回退列表（`speak: [tts_zonos, tts_ane, apple-speech]`）或完整映射配置。
- **`chain`：** 按顺序排列的本地 TTS 模型列表，末尾可选挂载零内存占用的 `apple-speech` 系统合成器。
- **`voice`：** 默认音色标识符（`Savannah`、`alloy`、`af_heart` 或 `AVSpeechSynthesisVoice` 标识符）。
- **`language`：** 默认语言或区域代码（`en_us`、`ru`、`de`、`cmn` 等）。
- **`reference_audio`：** 可选的本地参考 `.wav` / `.pcm` / `.caf` / `.aiff` / `.flac` 音频片段，用于零样本声音克隆。
- **`on_gpu_busy`：** 当 Metal 正忙于解码时的策略，可选 `keep`、`fallback_ane` 或 `fallback_apple_speech`。
- **`sentence_streaming`：** 随着 token 到达，将全双工分句块流式送入 TTS。
- **`format`：** 输出容器或采样编码格式（`wav`、`pcm`、`pcm_24k`、`mp3`、`opus`、`aac`、`flac`）。
- **`exaggeration`：** 支持该特性的模型中用于控制表现力韵律的参数，范围为 `0.0...2.0`。
- **`max_ttfa_ms`：** 沿回退链路降级路由前的最大首音频延迟（TTFA）预算。

### 图像生成（Imagine，`#517`, `#539`）
流水线可声明一个可选的 `imagine` 阶段，在 Apple Silicon 上通过最多四个显式阶段（`upsample_via` 改写器 -> `share_encoder` 文本/视觉编码器 -> `chain` DiT 绘制器 -> `decoder_via` 潜空间 VAE 解码器）合成或编辑图像，同时支持 TeaCache 跳步与 best-of-N VLM 评选：

```yaml
pipeline:
  answer: chat
  imagine:
    chain: [z_turbo, sana_fast]
    upsample_via: chat
    share_encoder: encoder
    decoder_via: vae
    steps: 8
    size: 1024x1024
    format: png
    teacache_threshold: 0.15
    preview_every_steps: 2
    candidates: 3
    judge_via: vlm_judge
    on_gpu_busy: fallback
```

- **`imagine`：** 支持单个生成器名称（`imagine: z_turbo`）、回退列表（`imagine: [z_turbo, sana_fast]`）或完整映射配置。
- **`chain`：** 按顺序排列的本地图像合成器（`FLUX.2/3-klein`、`Z-Image-Turbo`、`Iris-3B`、`SANA-1.5`、`Ming-Image`、`Bonsai-4B`、`Step1X-Edit`、`SonicDiffusion`）。
- **`upsample_via`：** 本地 LLM/VLM 改写器槽位，用于在合成前将简短提示词扩展为丰富的场景描述（针对 `FLUX.3` 时使用 BFL 的完整场景描述提示词规范）。
- **`share_encoder`：** 本地 LLM/VLM 槽位（例如 4-bit、8-bit 或 BF16 精度的 `Qwen3-VL-4B-Instruct`），其条件嵌入用于驱动图像合成器。
- **`decoder_via`：** 可选的本地潜空间 VAE 解码器槽位（如蒸馏版 `black-forest-labs/FLUX.2-small-decoder` 或完整版 `unsloth/FLUX.2-VAE` / `Felldude/FLUX.2-HDR-VAE`）。
- **`steps`：** 默认去噪步数，范围为 `1...100`。
- **`size`：** 默认输出分辨率 `<width>x<height>`（须为 16 的倍数，例如 `1024x1024` 或 `2048x2048`）。
- **`format`：** 输出图像格式（`png`、`webp` 或 `jpeg`）。
- **`teacache_threshold`：** 用于 TeaCache 中间步跳过的残差 L1 阈值（`>= 0`）。
- **`preview_every_steps`：** 在 SSE 流式传输期间每隔 N 步输出一次中间 TAEF 预览帧（`>= 0`）。
- **`candidates`：** 为 best-of-N 评选而生成的不同种子候选图像数量（`1...64`）。
- **`judge_via`：** 本地 VLM/LLM 槽位，用于根据提示词对 `candidates` 评分并返回胜出图像产物。
- **`on_gpu_busy`：** 当 Metal 正忙于解码时的策略，可选 `queue` 或 `fallback`。

内置 4 阶段 `FLUX.3` 预设（`basalt compose up <preset>`）：`flux3-16gb-fast`、`flux3-24gb-balanced`、`flux3-36gb-moe-vision`、`flux3-64gb-studio-base` 和 `flux3-128gb-reference`。

### Version 2：变量与配置文件（Profiles）
`version: 2` 新增了两个配置块，使单个文件能够适配多台不同的 Mac。version 1 文件仍可继续使用；在 version 1 文件中使用这两个配置块之一会被拒绝并提示 "needs version: 2"。

```yaml
version: 2
name: desk
variables:
  chat_repo: mlx-community/gemma-4-12B-it-qat-4bit
profiles:
  small:
    when: { memory_gb: { under: 20 } }
    variables: { chat_repo: mlx-community/gemma-4-E4B-it-4bit }
  big:
    when: { memory_gb: { over: 48 }, chip: Max }
models:
  chat:  { source: "${chat_repo}" }
  large: { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, profiles: [big] }
```

- **`variables`：** 小写变量名配合单行纯文本值。文件中的任何位置都可以用 `${name}` 进行替换。未声明的 `${…}` 会被拒绝，因此环境变量依然无法进入配置文件；密钥仅能通过 `api_key_env` 引入。
- **`profiles`：** 当某个 profile 的 `when` 条件在当前 Mac 上成立时即处于启用状态：支持按 `memory_gb` 的 `under` 和 `over`（统一内存大小，单位 GB）以及 `chip`（芯片名称的不区分大小写子串，如 `M4 Pro` 或 `Max`）匹配。未设置 `when` 的 profile 仅在通过 `--profile` 显式指定时启用。
  - 已启用 profile 中的 `variables` 会覆盖文件顶层的变量定义。
  - 带有 `profiles: [...]` 的模型或命名流水线仅在其中至少一个 profile 启用时才存在。
- **命令行覆盖：** 对 `config`、`plan` 和 `up` 命令，`--set name=value` 可覆盖已声明的变量，`--profile name` 可精确启用指定的 profiles。`compose up` 会将它们传递给服务器，diff 增量更新（up-as-diff）也会随文件一并发送。
- **路径：** 以 `./` 或 `../` 开头的模型路径相对于 compose 文件所在目录解析。
- **`basalt compose config -f <file>`** 会打印该文件在当前 Mac 上（结合已启用 profiles）解析后的完整结果。**`basalt compose schema`** 会打印该格式的 JSON Schema 供编辑器使用。

### 实验性选项与 MCP（`#485`）
compose 文件可以声明顶层 `experimental` 和 `mcp` 配置块（version 1 和 version 2 均支持），而不必在 `--` 后面附加冗长的命令行参数：

```yaml
experimental:
  zero_copy_body: true
  smallm_kernel: true
  speech_analyzer: true
  image_fallback: ocr
  ane_vision_tower: ./tower-bundle
  telemetry_confidence_every: 4
  adaptive_thinking_budget: true
  samplers: true
  mtp_draft_head: true
  expert_streaming: true
  expert_cache_gb: 2.5
  kv_head_split: 0.5            # or quest_top_k: 64 (mutually exclusive)

mcp:
  enabled: true
  rest: true
  research_tools: false
  enable: [surgery, intervene]
  disable: [switch_model]
  image_dirs: [./images]
```

- **`experimental`：** 会降级展开为对应的 `serve` 参数（`zero_copy_body`、`smallm_kernel`、`speech_analyzer`、`image_fallback`、`ane_vision_tower`、`telemetry_confidence_every`、`adaptive_thinking_budget`、`samplers`、`mtp_draft_head`、`expert_streaming`、`expert_cache_gb`、`kv_head_split`、`quest_top_k`），并在校验时精确报错到 YAML 行号（`telemetry_confidence_every` >= 0，`image_fallback` 为 `ocr`，`ane_vision_tower` 目录下包含 `metadata.json` 和 `tower.aimodel`，`expert_cache_gb` > 0 且需配合 `expert_streaming: true`，`kv_head_split` 在 `0.05...1` 之间，`quest_top_k` 在 `1...4096` 之间）。相对路径（`./`、`../`、`~`）相对于 compose 文件所在目录解析。
- **`mcp`：** 配置 MCP stdio 传输（`enabled`、`rest`、`research_tools`、`enable`、`disable`、`image_dirs`），使用与 `--mcp-enable` / `--mcp-disable` 相同的分组与工具选择器（`serving`、`gemma`、`media`、`harness`、`xray`、`surgery`、`research`、`all` 或单个工具名称；`disable` 始终优先）。
- **热重载与重启：** `speech_analyzer`、`image_fallback`、`telemetry_confidence_every` 和 `adaptive_thinking_budget` 在执行 `compose up` 时支持热更新（`POST /v1/compose/apply`，并在 `experimental:` 下打印）。而与引擎和传输层绑定的标志（`zero_copy_body`、`smallm_kernel`、`ane_vision_tower`、`samplers`、`mtp_draft_head`、`expert_streaming`、`expert_cache_gb`、`kv_head_split`、`quest_top_k` 以及 `mcp`）会在 `compose diff` 中报告 `needs a restart`，并在实时 `apply` 时被拒绝。

### 基准测试与榜单提交（version 2，`#492`）
`version: 2` 的 compose 文件可声明一个顶层 `benchmarks` 配置块，包含共享的 `defaults` 和命名的 `suites`（既可放在 `suites:` 下，也可作为简写 suite 键直接放在 `benchmarks:` 下）：

```yaml
benchmarks:
  defaults:
    seed: 42
    temperature: 0.0
    gpu_lock: true
    output_dir: ./bench-out
  suites:
    smoke:
      - adapter: basalt:throughput
        target: model:chat
        params: { iterations: 2, max_tokens: 64 }
        gates: { min_decode_tps: 15, max_peak_gb: 20, max_p50_ms: 5000 }
    frontier:
      profiles: [big]
      budget: { usd_max: 5.0, max_minutes: 30 }
      tasks:
        - adapter: arc-agi-2
          dataset: ./arc-eval.json
          target: pipeline
          gates: { min_accuracy: 0.25, max_ppl: 12.0, max_ece: 0.08 }
          submit:
            target: arc-prize
            api_key_env: ARC_PRIZE_API_KEY
            endpoint: https://arcprize.org/api/v1/submissions
            subset: public-eval
            split: test
```

- **`defaults`：** 可选的 `seed`、`temperature`、`gpu_lock` 和 `output_dir`。
- **`suites`：** 每个 suite 既可以是简写的任务列表（`smoke: [...]`），也可以是 `{ profiles, budget: { usd_max, max_minutes }, tasks: [...] }`。
- **`tasks`：** 每个任务指定一个 `adapter`（`basalt:throughput`、`basalt:perplexity`、`basalt:system-one`、`basalt:passkey`、`basalt:mom`、`basalt:harness`、`arc-agi`、`arc-agi-2`、`terminal-bench`、`swe-bench`、`lm-eval`、`livecodebench`、`bfcl`、`inspect-ai`、`tau-bench`、`gaia`、`evalplus`）、可选的 `dataset`、`target`（`pipeline`、`pipeline:<name>` 或 `model:<slot>`）、标量 `params`、回归门控 `gates`（`min_decode_tps`、`max_peak_gb`、`min_accuracy`、`max_p50_ms`、`max_ppl`、`max_ece`）以及可选的 `submit`（`target`、`api_key_env`、`endpoint`、`subset`、`split`）。
- **命令：** `basalt compose plan` 会打印解析后的 suites；`basalt compose bench -f basalt.yaml --suite smoke --submit-dry-run` 会运行对应 suite 并导出已校验的榜单提交包；`basalt bench list|compare|submit` 可检查适配器、对比测试产物，并在全部十个目标平台（`arc-prize`、`terminal-bench`、`swe-bench`、`hf-leaderboard`、`livecodebench`、`bfcl`、`inspect-ai`、`tau-bench`、`gaia`、`weave-otel`）上导出试运行（dry-run）提交包。

### 云端故障与费用控制
- 密钥被拒绝（HTTP 401/403）会禁用该云端模型直至服务器重启，并输出一条日志说明。
- 触发速率限制（429）会按 `Retry-After` 响应头暂停该模型，若无该响应头则暂停 60 s。
- 当某个云端模型被禁用或暂停时，请求会交由下一阶段或 fallback 模型回答。
- 每日按模型统计的支出和总支出保存在 `~/.basalt/cloud-spend.json` 中，因此重启不会重置预算上限。日期按 UTC 计算。

## 运行中的系统
- **`compose pull`：** 将文件中所有本地模型下载到 Hugging Face 缓存并校验每个文件，每完成十分之一打印一次进度。执行 pull 之后，`compose plan` 能获取精确体积，`compose up` 也可直接加载而无需等待下载。
- **`compose plan` 中的大小计算：** 缓存中已存在的模型按磁盘实际占用计算。未下载的模型通过 Hugging Face API 的文件列表计算大小。如果两者均无法获知大小，`plan` 会拒绝执行并指出该模型名称：此时可运行 `compose pull`，或传入 `--assume-size <GB>` 按指定大小估算。`plan` 不再将未知大小的模型计为 0 字节。
- **`compose diff`：** 展示如果对同名运行中系统执行 `compose up` 会产生哪些变更，但不会实际修改系统（`POST /v1/compose/diff`）。它会列出需要加载、卸载的模型、密钥和预算变更、是否需要重启，以及拒绝原因（例如缺少密钥或系统超出内存）。遇到需要重启或被拒绝的情况时，退出码为非零。
- **事件流：** `GET /v1/compose/events` 是展示系统状态的 Server-Sent Events（SSE）流。事件类型包括 `loading`、`ready`、`registered`（按需加载）、`failed`（启动探针失败）、`applied`、`refused` 和 `up`。新连接的客户端会先收到最近 100 条事件。`compose up --wait` 会在等待期间打印这些事件。
- **对运行中的系统执行 `compose up`：** 如果已有同名 `name` 的系统正在运行，`compose up` 会向其发送配置文件（`POST /v1/compose/apply`）并应用差异部分。
  - 当系统仍在加载或正在应用其他变更时，它会返回 503。`compose up` 会等待并重试发送文件，最长等待 `--wait-timeout` 秒（默认 1800）。
  - 配合 `--wait` 使用时，`compose up` 随后会等待直到 `GET /v1/health/ready` 返回 200，打印正在等待的模型列表，若超时仍未就绪则以非零退出码退出。
  - 它会加载新增的模型、重新加载发生变更的模型，并卸载被移除的模型。
  - 它会在请求间隙无缝切换流水线。
  - 它会像 `compose plan` 一样先对照当前 Mac 的内存检查新系统，并将所有流水线中的模型都计算在内。无法装入内存的系统会被直接拒绝，且不做任何更改。
  - 如果新系统本身能装下，但新旧模型无法同时并存，它会先卸载被替换的模型再加载新模型。在此窗口期内需要这些模型的请求要么等待加载完成，要么失败。如果新模型随后加载失败，系统会重新加载回旧模型。
  - 它会逐模型打印 `load`、`reload`、`unload` 和 `keep`，为每个预算发生变化的云端模型打印 `budget`（文件总预算打印为 `total`），并为每个 `api_key_env` 发生变化的云端模型打印 `key`。
  - 云端模型的预算或密钥环境变量修改无需重新加载。新限额会立即生效，且之前因密钥被拒而禁用的模型在其密钥变量变更后会重新启用。
  - 更换对话模型时，新模型会加载到当前服务模型的位置并打印为 `chat`。在此期间请求会排队等待加载。如果后续应用步骤失败，旧对话模型会被重新加载回来。
  - 流水线新增的本地决策模型会像其他新增模型一样加载。默认流水线的决策模型还会打印为 `decide`。各流水线和 `/v1/systemone` 会从下一个请求起使用它。若某个模型开始或停止承担决策职责，它会以新角色重新加载。
- **需要重启的变更：** 更改内存模式、`sequential` 感知器、`experimental` 下的引擎/HTTP 启动参数，或 `mcp` 下的配置。请先运行 `compose down`，然后再执行 `compose up`。
- **`compose ps`：** 列出每个正在运行的系统（`GET /v1/compose`）及其流水线和各模型的详情：
  - 状态为以下之一：`loading`、`ready`、`unloading`、`unloaded`、`idle-unloaded`（按需加载模型，当前未加载）、`decision`（由决策引擎持有）、`cloud`、`no-key`、`over-budget`、`rate-limited`、`refused` 和 `failed`；
  - 对于按需加载模型，显示上次使用时间以及何时卸载；
  - 对于云端模型，显示今日已用费用与上限，以及适用的拒绝原因。
- **健康检查：** 只要服务器能响应 HTTP，`GET /v1/health/live` 就返回 200。一旦所有模型都具备服务能力，`GET /v1/health/ready` 返回 200；在此之前返回 503 并列出仍在等待的模型，例如 `chat (loading)`。处于空闲状态的按需加载模型、超出预算的云端模型和触发速率限制的云端模型均视为具备服务能力；级联或路由器会自动绕过它们。
- **云端密钥：** 若某个云端模型的 `api_key_env` 未设置，`compose up` 会拒绝启动并指出该变量名。而增量更新（up-as-diff）则会检查运行中服务器的环境。
- **`compose down`：** 向服务器发送 SIGINT 信号。服务器会完成正在处理的请求并卸载模型。如果 30 s 后进程仍在运行，`down` 会给出提示并以非零退出码退出。
- **状态存储位置：** 运行中系统的 pid、端口和进程启动时间保存在 `~/.basalt/compose/<name>.json`。如果记录中的 pid 因系统复用已属于其他进程，则视为未在运行。
- **API 密钥：** 当服务器配置了 API 密钥时，为 `ps` 和增量更新（up-as-diff）设置 `BASALT_API_KEY`。

## 领域适配器（`version: 2`）
`version: 2` 文件可以声明 `adapters:`，在 Hugging Face NVIDIA Nemotron 预设数据集（`nemotron:post-v2`、`nemotron:post-v1`、`nemotron:code`、`nemotron:math`、`nemotron:crossthink`、`nemotron:personas`）、`hf:<org>/<repo>` 数据集或本地 `.jsonl` 文件上进行训练，并在单个常驻 `base` 模型上路由请求而无需重新加载权重：

```yaml
adapters:
  code_oft:
    base: chat
    method: oft              # lora | phm | oft
    rank: 16
    scale: 10.0
    hypercomplex_dim: 8
    factor_rank: 1
    shared_basis: true
    block_size: 16
    parameterization: cayley # cayley | matrixExp
    block_shared: false
    layers: [0, 1, 2, 3]
    output: ./adapters/code_oft
    iterations: 200
    batch_size: 2
    learning_rate: 0.0001
    max_seq_len: 2048
    checkpoint_every: 50
    corpus:
      source: nemotron:code
      subset: default
      split: split_0
      max_rows: 512
      min_pass_rate: 0.9
      include_reasoning: true
      seed: 42
    gate:
      max_kl: 0.2
      max_ppl_rise: 0.05
      held_out: ["Explain what a hash table is in two sentences."]
```

运行 `basalt compose train -f basalt.yaml`（可选 `--adapter`、`--force`、`--resume`）将声明的适配器训练并保存到磁盘，随后即可在 `pipeline.answer` 或 `pipeline.answer.routes` 的任意条目中通过 `adapter:` 选用。

## 执行回执（Receipts）
每个 `/v1/pipeline` 响应以及每条会话日志记录都会附带一份回执（receipt）。回执按阶段给出所用模型、耗时毫秒数和 token 数量。决策阶段还会记录其费用以及被跳过的云端阶段。回答阶段会记录命中的规则（`route`）、费用以及被跳过的模型。`swap_ms` 表示加载按需模型和顺序（sequential）模型所花费的时间。

## 内置硬件分级配置文件（`#526`）

`basalt compose` 和 `Basalt.app`（`Compose Profiles`、`⌥⌘C`，以及 **Settings → Engine → Bundled Profile**）内置了 **84 个经过硬件校准的配置文件**，覆盖 7 个 Apple Silicon 统一内存档位（`16gb`、`24gb`、`32-36gb` [别名 `32gb` 和 `36gb`]、`48gb`、`64gb`、`128gb`、`256gb` [别名 `192gb` 和 `236gb`]）× 12 种工作负载原型：

| 原型 | 主推 | 侧重点 |
|---|---|---|
| `daily-chat` | ★ | 日常对话、Apple Vision 文档感知、本地语音输入 |
| `coding-agent` | ★ | 投机采样 small-M 内核、代码专用路由、MCP 工具 |
| `voice-zonos` | ★ | Zyphra `ZONOS3-Mini` 44.1 kHz 神经 TTS + `Kokoro-82M` ANE 回退 |
| `multimodal-desk` | ★ | OCR、音频感知、System One 分流、对话与 ZONOS3 语音 |
| `reasoning-lab` | | 自适应思考预算 + 面向数学与逻辑的 System One 级联 |
| `document-ocr` | | 神经 OCR（`PaddleOCR-VL-1.5`）+ 四值 Belnap 认知门控 |
| `voice-duplex` | | 全双工语音转录与低于 80 ms TTFA 的流式语音合成 |
| `creative-studio` | | 结合 TeaCache 的原生 MLX 图像合成（`Z-Image-Turbo` / `FLUX`） |
| `rag-knowledge` | | 带冲突检测与弃权门控的可靠长上下文检索 |
| `hybrid-cloud` | | 本地隐私优先对话，配合每日 `$1/day` 上限的云端升级 |
| `moe-frontier` | | 稀疏 MoE（`Flint 40B-A2.5B`、`Nemotron-3.5`、`Qwen3.8`）+ 专家流式传输 |
| `research-xray` | | Model X-Ray、权重手术与 `basalt:throughput` 冒烟基准测试 |

启动时，`Basalt.app` 会检测物理统一内存（`ComposeMemoryTier.detect`），并将当前 Mac 对应的 4 个标星主推配置固定显示在 **Recommended for this Mac** 下，同时保持全部 84 个配置文件均可搜索和选用。任意配置文件均可直接从 CLI 以相同方式运行（`basalt compose plan 24gb-voice-zonos`、`basalt compose up 36gb-moe-frontier`、`basalt compose up 236gb-multimodal-desk`）。

## 示例
- `docs/examples/compose/any-mac.yaml`：单个 version 2 文件适配 16、24 和 64 GB 的 Mac。对话模型根据内存选中的 profile 确定，64 GB profile 还会按需增加一个 27B 模型和一条 `long` 流水线。
- `docs/examples/compose/mac-16gb.yaml`：本地 Ornith 9B，按需 OCR，长提示词在每日 $0.50 预算内交由 Claude Haiku 处理。
- `docs/examples/compose/mac-24gb.yaml`：Gemma 4 12B，按需 OCR，代码类请求交由 Haiku，另有第二条 `support` 流水线，每日总预算 $1。
- `docs/examples/compose/mac-64gb.yaml`：Gemma 4 26B-A4B，按需加载 Qwen3.8-27B 和 Nemotron Transcribe，不使用云端回答。
- `docs/examples/compose/document-qa-cloud.yaml`、`mom.yaml` 和 `too-big.yaml`：下文实测数据所对应的配置文件。

更多实测数据见 #412。

## 实测数据

```yaml
version: 1
name: document-qa-cloud
models:
  chat:  { source: mlx-community/gemma-4-12B-it-qat-4bit }
  ocr:   { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 30s }
  jev:   { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
pipeline:
  perceive: { documents: [ocr, apple-vision] }
  decide:   { cascade: [jev, chat], escalate_below: 0.6 }
  answer: chat
```

```bash
basalt compose plan -f docs/examples/compose/document-qa-cloud.yaml
basalt compose up -f docs/examples/compose/document-qa-cloud.yaml -- --port 8080
basalt compose up voice-agent            # bundled: document-qa, voice-agent
```

```yaml
pipeline:
  decide: { cascade: [jev, chat], escalate_below: 0.6 }
  answer:
    routes:
      - when: { decide: { route: code }, cost_left: true }
        model: haiku          # cloud:openrouter/anthropic/claude-haiku-5.5, budget: { usd_per_day: 0.5 }
        fallback: chat
      - model: chat
```

通过 `/v1/pipeline` 运行的 55 个请求混合负载（30 个决策、10 个代码、10 个对话、5 个文档），M4 Pro，Release 构建（`scripts/compose_mom_bench.swift`，`docs/examples/compose/mom.yaml`）：

| 配置方案 | 标签准确率 | 每 1000 次请求云端费用 | p50 | p95 | 本地阶段总耗时 |
|---|---|---|---|---|---|
| 仅本地（Gemma 4 12B 负责决策与回答） | 55/55 | $0 | 0.92 s | 10.0 s | 176 s |
| 云端优先（Jev 决策，Haiku 回答全部文本） | 55/55 | $0.046 | 1.25 s | 4.8 s | 24 s |
| MoM（Jev 决策，预算内代码转 Haiku，其余由 Gemma 回答） | 55/55 | $0.035 | 1.21 s | 4.8 s | 53 s |

在配备 24 GB 内存的 M4 Pro 上使用 `compose up`（Release 构建）的实测结果：

| 配置文件 | 请求 | 回执记录 |
|---|---|---|
| document-qa-cloud | 文档（首次） | 按需加载 OCR（489 ms），本地决策并回答：6.1 s |
| | 文档（再次） | OCR 已加载（0 ms）：5.0 s |
| | 空闲 40 s 后 | OCR 已卸载，内存占用 12 → 11 GB |
| | 纯文本提问 | 由云端 Jev 完成决策：0.6 s |
| | 文档 | 重新加载 OCR（267 ms）：5.3 s |
| voice-agent（内置） | 音频 | 在 Nemotron Transcribe 期间卸载 chat（交换耗时 6.9 s）：17.3 s |
| too-big（示例） | `compose plan` | 被拒绝：单个阶段需要 20.7 GB 外加 2 GB，超出 19.1 GB 工作集上限 |


## Schema `version: "2.2"` 扩展

Compose `version: "2.2"`（兼容 `version: 2`）增加了面向块扩散（block-diffusion）LLM、`DFlash` 投机采样草稿模型、全双工语音到语音、共享 VLM/DiT 文本编码器、1 步 `SeedVR2` 超分、`ACE-Step` 音乐 DiT、因果 3D-VAE 视频 DiT、分段偏好训练以及 Thunderbolt/LAN 多 Mac 集群的声明式配置：

- 按模型配置的 `speculative`（`mode`、`drafter`、`block_size`、`verify`）与 `diffusion_text`（`dual_kv_cache`、`confidence_threshold`、`contrastive_alpha`）。
- `imagine` 扩展字段：`share_text_encoder_with`、`kv_cache_references`、`rgba` 以及 `upscale`（`model`、`steps`、`target_resolution`）。
- 顶层 `voice`（`mode`、`sts_model`、`denoise`、包含 `model` 和 `diarization` 的 `turn_detection`、`tts_fallback`、`aligner`）。
- 顶层 `music`（`model`、`steps`、`duration_seconds`）和 `video`（`model`、`vae_tiling`、`sta_sparse`、`fps`、`frames`）。
- 顶层 `train`（`method`、`segments`、`ema_decay`）和 `cluster`（`topology`、`peers`、`compression`）。
- 企业级路由、缓存与治理（`v0.12`）：`BusinessOrchestratorCore` 5 组件路由、`LANTERN` 语义缓存（`SemanticCacheCore` / `SemanticCacheActor`，`#625`）、`HITL` 审批门控（`/v1/mcp/approvals`）+ LAAS-CNRS `SymbolicKnowledgeGraph`（`#626`），以及 UMA 感知的 L0/L1/L2 `TieredKVHierarchyCore` + `AdaptiveHeterogeneousPlanner`（`#627`）。
