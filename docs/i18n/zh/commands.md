# 命令参考

> Languages: [English](../../commands.md) · [Русский](../ru/commands.md) · **中文** · [Español](../es/commands.md) · [Français](../fr/commands.md) · [日本語](../ja/commands.md) · [Deutsch](../de/commands.md) · [한국어](../ko/commands.md) · [Bahasa Indonesia](../id/commands.md)

直接运行 `basalt`（不带子命令）将启动 `chat`。

| 命令 | 功能说明 |
|---|---|
| `basalt onboard` | 首次运行向导：为当前设备挑选并下载合适的模型 |
| `basalt fit` | 检测硬件配置，并据此对模型目录中的模型进行排序评估 |
| `basalt chat --model <id>` | 交互式终端对话 |
| `basalt serve --model <id> --rest` | 在 `:8080` 端口启动 REST 服务（同时兼容 OpenAI 与 Anthropic 协议） |
| `basalt serve --model <id> --mcp` | 为 Claude Desktop / Cursor 启动 MCP stdio 服务 |
| `basalt serve --model <id> --stats` | 每 5 秒向 stderr 输出一次遥测数据（可通过 `--stats-interval` 调整）；相同数据也可通过 `GET /api/v1/stats` 和 MCP `status` 获取 |
| `basalt serve --model <id> --canary-minutes 30` | 每 30 分钟在服务空闲时以后台优先级发送一次已知答案的金丝雀请求；通过/失败次数、退化输出计数以及按模型统计的 tok/s 与草稿接受率漂移均记录在 `GET /api/v1/stats` 的 `health` 字段下（#325） |
| `basalt models list --author <org>` | 浏览 HuggingFace 上指定作者/组织的模型列表 |
| `basalt models download <repo-id>` | 将模型下载至共享的 HuggingFace 缓存目录 |
| `basalt models info <repo-id>` | 查看模型体积、量化方式与上下文窗口大小 |
| `basalt models cache` | 查看本地磁盘上已缓存的模型及其占用空间 |
| `basalt models check <repo-id>` | 检查模型是否能装入当前内存（装不下时返回非零退出码） |
| `basalt models check --deep <repo-id>` | 在下载前执行深度检查：仅拉取 safetensors 文件头（几百 KB），据此仅构建模型形状并渲染对话模板；列出缺失、多余或形状不匹配的张量、不受支持的 `model_type` 或模板错误，并在异常时返回非零退出码 |
| `basalt models inspect <repo-id>` | 校验已缓存模型是否完整，并测量其实际位宽 |
| `basalt models inspect --verify <repo-id>` | 进一步将每个已缓存文件与 sidecar 摘要或 Hub 摘要逐一重新计算哈希校验：按文件返回 `true`、`false` 或 `unknown`，只读操作（#367） |
| `basalt models distill <model> --prompts FILE --teacher-url URL --teacher-model NAME` | 将位于 OpenAI 或 Anthropic 格式 API（通过 `--provider` 指定）背后的教师模型蒸馏为本地模型的 LoRA 适配器；密钥从 `BASALT_TEACHER_API_KEY` 读取，地址配合 `BASALT_TEACHER_URL`；`--budget-usd` 在达到花费上限前停止，`--filter verify` 仅保留与提示词中 `answer` 匹配的回答，`--on-policy` 让本地学生模型采样，同时由使用相同分词器的 basalt 教师模型在 `/v1/score` 上为其 token 评分（逐 token 反向 KL），离线运行每 `--checkpoint-every` 步保存一次检查点并支持 `--resume` 断点续训（#393） |
| `basalt models train-jepa <model> --dataset FILE` | 在多视角数据集（每行 JSONL 表示一个场景，至少包含两个视角）上训练 JEPA LoRA 适配器与投影层；输出训练前后的损失值与嵌入秩 |
| `basalt models convert <source> --out DIR` | 通用任意格式互转器（#546）：支持重新量化为 `--to macaq:affine\|mxfp4\|moe`（分块 Walsh–Hadamard 变换 + Hessian 翻转精调，路由器加权 2.8 bpw MoE #599）、`--to flatquant` / `--to lirquant`（#594）、`--to sinq` / `a-sinq` / `eaquant`（#561）、`--to e8p`（`QuIP#` 2-bit 格点量化 + `TurboBoA` Babai–LLL，#589）、`--experimental-qtip 2.0\|2.5\|3.0`（#494）、`nf4`、`SVD-LLM V2` / `OBR`（#592）、`.gguf` 原生 i-quant 与 `MACaQ-R`，并支持 `--pv-tune`（`--pv-preset fast\|full`，#545）、领域感知 QAT STE 精调以及量化后质量/公平性/泄露审计（`#623`）。亦可通过 REST 接口 `/v1/models/convert*` 与 macOS 界面 `ConvertModelSheet` 调用 |
| `basalt serve --video-model <id> --rest` | 通过 `/v1/videos/*` 提供分阶段 24 GB UMA 视频工作室服务（支持 `LTX-2.5 AV-DiT`、`Kandinsky 6.0`、`LingBot-Video MoE`、`Wan 2.2`），内置因果 3D-VAE、STA Metal 内核与 `AVAssetWriter` `.mp4` 封装（#608–#611, #614） |
| `basalt review --diff FILE\|- --model <id>` | 使用本地模型审查代码 diff；指定 `--gitlab` / `--gitea URL --repo R --number N` 可改为拉取合并请求，添加 `--post` 可将审查结果回传（需配置 `GITLAB_TOKEN` / `GITEA_TOKEN`）（#35） |
| `basalt compose` | 通过 YAML 编排并执行服务启动、规划（plan）、差异对比（diff）、拉取（pull）或停止（stop）多模型系统（支持 `models`、`pipeline`、`pipelines`、`experimental`、`mcp`、`variables`、`profiles`）；可通过 `/v1/compose/apply` 热重载模型、流水线、预算以及实时 `experimental` 设置（`speech_analyzer`、`image_fallback`、`telemetry_confidence_every`、`adaptive_thinking_budget`）（#412, #423, #485） |
| `basalt cloud configure` | 配置 OpenRouter 云端兜底，用于处理本地内存无法容纳的请求 |
| `basalt cloud cost` | 查看云端兜底至今的累计消费金额 |

在 `basalt chat` 交互模式中：

| 输入 | 效果 |
|---|---|
| `/clear` | 清空当前对话记录与终端屏幕 |
| `/color`, `/theme` | 轮换切换终端配色主题 |
| `exit`, `quit` | 退出对话（无需斜杠） |

`serve` 中值得了解的参数标志：

| 参数 | 适用场景 |
|---|---|
| `--reasoning-effort none` | 阻止推理模型将全部生成预算耗尽在 `<think>` 思考过程内 |
| `--reasoning-effort xhigh\|medium\|low` | 设置 Qwen 系列模型的思考预算档位 |
| `--quant 4bit` | 在包含多个量化版本子目录的仓库中指定所需的量化版本 |
| `--kv-bits 4\|8\|auto` | 量化 KV 缓存——在内存紧张的设备上为生成阶段腾出更多空间；会关闭提示词缓存（prompt-cache）复用。MLX 会先以 fp16 预填充提示词再进行量化，因此可支持的最大提示词长度不会超过 fp16（甚至略短：转换瞬间会同时持有两份副本）。`auto` 模式默认保持 fp16 并启用提示词缓存复用，直到某次请求的提示词加上 `max_tokens` 无法装入内存时，才自动对该请求切换为 8 bit，若仍不足则切换为 4 bit |
| `--kv-bits turbo3\|turbo4\|turbo8v4` | **实验性**（#471）：本分支在全注意力层（full-attention layers）上实现的 TurboQuant 缓存（旋转变换、Lloyd–Max 码本、每向量一个范数；首个及最后两个符合条件的层保持 8-bit 仿射量化）。不可与 `--kv-zones` 同用。在 M4 Pro 上 8k 上下文对比仿射 4-bit 的实测结果：解码速度慢 8–13%（Qwen3.8-27B 为 13.8 vs 15.0 tok/s，LFM2.5-24B 为 82 vs 94）；384 个 token 的解码困惑度变化在 Qwen3.8 上为 +0.3% / +1.3%（对比 4-bit 的 +1.5%），在 LFM2.5 上为 +10.1% / +3.6%（对比 +5.0%）；16k 上下文的 KV 占用并未减小（Qwen3.8 为 460 / 511 MB 对比 461，LFM2.5 为 118 / 131 对比 97，Gemma 4 26B-A4B 为 423 对比 306）。压缩仅在解码阶段生效：多 token 分块预填充读取的是原始 key |
| `--kv-bits polar4` | **实验性**（#351），需配合 `--kv-zones` 使用：分区中间段将每个 key 的 RoPE 维度对存储为单个字节（5-bit 相位，3-bit 模长，基于首个迁移批次按头和维度对缩放的 Rayleigh 分位数），其余维度采用仿射 4-bit；value 保持仿射 4-bit。仅支持 RoPE 配对维度为 i 与 i + d/2 的 Qwen3、Qwen3.5 和 LFM2；其他模型将保持仿射 key 并记录日志。在 M4 Pro 上 8k 之后解码 256 个 token 对比分区 q4 与 k8v4 的实测结果：困惑度方面 Qwen3.8-27B 为 8.61 vs 8.56 / 8.50，LFM2.5-24B 为 61.2 vs 76.5 / 70.9（256-token 样本波动较大）；分区 KV 占用分别为 196 vs 197 / 255 MB 与 59 vs 62 / 80 MB；解码速度为 12.4 vs 14.0 / 14.6 tok/s 与 11.3 vs 19.5 / 18.2（中间段每步都需反量化回 fp16，目前没有专用 polar 内核）；在两款模型上均成功找回深度 0.2 与 0.8 处的 16k passkey。离线实验测得在同等位宽下 polar 并不优于仿射量化 |
| `--kv-bits k8v4` | 需配合 `--kv-zones` 使用：量化中间段将 key 存为 8 bit、value 存为 4 bit（含分组缩放因子后每对 K+V 占 13 bit，约为 fp16 的 0.41×，介于 8-bit 的 0.53× 与 4-bit 的 0.28× 之间）。`auto` 配合 `--kv-zones` 时会在 8 bit 与 4 bit 之间尝试该档位。对于不适用分区的模型（如 Gemma 4），会自动回退为统一的 8-bit。**实验性：** `k8v4h` 会在 4-bit 量化前对 value 施加 Walsh–Hadamard 变换（并在读取时逆变换），以分散离群通道；该模式仅支持显式开启，`auto` 不会自动选用。在 8k 上下文后评测 8k token 的实测结果中，对比 `k8v4` 的困惑度为：Qwen3-4B 6.568 vs 6.562，Ornith-1.5-9B 4.937 vs 4.920，LFM2.5-8B 8.723 vs 8.750（各自单次运行，处于误差范围内），且 16k 提示词后的峰值内存分别上升约 85 MB（Ornith）与 30–150 MB（Qwen3-4B，源自参与旋转的 value fp32 副本）。该变换作用于头维度为 2^k 与 12·2^k 的情况（此时 MLX 的变换矩阵互为逆矩阵）；其他头维度保持不旋转并在 `serve` 加载时记录日志。它可能在更长上下文（32k+）或 V 中存在明显离群通道的模型上发挥作用，但尚未经过实测 |
| `--kv-zones` | 配合 `--kv-bits` 使用：在每个全注意力层中，将最前 64 个和最后 1024 个 token 保留为 fp16，仅在提示词预填充期间对两者之间的 token 进行量化。无需先构建完整的 fp16 预填充再转换，因此提示词可充分利用全部量化内存空间，峰值内存显著下降；在 Qwen3-4B 上，8k 上下文后评测 8k token，4-bit 困惑度为 6.58，对比 fp16 为 6.56，而统一 4-bit 为 8.66 |
| `--kv-evict F` | **实验性**，需配合 `--kv-zones` 使用：当 token 离开近期窗口时，按比率 `F`（0–0.9）丢弃每个全注意力层量化中间段的 token，按每个 KV 头淘汰其 key L2 范数最大的部分（arXiv:2406.11430）。无需读取注意力权重，因此注意力计算仍可走融合内核；sink（最前）与近期窗口的 token 永不丢弃，`serve` 与 `models check` 报告的容量空间仅统计保留下来的 token。在 16k 提示词与 k8v4 下，Qwen3-4B 的 KV 占用从 1.11 GB 降至 0.89 GB（0.25）与 0.66 GB（0.5）。这会带来质量损失：8k 后评测 8k token 的困惑度在 Qwen3-4B 上为 6.71 / 7.04（对比 6.56），在 LFM2.5-8B 上为 9.06 / 9.52（对比 8.75），Ornith-1.5-9B 上保持不变；在 0.25 淘汰率下，Qwen3-4B 与 Ornith 在 5 个测试深度中有 1 个丢失了 16k passkey，且贪心解码输出在最初几个 token 内即与 fp16 产生分歧。Gemma 4 以及单次预填充长度超过近期窗口的模型（在加载时检测）会保留全部 token 并按此估算容量 |
| `--kv-head-split R` | **实验性**，不可与统一 `--kv-bits` 同用：仅在擅长从长提示词中段提取事实的 `R` 比例（0.05–1）KV 头中保留完整上下文（每个模型通过合成 passkey 集评分一次，采用 DuoAttention，结果缓存于 `~/.basalt/head-scores`）；其余头仅保留 128 个 sink token 与 256 个近期 token。在 Qwen3-4B 32k 上下文中设置 `R` 为 0.25：KV 占用从 4.76 降至 1.26 GB，解码速度从 27.2 提升至 36.2 tok/s，16k 与 32k 下的 passkey 召回率均为 10/10。对于 Qwen3.5 系列、LFM2 / Nemotron-H 混合架构以及注意力不经过该缓存的模型（Gemma 4），该参数会被拒绝并给出警告。在混合架构上，通过消融选出的 25% 头在 16k 测试中丢失了 passkey（Qwen3.8-27B 为 0/3 深度，LFM2.5-24B 为 2/3），并导致困惑度上升 24% 和 42%；基于输出保真度选头见 #282。配合 `--kv-zones --kv-bits k8v4`（实验性，必须同时指定才生效）时，保留完整上下文的头采用分区存储：前 64 个和后 1024 个 token 精确存储，中间部分为 8-bit key 与 4-bit value；其余头策略同上。与 `--kv-zones` 一样，仅适用于未指定 prompt-cache 命名空间的请求 |
| `--prompt-cache-mb N` | 为提示词缓存前缀树保留的内存上限（默认值：加载模型后 Metal 推荐 GPU 工作集剩余内存的一半，加载时会输出日志；设为 `0` 可禁用），详见下文 |
| `--prompt-cache-disk-mb N` | 在优雅停机、切换模型前以及 `basalt chat` 退出时，将提示词缓存持久化到 `~/.basalt/prompt-cache/`（可通过 `--prompt-cache-dir` 指定），以便恢复会话或重启服务后直接从磁盘恢复前缀而无需重新预填充（默认 `0`，即关闭）。按最热分支优先写入，总容量不超过 `N` MB，且仅写入按假定 1 GB/s 速率能在停机截止时间内完成写入的分支；同一对话的更长路径会覆盖其包含的较短文件，空间不足时按 LRU 删除最久未使用的文件，超过 `--prompt-cache-disk-days`（7 天）未使用的文件会在启动时清理，`basalt sessions prune` 可在离线时执行相同清理。目标目录必须为空目录、新建目录或既有的 prompt-cache 目录（若包含其他无关文件将被拒绝）。缓存文件仅限同一模型（匹配权重 blob，或 HF 缓存外的采样权重内容、config、分词器及对话模板）、同一 basalt 版本与同一缓存格式复用；任何不匹配、文件截断或层布局变化都将回退至正常预填充。需要同时启用 `--session-log full`（磁盘文件以 KV 张量形式保存了对话内容）及内存提示词缓存；启用了 `--kv-bits`、`--kv-zones` 或包含多模态输入的提示词绝不会被持久化 |
| `--cache-blend` | 以近似方式按任意顺序复用已缓存的提示词分块（CacheBlend）；`--cache-blend-ratio R` 用于设置需重新计算的比例（默认 0.3），详见下文 |
| `--min-p`, `--top-k`, `--seed` | 采样参数；`--seed` 可让非贪心采样可复现；请求中的 `min_p` 会覆盖 `--min-p` |
| `--max-tokens` | 默认生成的最大 token 上限（2048–128000） |
| `--dry-run` | 执行内存可行性检查，随后直接退出而不加载模型 |
| `--config basalt.yaml` | 从 YAML 文件读取 serve 选项：`defaults.{model,serve,mcp}` 下的键会映射为对应的命令行标志，命令行显式传入的标志优先级更高，遇到未知键时会精确报告行号与列号（#344） |
| `--mcp-enable <tool\|group>`, `--mcp-disable <tool\|group>` | 配合 `--mcp` 使用，按需单独开启或关闭特定 MCP 工具或工具组（`serving`、`gemma`、`media`、`harness`、`xray`、`surgery`、`research`、`all`），无需暴露完整的 `--research-tools` 接口面；`--mcp-disable` 始终优先于 `--mcp-enable` 和 `--research-tools`，遇到未知名称会在启动时立即报错退出。亦可在 `basalt.yaml` 中通过 `defaults.mcp.enable` / `defaults.mcp.disable` 配置（#484） |
| `--adapter name=dir` | 在常驻基础模型上加载 mlx-lm LoRA 适配器（可多次指定）；请求可通过 `X-Basalt-Adapter: name` 选择要使用的适配器。使用相同适配器的请求可并发运行，切换适配器时会等待正在执行的生成任务完成，每个适配器拥有独立的提示词缓存命名空间；若适配器中的张量与基础模型的任何层都不匹配，会在启动时被拒绝（#179） |
| `--experimental-zero-copy-body` | **实验性**：将每个 REST 请求体零拷贝包装为 `Data`。在 Gemma 4 26B-A4B 携带 1–8 张图像的测试中，进程峰值内存降低了 70–95 MB，TTFT 无明显变化（#345） |
| `--experimental-smallm-kernel` | **实验性**，默认关闭（等价于 `BASALT_SMALLM_KERNEL=1`）：对 4-bit 仿射线性层的 4–8 行前向传播（如 speculative-lookup 与 MTP 验证窗口、批处理解码）改用 basalt 的 simdgroup-matrix 内核，使每份权重仅解码一次即可服务所有行。在 M4 Pro 上基于 Qwen3.8-27B 形状的实测结果：4–8 行耗时均为单 token 耗时的 1.5–1.7×，而原生实现在 4 行时为 1.7–1.9×、8 行时为 3.3–3.7×；输出与原生实现存在 bf16 舍入差异，真实模型上的贪心采样一致性尚待验证（#208） |
| `--experimental-adaptive-thinking-budget` | **实验性**，默认关闭：对于既未设置 `thinking_budget` 也未设置 `reasoning_effort` 的请求，根据其最后一条用户消息的难度估计（如 "prove"、"algorithm" 等关键词、文本长度、工具使用）动态分配思考预算：简单 512 token，中等 2048 token，困难不设上限。仅支持具备预算过渡机制的推理模型（Qwen3.x、LFM2.5）；在 #41 实测落地前尚无量化数据 |
| `--require-verified` | 在加载模型前，对照本地 sidecar 或 Hub 摘要重新计算快照中所有文件的哈希值，若发现任何未通过校验的文件则拒绝启动（#367） |
| `--port`, `--host` | 默认监听 `127.0.0.1:8080`；绑定至其他任何主机地址时均须配置 API 密钥（`BASALT_API_KEY` 或 `--auto-generate-key`） |
| `virtual_keys`（仅配置文件） | 为不同用户或团队配置带有独立云端预算的额外 API 密钥：在 `~/.gem/config.json` 中添加 `"virtual_keys": [{"key": "vk-…", "id": "team-a", "max_budget": 2.0}]`；密钥长度至少 16 个字符，`id` 和 `key` 均须唯一；各密钥的累计消费金额在服务重启后依然保留 |
| `--expert-streaming`, `--no-expert-streaming` | 从 SSD 的 safetensors 分片按需流式读取路由专家权重（通过 `F_NOCACHE` 批量异步读取到固定 LRU 槽位中，`--expert-cache-gb` 默认设为专家总权重的 25%）。当本地堆叠 `SwitchLinear` 检查点落入 `Streaming` 适配层级（`weights > workingSetBudget`，而常驻权重 + 专家缓存可装入）时会自动启用；可通过 `--no-expert-streaming` 或 `BASALT_NO_EXPERT_STREAMING=1` 关闭（`--experimental-expert-streaming` 作为别名保留，#225, #226） |
| `--speech-analyzer` | 当未加载专用音频编码器时，通过运行在神经网络引擎（Neural Engine）上的 macOS `SpeechAnalyzer` 来处理 `/v1/audio/transcriptions`、`basalt_transcribe` 以及 `/v1/pipeline` 中的音频任务（`--experimental-speech-analyzer` 为保留别名，#222, #226, #230） |
| `--ane-vision-tower DIR` | 从 Core AI bundle（`metadata.json` + `tower.aimodel`）在神经网络引擎上运行 PaddleOCR-VL 视觉编码器，并将特化编译产物书签与元数据按 bundle SHA-256 与 macOS 构建版本缓存在 `~/.basalt/cache/coreai-vision/` 下（`--experimental-ane-vision-tower` 为保留别名，#224, #226, #400） |
| `--max-concurrent N`, `--max-queue N` | 同时处于解码阶段的并发生成任务上限（默认 1；每个任务持有独立 KV 缓存）以及排队等待槽位的请求数上限（默认 8）。队列按优先级调度——`basalt chat` 最高，其次是服务内部请求（生成标题、摘要）和 harness 探测，然后是常规 API 流量，最后是基准测试——同优先级内按到达顺序 FIFO 调度；当高优先级请求遇到队列已满时，会淘汰队列中优先级最低的等待者；超出队列容量后服务端返回 503 并附带 `Retry-After`（若流式响应已发送响应头，则改发 error 事件）。切换模型会等待正在进行的生成结束，新请求会在其后排队；对同一模型的两次重复加载会合并共享，而期间请求加载另一模型将返回 503。收到 SIGTERM/SIGINT 信号时，会给正在进行的生成任务最多 5 秒时间收尾，超时后取消并退出进程 |
| `--batch-slots N` | 当 `--max-concurrent` 大于等于 2 时，允许最多 N 个生成任务合并在同一次批处理前向传播中解码（默认 1，即关闭）。支持全注意力层、循环状态层及滑动窗口层（Qwen3 稠密/MoE、Qwen3.5/3.8 与 Nemotron-H 混合架构、LFM2、Gemma 4：各行共享同一环形窗口，并像全注意力行一样左填充并施加掩码，#255），支持 fp16 KV（`BatchKVCache`）与统一仿射 `--kv-bits`（`4`、`8` 或同档位 `auto`，通过 `BatchQuantizedKVCache`，#243），并支持逐行的 `response_format` / 强制工具调用语法掩码；使用了 `--kv-zones`、`k8v4`、`polar4`、`turbo*`、语法跳转前进（jump-forward）、提示词查找（prompt lookup）、MTP、CacheBlend 或携带媒体附件的请求仍像以前一样回退为单独解码（详见下方矩阵）。预填充与提示词缓存逻辑保持不变；每一行保留各自独立的采样器、停止 token 与 `max_tokens`。批处理贪心解码在输出质量上与单行一致，但并非逐比特完全相同：MLX 会为多行与单行选择不同的内核（如量化矩阵乘），且为适配更长邻行而左填充的行会以不同顺序读取其 key，因此同一步中的同行组合可能会翻转接近平局的采样决策（#319）。彼此不引入额外填充的行之间不会改变各自的 logits |
| `--session-log full\|meta\|off` | 在 `~/.basalt/sessions/<date>/<session>.jsonl`（权限 `0700`/`0600`）下记录本地 JSONL 会话日志：每次调用引擎前记录一条 `request` 日志，调用后记录一条 `response` 日志（含状态、token 数、耗时、内存、结束原因），覆盖所有端点；被拒绝的请求会记录 `error`（仅记录大小，绝不记录请求体或请求头），切换模型会记录 `model_switch`。`full` 模式（默认）同时保存完整对话内容与回复，`meta` 模式仅保存大小和 SHA-256 哈希；相关配置包括 `--session-log-dir`、`--session-log-retention-days`（14 天）、`--session-log-max-total-mb`（1024 MB）；会话 ID 由 `X-Basalt-Session`（或 `x-session-id`）指定；未指定时，延续先前对话的请求会自动加入对应会话，其余请求则创建新会话。服务重启后，已知 ID 会追加写入既有日志文件（`session_resume`），空闲超过 `--session-idle-minutes`（30 分钟）的会话会被挂起。`basalt sessions list \| show <id\|latest> \| stats <id\|latest>` 无需加载模型即可直接读取日志文件；`basalt chat` 会在专属会话中保存多轮历史，并可通过 `basalt chat --resume <id\|latest>` 恢复（`meta` 会话不支持恢复）。在 `full` 模式下，私钥块和常见 API 令牌（`sk-`、`ghp_`/`github_pat_`、`AKIA`、`xox*-`、`Bearer`）会自动替换为 `[REDACTED:*]`（尽力脱敏，不作绝对担保；可通过 `--session-log-redact off` 关闭），延续同会话先前消息的请求仅会记录新增消息以及 `parent` 和 `prefix_count` |
| `--rate-limit N` | 全局每分钟请求数限制，超出返回 429——在回环地址上默认关闭，其他地址默认 120；设为 `0` 可禁用；亦可在 `~/.gem/config.json` 中通过 `rate_limit` 配置 |
| `--allow-origin URL` | 除服务端自身来源外，额外允许调用 API 的浏览器 Origin（如 `https://app.example:3000` 或本地前端 `http://localhost:3000`）；可重复指定；亦可在 `~/.gem/config.json` 中通过 `allowed_origins` 配置 |
| `--cloud-fallback MODEL` | 当本地模型无法胜任时，转由兼容 OpenAI 协议的远程服务器上的模型作答：触发条件包括请求的 KV 缓存超出生成内存预算（即限制 `max_tokens` 的同一 KV 空间）、提示词长度超过 `--local-context-limit N`，或成本路由器评估任务超出本地模型能力且收益高于成本。`--cloud-base-url URL` 指定目标服务器（默认为 OpenRouter；支持任何 `/v1` 端点，如 vLLM、Ollama、LM Studio 或另一台 Mac，本地局域网节点无需密钥）；`--no-cloud-fallback` 关闭该功能。亦可在 `~/.gem/config.json` 中配置 `cloud_fallback`：`{"enabled": true, "base_url": "http://192.168.1.20:8000/v1", "model": "qwen3-235b", "api_key_env": "LAN_KEY", "local_context_limit": 32768}`；`enabled: false` 保持关闭。OpenRouter 密钥仅会发送至 openrouter.ai。消费额受 `basalt cloud configure` 配置的日/月预算、`cloud_fallback.session_budget`（每个 `x-session-id` 的美元上限）以及各虚拟密钥的 `max_budget` 共同限制；超出预算后不再调用云端。`--router-params FILE`（训练好的 Bradley–Terry 参数：`strongEmbedding`、`weakEmbedding`、`weights`、`projection`）开启能力评估路由，并通过 `--cost-threshold θ`（默认 0.5）与 `--cache-stickiness λ`（默认 0.1；已预热的 `x-session-id` 会保持留在当前端，直到收益差超过 λ 倍切换成本）；`cloud_fallback` 中同样支持 `router_params`、`cost_threshold`、`cache_stickiness`。携带工具定义或 `response_format` 的请求以及服务内部请求始终保留在本地处理；云端调用失败时也会回退至本地作答 |

**批处理解码能力矩阵（`--batch-slots N`，#240, #243, #255）。** 每个符合条件的请求单独完成预填充后，只要其缓存布局与当前生效的 KV 位宽与运行中的批次一致，就会认领 `BatchedDecoder` 上的一个槽位；不符合条件或布局不匹配的请求将透明地回退到受 `ModelEvaluationLock` 保护的串行单独解码：

| 维度 | `BatchedDecoder` 支持范围 | 单独解码回退（Solo fallback） |
|---|---|---|
| 模型家族 | Llama、Qwen3（稠密与 MoE）、MiniMax-M2（`BatchKVCache` / `BatchQuantizedKVCache`）；Qwen3.5/3.6/3.8、Nemotron-H、LFM2/LFM2.5 混合架构（+ `ArraysCache`）；Gemma 4 稠密与 MoE（+ `BatchRotatingKVCache` 与 KV 共享层）；PaddleOCR-VL / DeepSeek-OCR 的纯文本对话轮次 | Gemma 3（带固定前缀的 `RotatingKVCache` / 未验证布局）、未验证的 `model_type` |
| KV 精度与布局 | 全精度 fp16/bf16（`BatchKVCache`，支持提示词缓存复用）、统一仿射 `--kv-bits`（`4`、`8`、`2`–`6` 以及同档位 `auto`，通过 `BatchQuantizedKVCache`）、当请求落在 `.stock` 路径时的 `--kv-head-split` / `--quest-top-k` | `--kv-zones`（`ZonedKVCache` / `EvictingKVCache`）、`k8v4` / `k8v4h`、`polar4`、`turbo3` / `turbo4` / `turbo8v4`、当前批次由另一 `auto` 档位占用时跨档位到达的请求 |
| 解码与约束 | 逐行贪心与随机采样器（`temperature`、`top_p`、`top_k`、`min_p`、各类惩罚项）、`response_format` JSON schema 掩码、强制 tool-choice 语法掩码、使用相同 `--adapter` 的请求 | `--grammar-jump-forward`、`--speculative-lookup` / `--speculative-recycling`、`--mtp`、`--cache-blend`、多模态提示词（图像/视频/音频）或自定义提示词掩码、内存极度紧张时 |

**提示词缓存复用（Prompt-cache reuse）。** 服务端将近期提示词的 KV 缓存维护在一棵前缀树中：拥有相同起始 token 序列的提示词共享该段前缀的一份存储副本，新请求只需预填充最长匹配前缀之后的新增部分。因此，对话的下一轮只需计算新增的 token（例如追加到长智能体记录末尾的一条工具返回结果，而无需重算整段历史），而以相同系统提示词与工具定义开头的第二个对话——例如 Claude Code 子智能体或另一个 Codex 会话线程——也能直接从共享部分接续，无需重复预填充。匹配完全基于 token 序列，不需要会话 ID（会忽略 `prompt_cache_key`、`metadata.user_id` 和 `x-session-id`）；若客户端希望隔离自己的提示词缓存，可在任意生成端点发送 `X-Basalt-Cache-Key: <name>`（1-256 字节），从而获得独立的缓存树（共享内存字节预算，但不与共享树或其他 key 共享前缀）；标题生成等服务内部请求绝不会触及该缓存。在 `--prompt-cache-mb` 预算约束下，缓存淘汰会从最久未使用分支的末端开始裁剪（混合模型的快照分支见下文说明，因其循环状态仅在末端有效，故须整支移除），从而确保空闲对话的系统提示词和工具定义保留最久；同时正在运行中的请求所复用的前缀节点绝不会被淘汰（对混合模型而言即其快照节点；该请求持有数据的独立副本，因此淘汰该分支的其余部分不会影响它）；此外前缀树最多保留 4096 个分支，超出时会丢弃最久未使用的空闲分支。用量统计会真实反映实际复用的数量：`prompt_tokens_details.cached_tokens`（Chat Completions）、`input_tokens_details.cached_tokens`（Responses）以及 `cache_read_input_tokens`（Anthropic，其中 `input_tokens` 仅统计剩余需要预填充的部分）。对于 Qwen3.5/3.6、Ornith-1.5 等混合注意力/SSM 模型以及带有滑动窗口层的 Gemma 4，由于其循环状态或环形缓存无法回退，服务端会在预填充提示词时，于请求文本末尾、对话模板闭合 token 之前保存一份状态快照；当下一轮对话延续该文本时，只需恢复该快照并预填充后续新增内容（在 Ornith-1.5-9B 的 12.6k-token 智能体提示词测试中，第二轮预填充耗时从约 36 秒降至约 1 秒）。复用轮次在首几个 token 之后可能会与无缓存运行存在细微差异：因为预填充的分块边界不同，会轻微影响 bf16 的数值舍入（在 Ornith 上实测：恢复出的状态与在相同节点切分的从头预填充结果逐比特相等，与一次性不分段预填充相比，末位置 logits 最大相差 1.56 且 top 5 排序完全一致；在 Gemma 4 12B 上恢复的缓存同样逐比特相等，在一组约 4k token 的 12 轮对话中，与单阶段预填充的 logits 差异为 0.60–2.33 且贪心输出 token 保持一致——这与将单阶段预填充的分块大小从 512 改为 256 token 造成的自身差异 2.53 处于同一量级，因此在 temperature 为 0 时接近平局的 token 仍有可能发生翻转）。此类混合模型仅能从这种快照处复用，因此拥有相同系统提示词的第二个独立对话仍需完整预填充。启用 `--kv-bits` 时始终完整预填充整个提示词。

**CacheBlend（显式开启，近似复用）。** 前缀树仅在提示词开头一致时生效；若智能体以不同顺序拼接了相同的几个文件，仍需全部重新预填充。开启 `--cache-blend` 后，系统还会额外保存每个提示词分块（chunk）的 KV 缓存——分块在每个空行后结束（无论分词器将其编码为单个 token 还是两个换行 token，因此由空行分隔的文件与消息各自保持独立分块；不足 16 个 token 的分块始终重新计算）——并在后续提示词的任意位置予以复用（key 在施加 RoPE 之前存储，并在复用时旋转至新位置）。由于缓存分块最初是在不同的上下文旁计算得到的，其 KV 存在上下文偏差：遵循 CacheBlend（arXiv:2405.16444）的做法，第一层会对所有 token 重新计算，而在后续层中仅重算占复用 token 总数 `--cache-blend-ratio` 比例的部分（默认 0.3）以及所有未命中缓存的 token。重算预算优先分配给每个复用分块开头的 32 个 token（分块位置移动时这部分上下文变化最大，参见 EPIC，arXiv:2410.15332），按其在第二层上 KV 偏差大小排序，然后再分配给剩余部分中偏差最大的 token。混合前向传播中每个重算 token 的计算开销约为模型原生预填充的 1.2×，因此仅当 `(1 − ratio) × 复用 token 占比` 至少达到 0.2，且整体缓存不超过 16 384 个 token 时才会触发混合计算；否则——如全新提示词、复用率过低、`--cache-blend-ratio 1` 或超长上下文——将直接走模型原生的分块预填充，并从中提取存储各分块（key 从施加 RoPE 后的状态反旋转还原，在缓存数据类型舍入精度内精确无损，bf16 下相对误差约 ~2⁻⁸）。该机制得到的是近似结果。在 M4 Pro 上使用 Qwen3-4B-4bit 对六个固定文件（550–900 token，`benchmarks/cacheblend`）按四种新顺序排列并分别提出新问题（约 5 066 个 token 中有 4 854 个来自缓存分块），采用教师强制（teacher forcing）对完整预填充的贪心参考答案（50 个 token）进行打分——统计 200 个位置样本上的参考 token 平均对数概率（log-prob）与 top-1 一致率（0.01 相当于 2 个位置）：

| 缓存策略 | log-prob | top-1 | 困惑度（perplexity） |
|---|---|---|---|
| full prefill（完整预填充） | −0.094 | 0.995 | 1.099 |
| ratio 0 | −0.180 | 0.925 | 1.198 |
| ratio 0.15（论文默认值） | −0.166 | 0.945 | 1.181 |
| **ratio 0.3（默认值）** | −0.123 | 0.960 | 1.131 |
| ratio 0.5 | −0.125 | 0.970 | 1.134 |
| ratio 1（原生预填充） | −0.094 | 0.995 | 1.099 |

优先重算分块头部正是将 0.3 设为默认值的原因：若仅按第 1 层偏差挑选，ratio 0.3 会留下 0.085 的困惑度差距，甚至 0.5 仍有 0.038 的差距；优先重算分块头部后，0.3 仅留有 0.032 差距（0.50 为 0.035）。在针对同一批文件的另外四种排列和问题测试中差距更大（0.3 时为 0.096，而无分块头部优先时为 0.169，旧默认值 0.5 时为 0.120），因此 0.05 的界限仅在该基准样本上成立，并不保证适用于所有提示词。`--cache-blend-ratio 0` 以牺牲质量换取最短的预填充耗时。仅支持 Llama 与 Qwen3 系列的稠密模型；其他模型保持精确前缀路径。开启该标志后，提示词缓存预算会在前缀树与分块存储之间对半平分。`cached_tokens` 仍只统计前缀树的精确命中 token；被混合复用的 token 计入 `prompt_tokens` 并输出日志（`[CacheBlend] Blended N prompt tokens: R from cached chunks, X of them recomputed`），前缀树也仅缓存混合提示词开头的精确前缀部分。执行混合前向传播时，内存中会短暂同时持有拼接后的缓存与各分块未旋转的 key/value（约为提示词本身 KV 占用的两倍，例如 16k token 的 Qwen3-4B 约需 ~5 GB），内存估算未将这部分峰值计入。在较低的 ratio 下，第 0–1 层的固定开销与偏差评估遍历占主导，因此 0.15 与 0.3 的实际耗时高于线性成本模型的预期。由于该特性会改变模型输出，默认处于关闭状态；设置 `--prompt-cache-mb 0` 同样会将其关闭。

隐私提示：所有客户端默认共享同一棵前缀树，且响应中的 `cached_tokens` 会暴露当前提示词与历史提示词有多少个起始 token 重合，因此调用方有可能借此探测其他客户端发送过的内容。在由互不信任的用户共享的服务器上运行服务时，请务必加上 `--prompt-cache-mb 0`；若启用了 `--cache-blend`，首字延迟（TTFT）同样可能泄露其他客户端发送过的文本分块。
