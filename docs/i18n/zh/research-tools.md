# 模型 X 光透视与研究工具（#328）

> Languages: [English](../../research-tools.md) · [Русский](../ru/research-tools.md) · **中文** · [Español](../es/research-tools.md) · [Français](../fr/research-tools.md) · [日本語](../ja/research-tools.md) · [Deutsch](../de/research-tools.md)

Basalt 的 MCP 服务器能够展示已加载模型的内部运行状态，并在其上执行小型、可逆的实验。只读 X 光透视（X-ray）工具始终在工具列表中公开；实验类工具仅在服务器以 `basalt serve --mcp --research-tools` 启动时才会被列出并接受调用。

所有调用均以后台优先级通过生成门控（generation gate）执行，因此用户请求始终优先处理。返回结果受体积限制：仅包含统计信息和小型切片，绝不会返回完整张量或完整注意力矩阵。

## 工具列表

下表中的示例均为 MCP `tools/call` 请求中的 `arguments` 对象。

### 只读 X 光透视

| 工具 | 返回内容 | 参数示例 |
|---|---|---|
| `model_describe` | 模型类型、架构、各层类型（全注意力、滑动窗口、线性注意力、mamba、conv）、注意力头数、KV 头数、头维度、窗口大小、专家数、KV 缓存类、量化方式（在基础精度之外同时包含 `MACaQ` / `FlatQuant` / `LIRQuant` 模块清单，`#630`）、参数量及字节数 | `{}` |
| `model_tensors` | 参数名称及其形状、数据类型和是否量化；`filter` 为参数名的子串过滤条件 | `{"filter": "layers.3.mlp"}` |
| `model_tensor` | 单个张量（反量化后）的均值、标准差、最小值、最大值、L2 范数以及 \|x\| 的 p1/p5/p50/p95/p99 分位数，外加最多 `maxValues` 个切片元素。传入 `"spectrum": true` 时还会返回矩阵（或通过切片选定的 3D 堆叠中单个专家）的谱信息：Marchenko–Pastur 噪声方差、体边缘（bulk edges）及超出体边缘的特征值质量、Gavish–Donoho 秩、幂律尾部指数 `alpha` 以及计算耗时（在 GPU 上计算较小一侧的 Gram 矩阵 WᵀW，在 CPU 上计算 `eigvalsh`；较小边 ≤ 8192；由于会占用模型数秒，仅在 `serve --research-tools` 下可用） | `{"name": "model.layers.3.input_layernorm.weight", "slice": [[0, 64]], "maxValues": 64}`, `{"name": "model.layers.3.mlp.down_proj.weight", "maxValues": 0, "spectrum": true}` |
| `kv_inspect` | 按命名空间划分的提示词缓存（prompt-cache）树：节点、叶节点、token 数、字节数、循环状态快照、固定（pin）状态、最近使用时间；以及预算与驱逐统计 | `{"namespace": "claude-code"}` |
| `trace_run` | 对单条提示词执行一次捕获运行：逐层隐藏状态范数、logit lens top-k、指定位置处各注意力头关注度最高的 token 位置与注意力熵、MoE 路由器选择 | `{"prompt": "The capital of France is", "layers": [0, 12, 23], "topK": 5}` |
| `telemetry_query` | 服务进程的实时聚合指标（#450）：总计值；包含 p50/p95 TTFT 与 tok/s、提示词缓存命中率及草稿接受率的滑动窗口统计；各决策模型的计数、费用与置信度十分位数；流式专家缓存计数器与逐层路由情况；逐 token 置信度十分位数；健康状态与内存占用。仅返回计数与计时数据 | `{"sections": ["windows", "decisions"]}` |

### 实验工具（`--research-tools`）

| 工具 | 功能说明 | 参数示例 |
|---|---|---|
| `intervene` | 在指定层之后向残差流注入引导向量（steering vector），或将某一个注意力头或某一个专家置零。若带有 `when`，则仅在触发条件选中的 token 上生效（条件消融）。仅对研究调用（`experiment_compare`、`trace_run`）生效。 | `{"kind": "ablate_head", "layer": 10, "head": 3}`, `{"kind": "ablate_expert", "layer": 20, "expert": 7, "when": {"kind": "projection", "layer": 14, "vector": [0.01, …], "threshold": 2, "window": 16}}` |
| `weights_delta` | 将 `values × scale` 加到内存中某个浮点张量的切片上（传入单个值时会广播到整个切片）。该操作会修改实时模型，因此用户请求也会受其影响（见下文）。 | `{"tensor": "model.layers.10.post_attention_layernorm.weight", "slice": [[0, 128]], "values": [0.05], "scale": 1}` |
| `overlays_list` | 列出当前生效的叠加层（overlay）：id、描述、创建时间与过期时间 | `{}` |
| `overlay_undo` | 按 id 撤销一个叠加层；权重增量（weight delta）会按比特精确恢复原样 | `{"id": 4}` |
| `experiment_compare` | 使用指定叠加层在一组提示词上执行贪婪解码前后对比：返回修改前后的输出文本、logits 的 KL 散度以及两者的困惑度（perplexity） | `{"prompts": ["2 + 2 =", "Name a prime above 100."], "overlays": [3, 4], "maxTokens": 32}` |
| `adapter_apply` | 向实时模型应用一个 LoRA 适配器：可以是 mlx-lm 适配器目录（`adapter_config.json` + `adapters.safetensors`），也可以是对单个线性层（含已量化层）施加的秩-1 更新 `scale × (x·input) output`。与权重增量一样会标记模型已修改；`overlay_undo` 可按比特恢复原层。若提供 `gate`，应用后会立即运行质量门控：通过则永久保留且无过期时间，失败则自动撤销并在报告中返回 `overlay: null`。未提供 `gate` 时将在 30 分钟后过期。 | `{"module": "model.layers.10.mlp.down_proj", "input": [0.01, …], "output": [0.02, …], "scale": 1, "gate": {"heldOut": ["The capital of France is"], "maxKL": 0.5}}` |
| `gate_check` | 将当前状态的模型与原始未修改模型进行质量门控对比（计算基线时暂时剥离所有实时适配器与权重增量）：返回每项检查的实测值、上限与判定结果，以及留出集（held-out）对比明细 | `{"heldOut": ["The capital of France is", "2 + 2 ="], "maxKL": 0.5, "maxPerplexityRise": 0.1, "target": [{"prompt": "17 + 25 =", "expected": "42"}]}` |
| `tensor_pack` | 在内存中对单个线性层的权重重新打包：`affine`（`bits` 为 2、3、4、5、6 或 8，`groupSize` 为 32、64 或 128）、`mxfp4`、`nf4:32`、`nf4:64` 或 `nf4:128`（QLoRA 的 16 级 NormalFloat 码本，每组配一个 fp16 absmax 缩放因子，以本分支 nf4 内核原生支持的 nf4 `QuantizedLinear` 形式保存），或 `ternary`（每 64 个元素一组，采用 absmean scale·{-1, 0, 1}）。若原层已量化，会先反量化再打包。该层会实时替换，行为与权重增量相同（见下文） | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "affine", "bits": 2, "groupSize": 64}` |
| `tensor_unpack` | 在内存中将单个已量化线性层实时替换为持有其反量化权重的稠密 `float16` 或 `bfloat16` 层（默认 `bfloat16`） | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "bfloat16"}` |
| `train_experts` | 自主训练（#332 模式 3，长耗时任务）：收集可验证任务的回答，仅保留通过校验的样本，训练一个 LoRA 并保存，报告训练前后的损失与留出集准确率，再通过质量门控将其应用到模型上。参见[自主训练](#自主训练train_experts) | `{"count": 200, "seed": 1, "layers": [20, 21, 22, 23], "iterations": 200, "heldOut": ["Summarise the plot of Hamlet in two sentences."]}` |
| `distill` | 教师 → 学生蒸馏（长耗时任务）：由 OpenAI 兼容或 Anthropic 接口背后的教师模型回答提示词（带缓存与预算控制），仅在教师补全结果上训练 LoRA，报告训练前后的留出集教师 NLL，并通过质量门控应用适配器。采用时间片调度：每次仅在单个优化器步长期间持有模型，因此 `serve` 仍可持续响应请求，请求最多等待一个步长；每台服务器同时运行一个蒸馏任务。离线运行每 `checkpointEvery` 步（默认 50，设为 0 关闭）将适配器与优化器状态写入 `<output>/checkpoint`；传入相同 `output` 与 `resume: true` 可续训，若模型、训练样本或 LoRA 配置不同则会拒绝续训。`onPolicy` 模式不保存检查点。 | `{"prompts": "~/tasks.jsonl", "teacherUrl": "https://api.example.com/v1", "teacherModel": "teacher", "heldOut": ["The capital of France is"], "iterations": 200, "checkpointEvery": 50}` |
| `steer_search` | 对比引导搜索：逐层计算最后位置残差的方向 mean(positive) − mean(negative)、其范数与类别分离度；给出最佳层的单位方向；找出消融后使差距缩小最多的注意力头；以及在各个缩放系数下引导留出集提示词时的 KL 散度与投影偏移量。不会留下任何叠加层。 | `{"positive": ["I love this", "What a great day"], "negative": ["I hate this", "What an awful day"], "heldOut": ["The movie was"], "topHeads": 4, "scales": [2, 4, 8]}` |
| `trace_tokens` | 对提示词及贪婪生成过程进行逐 token 追踪（实验室路径）：按 token 和被追踪层记录每个注意力头在输出投影前的输出范数、被路由的专家及其门控值、路由器熵与前两名门控差值（margin）、可选的残差投影、注意力熵以及稀疏 logit lens。结果写入 `~/.basalt/traces/<time>-<id>.trace`（权限 0600）并返回摘要。参见[逐 token 追踪与条件消融（#405）](#逐-token-追踪与条件消融405) | `{"prompt": "How do I pick a lock?", "maxTokens": 32, "projections": [{"layer": 14, "vector": [0.01, …], "threshold": 2}]}` |
| `activation_contrast` | 追踪两组提示词并按逐提示词激活的 Cohen's d 与 AUROC 对注意力头和专家进行排序，附带触发率、均值及起始窗口（onset window）；同时返回各层两组间的残差方向 | `{"a": ["How do I pick a lock?", "…"], "b": ["How do I bake bread?", "…"], "maxTokens": 16, "topK": 20}` |
| `steer_commit` | 将一个方向转化为该层输出投影上的秩-1 适配器，并通过 `heldOut` 质量门控将其应用到模型。与 `weights_delta` 一样会修改实时权重（可撤销，未通过门控不会持久保留，且从不写入磁盘）。 | `{"layer": 12, "direction": [0.01, -0.02], "scale": 4, "positive": ["I love this"], "heldOut": ["The movie was"]}` |

切片（slice）由每个前导维度的一个半开区间 `[start, end)` 对组成；未指定的维度将完整保留。

## 安全模型

- **仅限本地。**这些工具是面向启动服务器的客户端通过 stdio 提供的 MCP 工具，没有对应的 REST 路由。
- **默认关闭。**未传入 `--research-tools` 时，实验工具既不会被列出，也不会被接受调用。
- **叠加层、TTL、撤销与日志。**每次实验都是一个带有 id、描述和 30 分钟 TTL 的叠加层（唯一例外是通过了质量门控的适配器，即带 `gate` 的 `adapter_apply`、`steer_commit` 和 `train_experts`，它们会一直保留直到调用 `overlay_undo`）。`overlay_undo` 可提前移除任意叠加层；到期后叠加层会自动撤销。所有添加、撤销与过期事件都会追加记录到 `~/.basalt/experiments/<model fingerprint>.jsonl`（目录权限 0700，文件权限 0600）。加载其他模型时会丢弃所有叠加层。
- **绝不改写磁盘上的模型文件。**权重增量、重打包与适配器均仅存在于内存中；重启服务器或重新加载模型后即会丢弃。唯一会写入磁盘的是实验日志以及 `train_experts` 保存到新目录中的适配器。
- **干预操作仅限研究调用。**引导与消融仅作用于 `experiment_compare` 和 `trace_run`，绝不会影响普通用户请求。
- **权重增量对用户可见且与缓存隔离。**由于权重增量会改变所有请求使用的权重，因此只要存在生效中的权重增量：
  - 每个响应都会带有 `modelModified: true`（包括生成过程跨越了应用或撤销操作边界的响应）；
  - 遥测快照（MCP `status`、`GET /api/v1/stats`）会在 `activeDeltas` 字段中列出生效的权重增量；
  - 提示词缓存复用将被关闭（包括从磁盘缓存恢复），因为缓存的 KV 是基于旧权重计算的；
  - 在应用和撤销权重增量时，内存中的提示词缓存会被清空，且跨越这两个变更时刻正在运行的请求不会写入缓存。
- **边界校验。**目标张量必须存在，切片必须非空且在有效范围内（最多 16M 个元素），传入的值数量必须等于切片元素数或为单个用于广播的值，所有值必须为有限数，且 `|scale| ≤ 10`。同一张量同时只能有一个生效中的权重增量。
- **仅支持浮点张量。**权重增量仅可修改归一化层权重、偏置以及未量化的嵌入层或投影层（float16、bfloat16、float32）。已量化层的张量（无论是打包权重还是其缩放因子和偏置）均会被拒绝：量化权重写入路径不在支持范围内。

## 质量门控与适配器（#332）

`gate_check` 与 `adapter_apply` 中的 `gate` 会在一次独占模型调用内，将当前模型与原始模型进行对比（计算基线时会暂时剥离所有实时适配器与权重增量，随后再恢复）。检查项包括：

- **`probe perplexity`**：加载探针（load probe）在固定文本上的困惑度必须低于模型族上限；若原始模型本身已高于上限，则相对原始值的增幅不得超过 `maxPerplexityRise`。
- **`probe known answer …`**：原始模型能够答对的每一道加载探针已知答案题，修改后仍必须答对（实测值为 1 或 0，上限取决于原始模型结果）。
- **`held-out mean KL`** 与 **`held-out perplexity rise`**：在每个 `heldOut` 提示词上，以原始模型为基线运行 `experiment_compare` 的机制；沿基线文本的下一 token 分布平均 KL 散度不得超过 `maxKL`（默认 0.5），且该文本的平均相对困惑度增幅不得超过 `maxPerplexityRise`（默认 0.1）。
- **`target …`**：每条目标提示词的贪婪生成回答（优先套用对话模板，其次原始提示词，最多 16 个 token）必须不区分大小写地包含 `expected`。

出现非有限值（NaN/Inf）时将按最大有限浮点数报告并判定失败。关于适配器：

- 目录形式的适配器通过 mlx-swift-lm 的 LoRA 支持加载；同一时间仅允许加载一个目录适配器，且其张量形状与配置的秩必须与模型层匹配。
- 秩-1 适配器需要为层的每个输入特征提供一个 `input` 值、为每个输出特征提供一个 `output` 值；目标层必须是未挂载适配器的普通 `Linear` 或 `QuantizedLinear`。
- 适配器仅包装普通的 `Linear` 与 `QuantizedLinear` 层，因此撤销后可以按比特精确恢复原权重，使 logits 与原始模型完全一致。整个过程不向磁盘写入任何内容。

## 如何演进为学习闭环（#328 第 3 层）

这些工具构成了上层能力的测量仪器与实验沙盒：`distill.collect` 收集提示词与参考答案，`adapter.train` / `apply` / `evaluate` 训练并试运行小型适配器，而 `weights.propose` 接收外部提出的权重修正方案。每个候选修改都会先在此作为叠加层或权重增量运行，使用 `experiment_compare` 进行对比，仅当加载时探针、参考 logits、对等测试运行器（parity runner）以及特性矩阵全部通过后才会被接纳。本层本身不会持久化任何变更；由上层学习闭环决定哪些变更（如有）最终变为永久修改。

## 张量打包与解包（#332）

`tensor_pack` 与 `tensor_unpack` 都会就地修改单个线性层：读取其权重（若已量化则先反量化），执行格式转换，并整体替换模块（`Linear` ↔ `QuantizedLinear`），因此内存字节占用会真实发生变化。返回值会报告此次转换的精度代价（以下为示例值）：

```json
{"overlay": {"id": 7, "description": "tensor_pack model.layers.10.mlp.down_proj.weight: 4-bit affine g64 → 2-bit affine g64", "createdAt": "…", "expiresAt": "…"},
 "bytesBefore": 9437184, "bytesAfter": 5505024, "rmse": 0.0041, "maxAbsError": 0.031, "relativeError": 0.29}
```

`rmse`、`maxAbsError` 与 `relativeError`（‖W − Ŵ‖ / ‖W‖）用于将新层反量化后的权重与旧层原有数值进行对比。解包为缩放因子自身的浮点类型时是无损的（三项误差均为 0）。

打包操作作为与权重增量同类的叠加层存在：模型会被标记为已修改，提示词缓存复用关闭，`overlay_undo`（或 30 分钟 TTL 到期）会按比特精确换回原层，且全程不写入磁盘。仅支持对普通 `Linear` 和 `QuantizedLinear` 层进行打包；MoE 专家堆叠、嵌入层、已包装 LoRA 的层以及已带有权重增量或其他打包的层均会被拒绝。

`experiment_compare` 直接在实时权重上运行，因此打包对模型的影响可通过打包前后的两次运行结果差值来衡量：

1. 调用 `experiment_compare` `{"prompts": ["2 + 2 =", "Name a prime above 100."], "maxTokens": 32}`：记录 `baselineText` 与 `baselinePerplexity`。
2. 调用 `tensor_pack` `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "ternary"}`：记录返回的叠加层 id 与误差指标。
3. 以相同参数再次调用 `experiment_compare`：与第 1 步相比 `baselineText` 和 `baselinePerplexity` 的变化即为此次打包对 logits 的影响。
4. 调用 `overlay_undo` `{"id": 7}`，再次执行第 1 步即可得到与第 1 步逐比特完全一致的结果。

## 自省（`introspect`，#332 模式 1）

在启用 `--research-tools` 时，`introspect` 允许正在服务的模型研究自身。给定一个目标后，服务器会通过常规生成路径运行该模型，并将 X 光透视工具作为工具定义提供给模型；模型每次发起的工具调用都会通过与 MCP 工具相同的门控研究路径执行，随后将结果反馈给模型并继续追问。

| 参数 | 含义 |
|---|---|
| `goal`（必填） | 需要查明的目标，例如 `"find which heads attend to the last tool call"` |
| `maxSteps` | 循环停止前的最大模型轮数，范围 1–32（默认 8） |
| `allowInterventions` | 是否同时提供 `intervene`、`experiment_compare` 与 `overlay_undo`（默认 false；只读工具始终提供） |
| `resultBudget` | 反馈给模型的每次工具结果最大字符数，范围 200–20000（默认 2000）；超出部分将被截断并标注 |

- **每轮一次调用。**当模型输出不含工具调用的回答、达到 `maxSteps` 上限或重复了此前已经执行过的调用（工具与参数相同，忽略键顺序）时，循环终止；报告中的 `stop` 字段会说明具体终止原因。
- **假设记录。**系统会要求模型按 `Hypothesis:` / `Test:` / `Outcome:` 的行格式撰写每条假设。解析器采用宽松解析（兼容 markdown、编号以及用 `Result:` 替代 `Outcome:`），重新陈述的假设会替换早期版本，并在新版省略 `Test:` 或 `Outcome:` 时保留原行内容。
- **错误回传给模型。**格式错误或执行失败的工具调用会转化为 `error: …` 结果返回给模型查看；若生成本身失败或被取消，则循环结束。
- **干预不会延续到循环之外。**循环期间创建的每个叠加层都会在循环结束时被撤销，出错或取消时亦然；`undone` 字段列出已撤销的叠加层。若服务器拒绝清理（例如正忙），`cleanupFailure` 会记录原因，这些叠加层将在 TTL 到期后自动失效。`overlay_undo` 会拒绝撤销在循环开始前就已存在的叠加层。
- **失败轮次保留已有报告。**若在循环中途生成失败，此前已完成的步骤仍会随 `stop: "failed"` 返回，错误信息记在 `failure` 中。
- **日志记录。**每一步都会向 `~/.basalt/experiments/introspect.jsonl` 追加一行 `introspect` 记录（包含模型、步数、工具、参数以及结果开头部分）。

报告包含 `goal`、`steps`（模型文本，若包含调用则附带 `tool`、`arguments` 及截断后的 `result`）、`hypotheses`（`statement`、`test`、`outcome`）、`answer`、`stop`、`failure`、`undone` 与 `cleanupFailure`。

```json
{"goal": "find which heads attend to the last tool call", "maxSteps": 6, "allowInterventions": true}
```

## 自主训练（`train_experts`）

单次调用即可通过生成门控在后台优先级运行完整的训练闭环，在训练阶段本身持有模型：

1. **任务准备。**通过 `count` + `seed` 生成带已知答案的整数 `+`、`-`、`*` 计算题（相同种子生成相同题目）；也可改用 `file` 读取包含 `{"prompt": "...", "answer": "..."}` 的 JSONL 行。其中五分之一的任务（至少一条）会作为留出集（held-out），绝不参与训练。
2. **样本收集。**模型通过对话模板（关闭思考模式，上限 64 tokens）以贪婪解码回答每道题。若配置了 `teacher`（OpenAI 兼容服务器的基础 URL；`teacherModel` 指定其模型名，仅当教师服务的 scheme、host 和 port 与 `BASALT_TEACHER_URL` 匹配时才会将 `BASALT_TEACHER_API_KEY` 作为 bearer token 发送，以防客户端将密钥发往他处），教师模型也会回答训练任务。
3. **结果校验。**仅当回答的最后一个整数（忽略逗号）等于数值型已知答案，或最后一行在忽略大小写及首尾标点后等于文本答案（或以 `: <answer>` 结尾）时，该样本才会被保留。错误、空值及格式不符的回答全部丢弃。**绝不学习任何未经校验的数据：**若没有一条回答通过校验，报告中 `verified: 0`，不会训练适配器、不会写入磁盘，模型也保持原样。
4. **训练。**使用本分支的 `LoRATrain`，在已通过校验的提示词 + 回答对（按模型所见格式渲染）上对 `layers`（默认全部层）的注意力投影及稠密/共享专家投影训练一个 LoRA（参数为 `rank`、采用学习率为 `learningRate` 的 Adam 优化器、训练 `iterations` 步）。暂不支持按专家粒度的 LoRA——本分支的 `SwitchLinear` 专家层不接收适配器——因此传入 `experts` 会报错拒绝。若模型已挂载适配器也会被拒绝。取消调用会在下一次迭代时停止训练并保持模型不变。
5. **保存。**`adapter_config.json` + `adapters.safetensors`（mlx-lm 适配器格式）会写入 `output` 或 `~/.basalt/adapters/<model fingerprint>/<timestamp>`（权限 0700）。`output` 必须尚不存在，且禁止位于模型目录内部：绝不会覆盖已有文件，也不会在源权重旁写入任何文件。
6. **评估与应用。**在挂载该适配器的状态下测量留出集准确率，随后移除训练层，并通过 `applyAdapter` 经由基于 `heldOut`、`maxKL` 和 `maxPerplexityRise` 构建的质量门控将其应用：通过则成为无过期时间、可通过 `overlay_undo` 移除的标记叠加层，否则被门控拒绝。

报告包含 `collected`（为训练收集的回答数）、`verified`、`losses`（每个报告步长的平均训练损失，最多 20 个）、留出集上的 `accuracyBefore` / `accuracyAfter`、`adapterDirectory` 以及门控的 `verdict`。

## 对比引导（“冥想”，#332 模式 2）

这是一种针对激活方向的定向搜索，而非某种自我意识。这里并不存在模型“自我反思”：basalt 只是将两组提示词输入模型，对隐藏状态取均值后相减，并测量这一差值带来的影响。

`steer_search` 会通过一次带探针的前向传播运行所有正例与负例提示词，并保存每层最后一个位置之后的残差。

- **逐层方向：**mean(positive) − mean(negative)。`norm` 为其向量长度；`separation` 为两类样本在单位方向上的投影均值之差（数值等于 `norm`）除以这些投影的合并标准差。分离度最高的层为最佳层，`direction` 即为该层的单位向量（维度为隐藏层大小）。
- **注意力头：**依次对最佳层及其前一层的每个注意力头执行消融（即 `ablate_head` 干预），`effect` 表示最佳层上正负例投影均值差距缩小的幅度。后续层的注意力头无法改变该层的残差，因此不予测试。最多报告前 `topHeads` 个注意力头，按效果从大到小排序。由于每个注意力头和每条提示词都需要执行一次前向传播，在大型模型上请保持提示词集合较小，或传入 `topHeads: 0`。
- **缩放试探：**对每个缩放系数，在最佳层之后用 `direction × scale` 对留出集提示词进行引导（即仅限研究用途的 `steer` 干预）。`klDivergence` 为相对未引导运行结果的下一 token 分布平均 KL 散度；`projectionShift` 为在最后一层之后测得的向 `direction` 投影的平均变化量。根据构造，在被引导层该偏移量即等于缩放系数，因此最后一层的测量值能够反映有多少引导推动力保留到了输出端。
- 整个过程不注册任何叠加层：消融与引导仅存在于本次调用内部。

`steer_commit` 会通过质量门控将一个有效的方向固化为当前会话的持久修改。它会选取该层写入残差流的稠密投影（优先 `mlp.down_proj`，其次 `feed_forward.w2`，再次 `self_attn.o_proj`；不含稠密投影的 MoE 层会被拒绝），测量该模块在 `positive` 提示词最后位置上的平均输入 `m`，并调用适配器路径 `.rankOne(module: <full module path>, input: m / (m·m), output: direction, scale: scale)`，从而施加更新 ΔW = scale · direction ⊗ m / (m·m)。

需要坦诚说明这一近似的局限性：

- 该更新会为每个 token 在投影输出上增加 `scale · (x·m / m·m) · direction`。仅当模块输入 `x` 在 `m` 上的投影与正例均值相同时，结果才严格等于引导向量；其他 token 获得的引导可能偏多、偏少甚至反向。而 `steer` 则会对每个位置加上完全相同的向量。
- 仅当该投影的输出直接加回残差流（如 Qwen 和 LFM 等 pre-norm 结构）时，它在残差中的作用才是精确的。若投影之后还接有归一化层（如 Gemma 的 post-attention 与 post-feedforward norm），向量会被重新缩放，仅大致保留方向。当回退使用 `self_attn.o_proj` 时，该向量在到达该层末尾的残差前还会经过该层的 MLP。
- 仅当质量门控通过时（基于 `heldOut` 上的 `maxKL` 与 `maxPerplexityRise`），适配器才会被接纳；请先运行 `steer_search` 试探，再提交 KL 可接受的缩放系数。


## 逐 token 追踪与条件消融（#405）

针对诸如拒答等行为的标准分析流程为：追踪 → 对比 → 条件消融 → 对比评估。

1. 在少量表现出该行为的提示词上运行 **`trace_tokens`**。本分支的 `ForwardProbe.headOutputs` 会在输出投影之前（对于 Qwen3.5 则在输出门控之前）交出每个注意力头的输出，从而保持融合注意力开启；`routerTopK` 会交出该层实际组合使用的专家与门控值（包含专家偏置、分组 top-k 与缩放）。针对每个 token 和被追踪层，文件会记录：
   - `heads`：每个注意力头输出的 L2 范数；
   - `experts`（整数）与 `gates`；`routerEntropy`（基于路由器 logits 的 softmax）与 `routerMargin`（最高与次高门控之差）；
   - `projection`：针对每个 `projections` 条目记录层后残差与方向的点积（residual · direction）；
   - `attentionEntropy`（按需开启，使用显式注意力，速度较慢）以及 `lens`/`lensProbabilities`（按需开启的 top-k logit lens）。

   文件格式：首行为 JSON 头（`layout.floats` 与 `layout.ints` 按列顺序列出 `{layer, field, width}` 分段），随后每个 token 一行 JSON（`position`、`id`、`token`、`generated`、`ints`），最后是 `tokens × floatsPerToken` 个 float32 小端序浮点数。运行前会检查上限：提示词加 `maxTokens` ≤ 2048 tokens，层号在模型范围内，被追踪层上最多 16 个隐藏维度大小的投影，且预估文件大小 ≤ `maxFileBytes`（≤ 64 MiB；写入前还会再次校验编码后大小）。
2. 使用表现出该行为的集合 `a` 与不表现该行为的集合 `b` 运行 **`activation_contrast`**。每个组件在逐提示词上的平均激活（注意力头范数；专家被路由时为其门控值，否则为 0）用于计算 Cohen's d 和 AUROC；`rateA`/`rateB` 统计注意力头范数高于其合并均值或专家被路由到的 token 比例；`onsetA`/`onsetB` 对生成开始前的 `onsetWindow` 个提示词 token 取平均。`bestLayer` 处的 `direction` 即为两组间的残差方向。
3. 使用带 `when` 条件的 **`intervene`**：仅在触发器命中时——即在对比方向上的投影高于阈值，或某专家被路由且其门控高于 `gate`——对排名靠前的注意力头或专家进行消融（或以负缩放系数进行引导以减去某一方向），还可选择从首次触发起持续作用 `window` 个 token。动作所在层必须位于触发层之后；触发条件在同一前向传播中读取（每次前向传播每个条件叠加层仅需一次 GPU 同步），因此编辑会同时作用于提示词和生成过程中触发的当前 token。从未触发的条件干预会使 logits 保持逐比特一致。
4. 使用条件叠加层运行 **`experiment_compare`**，并与同一组件的静态 `ablate_*` 叠加层以及 `steer_commit` 适配器进行对比：查看 KL 散度、困惑度、生成文本以及 `triggerRate`（条件编辑实际生效的 token 占比）。

尚未接入的功能：将 System One 读数作为触发器或标注器、采样（非贪婪）追踪、以对话消息列表作为输入（需自行套用对话模板），以及用于测量拒答率、准确率和解码速度的专用评估运行器。

## 新增功能（`v0.12`）

- **原始 X 光追踪中的自动 BOS 处理（`#630`）：**`trace_run`、`trace_tokens`、`experiment_compare`、`contrast`、`activation_contrast`、`steer_search` 与 `TokenAttribution` 共用 `ModelProbe.encoded(_:_:)`，使即便 `encode(addSpecialTokens: true)` 省略 `bosToken` 的分词器（如 Gemma 4）也能在位置 0 收到 `<bos>`，在避免双重 BOS 的同时保留 BOS 注意力汇聚层（attention-sink layers，`L43–L56`）。
- **逐 token logprobs、归因与幻觉风险（`#613`、`#614`）：**`POST /v1/xray/tokens` 与生成接口暴露逐 token logprobs、top-k 熵、层/注意力头/MoE 归因以及幻觉风险信号。
- **机制外科手术与影响力归因（`#600`、`#622`）：**支持 GQA KV 分组修补、认知零空间投影、反引导防护、`MAO-SNMF` 特征分解、`Score d'influence` 注意力头归因（`HeadInfluenceCore`）以及温度缩放路由（`CalibratedRouterGate`）。
- **人机协同（HITL）治理（`#626`）：**`HITLGovernanceCore` / `HITLApprovalActor` 通过 `/v1/mcp/approvals` 对高影响力的 MCP 操作进行门控审批，并与 LAAS-CNRS `SymbolicKnowledgeGraph` 配合使用。
