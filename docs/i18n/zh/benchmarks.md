# 特性与基准测试

> Languages: [English](../../benchmarks.md) · [Русский](../ru/benchmarks.md) · **中文** · [Español](../es/benchmarks.md) · [Français](../fr/benchmarks.md) · [日本語](../ja/benchmarks.md) · [Deutsch](../de/benchmarks.md) · [한국어](../ko/benchmarks.md) · [Bahasa Indonesia](../id/benchmarks.md)

## 核心特性一览

除非表格中另有说明，所有加速特性均为可选开启，且不会改变贪心采样输出。实测数据基于连接电源的 M4 Pro 24 GB、release 构建。

| 特性 | 参数标志 | 作用机制 | 实测表现 |
|---|---|---|---|
| Prompt lookup（投机解码） | `serve --speculative-lookup` | 从提示词或已生成输出中复制并草拟最多 16 个 token，在单次前向传播中完成验证 | 代码编辑场景 1.84–2.66×（Bonsai 2 27B）、3.45–3.62×（Gemma 4 12B）；纯文本散文约 1.0× |
| MTP 头草稿生成 | `serve --mtp [--mtp-block N]` | 使用模型的多 token 预测头（`mtp.safetensors`，如 MTPLX 模型包）生成草稿；在任意温度下均保持数学等价。每轮从草拟 1 个 token 起步，连续四轮全部接受后逐步增至 N−1（默认 4）；一旦出现拒绝即回退至 1 | 代码 1.30×、散文 1.04×、推理 1.26×（Bonsai 2 27B，T=0.7）；后续轮次与普通解码一样复用提示词缓存：在 1,149-token 对话轮次中 TTFT 从 12.5 s 降至 0.75 s（Bonsai 2 27B） |
| Ternary Bonsai 2 | （自动识别） | 加载经 Hadamard 旋转的三值化检查点（`prism_hadamard_qwen35`） | 27B 模型仅占 7.4 GB |
| 向量嵌入与重排序 | `serve --embedding-model`、`--rerank-model` | 与对话模型并行提供 OpenAI `/v1/embeddings` 与 Cohere/Jina `/v1/rerank` 接口 | bge-small 229–299 texts/s，Qwen3-Embedding-0.6B 20 texts/s（batch 32） |
| 磁盘持久化提示词缓存 | `serve --prompt-cache-disk-mb N` | 恢复会话或重启服务时，直接加载最长匹配前缀而无需重新预填充 | — |
| KV 缓存量化与分区 | `serve --kv-bits 4\|8\|auto\|k8v4`、`--kv-zones` | 在同等内存中容纳更长上下文 | 16k 提示词峰值内存 1.03 GB 对比 3.11 GB（Qwen3-4B，4-bit 分区） |
| 并发请求控制 | `serve --max-concurrent N` | 每个运行中的请求预先保留其 KV 空间；空间不足时返回 503 `Retry-After` 而非内存溢出 | — |
| DRY 与 XTC 采样器（实验性） | `serve --experimental-samplers` | `dry_*` / `xtc_*` 请求字段 | 并非防死循环工具：在 2-bit 模型上 DRY 会把重复循环掩盖在失真事实背后（#177） |

## README 中数据的来源

README 首页上的每项数据均来自有记录的实际运行，且引入该特性的 PR 中附有完整表格、提交哈希与日志路径。除非 PR 另有说明，测试环境均为 M4 Pro 24 GB、release 构建。

| 数据声明 | 来源 |
|---|---|
| Qwen3-4B 78–89 tok/s；Ternary Bonsai 2 27B 占 7.4 GB 达 20.9–21.5 tok/s，代码场景开启 prompt lookup 达 31–38 tok/s | [#239](https://github.com/goodmai/basalt/pull/239)，源自 [#218](https://github.com/goodmai/basalt/pull/218)、[#231](https://github.com/goodmai/basalt/pull/231)、[#232](https://github.com/goodmai/basalt/pull/232) |
| Bonsai 2 27B 启用自带 MTP 预测头，代码场景达 27.1 tok/s | [#263](https://github.com/goodmai/basalt/pull/263) |
| Gemma 4 12B 开启 prompt lookup 提速 3.45–3.62×，贪心输出不变 | [#265](https://github.com/goodmai/basalt/pull/265) |
| Ornith-1.5-9B（12.6k-token 提示词）第二轮智能体响应从约 36 s 降至约 1 s | [#82](https://github.com/goodmai/basalt/pull/82) |
| Qwen3-4B 启用 KV 分区后 16k 提示词峰值内存 1.03 GB 对比 3.11 GB | [#159](https://github.com/goodmai/basalt/pull/159) |
| 多模型混合路由：所有标签准确率 100%，每千次请求 $0.035，p95 延迟 4.8 s 对比 10.0 s | [#418](https://github.com/goodmai/basalt/pull/418) |
| Mac 应用：热启动窗口 257 ms，空闲唤醒 0 次，流式 token p95 7.4 ms，快捷提问 40 ms | [#463](https://github.com/goodmai/basalt/pull/463) |
| 在 24 GB 上将 9B 裁判模型与 Gemma 4 12B 并行：4-bit 可装入（15.3 GB），准确率 98.9%，减少 77% LLM token | [#465](https://github.com/goodmai/basalt/pull/465) |
| Qwen3.6-35B-A3B-8bit（37.7 GB）在 24 GB 上从 SSD 流式推理达 8.77 tok/s，0 swap | [#477](https://github.com/goodmai/basalt/pull/477)、[#478](https://github.com/goodmai/basalt/pull/478) |
| 量化后 `--experimental-pv-tune`：3 分钟内恢复 2–3-bit 8.7–29% 困惑度损失 | [#486](https://github.com/goodmai/basalt/pull/486) |
| QTIP 移位网格 `--experimental-qtip`：`Qwen3.8-27B` 2.5 bpw 占 10.00 GB，16k PPL 4.621 优于 3-bit 仿射及 PV-Tuning（4.702 / 4.652）；2.0 bpw 占 8.48 GB，2-bit PPL 降低 72.1%（5.345 vs 19.146） | [#494](https://github.com/goodmai/basalt/pull/494) |
| Metal 上原生执行 `.gguf` GSQ-RCO / `i-quant`（`IQ4_XS` / `IQ3_S`） | [#433](https://github.com/goodmai/basalt/pull/433) |
| 递归模型 `Flint-40B-A2.5B`（`oQ3e`）在 M4 Pro 24 GB 上以 17.61 GB 内存实现 67.02 tok/s 解码（273.5 tok/s 预填充） | [#522](https://github.com/goodmai/basalt/pull/522) |
| `Zyphra/ZONOS3-Mini-hf` 44.1 kHz 神经 TTS 占 2.24 GB，在 EN/DE/RU 上实现 RTF 0.228–0.263（3.8–4.4× 实时） | [#527](https://github.com/goodmai/basalt/pull/527) |
| 84 套按硬件分档的 Compose 配置方案（`16–256 GB × 12`），支持 `Basalt.app` 按内存档位自动推荐 | [#528](https://github.com/goodmai/basalt/pull/528) |
| 单遍 `MACaQSwitchLinear` WHT + `gatherQuantizedMM`：MoE 内核提速 `3.41×`（`2.06 → 0.60 ms`），Gemma 4 26B-A4B 端到端提速 `+82%`（`49.2–54.8 tok/s`），Ornith 35B-A3B 3-bit 达 `64–75 tok/s` | [#628](https://github.com/goodmai/basalt/pull/628) |
| `Qwen3.5-9B-PV-MACaQ-4bit`：与 8-bit 保持 `4/5` 阶梯测试完全持平，解码提速 `+52%`（p50 `43.2 tok/s`），内存 `−33%`（`6.68 vs 9.97 GB`），最大层漂移 `3.37%` | [#545](https://github.com/goodmai/basalt/pull/545)、[#630](https://github.com/goodmai/basalt/pull/630) |
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`：在 24 GB Mac 上占用 `15.1 GB` 内存（峰值 `18.09 GB`），以 p50 `14.1 tok/s` 取得 `4/5` Ladder + `4/6` Lunar（4-bit 参考模型会 OOM） | [#628](https://github.com/goodmai/basalt/pull/628)、[#630](https://github.com/goodmai/basalt/pull/630) |

README 中列出的黄金测试模型会在 CI 中由 `scripts/check_readme_models.swift` 自动校验：其中列出的每个模型都必须链接到 `Tests/BasaltTests/Fixtures/golden` 中真实存在的测试固件。

独立的静态基准测试页面（[`docs/benchmark-page/index.html`](../../benchmark-page/index.html)）以及可用于评审的 Hugging Face 合集清单（[`docs/benchmark-page/hf-collection.json`](../../benchmark-page/hf-collection.json)）均由 `swift scripts/build_benchmark_page.swift` 基于上述表格、`benchmarks/` 目录和黄金固件生成（并在 CI 中通过 `--check` 校验；本地传入 `--logs logs` 可附加运行器 `summary` 表格）。是否对外公开发布仍由人工决定。

## 基准测试

两套测试套件均直连正在运行的服务器，并输出可在你自己机器上复现的真实指标。

**登月阶梯测试（The ladder to the moon）** — `benchmarks/ladder/run.py`。共五级阶梯，难度逐级递增，最终以完整的地月转移任务收官。评分完全不看文字表面是否顺畅：系统会把模型自己编写的代码放到提示词从未展示过的输入上实际运行，因此看似合理实则错误的回答同样无法通过。

| 阶梯 | 任务 | 实际考察内容 |
|---|---|---|
| 1 | 双二次方程（实根） | 解析闭式解及边界情况：无解、重根、零根 |
| 2 | 双二次方程（复根） | 分支处理——求取全部四个根及其重数 |
| 3 | 傅里叶系数 | 数值积分，使用锯齿波、方波与余弦函数校验 |
| 4 | 泵扬程计算 | 单位换算（mm 与 m、L/s 与 m³/s）+ 无闭式解的 Colebrook 方程求解 |
| 5 | **地球 → 月球任务** | 由 `sqrt(2ηQ)` 计算排气速度、逐级 Tsiolkovsky 方程、四体轨道传播、绕月制动、软着陆——所有物理常数均在提示词中给出，无法靠背诵教科书蒙混过关 |

最高级任务依据六个不变量计分，而非单一最终答案；这样即便模型算对了能量学部分但未能完成轨迹传播，仍能得到区分度明确的分数。任务提示词位于 `benchmarks/ladder/lunar_prompt.md`；`lunar_task.md` 中的参考数值绝不会暴露给模型。

`--warmup` 会在正式测试前追加三道廉价冒烟题（代数、斐波那契、翻译），任何指令微调模型都应顺利通过。它们不计入阶梯总分。

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`，严格遵循官方 pass@2 规则。数据集未内置在仓库中，请先将其下载至 `benchmarks/arc-agi/dataset/`（参见 `benchmarks/arc-agi/ARC_AGI.md`）。

### 测试结果

在配备 24 GB 统一内存的 M 系列 Mac 上测得。`thinking` 指对话模板中的 `<think>` 思考块：推理模型默认开启，可通过 `--reasoning-effort none` 关闭。

| 模型 | 热身 | 阶梯得分 | 最高阶梯 | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`（关闭思考，24 GB） | 3/3 | **4/5** | 4/6 不变量 | — | 11.6–13.5（p50 14.1） |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`（关闭思考，24 GB） | 3/3 | **4/5** | 0/6（超出长度） | — | 9.4–11.4（p50 11.6） |
| `Qwen3.5-9B-PV-MACaQ-4bit`（关闭思考） | 3/3 | **4/5** | 0/6（纯文本未出代码） | — | 35.6–41.6（p50 43.2） |
| `mlx-community/Qwen3.5-9B-8bit`（参考模型，关闭思考） | 3/3 | **4/5** | 0/6（纯文本未出代码） | — | 24.7–27.3 |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit`（关闭思考） | 3/3 | **4/5** | 5/6 不变量 | — | 56.0–63.4 |
| `goodmai/gemma-4-12B-it-qat-4bit`（关闭思考） | 3/3 | **4/5** | **6/6 PASS** | — | 18.8–22.2 |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX`（关闭思考，`#628`） | 3/3 | 3/5 | 0/6 | — | 49.2–54.8 |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`（关闭思考，`#628`） | 3/3 | 1/5 | 0/6 | — | 64.0–74.7 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit`（关闭思考） | 3/3 | 1/4 ² | 未运行 | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit`（开启思考） | 2/3 | 0/4 ² | 未运行 | 未运行 | ~43 |

Ornith 9B 关闭思考时的逐级表现：

| 阶梯 | 结果 | 详情 |
|---|---|---|
| 1 · 双二次方程实根 | ✅ | 根全部正确，包含重根与无解情况 |
| 2 · 双二次方程复根 | ❌ | `x⁴ + 1` 仅返回 2 个根而非 4 个 |
| 3 · 傅里叶系数 | ❌ | 对 `f(x) = x` 算得 `a0 = 10.86`（期望为 0）——问题出在积分实现而非公式 |
| 4 · 泵扬程 | ❌ | 算得 `v = 0.0298` m/s（期望 1.79）——未将 mm 换算为 m |
| 5 · 登月任务 | — | 未运行 |

热身测试（关闭思考）：代数 ✅（163 tok，4 s）、斐波那契 ✅（757 tok，16 s）、翻译 ✅（303 tok，7 s——德语地道自然，法语更偏向意译而非直译）。

² 阶梯 1 和 2 是在阶梯测试拆分为独立提示词之前测得的：当时的单个提示词同时要求编写两个函数，模型答对了实根部分，但复根部分出错。阶梯 5 尚未在该模型上运行。

¹ ARC-AGI 这一列为部分测试结果：仅运行了三道训练题（`007bbfb7`、`00d62c1b`、`017c7c7b`），pass@2 均未解出，总耗时 330 s。在其余两题运行前即提前中止，因此请将其视为冒烟测试而非正式得分。

验证器自身同样在无模型、无 GPU 环境下经过检验：

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 assertions, both directions
```

开启 **thinking** 时，三道高难度题与翻译题全部因 `finish_reason: length` 结束——token 预算全耗费在 `<think>` 块内部，未能输出最终答案。这属于预算耗尽而非能力上限，因此上表将两种模式分开汇报。

### 复现步骤

```bash
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

向表中新增一个模型只需分别运行一次上述两个命令——欢迎提交带有新数据行的 PR。
