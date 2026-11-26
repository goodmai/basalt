<p align="center"><img src="images/logo.svg" alt="basalt logo" width="128"></p>

# basalt — 你的 Mac 现在就是 AI 实验室。

> 在 Apple Silicon 上部署服务、训练、量化、编排并剖析开源 LLM、神经语音、视频、OCR 与图像模型。单个 Swift 6.4 二进制文件即可原生支持 **OpenAI API、Anthropic API 与 MCP**，上层配有原生 macOS 27 Liquid Glass 应用。运行时无需 Python，除非你主动请求，否则绝不访问云端。

[English](README.md) · [Русский](README.ru.md) · **中文** · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Bahasa Indonesia](README.id.md) · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

📖 **专著：**[神经网络架构的演进（1958–2027）](docs/monograph/evo.zh.md)，作者 Aleksei Boklag，提供联合国六种官方语言版本（[全部](docs/monograph/README.md)）· **习题集：** basalt 已能运行的内容，清单见 [#816](https://github.com/goodmai/basalt/issues/816)

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.13.0](https://img.shields.io/badge/release-v0.13.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.13.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2865%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/i18n/zh/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/i18n/zh/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/i18n/zh/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/i18n/zh/benchmarks.md#benchmarks)

[![Quality Metrics](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml)
[![Concurrency Hatches](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/concurrency-hatches.json)](scripts/concurrency-hatches.allow)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/swift-testing.json)](Tests/BasaltTests)
[![Files ≤600 LOC](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/file-size.json)](scripts/check_size_limits.swift)
[![Crash Traps](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/crash-traps.json)](scripts/check_quality_metrics.swift)
[![L10n EN/RU](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/l10n-parity.json)](Localization)
[![Supply Chain](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/supply-chain.json)](.github/workflows/quality-metrics.yml)
[![Golden Logits](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/golden-logits.json)](Tests/BasaltTests/Fixtures/ReferenceLogits)

<p align="center"><img src="images/models-hive.svg" alt="Open models basalt runs: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>在 Apple Silicon 上运行开源权重模型。此处已通过 logit 验证的有：Gemma 4、Qwen3.8 与 LFM2.5（见下表）；已在 <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a> 完成实测并打包的有：Ornith、Bonsai、Qwen3.6-35B-A3B、DeepSeek-OCR-2、OpenJev 与 Laya（见<a href="docs/i18n/zh/models.md#verified-models">模型列表</a>）；gpt-oss 与 Llama 兼容 MLX。各徽标均为其各自所有者的商标，仅用于说明兼容性。Anthropic 模型为闭源权重，因此不在上述模型簇中；但支持 Anthropic <i>API</i>，以便让 Claude Code 直接对接本地模型。</sub></p>

**一个仓库集齐五大能力：** 兼容你现有工具的本地推理服务器、LoRA 蒸馏与量化后微调、面向多模型系统的 docker-compose 编排、实时洞察模型内部运行状态的 X 光显微镜，以及一键即达的原生 Mac 应用。

**每次修改推理路径都会对照参考 logits 进行校验：** [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) 会重放已录制的提示词，一旦贪心采样 token 或 logit 发生偏移即判定失败。覆盖 `qwen3`、`qwen3_5`、`gemma4`、`lfm2` 与 `nemotron_h`（含稠密模型与 MoE）的分家族参考 logits 均存放在 [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits)。

| 黄金测试模型 | 架构类别 | 测试固件 |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | 滑动窗口 + 全注意力，MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | 混合线性/全注意力，稠密 | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | 混合卷积 + 注意力，MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### 从 [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) 一键启动（65+ 已验证模型）

[`goodmai`](https://huggingface.co/goodmai) 下的每张模型卡片都带有 **Run with Basalt** 按钮（[`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en)），可通过 `basalt://run?model=...` 直接打开 `Basalt.app`，或输出精确的 `basalt serve` / `basalt compose` CLI 命令：

| 模型 / 组合栈 | 定位与在 24 GB Mac 上的内存占用 | 启动 |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT 日常主力对话与视觉模型（约 7.6 GB，Ladder 4/5 + Lunar 6/6） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE（`A4B`），黄金测试通过（14.9 GB，56–63 tok/s，Ladder 4/5） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 适用于 24 GB Mac 的 31B 稠密 PV-MACaQ 3.2 bpw 模型（15.1 GB，14.1 tok/s，Ladder 4/5 + Lunar 4/6） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B 稠密 PV-MACaQ 4-bit（15.3 GB，p50 11.6 tok/s，PPL 1.23，MSE −19.5%，Ladder 4/5） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE（`A3B`）3-bit MACaQ（14.6 GB，结合 `#628` switch WHT 达 64–75 tok/s） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw 网格编码量化（10.0 GB，16k PPL 4.621） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO 原生 i-quant（11.8 GB，PPL 4.17，300/300 System One） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 37.7 GB 的 35B MoE 从 SSD 流式加载推理，8.77 tok/s，0 swap | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B Hadamard 三值量化，仅占 7.4 GB，开启 `--mtp` 达 27.1 tok/s | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B 混合卷积 + 注意力 MoE，黄金测试通过（约 12.6 GB） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | 44.1 kHz 神经 TTS，支持 43 种语言（2.24 GB，RTF 0.23–0.26） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | 带官方 768 分块的视觉编码器 OCR（`/v1/pipeline`） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | 9B 本地 System One 决策器（98.9% 准确率，减少 77% LLM token 消耗） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/i18n/zh/compose.md#bundled-hardware-tiered-profiles-526) | 推荐的 24 GB Compose 组合栈（对话 + ZONOS3 + Kokoro ANE） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

### v0.13.0 新特性（[更新日志](CHANGELOG.md)）

- **Inkling 2 Mini。** 循环式 MoE 端到端可用，支持工具调用，并在智能体多轮之间复用提示缓存。借助检查点自带的 DSpark 头，`--mtp` 让代码解码从 28.0 提升到 49.8 tok/s（M4 Pro，24 GB）；37.5K token 的提示峰值仅 14.5 GB，零 swap（[#650](https://github.com/goodmai/basalt/issues/650)）。
- **任意来源转 NVFP4。** `models convert --to nvfp4` 支持任意检查点；NVIDIA ModelOpt NVFP4 可无损导入，并在 `models download` 和 `serve` 时自动完成；KV 空间不足的 MoE 会改为流式加载专家，不再触发 Metal OOM（[#677](https://github.com/goodmai/basalt/issues/677)）。
- **新架构与 KV。** 支持专家流式加载的 Qwen3-Coder-Next（[#642](https://github.com/goodmai/basalt/issues/642)）、BailingHybrid / Ling-3.0（[#672](https://github.com/goodmai/basalt/issues/672)），以及实验性的 Gemma 4 K=V 共享行 KV 缓存，在 32K–128K 下 KV 内存减少 1.40–1.53 倍（[#645](https://github.com/goodmai/basalt/issues/645)）。
- **适配器与上下文。** PEFT LoRA 适配器可从目录或 Hub 加载，并可融合进导出的检查点（[#675](https://github.com/goodmai/basalt/issues/675)）；`--rope-yarn` 扩展 Qwen 3.5–3.8 的上下文（[#729](https://github.com/goodmai/basalt/issues/729)）。
- **翻译与自我改写。** `basalt:translation` 基准（chrF++、COMET、MetricX-24）及 TranslateGemma-4 配置；基于写时复制权重的实验性自我改写（`basalt selfedit`，[#676](https://github.com/goodmai/basalt/issues/676)）。
- **加固。** 双智能体缺陷排查修复了 64 个问题，覆盖 REST/MCP 边界、云端预算、评分、转换、缓存和 Mac 应用（[#809](https://github.com/goodmai/basalt/pull/809), [#810](https://github.com/goodmai/basalt/pull/810), [#811](https://github.com/goodmai/basalt/pull/811)）。

---

## 推理引擎：40B MoE 达 67 tok/s、27B 仅占 7.4 GB、37.7 GB MoE 零 swap 运行——无缝接入你熟悉的 API

- **一行命令让 Claude Code 跑在你自己的 GPU 上。** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude`——完全不触碰你真实的 API 密钥。OpenAI SDK、OpenCode、Continue、Cline 和 aider 通过 OpenAI API 连接；Claude Desktop 和 Cursor 通过 MCP 连接（支持按工具和按分组使用 `--mcp-enable` / `--mcp-disable` 精细开关，[#488](https://github.com/goodmai/basalt/pull/488)）。
- **在笔记本上飞速运行。** 在 24 GB M4 Pro 上：递归模型 `Flint-40B-A2.5B`（`oQ3e`）以 17.61 GB 内存达到 67.02 tok/s 解码速度（[#522](https://github.com/goodmai/basalt/pull/522)）；`Ornith-1.5-35B-A3B-MACaQ-3bit` 借助我们的单遍 `MACaQSwitchLinear` WHT + `gatherQuantizedMM` 内核，以 14.6 GB 达到 64–75 tok/s（MoE 内核提速 `3.41×`，Gemma 4 26B-A4B 端到端提升 `+82%`，[#628](https://github.com/goodmai/basalt/pull/628)）；`Qwen3.5-9B-PV-MACaQ-4bit` 达到 p50 `43.2 tok/s`（对比 8-bit 提速 `+52%`、内存 `−33%`），且与 8-bit 保持完全一致的 `4/5` Ladder 表现；`Gemma-4-31B-it-PV-MACaQ-3.2bpw` 让 31B 稠密模型以 `15.1 GB`（`14.1 tok/s`，Ladder `4/5` + Lunar `4/6`）在 4-bit 会直接 OOM 的 24 GB Mac 上稳定运行；Qwen3-4B 解码速度达 78–89 tok/s；Ternary Bonsai 2 27B 仅占 7.4 GB，裸跑可达 20.9–21.5 tok/s，启用自带 MTP 预测头达 27.1 tok/s，在代码场景启用 prompt lookup 可达 31–38 tok/s（[#239](https://github.com/goodmai/basalt/pull/239)、[#263](https://github.com/goodmai/basalt/pull/263)）；`.gguf` 格式的 GSQ-RCO / `i-quant` 检查点可直接在自定义 Metal 内核上执行（[#433](https://github.com/goodmai/basalt/pull/433)、[#515](https://github.com/goodmai/basalt/pull/515)）。
- **大于物理内存的模型也能零 swap 运行。** `Qwen3.6-35B-A3B-8bit`（37.7 GB）在 24 GB Mac 上通过批量 POSIX AIO 与 `LIO_NOWAIT` 下一层 SSD 专家预取技术，实现 8.77 tok/s 推理与 0 swap（`--expert-streaming`，[#477](https://github.com/goodmai/basalt/pull/477)、[#478](https://github.com/goodmai/basalt/pull/478)、[#516](https://github.com/goodmai/basalt/pull/516)），并配合感知 UMA 的 L0/L1/L2 分层 KV 缓存（`TieredKVHierarchyCore`）与自适应异构规划（[#627](https://github.com/goodmai/basalt/pull/627)）。
- **严格保持精确性的投机解码。** Prompt lookup 在代码编辑任务上提供 1.84–2.66× 加速（Bonsai 2 27B，[#239](https://github.com/goodmai/basalt/pull/239)）以及 3.45–3.62× 加速（Gemma 4 12B，[#265](https://github.com/goodmai/basalt/pull/265)），且贪心输出分毫不变；`SpecTr` OT + `Residual Race`、`EAGLE-3`、`SmallM`（`M=2..16`）、`Fast-dLLM` + `DFlash` 与 `DiffuCD` 进一步扩展了无损多草稿与块扩散解码能力（[#593](https://github.com/goodmai/basalt/pull/593)、[#603](https://github.com/goodmai/basalt/pull/603)）。
- **下一轮对话只需计算新增 token。** 基于前缀树的提示词缓存同样适用于混合架构模型：在 Ornith-1.5-9B 面对 12.6k-token 智能体提示词时，第二轮响应从约 36 s 缩短至约 1 s（[#82](https://github.com/goodmai/basalt/pull/82)）。
- **更小内存承载更长上下文。** 带精确分区的 KV 量化（`4|8|k8v4`，外加 `KVarN` 二维均衡、无偏 1-bit `QJL`、`ChanMix`/`ThinKV`、跨语言 `HeadKV` RBO 迁移、质心 `RQ-KV`、`turbo3|turbo4|turbo8v4` 与 `polar4` RoPE 成对键量化，[#472](https://github.com/goodmai/basalt/pull/472)、[#590](https://github.com/goodmai/basalt/pull/590)、[#602](https://github.com/goodmai/basalt/pull/602)）：在 Qwen3-4B 上处理 16k 提示词时，峰值内存从 3.11 GB 降至 1.03 GB（[#159](https://github.com/goodmai/basalt/pull/159)）。
- **视频、神经 TTS、全双工语音、音乐与 2K MLX 图像生成。** 分阶段 24 GB UMA 视频工作室支持 `LTX-2.5 AV-DiT`、`Kandinsky 6.0 CrossDiT`、`LingBot-Video MoE` 与 `Wan 2.2`，配备因果 3D-VAE、分块 STA Metal 内核与 `AVAssetWriter` `.mp4` 封装（[#608](https://github.com/goodmai/basalt/pull/608)–[#611](https://github.com/goodmai/basalt/pull/611)）；`LFM2.5-Audio` Mimi/SNAC 双工编解码器、`SmartTurn` + `Sortformer`、`MossFormer2`、`Soprano-80M`、`Qwen3ForcedAligner`、`Zyphra/ZONOS3-Mini-hf`（`44.1 kHz`、43 种语言、2.24 GB、RTF 0.228–0.263，[#527](https://github.com/goodmai/basalt/pull/527)）及 `Kokoro-82M` 驱动 `/v1/audio/*`（[#604](https://github.com/goodmai/basalt/pull/604)）；`FLUX.3` 四阶段生成管线、`SeedVR2` 超分、`ACE-Step 1.5` 音乐生成、`Z-Image-Turbo`、`Iris-3B` 与 `SANA-1.5` 驱动 `/v1/images/*` 和 `/v1/music/*`（[#542](https://github.com/goodmai/basalt/pull/542)、[#605](https://github.com/goodmai/basalt/pull/605)）。
- **完整本地软件栈 + Swift SDK。** 工具调用、JSON-schema / 正则表达式 / 语法约束输出（XGrammar）、Meta `Muse-Glimmer-30B`（[#534](https://github.com/goodmai/basalt/pull/534)）、`Ripple-65B`、`Nemotron-3.5-Lightning`（[#522](https://github.com/goodmai/basalt/pull/522)）、`DeepSeek-OCR-2`（[#476](https://github.com/goodmai/basalt/pull/476)）、`/v1/embeddings`、`/v1/rerank`、`BasaltFoundationModels` Swift SDK（[#606](https://github.com/goodmai/basalt/pull/606)）、`swift-huggingface` Xet 下载（[#587](https://github.com/goodmai/basalt/pull/587)）、按请求动态切换的 LoRA/PHM/OFT 适配器，以及局域网优先于云端的有序回退层级（[#480](https://github.com/goodmai/basalt/pull/480)）。
- **PV-MACaQ 3-bit 让 27B 在 24 GB 上装得下、跑得快。** `Qwen3.8-27B-PV-MACaQ-3bit` 在 M4 Pro 上以 11.5–12.4 GB、15.6–16.7 tok/s 通过 Official Ladder 5/5（含 `lunar` 6/6），而原版 4-bit 为 4/5 级、15.3 GB、11.6 tok/s。`--speculative-lookup` 现已支持带视觉塔的 Qwen3.5/3.8 检查点（草稿接受率 43–51%，代码编辑提速 17%），`--kv-bits kvarn|kvarn-qjl`（2-bit KV）与 `--attention-backend moba` 已接入 `basalt serve` 并通过 4k passkey（[#632](https://github.com/goodmai/basalt/issues/632)、[#634](https://github.com/goodmai/basalt/issues/634)）。检查点：[`goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX)、[`goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX)。

详情请参阅：[REST API](docs/i18n/zh/rest-api.md) · [集成指南](docs/i18n/zh/integrations.md) · [命令与参数](docs/i18n/zh/commands.md) · [模型列表](docs/i18n/zh/models.md) · [基准测试](docs/i18n/zh/benchmarks.md)

## 训练与量化：通用任意格式转换器、PV-MACaQ、分布式训练与 QAT

- **通用任意到任意量化转换器（`basalt models convert`）。** 支持在以下格式间自由转换：`MACaQ`（`affine` / `mxfp4` / `moe` 2.8 bpw 路由加权 Hessian，[#537](https://github.com/goodmai/basalt/pull/537)、[#599](https://github.com/goodmai/basalt/pull/599)）、`FlatQuant` 与 `LIRQuant`（[#588](https://github.com/goodmai/basalt/pull/588)、[#594](https://github.com/goodmai/basalt/pull/594)）、`SINQ` / `A-SINQ` / `EAQuant` 双轴归一化（[#561](https://github.com/goodmai/basalt/pull/561)）、`QuIP# E8P` 2-bit 格点与 `TurboBoA` Babai–LLL（[#589](https://github.com/goodmai/basalt/pull/589)）、`QTIP` 移位网格（`2.0–3.0 bpw`，[#494](https://github.com/goodmai/basalt/pull/494)）、`SVD-LLM V2` / `OBR`（[#592](https://github.com/goodmai/basalt/pull/592)）、`.gguf` i-quant 与 `MACaQ-R`，外加领域感知 `QAT STE` 精调与量化后质量/公平性/泄露审计（`QuantizationAuditCore`，[#623](https://github.com/goodmai/basalt/pull/623)）——CLI、REST `/v1/models/convert*` 与 macOS `ConvertModelSheet` 三端完全打通（[#546](https://github.com/goodmai/basalt/issues/546)）。
- **量化后 PV-Tuning 与 PV-MACaQ。** `basalt models convert --to macaq:affine:4:64 --pv-tune` 会在 Walsh–Hadamard 分块上交替执行连续缩放/偏置更新与离散 Hessian 翻转精调：在 `Qwen3.5-9B` 上，`8-bit → 4-bit PV-MACaQ` 降低 33% 内存（`9.97 → 6.68 GB`）并将解码提速 `+52%`（`27.3 → 41.6 tok/s`，p50 `43.2`），同时保持与 8-bit 完全一致的 `4/5` Ladder 成绩（`valKL = 0.0044`）；在 2–3-bit 检查点上，PV-Tuning 可在 3 分钟内减少 8.7–29% 的困惑度损失（[#486](https://github.com/goodmai/basalt/pull/486)、[#545](https://github.com/goodmai/basalt/pull/545)）。
- **跨 Mac 蒸馏、后训练与分布式训练。** `basalt models distill` 可将 OpenAI/Anthropic 教师模型蒸馏至 `lora`、`phm`、`oft` 或量化 `AdaLoRA (LR/40)` 适配器（[#501](https://github.com/goodmai/basalt/pull/501)、[#508](https://github.com/goodmai/basalt/pull/508)、[#624](https://github.com/goodmai/basalt/pull/624)）；两阶段 `SFT → DPO`、零拷贝 `ADPO`/`cADPO`、`EMA-QLoRA`、`Self-Appraisal` 与 `RLVR` 反捷径审计保障偏好对齐质量（[#601](https://github.com/goodmai/basalt/pull/601)、[#624](https://github.com/goodmai/basalt/pull/624)）；多 Mac 集群训练支持鉴权传输、`DDP + ZeRO-2`、`DiLoCo` + `PowerSGD`、`Zero-Bubble ZB-H1` 与 `Expert-Parallel MoE`（[#595](https://github.com/goodmai/basalt/pull/595)–[#598](https://github.com/goodmai/basalt/pull/598)）。
- **边服务边训练，带质量门禁。** 支持时间分片式 MCP `distill`、仅限已验证样本的 `train_experts`、`train-jepa`，以及在验证集 KL 或困惑度退化时逐位无损回滚。

详情请参阅：[训练命令](docs/i18n/zh/commands.md) · [自训练、蒸馏与质量门禁](docs/i18n/zh/research-tools.md#self-training-train_experts)

## Compose 编排：声明式 `basalt.yaml v2.2`、业务治理与 117 套硬件配置方案

- **面向模型的 docker-compose（`basalt.yaml v2.2`）。** 在同一份声明式配置中统一管理本地与云端模型、内存去重规划器、专家配置文件、`resident` / `on-demand` / `sequential` 三种内存模式、`BusinessOrchestrator` 五分量路由、`LANTERN` 语义缓存、`HITL` 人工审批门禁（`/v1/mcp/approvals`）以及 `LAAS-CNRS` 神经符号知识核心（`SymbolicKnowledgeGraph`）（[#607](https://github.com/goodmai/basalt/pull/607)、[#625](https://github.com/goodmai/basalt/pull/625)、[#626](https://github.com/goodmai/basalt/pull/626)）。
- **内置 117 套覆盖全系 Mac（`16 GB` 至 `256 GB`）的配置方案。** 九档统一内存规格（`16gb`、`24gb`、`32-36gb`、`48gb`、`64gb`、`96gb`、`128gb`、`192gb`、`256gb`）× 13 类典型工作负载，一条命令即可拉起（`basalt compose up 24gb-voice-zonos`，[#528](https://github.com/goodmai/basalt/pull/528)）。
- **放不下就拒绝，支持热更新。** `basalt compose plan` 会在下载前从 Hub 测算模型体量并拦截内存溢出；`compose diff` / `compose up` 可干净地热重载差异；`compose bench` 运行声明式基准测试套件并导出排行榜数据包（[#505](https://github.com/goodmai/basalt/pull/505)）。
- **本地裁判、校准路由与云端预算控制。** `OpenJev-Flash-9B-4bit` 在分诊任务上达到 98.9% 准确率，同时削减 77% 下游 LLM token 消耗（[#465](https://github.com/goodmai/basalt/pull/465)）；经过温度缩放的 `CalibratedRouterGate` + `Gemini Decision 1` 在每日美元预算上限内精准路由疑难请求（[#418](https://github.com/goodmai/basalt/pull/418)、[#622](https://github.com/goodmai/basalt/pull/622)）；默认情况下，本地感知到的文档与音频文本绝不离开你的 Mac。

详情请参阅：[Compose 参考与实测数据](docs/i18n/zh/compose.md) · [管线预设与内存模式](docs/i18n/zh/pipelines.md) · [基于 Gemini 的云端决策](docs/i18n/zh/cloud-decisions.md)

## 研究工具：在模型运行时透视其内部

- **模型 X 光、逐 token 归因与 Belnap 认知门控。** 提供 `model_describe`（支持报告 `MACaQ` / `FlatQuant` / `LIRQuant` 模块清单，[#630](https://github.com/goodmai/basalt/pull/630)）、`model_tensors`、`model_tensor`、`kv_inspect`、`trace_run`、`trace_tokens`（自动处理 Gemma BOS，[#630](https://github.com/goodmai/basalt/pull/630)）、`/v1/xray/tokens` 逐 token 对数概率与幻觉风险评估（[#613](https://github.com/goodmai/basalt/pull/613)）、`Score d'influence` 注意力头归因（[#622](https://github.com/goodmai/basalt/pull/622)）、Marchenko–Pastur 谱分析、四值 Belnap `SIXTEEN_3` 证据状态（[#500](https://github.com/goodmai/basalt/pull/500)）、复数 MVN 相量探针（[#509](https://github.com/goodmai/basalt/pull/509)）、Fisher–Rao 漂移监测（[#503](https://github.com/goodmai/basalt/pull/503)），以及位于 `/api/v1/metrics` 的 OpenMetrics 指标（[#458](https://github.com/goodmai/basalt/pull/458)）。
- **可逆手术与机制电路分析。** 启用 `serve --research-tools` 后可使用：`intervene`（含条件消融、GQA 分组修补、认知零空间与反引导保护，[#600](https://github.com/goodmai/basalt/pull/600)）、`MAO-SNMF` 特征分解、`weights_delta`、实时 `tensor_pack`/`tensor_unpack`、`activation_contrast`、`steer_search`/`steer_commit`、`experiment_compare` 以及 `introspect`。每次实验都作为带 TTL 与操作日志的可撤销覆盖层存在，磁盘上的原始权重绝不会被修改。
- **真正运行模型所写代码的基准测试。** 五级阶梯挑战最终直通地月转移任务，将模型自己编写的代码直接放到未见输入上执行验证；ARC-AGI 则严格按照官方 pass@2 规则运行。

详情请参阅：[研究工具](docs/i18n/zh/research-tools.md) · [研究基础（95+ 篇论文）](docs/i18n/zh/research-foundations.md) · [基准测试](docs/i18n/zh/benchmarks.md) · [serve 参数说明](docs/i18n/zh/commands.md)

## Basalt for Mac：macOS 27 Liquid Glass，一键或轻点即达

- **运行在受监管引擎上的原生 SwiftUI 应用。** 专为 macOS 27 Liquid Glass 打造（[#479](https://github.com/goodmai/basalt/pull/479)），支持英文与俄文界面（[#456](https://github.com/goodmai/basalt/pull/456)）：热启动窗口 257 ms、空闲唤醒 0 次、2,000 条消息长对话中流式 token p95 延迟 7.4 ms、`⌥Space` 快捷提问仅 40 ms 开销（[#463](https://github.com/goodmai/basalt/pull/463)）。
- **Compose 视频工作室、2K 图像工作室与通用转换器界面。** 在 24 GB Mac 上生成并播放音视频 `.mp4`（[#611](https://github.com/goodmai/basalt/pull/611)）、使用前后对比滑块编辑 2K RGBA 图像（[#519](https://github.com/goodmai/basalt/pull/519)）、直接在 `ConvertModelSheet` 中完成模型转换与 QAT 审计（[#623](https://github.com/goodmai/basalt/pull/623)），并在应用内以 10 Hz SSE 遥测和 Xet 下载浏览/发布 Hugging Face 模型（[#587](https://github.com/goodmai/basalt/pull/587)）。
- **Compose 配置档案（`⌥⌘C`）、全双工语音与 `basalt://` 一键启动。** 根据当前 Mac 的内存档位自动推荐最佳配置（`16–256 GB`，[#528](https://github.com/goodmai/basalt/pull/528)），通过 `SpeechAnalyzer` / `LFM2.5-Audio` / `ZONOS3-Mini` 实现免提流式语音对话（[#604](https://github.com/goodmai/basalt/pull/604)），并支持 `basalt://run?model=...` 链接、`⌥Space` 快捷提问、Shortcuts 快捷指令、Spotlight 搜索与已公证 `.dmg` 打包。


**Basalt.app：引擎上运行中的 Compose 与带检视器的聊天界面**（macOS 27 Liquid Glass，深色与浅色）：

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-compose-dark.png"><img src="docs/assets/screenshots/app-compose-light.png" alt="Basalt.app — 引擎上已激活的 Compose 配置" width="49%"></picture></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-chat-dark.png"><img src="docs/assets/screenshots/app-chat-light.png" alt="Basalt.app — 打开检视器的聊天" width="49%"></picture></p>

详情请参阅：[Basalt for Mac](docs/i18n/zh/mac-app.md)

---

## 快速开始

环境要求：macOS 27+、Xcode 27.0+ / Swift 6.4 及 Metal 工具链（`xcodebuild -downloadComponent MetalToolchain`）、Apple Silicon M1–M4 芯片，以及视模型而定 2–38 GB 可用磁盘空间。完整安装步骤、源码编译与 `./basalt-claude` 启动器请参阅 [docs/i18n/zh/install.md](docs/i18n/zh/install.md)。

**无需 Xcode、无需编译——两行命令运行预编译的 Apple Silicon 二进制**（macOS 27+、Apple Silicon；`mlx.metallib` 已包含在压缩包内；首次运行会下载 12 GB 模型，想轻量启动可改用 `goodmai/gemma-4-12B-it-qat-4bit`）：

```bash
curl -fsSL https://github.com/goodmai/basalt/releases/latest/download/basalt-macos-arm64.tar.gz | tar -xz
./basalt-macos-arm64/basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --rest
```

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds CLI & server from source

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model goodmai/gemma-4-12B-it-qat-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

在 24 GB Mac 上流式运行 37.7 GB MoE 模型，或启动完整的多模型 Compose 系统：

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 117 profiles across 16–256 GB Macs
```

从源码检出目录构建并运行 Mac 应用（首次打开时会自动注册 `basalt://` 一键链接）：

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

4B–12B 本地模型并非前沿超大模型的平替——它们专为隐私、离线、零成本的智能体工作流以及本地/云端混合系统而生。`basalt fit` 会告诉你当前设备究竟能跑得动什么，模型列表也会如实标注哪些模型曾*无法正常工作*。

如果你希望看到更多基于 Swift 的 MLX 原生工具，欢迎**点亮 Star**，也欢迎提 Issue 分享你的 Mac 型号和 tok/s 数据——同样欢迎提交 PR，[为基准测试表添上一行](docs/i18n/zh/benchmarks.md#benchmarks)。

---

## 文档索引

> 多语言翻译文档位于 `docs/i18n/`：[Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/) · [한국어](docs/i18n/ko/) · [Bahasa Indonesia](docs/i18n/id/)。

| 主题 | 文档 |
|---|---|
| 安装、系统要求、源码构建、`./basalt-claude` | [docs/i18n/zh/install.md](docs/i18n/zh/install.md) |
| CLI 命令、`serve` 参数、提示词缓存、CacheBlend、PV-Tuning | [docs/i18n/zh/commands.md](docs/i18n/zh/commands.md) |
| REST API：安全机制、模型管理、OpenAI、Anthropic、图像、SSE、OpenMetrics、Swagger | [docs/i18n/zh/rest-api.md](docs/i18n/zh/rest-api.md) |
| Claude Code、OpenCode、OpenAI SDK、面向 Claude Desktop 与 Cursor 的 MCP | [docs/i18n/zh/integrations.md](docs/i18n/zh/integrations.md) |
| 已验证模型、`goodmai` Hub 模型栈与下载方式 | [docs/i18n/zh/models.md](docs/i18n/zh/models.md) |
| 核心特性一览、阶梯基准测试、ARC-AGI、数据来源 | [docs/i18n/zh/benchmarks.md](docs/i18n/zh/benchmarks.md) |
| 管线预设与内存模式 | [docs/i18n/zh/pipelines.md](docs/i18n/zh/pipelines.md) |
| 云端决策：Gemini Decision 1、多模态媒体、校准教师模型 | [docs/i18n/zh/cloud-decisions.md](docs/i18n/zh/cloud-decisions.md) |
| `basalt compose` 参考手册与实测数据 | [docs/i18n/zh/compose.md](docs/i18n/zh/compose.md) |
| Basalt for Mac 与 `basalt://` 深度链接 | [docs/i18n/zh/mac-app.md](docs/i18n/zh/mac-app.md) |
| 模型 X 光、遥测、实验系统、自训练、蒸馏 | [docs/i18n/zh/research-tools.md](docs/i18n/zh/research-tools.md) |
| 数学与科学研究基础（85+ 篇论文与致谢） | [docs/i18n/zh/research-foundations.md](docs/i18n/zh/research-foundations.md) |
| 项目结构、架构说明、脚本与依赖项 | [docs/i18n/zh/architecture.md](docs/i18n/zh/architecture.md) |
| 架构模式与模块说明（俄文） | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| 云端接入：OpenRouter 配置、成本优化、模型对比 | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| 使用 AI 智能体协作开发 basalt | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| 路线图、隐私政策 | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| 智能体 CLI 调研笔记、检查清单与规范（俄文） | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| 贡献指南、更新日志 | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## 许可证

[MIT](LICENSE) © 2026 goodmai

模型权重**不受**本许可证覆盖——HuggingFace 上的每个模型均遵循其各自附带的许可条款。
