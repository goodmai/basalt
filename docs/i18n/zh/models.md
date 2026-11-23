# 模型

> Languages: [English](../../models.md) · [Русский](../ru/models.md) · **中文** · [Español](../es/models.md) · [Français](../fr/models.md) · [日本語](../ja/models.md) · [Deutsch](../de/models.md) · [한국어](../ko/models.md) · [Bahasa Indonesia](../id/models.md)

## 已验证模型

模型从 HuggingFace 下载并缓存在 `~/.cache/huggingface/hub/` 中。[`goodmai` Hugging Face 合集](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)已发布超过 42 个经过验证的对话、MoE、QTIP、PV-Tuned、GSQ-RCO、OCR、神经 TTS、向量嵌入、重排序及 System One 决策检查点，均附带一键 **Run with Basalt** 按钮（[`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en) → `basalt://run?model=...`）。

```bash
basalt models download goodmai/gemma-4-12B-it-qat-4bit
basalt models download goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native
```

### 0.12 版本实测

测试环境：M4 Pro，24 GB 内存，release 构建（`620641c` / `82a99d4`），2026 年 11 月（`--reasoning-effort none --kv-bits 4`）。

| 模型 | 定位 / 架构 | 活跃内存 | 解码 tok/s | 阶梯测试 / 说明 |
|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` | 31B 稠密混合架构（4-bit 注意力/嵌入 + 3-bit 仿射 PV-MACaQ MLP） | 15.1 GB（峰值 18.09 GB） | 11.6–13.5（p50 `14.07`，p95 `14.42`） | **热身 3/3，阶梯 4/5 + Lunar 4/6**，`probe_ppl = 2.39`，lookup 接受率 50.6%；在 4-bit 会 OOM 的 24 GB Mac 上可稳定运行（`--prompt-cache-mb 0 --wired-limit`） |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` | 27B 稠密混合架构（48 个线性注意力 + 16 个全注意力，496 个 MACaQ 4-bit g64 模块） | 15.3 GB（峰值 17.45 GB） | 9.4–11.4（p50 `11.59`，p95 `11.86`） | **热身 3/3，阶梯 4/5**，`probe_ppl = 1.23`，`SQNR 20.61 → 21.59 dB`（`MSE −19.5%`），64 层最大漂移 `5.53%`（`#630`） |
| `Qwen3.5-9B-PV-MACaQ-4bit`（由 `Qwen3.5-9B-8bit` 转换） | 9B 混合架构（24 个线性注意力 + 8 个全注意力，200 个 MACaQ 4-bit g64 模块） | 6.68 GB（比 8-bit 减少 33%） | 35.6–41.6（p50 `43.22`，比 8-bit 快 +52%） | **热身 3/3，阶梯 4/5**（与 8-bit 参考 `24.7–27.3 tok/s` 完全持平），`valKL = 0.0044`，`probe_ppl = 1.28`，最大层漂移 `3.37%` |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` | 26B MoE（`A4B`，128 个专家，top-8，g32 仿射 MACaQ + 8-bit 共享 MLP） | 16.75 GB | 49.2–54.8（经 `#628` switch WHT 优化提升 +82%） | **热身 3/3，阶梯 3/5**（`biquad-complex`、`fourier`、`pump`）；内核提速 `3.41×`（`2.06 → 0.60 ms`，`#628`） |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` | 35B MoE（`A3B`，120 个 `MACaQSwitchLinear` 层） | 14.64 GB | 64.0–74.7（8k 上下文为 68.7） | **热身 3/3，阶梯 1/5**，`probe_ppl = 1.46`，单遍 switch WHT（`#628`） |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` | 26B MoE（`A4B`，QAT 4-bit） | 14.92 GB | 56.0–63.4 | **热身 3/3，阶梯 4/5**（`pump` ✅，`lunar 5/6`） |
| `goodmai/gemma-4-12B-it-qat-4bit` | 12B 稠密 QAT 4-bit | 10.51 GB | 18.8–22.2 | **热身 3/3，阶梯 4/5**（`lunar` ✅ 6/6） |
| 视频 / 语音 / 音乐（`LTX-2.5 AV-DiT`、`Kandinsky 6.0`、`LingBot-Video`、`Wan 2.2`、`LFM2.5-Audio`、`FLUX.3`、`SeedVR2`、`ACE-Step 1.5`） | 分阶段 24 GB UMA 视频工作室、双工 Mimi/SNAC 音频与四阶段图像/音乐管线 | ≤ 18 GB（分阶段） | 3D-VAE + STA Metal 内核 | 已于 `#542`、`#604`、`#605`、`#608`–`#611` 验证 |

### 0.11 版本实测

测试环境：M4 Pro，24 GB 内存，release 构建，2026 年 11 月。

| 模型 | 定位 / 架构 | 活跃内存 | 吞吐量 / 延迟 | 说明 |
|---|---|---|---|---|
| `Recursive/Flint-40B-A2.5B`（`flint-oQ3e`） | 40B-A2.5B 混合滑动窗口/全注意力 MoE（`512` 路由专家，top-10 + 1 共享专家） | 17.61 GB | 67.02 tok/s 解码 · 273.5 tok/s 预填充 | 原生 `oQ3e`（`groupSize = 32`）加载器（#518, #522），在 24 GB 上 0 swap |
| `Zyphra/ZONOS3-Mini-hf` | 1.6B Llama + 32 码本 FSQ 多头 DAC TTS（`44.1 kHz`） | 2.24 GB | RTF 0.228–0.263（3.8–4.4× 实时速度） | 支持 43 种语言、27 种音色、跨语言 `embedding` + CFG `1.5`（#525, #527） |
| `hexgrad/Kokoro-82M` | 82M 神经 TTS（ANE + MLX） | ~0.35 GB | 首字音频延迟（TTFA）低于 80 ms | 用作 `pipeline.speak` 中的按句流式回退阶段（#493, #513） |
| `Tongyi-MAI/Z-Image-Turbo` | DiT 图像合成 | ~6.4 GB | 4–8 步（`1024×1024`） | 集成于 `pipeline.imagine`，支持 TeaCache 与共享文本编码器（#517, #519） |
| `Ricursive/Ripple-65B-4bit` | 65B 门控注意力稀疏 MoE（`128` 专家，top-8） | ~35 GB（64 GB+ 常驻，或 SSD 流式加载） | 架构已验证 | NoPE / 部分 RoPE（#520, #522） |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | 30B-A3B Mamba-2 + 潜空间路由 MoE（`lr_moe`） | ~17 GB（4-bit） | 架构已验证 | 交错 GQA + Mamba-2（#521, #522） |
| `meta-models/Muse-Glimmer-30B`（`mlx-community/Muse-Glimmer-30B-4bit`） | 30B 混合 SWA（`2048`）+ NoPE 全注意力 + ViT-G/14 VLM | ~18.1 GB（4-bit）· ~19.8 GB（`oQ4e`） | 架构与 HF 分片已验证 | Onyx `to=self` 推理 + ATEM 工具调用 + GGUF 去中心化归一化导入（#533） |

### 0.8 版本实测

测试环境：M4 Pro，24 GB 内存，连接 68 W 电源适配器，release 构建，贪心解码，2026 年 11 月。

| 模型 | 参数量 | 活跃内存 | 解码速度，tok/s | 说明 |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | ~2.2 GB | 78–89 | 稠密模型；支持 prompt lookup 及所有 KV 模式 |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B 三值 | 7.4 GB | 普通 20.9–21.5 · 代码场景 `--mtp` 27.1 · 代码场景 `--speculative-lookup` 31–38 | 混合架构（Gated DeltaNet）；镜像位于 `goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | ~11.8 GB | 15–19 | 混合架构；已通过黄金测试；镜像位于 `goodmai/Qwen3.8-27B-3bit-MLX-TextOnly` |
| `goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native` | 27B i-quant | ~12.4 GB | 5.2 | 原生 GSQ-RCO `IQ3_S` 导入（#433），WikiText-2 PPL 4.169，300/300 System One |
| `goodmai/Qwen3.6-35B-A3B-8bit` | 35B MoE（37.7 GB） | ~13.9 GB | 8.77 | `--experimental-expert-streaming`（#478），通过批量 AIO + `LIO_NOWAIT` 在 24 GB 上实现 0 swap |
| `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly` | 27B | 未测量 | 18–22 | 部分提示词会陷入循环（8 次中出现 3 次）；`RepetitionGuard` 会及时截断 |

| 嵌入 / 重排序模型 | 任务 | 实测表现 |
|---|---|---|
| `mlx-community/bge-small-en-v1.5-bf16` | 向量嵌入（384 维） | 229–299 texts/s，batch 32 × 约 240 tokens |
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | 向量嵌入（1024 维） | 20 texts/s，同批次规格 |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | 重排序 | 已知样本对排序正确 |
| `jinaai/jina-reranker-v3-mlx` | 重排序 | 已知样本对排序正确 |

### 早期报告（未针对 0.8 重新测量）

| 模型 | 参数量 | 内存占用 | 在 24GB Mac 上的状态 |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 GB | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 GB | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 GB | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 GB | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 GB | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 GB | ✅ ~12 TPS（去拒绝版 / Abliterated） |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 GB | ✅ ~14 TPS（动态量化） |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 GB | ✅ ~8 TPS（NVFP4） |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 GB | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 GB | ✅ ~48 TPS（推理模式——见下方说明） |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ 未在 24 GB 以上验证 |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 GB | ✅ ~76 TPS（始终开启推理思考） |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 GB | ✅ ~40 TPS（始终开启推理思考） |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated`（Base BF16） | 27B | ~54 GB | ❌ 需要 64GB+（24GB 请改用 4-bit MLX 版本） |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ 输出乱码 |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ 输出乱码 |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM（需要 32GB+） |

> **24GB 内存 Mac 注意事项：**
> - 未量化的 27B/35B 模型（权重约 54 GB）超出物理内存上限。请使用 MLX 4-bit / 动态量化（DQ）权重（例如 `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`、`Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`），它们可轻松放入约 15 GB 内存中。
> - 体积大于 10GB 的模型在重度多轮上下文场景下会限制 KV 缓存空间。KV 缓存按每个 token 占用 `2 · layers · kv_heads · head_dim · 2 B`（fp16）计算，具体参数从模型的 `config.json` 读取；混合架构模型仅计入其全注意力层加固定循环状态，滑动窗口层在达到窗口长度后停止增长，`--kv-bits 4|8` 以每个元素 `bits + 0.5` 位存储（`--kv-bits` 不适用于滑动窗口层）。例如 Qwen3-4B 为每 token 144 KiB，32k tokens 占用 4.5 GiB；Ornith-1.5-9B（32 层中有 8 层为全注意力）为每 token 32 KiB，32k tokens 占用 1.05 GiB。`serve --dry-run`、`models check`、`fit`（其量化网格与解码速度估算——其中的 RAM 列和适配等级仍仅统计权重大小）以及基于内存推导的 token 预算均采用该数值（上限 = min(请求的 max_tokens, --max-tokens, KV 可用空间 − 提示词 token 数)，其中 KV 可用空间等于 Metal 建议工作集减去已加载权重与提示词缓存预算，提示词由模型自带分词器计数；若提示词完全放不下，会在生成前直接返回 413 拒绝）；在尚未下载配置文件时，上述命令会回退到按参数量估算，这对 7B–32B 稠密模型较为接近，但对小模型、混合架构及 MoE 模型会有数倍偏差。需注意这些是估算值而非实测值：Mamba 层（Jamba、NemotronH）的循环状态未被计入，仅计入其注意力层；MLA 模型（DeepSeek V2/V3）按 MLX 实际缓存的解压后逐头 K 和 V 计入，而非压缩潜向量。

## 下载模型（mlx-community、huihui-ai、Ex0bit）

Basalt 支持浏览、检查并下载 Hugging Face 上任意作者或组织的模型：

```bash
# List models by author
basalt models list --author Ex0bit
basalt models list --author huihui-ai
basalt models list --author mlx-community

# Search with filters
basalt models list --author Ex0bit --search PRISM
basalt models list --search Huihui-Qwen3.8

# Download specific models
basalt models download AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit
basalt models download Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX
basalt models download Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4

# Interactive picker with custom author
basalt models download --author Ex0bit
```
