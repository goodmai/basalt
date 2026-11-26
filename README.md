<p align="center"><img src="images/logo.svg" alt="basalt logo" width="128"></p>

# basalt — your Mac is the AI lab now.

> Serve, train, quantize, compose and dissect open LLMs, neural speech, video, OCR and image models on Apple Silicon. One Swift 6.4 binary speaks the **OpenAI API, the Anthropic API and MCP**, and a native macOS 27 Liquid Glass app sits on top. No Python at runtime. No cloud unless you ask for it.

[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Bahasa Indonesia](README.id.md) · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

📖 **Monograph:** [Evolution of neural network architectures, 1958–2027](docs/monograph/evo.en.md) by Aleksei Boklag, in the six UN languages ([all](docs/monograph/README.md)) · **problem book:** what basalt already runs from it, checklist [#816](https://github.com/goodmai/basalt/issues/816)

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.13.0](https://img.shields.io/badge/release-v0.13.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.13.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2865%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/benchmarks.md#benchmarks)

[![Quality Metrics](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml)
[![Concurrency Hatches](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/concurrency-hatches.json)](scripts/concurrency-hatches.allow)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/swift-testing.json)](Tests/BasaltTests)
[![Files ≤600 LOC](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/file-size.json)](scripts/check_size_limits.swift)
[![Crash Traps](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/crash-traps.json)](scripts/check_quality_metrics.swift)
[![L10n EN/RU](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/l10n-parity.json)](Localization)
[![Supply Chain](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/supply-chain.json)](.github/workflows/quality-metrics.yml)
[![Golden Logits](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/golden-logits.json)](Tests/BasaltTests/Fixtures/ReferenceLogits)

<p align="center"><img src="images/models-hive.svg" alt="Open models basalt runs: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Runs open-weight models on Apple Silicon. Logit-verified here: Gemma 4, Qwen3.8 and LFM2.5 (the table below); measured and packaged in <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a>: Ornith, Bonsai, Qwen3.6-35B-A3B, DeepSeek-OCR-2, OpenJev and Laya (see <a href="docs/models.md#verified-models">the models table</a>); gpt-oss and Llama are MLX-compatible. Logos are trademarks of their owners, shown only to indicate compatibility. Anthropic's models are closed-weight, so they are not in the hive; the Anthropic <i>API</i> is supported so that Claude Code can talk to a local model.</sub></p>

**Five things in one repo:** a local inference server your tools already know how to talk to, LoRA distillation and post-quantization tuning, a docker-compose for multi-model systems, a live X-ray microscope for what happens inside the model, and a native Mac app that puts it all one keystroke away.

**Checked against reference logits on every change** to the inference path: [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) replays recorded prompts and fails when a greedy token or a logit moves. Per-family reference logits for qwen3, qwen3_5, gemma4, lfm2 and nemotron_h, dense and MoE, are in [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits).

| Golden-tested model | Class | Fixture |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | sliding-window + full attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | hybrid linear/full attention, dense | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | hybrid conv + attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### One-click launch from [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) (65+ verified models)

Every model card under [`goodmai`](https://huggingface.co/goodmai) carries a **Run with Basalt** button ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en)) that opens `Basalt.app` via `basalt://run?model=...` or prints the exact `basalt serve` / `basalt compose` CLI command:

| Model / Stack | Role & footprint on 24 GB Mac | Launch |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT daily-driver chat & vision (~7.6 GB, Ladder 4/5 + Lunar 6/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE (`A4B`), golden-verified (14.9 GB, 56–63 tok/s, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 31B dense PV-MACaQ 3.2 bpw for 24 GB Mac (15.1 GB, 14.1 tok/s, Ladder 4/5 + Lunar 4/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B dense PV-MACaQ 4-bit (15.3 GB, 11.6 tok/s p50, PPL 1.23, MSE −19.5%, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE (`A3B`) 3-bit MACaQ (14.6 GB, 64–75 tok/s with `#628` switch WHT) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw trellis (10.0 GB, 16k PPL 4.621) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO native i-quant (11.8 GB, PPL 4.17, 300/300 System One) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 35B MoE in 37.7 GB streamed from SSD at 8.77 tok/s, 0 swap | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B Hadamard ternary in 7.4 GB, 27.1 tok/s with `--mtp` | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B hybrid conv + attention MoE, golden-verified (~12.6 GB) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | 44.1 kHz neural TTS across 43 languages (2.24 GB, RTF 0.23–0.26) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | Vision-encoder OCR with official 768 tiling (`/v1/pipeline`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | 9B local System One decider (98.9% acc, −77% LLM tokens) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/compose.md#bundled-hardware-tiered-profiles-526) | Recommended 24 GB Compose stack (Chat + ZONOS3 + Kokoro ANE) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

### What's new in v0.13.0 ([changelog](CHANGELOG.md))

- **Inkling 2 Mini.** The looped MoE is served end to end, with tool calls and prompt-cache reuse across agent turns. `--mtp` with the checkpoint's DSpark head takes decode from 28.0 to 49.8 tok/s on code (M4 Pro, 24 GB), and a 37.5K-token prompt peaks at 14.5 GB with 0 swap ([#650](https://github.com/goodmai/basalt/issues/650)).
- **NVFP4 from any source.** `models convert --to nvfp4` works from any checkpoint. NVIDIA ModelOpt NVFP4 imports losslessly, on its own at `models download` and `serve`. A MoE with too little KV room streams its experts instead of hitting a Metal OOM ([#677](https://github.com/goodmai/basalt/issues/677)).
- **New architectures and KV.** Qwen3-Coder-Next with streamed experts ([#642](https://github.com/goodmai/basalt/issues/642)), BailingHybrid / Ling-3.0 ([#672](https://github.com/goodmai/basalt/issues/672)), and an experimental Gemma 4 K=V shared-row KV cache that needs 1.40–1.53× less KV memory at 32K–128K ([#645](https://github.com/goodmai/basalt/issues/645)).
- **Adapters and context.** PEFT LoRA adapters load from a directory or the Hub and fuse into exported checkpoints ([#675](https://github.com/goodmai/basalt/issues/675)). `--rope-yarn` extends the context of Qwen 3.5–3.8 ([#729](https://github.com/goodmai/basalt/issues/729)).
- **Translation and self-rewriting.** `basalt:translation` benchmarks (chrF++, COMET, MetricX-24) with TranslateGemma-4 profiles, and experimental self-rewriting on copy-on-write weights (`basalt selfedit`, [#676](https://github.com/goodmai/basalt/issues/676)).
- **Hardening.** A two-agent defect hunt fixed 64 findings across the REST/MCP perimeter, cloud budgets, scoring, conversion, caches and the Mac app ([#809](https://github.com/goodmai/basalt/pull/809), [#810](https://github.com/goodmai/basalt/pull/810), [#811](https://github.com/goodmai/basalt/pull/811)).

---

## Inference: 40B MoE at 67 tok/s, 27B in 7.4 GB, or 37.7 GB MoE with zero swap — behind the APIs you already use

- **Claude Code on your own GPU in one line.** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — your real API key is never touched. OpenAI SDKs, OpenCode, Continue, Cline and aider connect over the OpenAI API, Claude Desktop and Cursor over MCP (with per-tool and per-group `--mcp-enable` / `--mcp-disable`, [#488](https://github.com/goodmai/basalt/pull/488)).
- **Fast on a laptop.** On an M4 Pro with 24 GB: Recursive `Flint-40B-A2.5B` (`oQ3e`) decodes at 67.02 tok/s in 17.61 GB ([#522](https://github.com/goodmai/basalt/pull/522)); `Ornith-1.5-35B-A3B-MACaQ-3bit` decodes at 64–75 tok/s in 14.6 GB with our single-pass `MACaQSwitchLinear` WHT + `gatherQuantizedMM` kernel (`3.41×` faster MoE kernel, `+82%` end-to-end on Gemma 4 26B-A4B, [#628](https://github.com/goodmai/basalt/pull/628)); `Qwen3.5-9B-PV-MACaQ-4bit` reaches `43.2 tok/s` p50 (`+52%` speed, `−33%` RAM vs 8-bit) with exact `4/5` Ladder parity; `Gemma-4-31B-it-PV-MACaQ-3.2bpw` fits 31B dense in `15.1 GB` (`14.1 tok/s`, Ladder `4/5` + Lunar `4/6`) on 24 GB Mac where 4-bit OOMs; Qwen3-4B decodes at 78–89 tok/s; Ternary Bonsai 2 27B fits in 7.4 GB and runs at 20.9–21.5 tok/s, 27.1 with its MTP head and 31–38 with prompt lookup on code ([#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)); `.gguf` GSQ-RCO / `i-quant` checkpoints execute directly on custom Metal kernels ([#433](https://github.com/goodmai/basalt/pull/433), [#515](https://github.com/goodmai/basalt/pull/515)).
- **Models larger than RAM, zero swap.** `Qwen3.6-35B-A3B-8bit` (37.7 GB) decodes at 8.77 tok/s on a 24 GB Mac with 0 swap via batched POSIX AIO and `LIO_NOWAIT` next-layer SSD expert prefetch (`--expert-streaming`, [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478), [#516](https://github.com/goodmai/basalt/pull/516)), alongside UMA-aware L0/L1/L2 tiered KV caching (`TieredKVHierarchyCore`) and adaptive heterogeneous planning ([#627](https://github.com/goodmai/basalt/pull/627)).
- **Speculative decoding that stays exact.** Prompt lookup gives 1.84–2.66× on code edits (Bonsai 2 27B, [#239](https://github.com/goodmai/basalt/pull/239)) and 3.45–3.62× (Gemma 4 12B, [#265](https://github.com/goodmai/basalt/pull/265)), with greedy output unchanged; `SpecTr` OT + `Residual Race`, `EAGLE-3`, `SmallM` (`M=2..16`), `Fast-dLLM` + `DFlash` and `DiffuCD` extend lossless multi-draft and block-diffusion decoding ([#593](https://github.com/goodmai/basalt/pull/593), [#603](https://github.com/goodmai/basalt/pull/603)).
- **The next turn costs only its new tokens.** A prefix-tree prompt cache, even for hybrid models: on Ornith-1.5-9B with a 12.6k-token agent prompt, the second turn takes about 1 s instead of about 36 s ([#82](https://github.com/goodmai/basalt/pull/82)).
- **Long contexts in less memory.** KV quantization with exact zones (`4|8|k8v4`, plus `KVarN` 2D balancing, unbiased 1-bit `QJL`, `ChanMix`/`ThinKV`, cross-lingual `HeadKV` RBO transfer, centroid `RQ-KV`, `turbo3|turbo4|turbo8v4` and `polar4` RoPE-pair keys, [#472](https://github.com/goodmai/basalt/pull/472), [#590](https://github.com/goodmai/basalt/pull/590), [#602](https://github.com/goodmai/basalt/pull/602)): a 16k prompt peaks at 1.03 GB instead of 3.11 GB on Qwen3-4B ([#159](https://github.com/goodmai/basalt/pull/159)).
- **Video, neural TTS, full-duplex speech, music & 2K MLX image synthesis.** Phased 24 GB UMA Video Studio runs `LTX-2.5 AV-DiT`, `Kandinsky 6.0 CrossDiT`, `LingBot-Video MoE` and `Wan 2.2` with causal 3D-VAE, brick-tiled STA Metal kernels and `AVAssetWriter` `.mp4` muxing ([#608](https://github.com/goodmai/basalt/pull/608)–[#611](https://github.com/goodmai/basalt/pull/611)); `LFM2.5-Audio` Mimi/SNAC duplex codecs, `SmartTurn` + `Sortformer`, `MossFormer2`, `Soprano-80M`, `Qwen3ForcedAligner`, `Zyphra/ZONOS3-Mini-hf` (`44.1 kHz`, 43 languages, 2.24 GB, RTF 0.228–0.263, [#527](https://github.com/goodmai/basalt/pull/527)) and `Kokoro-82M` power `/v1/audio/*` ([#604](https://github.com/goodmai/basalt/pull/604)); `FLUX.3` 4-stage compose, `SeedVR2` upscaling, `ACE-Step 1.5` music, `Z-Image-Turbo`, `Iris-3B` and `SANA-1.5` power `/v1/images/*` and `/v1/music/*` ([#542](https://github.com/goodmai/basalt/pull/542), [#605](https://github.com/goodmai/basalt/pull/605)).
- **A whole local stack + Swift SDK.** Tool calls, JSON-schema, regex and grammar output (XGrammar), Meta `Muse-Glimmer-30B` ([#534](https://github.com/goodmai/basalt/pull/534)), `Ripple-65B`, `Nemotron-3.5-Lightning` ([#522](https://github.com/goodmai/basalt/pull/522)), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)), `/v1/embeddings`, `/v1/rerank`, `BasaltFoundationModels` Swift SDK ([#606](https://github.com/goodmai/basalt/pull/606)), `swift-huggingface` Xet downloads ([#587](https://github.com/goodmai/basalt/pull/587)), LoRA/PHM/OFT adapters picked per request, and ordered LAN-before-cloud fallback tiers ([#480](https://github.com/goodmai/basalt/pull/480)).
- **PV-MACaQ 3-bit makes 27B fit and fly on 24 GB.** `Qwen3.8-27B-PV-MACaQ-3bit` passes the Official Ladder 5/5 (including `lunar` 6/6) in 11.5–12.4 GB at 15.6–16.7 tok/s on an M4 Pro, against 4/5 rungs, 15.3 GB and 11.6 tok/s for the stock 4-bit. `--speculative-lookup` now also runs on Qwen3.5/3.8 checkpoints that ship a vision tower (43–51% draft acceptance, +17% on code edits), and `--kv-bits kvarn|kvarn-qjl` (2-bit KV) and `--attention-backend moba` are wired into `basalt serve` and pass a 4k passkey ([#632](https://github.com/goodmai/basalt/issues/632), [#634](https://github.com/goodmai/basalt/issues/634)). Checkpoints: [`goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX), [`goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX).

Details: [REST API](docs/rest-api.md) · [integrations](docs/integrations.md) · [commands and flags](docs/commands.md) · [models](docs/models.md) · [benchmarks](docs/benchmarks.md)

## Training & Quantization: universal any-to-any converter, PV-MACaQ, distributed & QAT

- **Universal any-to-any quantization converter (`basalt models convert`).** Convert across `MACaQ` (`affine` / `mxfp4` / `moe` 2.8 bpw router-weighted Hessian, [#537](https://github.com/goodmai/basalt/pull/537), [#599](https://github.com/goodmai/basalt/pull/599)), `FlatQuant` & `LIRQuant` ([#588](https://github.com/goodmai/basalt/pull/588), [#594](https://github.com/goodmai/basalt/pull/594)), `SINQ` / `A-SINQ` / `EAQuant` dual-axis normalization ([#561](https://github.com/goodmai/basalt/pull/561)), `QuIP# E8P` 2-bit lattice & `TurboBoA` Babai–LLL ([#589](https://github.com/goodmai/basalt/pull/589)), `QTIP` bitshift trellis (`2.0–3.0 bpw`, [#494](https://github.com/goodmai/basalt/pull/494)), `SVD-LLM V2` / `OBR` ([#592](https://github.com/goodmai/basalt/pull/592)), `.gguf` i-quants & `MACaQ-R`, plus domain-aware `QAT STE` refinement and post-quantization quality/fairness/leakage audit (`QuantizationAuditCore`, [#623](https://github.com/goodmai/basalt/pull/623)) — wired identically across CLI, REST `/v1/models/convert*` and macOS `ConvertModelSheet` ([#546](https://github.com/goodmai/basalt/issues/546)).
- **Post-quantization PV-Tuning & PV-MACaQ.** `basalt models convert --to macaq:affine:4:64 --pv-tune` alternates continuous scale/bias updates with discrete Hessian flip-refinement over Walsh–Hadamard blocks: on `Qwen3.5-9B`, `8-bit → 4-bit PV-MACaQ` cuts RAM by 33% (`9.97 → 6.68 GB`) and speeds up decode by `+52%` (`27.3 → 41.6 tok/s`, p50 `43.2`) with exact `4/5` Ladder parity (`valKL = 0.0044`); on 2–3-bit checkpoints, PV-Tuning cuts perplexity loss by 8.7–29% in under 3 minutes ([#486](https://github.com/goodmai/basalt/pull/486), [#545](https://github.com/goodmai/basalt/pull/545)).
- **Distill, post-train & distribute across Macs.** `basalt models distill` distills OpenAI/Anthropic teachers into `lora`, `phm`, `oft` or quantized `AdaLoRA (LR/40)` adapters ([#501](https://github.com/goodmai/basalt/pull/501), [#508](https://github.com/goodmai/basalt/pull/508), [#624](https://github.com/goodmai/basalt/pull/624)); two-stage `SFT → DPO`, zero-copy `ADPO`/`cADPO`, `EMA-QLoRA`, `Self-Appraisal`, and `RLVR` anti-shortcut audit keep preference alignment sharp ([#601](https://github.com/goodmai/basalt/pull/601), [#624](https://github.com/goodmai/basalt/pull/624)); multi-Mac cluster training supports authenticated transport, `DDP + ZeRO-2`, `DiLoCo` + `PowerSGD`, `Zero-Bubble ZB-H1` and `Expert-Parallel MoE` ([#595](https://github.com/goodmai/basalt/pull/595)–[#598](https://github.com/goodmai/basalt/pull/598)).
- **Train while you serve, with a quality gate.** Time-sliced MCP `distill`, verified-only `train_experts`, `train-jepa`, and bit-for-bit rollback if held-out KL or perplexity regresses.

Details: [training commands](docs/commands.md) · [self-training, distillation and the quality gate](docs/research-tools.md#self-training-train_experts)

## Compose: declarative `basalt.yaml v2.2`, business governance & 117 hardware profiles

- **docker-compose, for models (`basalt.yaml v2.2`).** Local and cloud models, RAM deduplication planner, specialist profiles, `resident` / `on-demand` / `sequential` memory modes, `BusinessOrchestrator` 5-component routing, `LANTERN` semantic caching, `HITL` approval gate (`/v1/mcp/approvals`), and `LAAS-CNRS` neuro-symbolic knowledge core (`SymbolicKnowledgeGraph`) — in one declarative file ([#607](https://github.com/goodmai/basalt/pull/607), [#625](https://github.com/goodmai/basalt/pull/625), [#626](https://github.com/goodmai/basalt/pull/626)).
- **117 built-in profiles for every Mac (`16 GB` to `256 GB`).** Nine unified-memory tiers (`16gb`, `24gb`, `32-36gb`, `48gb`, `64gb`, `96gb`, `128gb`, `192gb`, `256gb`) × 13 workload archetypes launch in one command (`basalt compose up 24gb-voice-zonos`, [#528](https://github.com/goodmai/basalt/pull/528)).
- **Refuses what will not fit, changes live.** `basalt compose plan` sizes models from the Hub before download and refuses overflows; `compose diff` / `compose up` hot-reload diffs cleanly; `compose bench` runs declarative benchmark suites and exports leaderboard bundles ([#505](https://github.com/goodmai/basalt/pull/505)).
- **Local judges, calibrated routing & cloud budgets.** `OpenJev-Flash-9B-4bit` reaches 98.9% triage accuracy while cutting downstream LLM tokens by 77% ([#465](https://github.com/goodmai/basalt/pull/465)), temperature-scaled `CalibratedRouterGate` + `Gemini Decision 1` route hard queries cleanly within daily USD caps ([#418](https://github.com/goodmai/basalt/pull/418), [#622](https://github.com/goodmai/basalt/pull/622)), and perceived document/audio text never leaves your Mac by default.

Details: [compose reference and measurements](docs/compose.md) · [pipeline presets and memory modes](docs/pipelines.md) · [cloud decisions with Gemini](docs/cloud-decisions.md)

## Research: open the model up while it runs

- **Model X-ray, per-token attribution & Belnap epistemic gating.** `model_describe` (reporting `MACaQ` / `FlatQuant` / `LIRQuant` module manifests, [#630](https://github.com/goodmai/basalt/pull/630)), `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run`, `trace_tokens` (with automatic Gemma BOS handling, [#630](https://github.com/goodmai/basalt/pull/630)), `/v1/xray/tokens` per-token logprobs & hallucination risk ([#613](https://github.com/goodmai/basalt/pull/613)), `Score d'influence` head attribution ([#622](https://github.com/goodmai/basalt/pull/622)), Marchenko–Pastur spectra, four-valued Belnap `SIXTEEN_3` evidence states ([#500](https://github.com/goodmai/basalt/pull/500)), complex MVN phasor probes ([#509](https://github.com/goodmai/basalt/pull/509)), Fisher–Rao drift ([#503](https://github.com/goodmai/basalt/pull/503)), and OpenMetrics at `/api/v1/metrics` ([#458](https://github.com/goodmai/basalt/pull/458)).
- **Reversible surgery & mechanistic circuits.** With `serve --research-tools`: `intervene` (including conditional ablation, GQA group patching, cognitive nullspace & anti-steering guard, [#600](https://github.com/goodmai/basalt/pull/600)), `MAO-SNMF` feature decomposition, `weights_delta`, live `tensor_pack`/`tensor_unpack`, `activation_contrast`, `steer_search`/`steer_commit`, `experiment_compare`, and `introspect`. Every experiment is an undoable overlay with a TTL and journal; weights on disk are never modified.
- **Benchmarks that run the model's code.** A five-rung ladder ending in an Earth-to-Moon mission executes the model's own code against unseen inputs, and ARC-AGI runs under official pass@2 rules.

Details: [research tools](docs/research-tools.md) · [research foundations (95+ papers)](docs/research-foundations.md) · [benchmarks](docs/benchmarks.md) · [serve flags](docs/commands.md)

## Basalt for Mac: macOS 27 Liquid Glass, one keystroke or one click away

- **A native SwiftUI app on a supervised engine.** Built for macOS 27 Liquid Glass ([#479](https://github.com/goodmai/basalt/pull/479)) in English and Russian ([#456](https://github.com/goodmai/basalt/pull/456)): 257 ms warm window, 0 idle wakeups, 7.4 ms p95 per streamed token in a 2,000-message transcript, and 40 ms `⌥Space` quick-ask overhead ([#463](https://github.com/goodmai/basalt/pull/463)).
- **Compose Video Studio, 2K Image Studio & Universal Converter UI.** Generate and play A/V `.mp4` videos on 24 GB Macs ([#611](https://github.com/goodmai/basalt/pull/611)), edit 2K RGBA images with Before/After slider ([#519](https://github.com/goodmai/basalt/pull/519)), convert & QAT-audit models directly in `ConvertModelSheet` ([#623](https://github.com/goodmai/basalt/pull/623)), and browse/publish on Hugging Face with 10 Hz SSE telemetry & Xet downloads ([#587](https://github.com/goodmai/basalt/pull/587)).
- **Compose Profiles (`⌥⌘C`), full-duplex voice & `basalt://` 1-click launch.** Auto-recommends top profiles for your Mac's RAM tier (`16–256 GB`, [#528](https://github.com/goodmai/basalt/pull/528)), streams hands-free voice via `SpeechAnalyzer` / `LFM2.5-Audio` / `ZONOS3-Mini` ([#604](https://github.com/goodmai/basalt/pull/604)), and handles `basalt://run?model=...` links, `⌥Space` quick ask, Shortcuts, Spotlight, and notarized `.dmg` packaging.


**Basalt.app: Compose running on the engine and the chat with its inspector** (macOS 27 Liquid Glass, dark & light):

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-compose-dark.png"><img src="docs/assets/screenshots/app-compose-light.png" alt="Basalt.app — Compose profile active on the engine" width="49%"></picture></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-chat-dark.png"><img src="docs/assets/screenshots/app-chat-light.png" alt="Basalt.app — chat with the inspector open" width="49%"></picture></p>

Details: [Basalt for Mac](docs/mac-app.md)

---

## Quick start

Requirements: macOS 27+, Xcode 27.0+ / Swift 6.4 with the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), Apple Silicon M1–M4, 2–38 GB of disk depending on the model. Full steps, building from source and the `./basalt-claude` launcher are in [docs/install.md](docs/install.md).

**No Xcode, no build — run the prebuilt Apple Silicon binary in two lines** (macOS 27+, Apple Silicon; `mlx.metallib` ships inside the archive; the first run downloads the 12 GB model, use `goodmai/gemma-4-12B-it-qat-4bit` for a lighter start):

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

Stream a 37.7 GB MoE model on a 24 GB Mac, or run a full multi-model compose system:

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 117 profiles across 16–256 GB Macs
```

The Mac app, from a checkout (registers `basalt://` 1-click links automatically on first open):

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

Local 4B–12B models are not a drop-in replacement for a frontier model — they are for private, offline, zero-cost agentic work and for hybrid local/cloud systems. `basalt fit` tells you what your machine can actually run. The model table lists what *broke* too.

**Star it** if you want more MLX-native tooling in Swift, and open an issue with your Mac and your tok/s — [PRs adding a row to the benchmark table](docs/benchmarks.md#benchmarks) are welcome.

---

## Documentation

> Translated documentation is available under `docs/i18n/`: [Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/) · [한국어](docs/i18n/ko/) · [Bahasa Indonesia](docs/i18n/id/).

| Topic | Doc |
|---|---|
| Install, requirements, build from source, `./basalt-claude` | [docs/install.md](docs/install.md) |
| CLI commands, `serve` flags, prompt cache, CacheBlend, PV-Tuning | [docs/commands.md](docs/commands.md) |
| REST API: security, model management, OpenAI, Anthropic, images, SSE, OpenMetrics, Swagger | [docs/rest-api.md](docs/rest-api.md) |
| Claude Code, OpenCode, OpenAI SDKs, MCP for Claude Desktop and Cursor | [docs/integrations.md](docs/integrations.md) |
| Verified models, `goodmai` Hub stack and downloading | [docs/models.md](docs/models.md) |
| Features at a glance, benchmark ladder, ARC-AGI, provenance | [docs/benchmarks.md](docs/benchmarks.md) |
| Pipeline presets and memory modes | [docs/pipelines.md](docs/pipelines.md) |
| Cloud decisions: Gemini Decision 1, multimodal media, the calibration teacher | [docs/cloud-decisions.md](docs/cloud-decisions.md) |
| `basalt compose` reference and measurements | [docs/compose.md](docs/compose.md) |
| Basalt for Mac & `basalt://` deep links | [docs/mac-app.md](docs/mac-app.md) |
| Model X-ray, telemetry, experiments, self-training, distillation | [docs/research-tools.md](docs/research-tools.md) |
| Mathematical & scientific research foundations (85+ papers & acknowledgements) | [docs/research-foundations.md](docs/research-foundations.md) |
| Project structure, architecture notes, scripts, dependencies | [docs/architecture.md](docs/architecture.md) |
| Architecture patterns and modules (Russian) | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| Cloud: OpenRouter setup, cost optimization, model comparison | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| Working on basalt with AI agents | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| Roadmap, privacy policy | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| Research notes on agent CLIs, checklists, specs (Russian) | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| Contributing, changelog | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## License

[MIT](LICENSE) © 2026 goodmai

Model weights are **not** covered by this license — each model on HuggingFace carries its own terms.
