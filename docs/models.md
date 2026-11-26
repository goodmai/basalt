# Models

> Languages: **English** · [Русский](i18n/ru/models.md) · [中文](i18n/zh/models.md) · [Español](i18n/es/models.md) · [Français](i18n/fr/models.md) · [日本語](i18n/ja/models.md) · [Deutsch](i18n/de/models.md) · [한국어](i18n/ko/models.md) · [Bahasa Indonesia](i18n/id/models.md)

## Verified Models

> **Backend status (`#828`):** Real-weight tests and measurements cover LLMs/VLMs (tables below), embeddings/rerank, `DeepSeek-OCR-2`, `Nemotron Transcribe` / Apple `SpeechAnalyzer`, and `Zyphra/ZONOS3-Mini-hf` neural TTS. Image/video generators (`LTX-2.5`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`, `FLUX`, `Z-Image-Turbo`, `SeedVR2`, `ACE-Step 1.5`) and other TTS/duplex backends (`Kokoro-82M`, `Qwen3-TTS`, `CosyVoice 3`, `Chatterbox`, `Pocket TTS`, `LFM2.5-Audio`) currently ship as **scaffolds** (fixed random seed, partial weight loading) — real-weight implementations are tracked in [#824](https://github.com/goodmai/basalt/issues/824), [#821](https://github.com/goodmai/basalt/issues/821), [#818](https://github.com/goodmai/basalt/issues/818) and [#826](https://github.com/goodmai/basalt/issues/826).

Models are downloaded from HuggingFace and cached in `~/.cache/huggingface/hub/`. Over 42 verified chat, MoE, QTIP, PV-Tuned, GSQ-RCO, OCR, neural TTS, embedding, rerank and System One decision checkpoints are published with one-click **Run with Basalt** ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en) → `basalt://run?model=...`) buttons in the [`goodmai` Hugging Face collection](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc).

```bash
basalt models download goodmai/gemma-4-12B-it-qat-4bit
basalt models download goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native
```

### Measured for 0.12

M4 Pro, 24 GB, release build (`620641c` / `82a99d4`), November 2026 (`--reasoning-effort none --kv-bits 4`).

| Model | Role / Architecture | Active memory | Decode tok/s | Ladder / Notes |
|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` | 31B dense hybrid (4-bit attn/embed + 3-bit affine PV-MACaQ MLP) | 15.1 GB (18.09 GB peak) | 11.6–13.5 (p50 `14.07`, p95 `14.42`) | **3/3 warmup, 4/5 rungs + Lunar 4/6**, `probe_ppl = 2.39`, 50.6% lookup accept; fits 24 GB Mac where 4-bit OOMs (`--prompt-cache-mb 0 --wired-limit`) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` | 27B dense hybrid (48 linear + 16 full attn, 496 MACaQ 4-bit g64 modules) | 15.3 GB (17.45 GB peak) | 9.4–11.4 (p50 `11.59`, p95 `11.86`) | **3/3 warmup, 4/5 rungs**, `probe_ppl = 1.23`, `SQNR 20.61 → 21.59 dB` (`MSE −19.5%`), max 64-layer drift `5.53%` (`#630`) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (from `Qwen3.5-9B-8bit`) | 9B hybrid (24 linear + 8 full attn, 200 MACaQ 4-bit g64 modules) | 6.68 GB (−33% vs 8-bit) | 35.6–41.6 (p50 `43.22`, +52% vs 8-bit) | **3/3 warmup, 4/5 rungs** (100% parity with 8-bit ref `24.7–27.3 tok/s`), `valKL = 0.0044`, `probe_ppl = 1.28`, max layer drift `3.37%` |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` | 26B MoE (`A4B`, 128 experts, top-8, g32 affine MACaQ + 8-bit shared MLP) | 16.75 GB | 49.2–54.8 (+82% from `#628` switch WHT) | **3/3 warmup, 3/5 rungs** (`biquad-complex`, `fourier`, `pump`); `3.41×` kernel speedup (`2.06 → 0.60 ms`, `#628`) |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` | 35B MoE (`A3B`, 120 `MACaQSwitchLinear` layers) | 14.64 GB | 64.0–74.7 (68.7 at 8k) | **3/3 warmup, 1/5 rungs**, `probe_ppl = 1.46`, single-pass switch WHT (`#628`) |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` | 26B MoE (`A4B`, QAT 4-bit) | 14.92 GB | 56.0–63.4 | **3/3 warmup, 4/5 rungs** (`pump` ✅, `lunar 5/6`) |
| `mlx-community/Inkling-2-Mini-oQ3e-mtp` | Inkling 2 Mini: looped MoE (48 visits), relative-bias attention, DSpark drafter (oQ3e) | 11.6 GB (14.5 GB peak at 37.5K tokens, swap 0) | 27.4–27.7; `--mtp` 48.2 on code at T 0, 39.1 at T 0.7, even on prose | **3/3 warmup, 3/5 rungs**, 32K passkey 5/5 (`reasoning_effort: none` still reasons for 119–314 tokens, so give it ≥ 512); prefill ≈ 330 tok/s at 4K, ≈ 155 tok/s at 32K until K1 (#650); measured on `0c3e247` |
| `goodmai/gemma-4-12B-it-qat-4bit` | 12B dense QAT 4-bit | 10.51 GB | 18.8–22.2 | **3/3 warmup, 4/5 rungs** (`lunar` ✅ 6/6) |
| Video / Image / Duplex Speech / Music (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`, `LFM2.5-Audio`, `FLUX.3`, `SeedVR2`, `ACE-Step 1.5`) | **Scaffold** (fixed random seed, partial tensor match; not real-weight verified) | — | — | Pipeline & Metal-kernel scaffolding only; real-weight work tracked in [#824](https://github.com/goodmai/basalt/issues/824), [#826](https://github.com/goodmai/basalt/issues/826) |

### Measured for 0.11

M4 Pro, 24 GB, release build, November 2026.

| Model | Role / Architecture | Active memory | Throughput / Latency | Notes |
|---|---|---|---|---|
| `Recursive/Flint-40B-A2.5B` (`flint-oQ3e`) | 40B-A2.5B hybrid sliding/full MoE (`512` routed, top-10 + 1 shared) | 17.61 GB | 67.02 tok/s decode · 273.5 tok/s prefill | native `oQ3e` (`groupSize = 32`) loader (#518, #522), 0 swap on 24 GB |
| `Zyphra/ZONOS3-Mini-hf` | 1.6B Llama + 32-codebook FSQ multi-head DAC TTS (`44.1 kHz`) | 2.24 GB | RTF 0.228–0.263 (3.8–4.4× RT) | 43 languages, 27 voices, cross-lingual `embedding` + CFG `1.5` (#525, #527) |
| `hexgrad/Kokoro-82M` (and `Qwen3-TTS`, `CosyVoice 3`, `Chatterbox`, `Pocket TTS`) | **Scaffold** (fixed random seed, partial tensor match) | — | — | Not real-weight verified yet; real-weight Kokoro/CosyVoice 3/F5-TTS tracked in [#821](https://github.com/goodmai/basalt/issues/821) (use `Zyphra/ZONOS3-Mini-hf` for real-weight TTS today) |
| `Tongyi-MAI/Z-Image-Turbo` (and `FLUX` / `Iris-3B` / `SANA-1.5`) | **Scaffold** (hash-embedding + partial DiT weight match) | — | — | Not real-weight verified yet; real-weight Z-Image-Turbo & FLUX-class DiT tracked in [#824](https://github.com/goodmai/basalt/issues/824) |
| `Ricursive/Ripple-65B-4bit` | 65B gated-attention sparse MoE (`128` experts, top-8) | ~35 GB (64 GB+ resident, or SSD streaming) | architecture verified | NoPE / partial RoPE (#520, #522) |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | 30B-A3B Mamba-2 + Latently-Routed MoE (`lr_moe`) | ~17 GB (4-bit) | architecture verified | interleaved GQA + Mamba-2 (#521, #522) |
| `meta-models/Muse-Glimmer-30B` (`mlx-community/Muse-Glimmer-30B-4bit`) | 30B hybrid SWA (`2048`) + NoPE full attention + ViT-G/14 VLM | ~18.1 GB (4-bit) · ~19.8 GB (`oQ4e`) | architecture & HF shards verified | Onyx `to=self` reasoning + ATEM tool calls + GGUF de-centered norm import (#533) |

### Measured for 0.8

M4 Pro, 24 GB, on a 68 W adapter, release build, greedy decoding, November 2026.

| Model | Params | Active memory | Decode, tok/s | Notes |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | ~2.2 GB | 78–89 | dense; prompt lookup and every KV mode apply |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B ternary | 7.4 GB | 20.9–21.5 plain · 27.1 with `--mtp` on code · 31–38 with `--speculative-lookup` on code | hybrid (Gated DeltaNet); mirrored at `goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | ~11.8 GB | 15–19 | hybrid; golden-tested; mirrored at `goodmai/Qwen3.8-27B-3bit-MLX-TextOnly` |
| `goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native` | 27B i-quant | ~12.4 GB | 5.2 | native GSQ-RCO `IQ3_S` import (#433), WikiText-2 PPL 4.169, 300/300 System One |
| `goodmai/Qwen3.6-35B-A3B-8bit` | 35B MoE (37.7 GB) | ~13.9 GB | 8.77 | `--experimental-expert-streaming` (#478), 0 swap on 24 GB via batched AIO + `LIO_NOWAIT` |
| `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly` | 27B | not measured | 18–22 | loops on some prompts (3 of 8); `RepetitionGuard` stops them |

| Embedding / rerank model | Task | Measured |
|---|---|---|
| `mlx-community/bge-small-en-v1.5-bf16` | embeddings (384-d) | 229–299 texts/s, batch 32 × ~240 tokens |
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | embeddings (1024-d) | 20 texts/s, same batch |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | rerank | known pairs ranked correctly |
| `jinaai/jina-reranker-v3-mlx` | rerank | known pairs ranked correctly |

### Earlier reports (not re-measured for 0.8)

| Model | Params | RAM | Status on 24GB Mac |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 GB | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 GB | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 GB | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 GB | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 GB | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 GB | ✅ ~12 TPS (Abliterated) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 GB | ✅ ~14 TPS (Dynamic Quant) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 GB | ✅ ~8 TPS (NVFP4) |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 GB | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 GB | ✅ ~48 TPS (reasoning — see note) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ untested above 24 GB |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 GB | ✅ ~76 TPS (always reasons) |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 GB | ✅ ~40 TPS (always reasons) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (Base BF16) | 27B | ~54 GB | ❌ Needs 64GB+ (Use 4-bit MLX version on 24GB) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ Gibberish output |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ Gibberish output |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM (Needs 32GB+) |

> **Note on 24GB RAM Macs:** 
> - Unquantized 27B/35B models (~54 GB weights) exceed physical memory. Use MLX 4-bit / Dynamic Quant (DQ) quantized weights (e.g. `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`, `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`) which comfortably fit in ~15 GB RAM.
> - Models > 10GB constrain KV-cache size under heavy multi-turn contexts. The KV cache takes `2 · layers · kv_heads · head_dim · 2 B` per token (fp16), read from the model's `config.json`; hybrid models count only their full-attention layers plus a fixed recurrent state, sliding-window layers stop at the window, `--kv-bits 4|8` stores `bits + 0.5` bits per element (`--kv-bits` does not apply to sliding-window layers). Qwen3-4B: 144 KiB per token, 4.5 GiB at 32k tokens; Ornith-1.5-9B (8 of 32 layers full attention): 32 KiB per token, 1.05 GiB at 32k. `serve --dry-run`, `models check`, `fit` (its quantization grid and decode-speed estimate — the RAM column and fit level still count weights only) and the memory-derived token budget use this figure (cap = min(requested max_tokens, --max-tokens, KV room − prompt tokens), where the KV room is Metal's recommended working set minus the loaded weights and the prompt-cache budget, and the prompt is counted with the model's tokenizer; a prompt that does not fit at all is refused with 413 before generation); without a downloaded config they fall back to a parameter-count estimate that is close for dense 7B–32B models but several times off for small, hybrid and MoE ones. Estimates, not measurements: the recurrent state of Mamba layers (Jamba, NemotronH) is not counted, only their attention layers; MLA models (DeepSeek V2/V3) are counted as the decompressed per-head K and V that MLX caches, not the compressed latent.

## Downloading Models (mlx-community, huihui-ai, Ex0bit)

Basalt can browse, inspect, and download models from any creator or organization on Hugging Face:

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
