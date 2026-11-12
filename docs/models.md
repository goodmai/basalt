# Models

## Verified Models

Models are downloaded from HuggingFace and cached in `~/.cache/huggingface/hub/`.

```bash
huggingface-cli download mlx-community/Qwen3.5-4B-4bit
huggingface-cli download mlx-community/gemma-4-e4b-it-4bit
```

### Measured for 0.8

M4 Pro, 24 GB, on a 68 W adapter, release build, greedy decoding, November 2026.

| Model | Params | Active memory | Decode, tok/s | Notes |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | ~2.2 GB | 78–89 | dense; prompt lookup and every KV mode apply |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B ternary | 7.4 GB | 20.9–21.5 plain · 27.1 with `--mtp` on code · 31–38 with `--speculative-lookup` on code | hybrid (Gated DeltaNet); MTP head from Qwen3.8-27B |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | not measured | 15–19 | hybrid |
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
