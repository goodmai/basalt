# Modelle

> Languages: [English](../../models.md) · [Русский](../ru/models.md) · [中文](../zh/models.md) · [Español](../es/models.md) · [Français](../fr/models.md) · [日本語](../ja/models.md) · **Deutsch**

## Verifizierte Modelle

Modelle werden von HuggingFace heruntergeladen und unter `~/.cache/huggingface/hub/` zwischengespeichert. Über 42 verifizierte Chat-, MoE-, QTIP-, PV-Tuned-, GSQ-RCO-, OCR-, neuronale TTS-, Embedding-, Rerank- und System-One-Decision-Checkpoints sind mit Ein-Klick-Schaltflächen **Run with Basalt** ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en) → `basalt://run?model=...`) in der [`goodmai` Hugging Face Collection](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) veröffentlicht.

```bash
basalt models download goodmai/gemma-4-12B-it-qat-4bit
basalt models download goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native
```

### Messwerte für 0.12

M4 Pro, 24 GB, Release-Build (`620641c` / `82a99d4`), November 2026 (`--reasoning-effort none --kv-bits 4`).

| Modell | Rolle / Architektur | Aktiver Speicher | Decode tok/s | Ladder / Hinweise |
|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` | 31B Dense Hybrid (4-bit Attn/Embed + 3-bit affines PV-MACaQ MLP) | 15.1 GB (18.09 GB Peak) | 11.6–13.5 (p50 `14.07`, p95 `14.42`) | **3/3 Warm-up, 4/5 Stufen + Lunar 4/6**, `probe_ppl = 2.39`, 50.6% Lookup-Akzeptanz; passt auf 24-GB-Mac, wo 4-bit OOM läuft (`--prompt-cache-mb 0 --wired-limit`) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` | 27B Dense Hybrid (48 Linear + 16 Full Attn, 496 MACaQ 4-bit g64 Module) | 15.3 GB (17.45 GB Peak) | 9.4–11.4 (p50 `11.59`, p95 `11.86`) | **3/3 Warm-up, 4/5 Stufen**, `probe_ppl = 1.23`, `SQNR 20.61 → 21.59 dB` (`MSE −19.5%`), max. 64-Layer-Drift `5.53%` (`#630`) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (aus `Qwen3.5-9B-8bit`) | 9B Hybrid (24 Linear + 8 Full Attn, 200 MACaQ 4-bit g64 Module) | 6.68 GB (−33% ggü. 8-bit) | 35.6–41.6 (p50 `43.22`, +52% ggü. 8-bit) | **3/3 Warm-up, 4/5 Stufen** (100% Parität mit 8-bit Ref `24.7–27.3 tok/s`), `valKL = 0.0044`, `probe_ppl = 1.28`, max. Layer-Drift `3.37%` |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` | 26B MoE (`A4B`, 128 Experten, Top-8, g32 affines MACaQ + 8-bit Shared-MLP) | 16.75 GB | 49.2–54.8 (+82% durch `#628` Switch-WHT) | **3/3 Warm-up, 3/5 Stufen** (`biquad-complex`, `fourier`, `pump`); `3.41×` Kernel-Speedup (`2.06 → 0.60 ms`, `#628`) |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` | 35B MoE (`A3B`, 120 `MACaQSwitchLinear`-Layer) | 14.64 GB | 64.0–74.7 (68.7 bei 8k) | **3/3 Warm-up, 1/5 Stufen**, `probe_ppl = 1.46`, Single-Pass-Switch-WHT (`#628`) |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` | 26B MoE (`A4B`, QAT 4-bit) | 14.92 GB | 56.0–63.4 | **3/3 Warm-up, 4/5 Stufen** (`pump` ✅, `lunar 5/6`) |
| `goodmai/gemma-4-12B-it-qat-4bit` | 12B Dense QAT 4-bit | 10.51 GB | 18.8–22.2 | **3/3 Warm-up, 4/5 Stufen** (`lunar` ✅ 6/6) |
| Video / Sprache / Musik (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`, `LFM2.5-Audio`, `FLUX.3`, `SeedVR2`, `ACE-Step 1.5`) | Phasenweises 24-GB-UMA-Video-Studio, Duplex-Mimi/SNAC-Audio & 4-stufige Bild-/Musiksynthese | ≤ 18 GB phasenweise | 3D-VAE + STA-Metal-Kernel | Verifiziert in `#542`, `#604`, `#605`, `#608`–`#611` |

### Messwerte für 0.11

M4 Pro, 24 GB, Release-Build, November 2026.

| Modell | Rolle / Architektur | Aktiver Speicher | Durchsatz / Latenz | Hinweise |
|---|---|---|---|---|
| `Recursive/Flint-40B-A2.5B` (`flint-oQ3e`) | 40B-A2.5B Hybrid Sliding/Full MoE (`512` geroutete, Top-10 + 1 Shared) | 17.61 GB | 67.02 tok/s Decode · 273.5 tok/s Prefill | nativer `oQ3e` (`groupSize = 32`)-Loader (#518, #522), 0 Swap auf 24 GB |
| `Zyphra/ZONOS3-Mini-hf` | 1.6B Llama + 32-Codebook-FSQ-Multi-Head-DAC-TTS (`44.1 kHz`) | 2.24 GB | RTF 0.228–0.263 (3.8–4.4× RT) | 43 Sprachen, 27 Stimmen, sprachübergreifendes `embedding` + CFG `1.5` (#525, #527) |
| `hexgrad/Kokoro-82M` | 82M neuronales TTS (ANE + MLX) | ~0.35 GB | unter 80 ms TTFA | satzweises Streaming als Fallback-Stufe in `pipeline.speak` (#493, #513) |
| `Tongyi-MAI/Z-Image-Turbo` | DiT-Bildsynthese | ~6.4 GB | 4–8 Schritte (`1024×1024`) | `pipeline.imagine` mit TeaCache & gemeinsamem Text-Encoder (#517, #519) |
| `Ricursive/Ripple-65B-4bit` | 65B Gated-Attention Sparse MoE (`128` Experten, Top-8) | ~35 GB (64 GB+ resident oder SSD-Streaming) | Architektur verifiziert | NoPE / partielles RoPE (#520, #522) |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | 30B-A3B Mamba-2 + Latently-Routed MoE (`lr_moe`) | ~17 GB (4-bit) | Architektur verifiziert | verzahntes GQA + Mamba-2 (#521, #522) |
| `meta-models/Muse-Glimmer-30B` (`mlx-community/Muse-Glimmer-30B-4bit`) | 30B Hybrid SWA (`2048`) + NoPE Full Attention + ViT-G/14 VLM | ~18.1 GB (4-bit) · ~19.8 GB (`oQ4e`) | Architektur & HF-Shards verifiziert | Onyx-`to=self`-Reasoning + ATEM-Tool-Calls + dezentrierter GGUF-Norm-Import (#533) |

### Messwerte für 0.8

M4 Pro, 24 GB, am 68-W-Netzteil, Release-Build, Greedy-Decoding, November 2026.

| Modell | Parameter | Aktiver Speicher | Decode, tok/s | Hinweise |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | ~2.2 GB | 78–89 | Dense; Prompt-Lookup und jeder KV-Modus anwendbar |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B ternär | 7.4 GB | 20.9–21.5 pur · 27.1 mit `--mtp` bei Code · 31–38 mit `--speculative-lookup` bei Code | Hybrid (Gated DeltaNet); gespiegelt unter `goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | ~11.8 GB | 15–19 | Hybrid; golden-getestet; gespiegelt unter `goodmai/Qwen3.8-27B-3bit-MLX-TextOnly` |
| `goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native` | 27B i-Quant | ~12.4 GB | 5.2 | nativer GSQ-RCO-`IQ3_S`-Import (#433), WikiText-2 PPL 4.169, 300/300 System One |
| `goodmai/Qwen3.6-35B-A3B-8bit` | 35B MoE (37.7 GB) | ~13.9 GB | 8.77 | `--experimental-expert-streaming` (#478), 0 Swap auf 24 GB durch gebatchtes AIO + `LIO_NOWAIT` |
| `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly` | 27B | nicht gemessen | 18–22 | gerät bei manchen Prompts (3 von 8) in Schleifen; `RepetitionGuard` stoppt sie |

| Embedding- / Rerank-Modell | Aufgabe | Messwert |
|---|---|---|
| `mlx-community/bge-small-en-v1.5-bf16` | Embeddings (384-d) | 229–299 Texte/s, Batch 32 × ~240 Tokens |
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | Embeddings (1024-d) | 20 Texte/s, gleicher Batch |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | Rerank | bekannte Paare korrekt gerankt |
| `jinaai/jina-reranker-v3-mlx` | Rerank | bekannte Paare korrekt gerankt |

### Frühere Berichte (für 0.8 nicht erneut gemessen)

| Modell | Parameter | RAM | Status auf 24-GB-Mac |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 GB | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 GB | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 GB | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 GB | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 GB | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 GB | ✅ ~12 TPS (abliterated) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 GB | ✅ ~14 TPS (Dynamic Quant) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 GB | ✅ ~8 TPS (NVFP4) |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 GB | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 GB | ✅ ~48 TPS (Reasoning — siehe Hinweis) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ ungetestet oberhalb von 24 GB |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 GB | ✅ ~76 TPS (denkt immer) |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 GB | ✅ ~40 TPS (denkt immer) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (Base BF16) | 27B | ~54 GB | ❌ Benötigt 64 GB+ (auf 24 GB die 4-bit-MLX-Version nutzen) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ Wirre Ausgabe |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ Wirre Ausgabe |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM (benötigt 32 GB+) |

> **Hinweis zu Macs mit 24 GB RAM:**
> - Unquantisierte 27B/35B-Modelle (~54 GB Gewichte) überschreiten den physischen Speicher. Nutze MLX-4-bit- oder Dynamic-Quant-(DQ-)Gewichte (z. B. `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`, `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`), die bequem in ~15 GB RAM passen.
> - Modelle > 10 GB begrenzen die KV-Cache-Größe bei umfangreichen Multi-Turn-Kontexten. Der KV-Cache belegt `2 · layers · kv_heads · head_dim · 2 B` pro Token (fp16), gelesen aus der `config.json` des Modells; Hybrid-Modelle zählen nur ihre Full-Attention-Layer zuzüglich eines festen rekurrenten Zustands, Sliding-Window-Layer stoppen am Fenster, `--kv-bits 4|8` speichert `bits + 0.5` Bits pro Element (`--kv-bits` gilt nicht für Sliding-Window-Layer). Qwen3-4B: 144 KiB pro Token, 4.5 GiB bei 32k Tokens; Ornith-1.5-9B (8 von 32 Layern Full Attention): 32 KiB pro Token, 1.05 GiB bei 32k. `serve --dry-run`, `models check`, `fit` (dessen Quantisierungsraster und Decode-Geschwindigkeitsschätzung — die RAM-Spalte und die Fit-Stufe zählen weiterhin nur Gewichte) sowie das speicherabgeleitete Token-Budget verwenden diesen Wert (cap = min(angeforderte max_tokens, --max-tokens, KV-Platz − Prompt-Tokens), wobei der KV-Platz das von Metal empfohlene Working Set minus geladener Gewichte und Prompt-Cache-Budget ist und der Prompt mit dem Tokenizer des Modells gezählt wird; ein Prompt, der gar nicht passt, wird vor der Generierung mit 413 abgelehnt); ohne heruntergeladene Konfiguration fallen sie auf eine Parameterzahl-Schätzung zurück, die bei Dense-7B–32B-Modellen genau ist, bei kleinen, hybriden und MoE-Modellen jedoch um ein Mehrfaches danebenliegt. Schätzungen, keine Messungen: Der rekurrente Zustand von Mamba-Layern (Jamba, NemotronH) wird nicht mitgezählt, sondern nur deren Attention-Layer; MLA-Modelle (DeepSeek V2/V3) werden als die dekomprimierten K und V pro Head gezählt, die MLX cached, nicht als der komprimierte Latent.

## Modelle herunterladen (mlx-community, huihui-ai, Ex0bit)

Basalt kann Modelle jedes Autors und jeder Organisation auf Hugging Face durchsuchen, prüfen und herunterladen:

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
