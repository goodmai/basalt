# Modelos

> Languages: [English](../../models.md) · [Русский](../ru/models.md) · [中文](../zh/models.md) · **Español** · [Français](../fr/models.md) · [日本語](../ja/models.md) · [Deutsch](../de/models.md)

## Modelos verificados

Los modelos se descargan desde HuggingFace y se guardan en caché en `~/.cache/huggingface/hub/`. Hay más de 42 checkpoints verificados de chat, MoE, QTIP, PV-Tuned, GSQ-RCO, OCR, TTS neuronal, embeddings, rerank y decisión System One publicados con botones **Run with Basalt** de un clic ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en) → `basalt://run?model=...`) en la [colección de Hugging Face de `goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc).

```bash
basalt models download goodmai/gemma-4-12B-it-qat-4bit
basalt models download goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native
```

### Medidos para 0.12

M4 Pro, 24 GB, compilación release (`620641c` / `82a99d4`), noviembre de 2026 (`--reasoning-effort none --kv-bits 4`).

| Modelo | Rol / Arquitectura | Memoria activa | Decodificación tok/s | Escalera / Notas |
|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` | 31B híbrido denso (attn/embed 4-bit + MLP PV-MACaQ afín 3-bit) | 15.1 GB (18.09 GB pico) | 11.6–13.5 (p50 `14.07`, p95 `14.42`) | **3/3 calentamiento, 4/5 peldaños + Lunar 4/6**, `probe_ppl = 2.39`, 50.6% de aceptación de lookup; cabe en Mac de 24 GB donde 4-bit da OOM (`--prompt-cache-mb 0 --wired-limit`) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` | 27B híbrido denso (48 attn lineales + 16 completas, 496 módulos MACaQ 4-bit g64) | 15.3 GB (17.45 GB pico) | 9.4–11.4 (p50 `11.59`, p95 `11.86`) | **3/3 calentamiento, 4/5 peldaños**, `probe_ppl = 1.23`, `SQNR 20.61 → 21.59 dB` (`MSE −19.5%`), deriva máx. en 64 capas `5.53%` (`#630`) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (desde `Qwen3.5-9B-8bit`) | 9B híbrido (24 attn lineales + 8 completas, 200 módulos MACaQ 4-bit g64) | 6.68 GB (−33% frente a 8-bit) | 35.6–41.6 (p50 `43.22`, +52% frente a 8-bit) | **3/3 calentamiento, 4/5 peldaños** (100% paridad con ref. 8-bit `24.7–27.3 tok/s`), `valKL = 0.0044`, `probe_ppl = 1.28`, deriva máx. por capa `3.37%` |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` | 26B MoE (`A4B`, 128 expertos, top-8, MACaQ afín g32 + MLP compartido 8-bit) | 16.75 GB | 49.2–54.8 (+82% por switch WHT de `#628`) | **3/3 calentamiento, 3/5 peldaños** (`biquad-complex`, `fourier`, `pump`); aceleración de kernel `3.41×` (`2.06 → 0.60 ms`, `#628`) |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` | 35B MoE (`A3B`, 120 capas `MACaQSwitchLinear`) | 14.64 GB | 64.0–74.7 (68.7 en 8k) | **3/3 calentamiento, 1/5 peldaños**, `probe_ppl = 1.46`, switch WHT de una pasada (`#628`) |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` | 26B MoE (`A4B`, QAT 4-bit) | 14.92 GB | 56.0–63.4 | **3/3 calentamiento, 4/5 peldaños** (`pump` ✅, `lunar 5/6`) |
| `goodmai/gemma-4-12B-it-qat-4bit` | 12B denso QAT 4-bit | 10.51 GB | 18.8–22.2 | **3/3 calentamiento, 4/5 peldaños** (`lunar` ✅ 6/6) |
| Vídeo / Voz / Música (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`, `LFM2.5-Audio`, `FLUX.3`, `SeedVR2`, `ACE-Step 1.5`) | Video Studio por fases para 24 GB UMA, audio dúplex Mimi/SNAC e imagen/música en 4 etapas | ≤ 18 GB por fases | 3D-VAE + kernel Metal STA | Verificado en `#542`, `#604`, `#605`, `#608`–`#611` |

### Medidos para 0.11

M4 Pro, 24 GB, compilación release, noviembre de 2026.

| Modelo | Rol / Arquitectura | Memoria activa | Rendimiento / Latencia | Notas |
|---|---|---|---|---|
| `Recursive/Flint-40B-A2.5B` (`flint-oQ3e`) | 40B-A2.5B MoE híbrido deslizante/completo (`512` enrutados, top-10 + 1 compartido) | 17.61 GB | 67.02 tok/s decodificación · 273.5 tok/s prefill | cargador nativo `oQ3e` (`groupSize = 32`) (#518, #522), 0 swap en 24 GB |
| `Zyphra/ZONOS3-Mini-hf` | 1.6B Llama + TTS DAC multicabezal FSQ de 32 libros de códigos (`44.1 kHz`) | 2.24 GB | RTF 0.228–0.263 (3.8–4.4× TR) | 43 idiomas, 27 voces, `embedding` multilingüe + CFG `1.5` (#525, #527) |
| `hexgrad/Kokoro-82M` | TTS neuronal de 82M (ANE + MLX) | ~0.35 GB | TTFA por debajo de 80 ms | etapa de respaldo por streaming de oraciones en `pipeline.speak` (#493, #513) |
| `Tongyi-MAI/Z-Image-Turbo` | síntesis de imagen DiT | ~6.4 GB | 4–8 pasos (`1024×1024`) | `pipeline.imagine` con TeaCache y codificador de texto compartido (#517, #519) |
| `Ricursive/Ripple-65B-4bit` | 65B MoE disperso con atención con compuertas (`128` expertos, top-8) | ~35 GB (residente en 64 GB+, o streaming desde SSD) | arquitectura verificada | NoPE / RoPE parcial (#520, #522) |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | 30B-A3B Mamba-2 + MoE de enrutamiento latente (`lr_moe`) | ~17 GB (4-bit) | arquitectura verificada | GQA intercalado + Mamba-2 (#521, #522) |
| `meta-models/Muse-Glimmer-30B` (`mlx-community/Muse-Glimmer-30B-4bit`) | 30B híbrido SWA (`2048`) + atención completa NoPE + VLM ViT-G/14 | ~18.1 GB (4-bit) · ~19.8 GB (`oQ4e`) | arquitectura y fragmentos HF verificados | razonamiento Onyx `to=self` + llamadas a herramientas ATEM + importación de normas descentradas GGUF (#533) |

### Medidos para 0.8

M4 Pro, 24 GB, con adaptador de 68 W, compilación release, decodificación greedy, noviembre de 2026.

| Modelo | Parámetros | Memoria activa | Decodificación, tok/s | Notas |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | ~2.2 GB | 78–89 | denso; se aplican prompt lookup y todos los modos KV |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B ternario | 7.4 GB | 20.9–21.5 estándar · 27.1 con `--mtp` en código · 31–38 con `--speculative-lookup` en código | híbrido (Gated DeltaNet); replicado en `goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | ~11.8 GB | 15–19 | híbrido; verificado con golden; replicado en `goodmai/Qwen3.8-27B-3bit-MLX-TextOnly` |
| `goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native` | 27B i-quant | ~12.4 GB | 5.2 | importación nativa GSQ-RCO `IQ3_S` (#433), WikiText-2 PPL 4.169, 300/300 System One |
| `goodmai/Qwen3.6-35B-A3B-8bit` | 35B MoE (37.7 GB) | ~13.9 GB | 8.77 | `--experimental-expert-streaming` (#478), 0 swap en 24 GB mediante AIO por lotes + `LIO_NOWAIT` |
| `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly` | 27B | no medido | 18–22 | entra en bucle en algunos prompts (3 de 8); `RepetitionGuard` los detiene |

| Modelo de embedding / rerank | Tarea | Medido |
|---|---|---|
| `mlx-community/bge-small-en-v1.5-bf16` | embeddings (384-d) | 229–299 textos/s, lote de 32 × ~240 tokens |
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | embeddings (1024-d) | 20 textos/s, mismo lote |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | rerank | pares conocidos ordenados correctamente |
| `jinaai/jina-reranker-v3-mlx` | rerank | pares conocidos ordenados correctamente |

### Informes anteriores (no vueltos a medir para 0.8)

| Modelo | Parámetros | RAM | Estado en Mac de 24 GB |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 GB | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 GB | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 GB | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 GB | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 GB | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 GB | ✅ ~12 TPS (Abliterado) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 GB | ✅ ~14 TPS (Cuantización Dinámica) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 GB | ✅ ~8 TPS (NVFP4) |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 GB | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 GB | ✅ ~48 TPS (razonamiento — ver nota) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ sin probar por encima de 24 GB |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 GB | ✅ ~76 TPS (razona siempre) |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 GB | ✅ ~40 TPS (razona siempre) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (Base BF16) | 27B | ~54 GB | ❌ Necesita 64 GB+ (Usa la versión MLX 4-bit en 24 GB) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ Salida ininteligible |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ Salida ininteligible |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM (Necesita 32 GB+) |

> **Nota sobre Macs de 24 GB de RAM:**
> - Los modelos de 27B/35B sin cuantizar (~54 GB de pesos) superan la memoria física. Usa pesos cuantizados MLX 4-bit / Dynamic Quant (DQ) (por ejemplo, `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`, `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`) que caben cómodamente en ~15 GB de RAM.
> - Los modelos > 10 GB limitan el tamaño de la caché KV en contextos multiturno intensos. La caché KV ocupa `2 · layers · kv_heads · head_dim · 2 B` por token (fp16), leído del `config.json` del modelo; los modelos híbridos cuentan solo sus capas de atención completa más un estado recurrente fijo, las capas de ventana deslizante se detienen en la ventana, `--kv-bits 4|8` almacena `bits + 0.5` bits por elemento (`--kv-bits` no se aplica a capas de ventana deslizante). Qwen3-4B: 144 KiB por token, 4.5 GiB a 32k tokens; Ornith-1.5-9B (8 de 32 capas con atención completa): 32 KiB por token, 1.05 GiB a 32k. `serve --dry-run`, `models check`, `fit` (su cuadrícula de cuantización y estimación de velocidad de decodificación — la columna RAM y el nivel de ajuste siguen contando solo los pesos) y el presupuesto de tokens derivado de la memoria usan esta cifra (cap = min(requested max_tokens, --max-tokens, KV room − prompt tokens), donde el espacio KV es el conjunto de trabajo recomendado de Metal menos los pesos cargados y el presupuesto de caché de prompts, y el prompt se cuenta con el tokenizador del modelo; un prompt que no cabe en absoluto se rechaza con 413 antes de generar); sin un config descargado, recurren a una estimación por número de parámetros que se aproxima para modelos densos de 7B–32B, pero difiere varias veces para modelos pequeños, híbridos y MoE. Son estimaciones, no mediciones: no se cuenta el estado recurrente de las capas Mamba (Jamba, NemotronH), solo sus capas de atención; los modelos MLA (DeepSeek V2/V3) se cuentan como las K y V por cabezal descomprimidas que MLX guarda en caché, no como el latente comprimido.

## Descarga de modelos (mlx-community, huihui-ai, Ex0bit)

Basalt puede explorar, inspeccionar y descargar modelos de cualquier creador u organización en Hugging Face:

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
