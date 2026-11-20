# Модели

> Languages: [English](../../models.md) · **Русский** · [中文](../zh/models.md) · [Español](../es/models.md) · [Français](../fr/models.md) · [日本語](../ja/models.md) · [Deutsch](../de/models.md)

## Проверенные модели

Модели скачиваются с HuggingFace и кэшируются в `~/.cache/huggingface/hub/`. Более 42 проверенных чекпоинтов для чата, MoE, QTIP, PV-Tuned, GSQ-RCO, OCR, нейросетевого TTS, эмбеддингов, переранжирования и диспетчеров System One опубликованы с кнопками запуска в один клик **Run with Basalt** ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en) → `basalt://run?model=...`) в [коллекции `goodmai` на Hugging Face](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc).

```bash
basalt models download goodmai/gemma-4-12B-it-qat-4bit
basalt models download goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native
```

### Замеры для 0.12

M4 Pro, 24 ГБ, релизная сборка (`620641c` / `82a99d4`), ноябрь 2026 (`--reasoning-effort none --kv-bits 4`).

| Модель | Роль / Архитектура | Активная память | Декод, tok/s | Ladder / Примечания |
|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` | 31B плотный гибрид (4-bit attn/embed + 3-bit affine PV-MACaQ MLP) | 15.1 ГБ (пик 18.09 ГБ) | 11.6–13.5 (p50 `14.07`, p95 `14.42`) | **3/3 разминка, 4/5 ступеней + Lunar 4/6**, `probe_ppl = 2.39`, 50.6% принятия lookup; помещается на 24 ГБ Mac, где 4-bit уходит в OOM (`--prompt-cache-mb 0 --wired-limit`) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` | 27B плотный гибрид (48 linear + 16 full attn, 496 модулей MACaQ 4-bit g64) | 15.3 ГБ (пик 17.45 ГБ) | 9.4–11.4 (p50 `11.59`, p95 `11.86`) | **3/3 разминка, 4/5 ступеней**, `probe_ppl = 1.23`, `SQNR 20.61 → 21.59 dB` (`MSE −19.5%`), макс. дрейф по 64 слоям `5.53%` (`#630`) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (из `Qwen3.5-9B-8bit`) | 9B гибрид (24 linear + 8 full attn, 200 модулей MACaQ 4-bit g64) | 6.68 ГБ (−33% от 8-bit) | 35.6–41.6 (p50 `43.22`, +52% к 8-bit) | **3/3 разминка, 4/5 ступеней** (100% паритет с 8-bit референсом `24.7–27.3 tok/s`), `valKL = 0.0044`, `probe_ppl = 1.28`, макс. дрейф слоя `3.37%` |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` | 26B MoE (`A4B`, 128 экспертов, top-8, g32 affine MACaQ + 8-bit общий MLP) | 16.75 ГБ | 49.2–54.8 (+82% от switch WHT `#628`) | **3/3 разминка, 3/5 ступеней** (`biquad-complex`, `fourier`, `pump`); ускорение ядра `3.41×` (`2.06 → 0.60 ms`, `#628`) |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` | 35B MoE (`A3B`, 120 слоёв `MACaQSwitchLinear`) | 14.64 ГБ | 64.0–74.7 (68.7 на 8k) | **3/3 разминка, 1/5 ступеней**, `probe_ppl = 1.46`, однопроходный switch WHT (`#628`) |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` | 26B MoE (`A4B`, QAT 4-bit) | 14.92 ГБ | 56.0–63.4 | **3/3 разминка, 4/5 ступеней** (`pump` ✅, `lunar 5/6`) |
| `goodmai/gemma-4-12B-it-qat-4bit` | 12B плотная QAT 4-bit | 10.51 ГБ | 18.8–22.2 | **3/3 разминка, 4/5 ступеней** (`lunar` ✅ 6/6) |
| Видео / Речь / Музыка (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`, `LFM2.5-Audio`, `FLUX.3`, `SeedVR2`, `ACE-Step 1.5`) | Пофазная 24 ГБ UMA Video Studio, дуплексный аудио Mimi/SNAC и 4-этапная генерация изображений/музыки | ≤ 18 ГБ по фазам | 3D-VAE + STA Metal-ядро | Проверено в `#542`, `#604`, `#605`, `#608`–`#611` |

### Замеры для 0.11

M4 Pro, 24 ГБ, релизная сборка, ноябрь 2026.

| Модель | Роль / Архитектура | Активная память | Пропускная способность / Задержка | Примечания |
|---|---|---|---|---|
| `Recursive/Flint-40B-A2.5B` (`flint-oQ3e`) | 40B-A2.5B гибрид sliding/full MoE (`512` маршрутизируемых, top-10 + 1 общий) | 17.61 ГБ | 67.02 tok/s декод · 273.5 tok/s префилл | нативный загрузчик `oQ3e` (`groupSize = 32`) (#518, #522), 0 swap на 24 ГБ |
| `Zyphra/ZONOS3-Mini-hf` | 1.6B Llama + 32-кодовый FSQ multi-head DAC TTS (`44.1 kHz`) | 2.24 ГБ | RTF 0.228–0.263 (3.8–4.4× быстрее реального времени) | 43 языка, 27 голосов, кросс-языковой `embedding` + CFG `1.5` (#525, #527) |
| `hexgrad/Kokoro-82M` | 82M нейросетевой TTS (ANE + MLX) | ~0.35 ГБ | < 80 мс TTFA | попредложенный стриминговый фолбэк в `pipeline.speak` (#493, #513) |
| `Tongyi-MAI/Z-Image-Turbo` | DiT-синтез изображений | ~6.4 ГБ | 4–8 шагов (`1024×1024`) | `pipeline.imagine` с TeaCache и общим текстовым энкодером (#517, #519) |
| `Ricursive/Ripple-65B-4bit` | 65B разреженный MoE с gated-attention (`128` экспертов, top-8) | ~35 ГБ (резидентно 64 ГБ+ или стриминг с SSD) | архитектура проверена | NoPE / частичный RoPE (#520, #522) |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | 30B-A3B Mamba-2 + Latently-Routed MoE (`lr_moe`) | ~17 ГБ (4-bit) | архитектура проверена | чередование GQA + Mamba-2 (#521, #522) |
| `meta-models/Muse-Glimmer-30B` (`mlx-community/Muse-Glimmer-30B-4bit`) | 30B гибрид SWA (`2048`) + NoPE full attention + ViT-G/14 VLM | ~18.1 ГБ (4-bit) · ~19.8 ГБ (`oQ4e`) | архитектура и HF-шарды проверены | рассуждения Onyx `to=self` + вызовы инструментов ATEM + импорт децентрированных норм GGUF (#533) |

### Замеры для 0.8

M4 Pro, 24 ГБ, от адаптера 68 Вт, релизная сборка, жадное декодирование, ноябрь 2026.

| Модель | Параметры | Активная память | Декод, tok/s | Примечания |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | ~2.2 ГБ | 78–89 | плотная; работают prompt lookup и все режимы KV |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B тернарная | 7.4 ГБ | 20.9–21.5 обычный · 27.1 с `--mtp` на коде · 31–38 со `--speculative-lookup` на коде | гибридная (Gated DeltaNet); зеркало в `goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | ~11.8 ГБ | 15–19 | гибридная; проверена по golden-тестам; зеркало в `goodmai/Qwen3.8-27B-3bit-MLX-TextOnly` |
| `goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native` | 27B i-quant | ~12.4 ГБ | 5.2 | нативный импорт GSQ-RCO `IQ3_S` (#433), WikiText-2 PPL 4.169, 300/300 System One |
| `goodmai/Qwen3.6-35B-A3B-8bit` | 35B MoE (37.7 ГБ) | ~13.9 ГБ | 8.77 | `--experimental-expert-streaming` (#478), 0 swap на 24 ГБ через пакетный AIO + `LIO_NOWAIT` |
| `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly` | 27B | не измерялась | 18–22 | зацикливается на некоторых промптах (3 из 8); `RepetitionGuard` останавливает циклы |

| Модель эмбеддингов / переранжирования | Задача | Измерено |
|---|---|---|
| `mlx-community/bge-small-en-v1.5-bf16` | эмбеддинги (384-d) | 229–299 текстов/с, батч 32 × ~240 токенов |
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | эмбеддинги (1024-d) | 20 текстов/с на том же батче |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | переранжирование | известные пары ранжируются верно |
| `jinaai/jina-reranker-v3-mlx` | переранжирование | известные пары ранжируются верно |

### Более ранние отчёты (не перемерялись для 0.8)

| Модель | Параметры | RAM | Статус на Mac с 24 ГБ |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 ГБ | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 ГБ | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 ГБ | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 ГБ | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 ГБ | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 ГБ | ✅ ~12 TPS (Abliterated) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 ГБ | ✅ ~14 TPS (Dynamic Quant) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 ГБ | ✅ ~8 TPS (NVFP4) |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 ГБ | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 ГБ | ✅ ~48 TPS (рассуждающая — см. примечание) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 ГБ | ✅ не тестировалась сверх 24 ГБ |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 ГБ | ✅ ~76 TPS (всегда рассуждает) |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 ГБ | ✅ ~40 TPS (всегда рассуждает) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (Base BF16) | 27B | ~54 ГБ | ❌ Требует 64 ГБ+ (на 24 ГБ используйте 4-bit MLX-версию) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 ГБ | ❌ Бессмысленный вывод |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 ГБ | ❌ Бессмысленный вывод |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 ГБ | ❌ OOM (требует 32 ГБ+) |

> **Примечание о Mac с 24 ГБ RAM:**
> - Неквантованные модели 27B/35B (~54 ГБ весов) превышают физическую память. Используйте веса, квантованные в MLX 4-bit / Dynamic Quant (DQ) (например, `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`, `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`), которые спокойно помещаются примерно в 15 ГБ RAM.
> - Модели > 10 ГБ ограничивают размер KV-кэша при тяжёлых многоходовых диалогах. KV-кэш занимает `2 · layers · kv_heads · head_dim · 2 B` на токен (fp16), считываемых из `config.json` модели; гибридные модели учитывают только слои с полным вниманием плюс фиксированное рекуррентное состояние, слои со скользящим окном останавливаются на размере окна, а `--kv-bits 4|8` хранит `bits + 0.5` бита на элемент (`--kv-bits` не применяется к слоям со скользящим окном). Qwen3-4B: 144 КиБ на токен, 4.5 ГиБ на 32k токенах; Ornith-1.5-9B (8 из 32 слоёв с полным вниманием): 32 КиБ на токен, 1.05 ГиБ на 32k. Команды `serve --dry-run`, `models check`, `fit` (её сетка квантования и оценка скорости декодирования — столбец RAM и уровень совместимости по-прежнему считают только веса) и бюджет токенов по памяти используют это значение (`cap = min(requested max_tokens, --max-tokens, KV room − prompt tokens)`, где `KV room` — рекомендуемый рабочий набор Metal минус загруженные веса и бюджет кэша промптов, а промпт подсчитывается токенизатором модели; промпт, который не помещается вовсе, отклоняется с кодом 413 ещё до генерации); без скачанного конфига они откатываются к оценке по числу параметров, которая близка к истине для плотных моделей 7B–32B, но ошибается в несколько раз на малых, гибридных и MoE-моделях. Это оценки, а не замеры: рекуррентное состояние слоёв Mamba (Jamba, NemotronH) не учитывается — считаются только слои внимания; модели MLA (DeepSeek V2/V3) оцениваются по развёрнутым K и V на голову, которые кэширует MLX, а не по сжатому латенту.

## Скачивание моделей (mlx-community, huihui-ai, Ex0bit)

Basalt умеет искать, просматривать и скачивать модели любого автора или организации на Hugging Face:

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
