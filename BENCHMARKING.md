# Отчет по бенчмаркингу Gem

Тестирование проводилось на различных моделях семейства Gemma через REST API.

## 1. Легкая модель (Gemma 4 2B IT 4-bit)
Модель: `mlx-community/gemma-4-e2b-it-4bit`.

### 1.1. Арифметика
**Задача:** Вычислить 256 * 144.
- **Результат:** `36864` (Верно)
- **TTFT:** 0.265 s | **TPS:** 56.10
- **Статус:** ✅ Успешно

### 1.2. Алгебра
**Задача:** Решить x^2 - 5x + 6 = 0.
- **Результат:** `2, 3` (Верно)
- **TTFT:** 0.284 s | **TPS:** 62.05
- **Статус:** ✅ Успешно

### 1.3. Перевод
**Задача:** "The quick brown fox jumps over the lazy dog."
- **Результат:** "Быстрая коричневая лиса прыгает через ленивую собаку." (Верно)
- **TTFT:** 0.053 s | **TPS:** 62.40
- **Статус:** ✅ Успешно

---

## 2. Средняя модель (Gemma 2 9B IT 4-bit)
Модель: `mlx-community/gemma-2-9b-it-4bit`.

### 2.1. Арифметика
**Задача:** Вычислить 123 * 456.
- **Результат:** `56088` (Верно)
- **TTFT:** 1.778 s (холодный старт) | **TPS:** 40.97
- **Статус:** ✅ Успешно

### 2.2. Алгебра
**Задача:** Решить 2x + 10 = 20.
- **Результат:** `5` (Верно)
- **TTFT:** 0.148 s | **TPS:** 40.41
- **Статус:** ✅ Успешно

### 2.3. Перевод
**Задача:** "I love exploring new technologies."
- **Результат:** "Мне нравится исследовать новые технологии." (Верно)
- **TTFT:** 0.146 s | **TPS:** 41.40
- **Статус:** ✅ Успешно

---

## 3. Тяжелая модель (Gemma 4 26B MoE)
- **Статус:** ❌ Ошибка выполнения (Runtime Error)
- **Причина:** Несовместимость текущего ядра генерации с архитектурой Mixture of Experts.
- **Память:** Занимает ~15 ГБ RAM.

---

## Сводная таблица (Средние значения)
| Модель | Avg TTFT | Avg TPS | RAM Usage |
|---|---|---|---|
| **Gemma-4-2B** | 0.165 s | 60.40 | ~2.7 GB |
| **Gemma-2-9B** | 0.147 s* | 40.92 | ~5.3 GB |

*\* Без учета холодного старта.*

---

## Speculative decoding

Prompt lookup (`--speculative-lookup`, #146/#138/#25) drafts up to 16 tokens per verify round by copying the continuation of the longest matching n-gram from the context. Numbers are ratios between neighbouring cells of the same pass, never absolute tok/s: each model runs every cell once in order and once in reverse. Release build, 3 runs × 300 tokens per cell, `--seed 1`, engine default top-p 0.9, Apple M4 Pro 24 GB. The code workload is a rename-and-reprint of a Swift file (~1.1k-token prompt), prose is a short story (`benchmarks/speculative-lookup/`).

```
./scripts/speculative_lookup_bench.swift
./scripts/speculative_lookup_bench.swift --variant k8=/path/to/k8/PerformanceBenchmark
```

The runner builds and measures the checkout it lives in. `--variant` adds cells for another `PerformanceBenchmark` binary, labelled by what that binary is: the second command, run from this checkout (draft limit 16), compares a draft-limit-8 build against it, and its column reads k8 / on. The k16 / k8 table below came from the opposite setup, a k8 checkout with a k16 variant.

Lookup on / off at commit `7e2a0ea` (draft limit 16). `*` marks a ratio with an invalid cell (σ above 3% of the mean). The two passes disagree by more than the gate on the code rows: the host's throughput swung between neighbouring cells in this run, so read the code speedup as a range.

| model | workload | T | on / off (pass 1, pass 2) | α | same text as off |
|---|---|---|---|---|---|
| Qwen3-4B-4bit | code | 0 | 2.41×\*, 3.78× | 0.986 | yes |
| Qwen3-4B-4bit | code | 0.7 | 3.27×\*, 3.53× | 0.986 | yes |
| Qwen3-4B-4bit | prose | 0 | 1.04×, 0.96× | 0 (0 of 48) | no |
| Qwen3-4B-4bit | prose | 0.7 | 1.00×, 0.98× | no drafts | yes |
| Llama-3.1-8B-Instruct-4bit | code | 0 | 3.65×, 2.78×\* | 0.958 | yes |
| Llama-3.1-8B-Instruct-4bit | code | 0.7 | 3.53×, 4.23×\* | 0.958 | yes |
| Llama-3.1-8B-Instruct-4bit | prose | 0 | 1.01×, 0.96×\* | 0 (0 of 96) | yes |
| Llama-3.1-8B-Instruct-4bit | prose | 0.7 | 1.01×, 0.99× | 0.429 (9 of 21) | yes |

Draft limit 16 against 8, both with lookup on, at commit `f9fa34b`:

| model | workload | T | k16 / k8 (pass 1, pass 2) |
|---|---|---|---|
| Qwen3-4B-4bit | code | 0 | 1.47×, 1.47× |
| Qwen3-4B-4bit | code | 0.7 | 1.40×, 1.71×\* |
| Qwen3-4B-4bit | prose | 0 / 0.7 | 1.00×, 1.00× |
| Llama-3.1-8B-Instruct-4bit | code | 0 | 1.42×\*, 1.51×\* |
| Llama-3.1-8B-Instruct-4bit | code | 0.7 | 1.45×, 1.48× |
| Llama-3.1-8B-Instruct-4bit | prose | 0 / 0.7 | 1.00×, 1.00× |

Token Recycling (`--speculative-recycling`, #38) adds a `recycling` mode to the runner: lookup on, plus chains of recycled argmax successors when no n-gram matches. The reasoning workload is a step-by-step arithmetic word problem. Commit `fb4d56e`, same settings as above; the log is `logs/speculative_lookup_bench-2026-10-29T120031Z.log`.

| model | workload | T | on / off (pass 1, pass 2) | recycling / on (pass 1, pass 2) | α on | α recycling |
|---|---|---|---|---|---|---|
| Qwen3-4B-4bit | code | 0 | 2.51×\*, 3.85× | 0.82×\*, 1.01× | 0.986 | 0.986 |
| Qwen3-4B-4bit | code | 0.7 | 3.45×\*, 3.04×\* | 1.11×\*, 1.20×\* | 0.986 | 0.986 |
| Qwen3-4B-4bit | prose | 0 | 1.03×\*, 1.04×\* | 1.01×, 0.99× | 0 (0 of 48) | 0 (0 of 48) |
| Qwen3-4B-4bit | prose | 0.7 | 1.00×, 1.00× | 0.99×, 0.97× | no drafts | 0 (0 of 3) |
| Qwen3-4B-4bit | reasoning | 0 | 0.85×, 0.80× | 1.15×, 1.02× | 0.055 | 0.070 |
| Qwen3-4B-4bit | reasoning | 0.7 | 0.86×, 0.82× | 1.04×, 1.04× | 0.096 | 0.148 |

Llama-3.1-8B-Instruct-4bit at the same commit (`logs/speculative_lookup_bench-2026-10-29T131707Z.log`). This run overlapped other load on the host: plain decoding dropped to 10–25 tok/s in some cells and a CI job ran once. Most ratios are therefore invalid and only bound the effect. The valid ones are prose 1.01×/1.03× (T 0) and 1.01× (T 0.7, pass 2), and reasoning 1.02× (T 0, pass 1).

| model | workload | T | on / off (pass 1, pass 2) | recycling / on (pass 1, pass 2) | α on | α recycling |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct-4bit | code | 0 | 3.64×\*, 3.70× | 0.98×\*, 1.02×\* | 0.958 | 0.958 |
| Llama-3.1-8B-Instruct-4bit | code | 0.7 | 4.15×\*, 2.64×\* | 1.01×\*, 1.09×\* | 0.958 | 0.958 |
| Llama-3.1-8B-Instruct-4bit | prose | 0 | 1.03×, 1.04×\* | 1.01×, 1.03× | 0 (0 of 96) | 0 (0 of 99) |
| Llama-3.1-8B-Instruct-4bit | prose | 0.7 | 1.02×, 1.02× | 0.92×\*, 1.01× | 0.429 | 0.429 |
| Llama-3.1-8B-Instruct-4bit | reasoning | 0 | 0.84×, 1.10×\* | 1.02×, 1.54×\* | 0.128 | 0.146 |
| Llama-3.1-8B-Instruct-4bit | reasoning | 0.7 | 1.29×\*, 0.49×\* | 0.89×\*, 1.02×\* | 0.053 | 0.232 |

"Same text as off" is what these runs observed, not a guarantee. Lookup spends exactly one sampler draw per emitted token, in plain-decoding order, so a seeded run follows plain decoding as long as the logits match. But a multi-token verify uses different matmul kernels than a one-token step, so on a quantized model a near-tie can flip, at T 0 and at T > 0 alike (#146). That happened on Qwen3-4B's greedy prose. The Qwen3-4B prose row at T 0.7 drafted nothing, so its "yes" shows nothing about lookup.
