<p align="center"><img src="images/logo.svg" alt="Логотип basalt" width="128"></p>

# basalt — ваш Mac теперь AI-лаборатория.

> Запускайте, дообучайте, квантуйте, объединяйте в стеки и разбирайте изнутри открытые LLM, нейросетевые модели речи, видео, OCR и генерации изображений на Apple Silicon. Один бинарный файл на Swift 6.4 говорит на **OpenAI API, Anthropic API и MCP**, а поверх работает нативное приложение для macOS 27 Liquid Glass. Без Python в рантайме. Без облака — пока вы сами его не запросите.

[English](README.md) · **Русский** · [中文](README.zh.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Bahasa Indonesia](README.id.md) · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

📖 **Монография:** [Эволюция архитектур нейронных сетей, 1958–2027](docs/monograph/evo.ru.md), автор Алексей Боклаг, на шести языках ООН ([все](docs/monograph/README.md)) · **задачник:** что из неё уже запускает basalt, чек-лист [#816](https://github.com/goodmai/basalt/issues/816)

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.13.0](https://img.shields.io/badge/release-v0.13.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.13.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2865%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/i18n/ru/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/i18n/ru/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/i18n/ru/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/i18n/ru/benchmarks.md#benchmarks)

[![Quality Metrics](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml)
[![Concurrency Hatches](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/concurrency-hatches.json)](scripts/concurrency-hatches.allow)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/swift-testing.json)](Tests/BasaltTests)
[![Files ≤600 LOC](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/file-size.json)](scripts/check_size_limits.swift)
[![Crash Traps](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/crash-traps.json)](scripts/check_quality_metrics.swift)
[![L10n EN/RU](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/l10n-parity.json)](Localization)
[![Supply Chain](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/supply-chain.json)](.github/workflows/quality-metrics.yml)
[![Golden Logits](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/golden-logits.json)](Tests/BasaltTests/Fixtures/ReferenceLogits)

<p align="center"><img src="images/models-hive.svg" alt="Открытые модели, которые запускает basalt: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Запускает модели с открытыми весами на Apple Silicon. Проверены по эталонным логитам: Gemma 4, Qwen3.8 и LFM2.5 (таблица ниже); измерены и собраны в <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a>: Ornith, Bonsai, Qwen3.6-35B-A3B, DeepSeek-OCR-2, OpenJev и Laya (см. <a href="docs/i18n/ru/models.md#verified-models">таблицу моделей</a>); gpt-oss и Llama совместимы с MLX. Логотипы являются товарными знаками их владельцев и приведены только для указания совместимости. Модели Anthropic имеют закрытые веса, поэтому их нет в этом списке; <i>API</i> Anthropic поддерживается, чтобы Claude Code мог работать с локальной моделью.</sub></p>

**Пять инструментов в одном репозитории:** локальный сервер инференса, с которым уже умеют работать ваши инструменты, дистилляция LoRA и доводка после квантования, аналог docker-compose для мультимодельных систем, живой рентген-микроскоп того, что происходит внутри модели, и нативное приложение для Mac, где всё доступно по одной горячей клавише.

**Проверка по эталонным логитам при каждом изменении** в пути инференса: [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) прогоняет записанные промпты и падает, если сдвигается хотя бы один жадный токен или значение логита. Эталонные логиты по семействам для qwen3, qwen3_5, gemma4, lfm2 и nemotron_h — как плотных, так и MoE — лежат в [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits).

| Модель с golden-проверкой | Класс | Фикстура |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | sliding-window + full attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | гибридное linear/full attention, плотная | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | гибрид conv + attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### Запуск в один клик из [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) (65+ проверенных моделей)

Каждая карточка модели в [`goodmai`](https://huggingface.co/goodmai) содержит кнопку **Run with Basalt** ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en)), которая открывает `Basalt.app` по ссылке `basalt://run?model=...` или выводит точную CLI-команду `basalt serve` / `basalt compose`:

| Модель / Стек | Назначение и объём памяти на Mac с 24 ГБ | Запуск |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT на каждый день для чата и vision (~7.6 ГБ, Ladder 4/5 + Lunar 6/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE (`A4B`), проверена по golden-логитам (14.9 ГБ, 56–63 tok/s, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 31B плотная PV-MACaQ 3.2 bpw для Mac с 24 ГБ (15.1 ГБ, 14.1 tok/s, Ladder 4/5 + Lunar 4/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B плотная PV-MACaQ 4-bit (15.3 ГБ, 11.6 tok/s p50, PPL 1.23, MSE −19.5%, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE (`A3B`) 3-bit MACaQ (14.6 ГБ, 64–75 tok/s с switch WHT из `#628`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw trellis (10.0 ГБ, 16k PPL 4.621) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO нативный i-quant (11.8 ГБ, PPL 4.17, 300/300 System One) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 35B MoE на 37.7 ГБ со стримингом с SSD на 8.77 tok/s, 0 swap | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B Hadamard ternary в 7.4 ГБ, 27.1 tok/s с `--mtp` | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B гибрид conv + attention MoE, проверена по golden-логитам (~12.6 ГБ) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | Нейросетевой TTS 44.1 кГц на 43 языках (2.24 ГБ, RTF 0.23–0.26) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | OCR с vision-энкодером и официальным тайлингом 768 (`/v1/pipeline`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | 9B локальный диспетчер System One (точность 98.9%, −77% токенов LLM) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/i18n/ru/compose.md#bundled-hardware-tiered-profiles-526) | Рекомендуемый Compose-стек для 24 ГБ (Chat + ZONOS3 + Kokoro ANE) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

### Что нового в v0.13.0 ([список изменений](CHANGELOG.md))

- **Inkling 2 Mini.** Зацикленная MoE работает от начала до конца: вызовы инструментов и переиспользование кэша промпта между ходами агента. `--mtp` с DSpark-головой из чекпойнта поднимает декодирование с 28.0 до 49.8 tok/s на коде (M4 Pro, 24 ГБ), а промпт на 37.5K токенов занимает в пике 14.5 ГБ без swap ([#650](https://github.com/goodmai/basalt/issues/650)).
- **NVFP4 из любого источника.** `models convert --to nvfp4` работает с любым чекпойнтом. NVIDIA ModelOpt NVFP4 импортируется без потерь, сам, при `models download` и `serve`. MoE, которой не хватает места под KV, стримит экспертов вместо Metal OOM ([#677](https://github.com/goodmai/basalt/issues/677)).
- **Новые архитектуры и KV.** Qwen3-Coder-Next со стримингом экспертов ([#642](https://github.com/goodmai/basalt/issues/642)), BailingHybrid / Ling-3.0 ([#672](https://github.com/goodmai/basalt/issues/672)) и экспериментальный KV-кэш Gemma 4 с общей строкой K=V, которому нужно в 1.40–1.53 раза меньше памяти на 32K–128K ([#645](https://github.com/goodmai/basalt/issues/645)).
- **Адаптеры и контекст.** PEFT LoRA-адаптеры загружаются из папки или с Hub и вплавляются в экспортируемые чекпойнты ([#675](https://github.com/goodmai/basalt/issues/675)). `--rope-yarn` расширяет контекст Qwen 3.5–3.8 ([#729](https://github.com/goodmai/basalt/issues/729)).
- **Перевод и самопереписывание.** Бенчмарки `basalt:translation` (chrF++, COMET, MetricX-24) с профилями TranslateGemma-4 и экспериментальное самопереписывание на копируемых при записи весах (`basalt selfedit`, [#676](https://github.com/goodmai/basalt/issues/676)).
- **Надёжность.** Совместный поиск дефектов двумя агентами закрыл 64 находки: периметр REST/MCP, облачные бюджеты, оценка, конвертация, кэши и приложение для Mac ([#809](https://github.com/goodmai/basalt/pull/809), [#810](https://github.com/goodmai/basalt/pull/810), [#811](https://github.com/goodmai/basalt/pull/811)).

---

## Инференс: 40B MoE на 67 tok/s, 27B в 7.4 ГБ или 37.7 ГБ MoE без единого байта swap — за привычными вам API

- **Claude Code на собственном GPU в одну строку.** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — ваш настоящий API-ключ даже не задействуется. OpenAI SDK, OpenCode, Continue, Cline и aider подключаются по OpenAI API, а Claude Desktop и Cursor — по MCP (с гранулярным включением и отключением инструментов и групп через `--mcp-enable` / `--mcp-disable`, [#488](https://github.com/goodmai/basalt/pull/488)).
- **Быстро на ноутбуке.** На M4 Pro с 24 ГБ: рекурсивная `Flint-40B-A2.5B` (`oQ3e`) декодирует на 67.02 tok/s в 17.61 ГБ ([#522](https://github.com/goodmai/basalt/pull/522)); `Ornith-1.5-35B-A3B-MACaQ-3bit` декодирует на 64–75 tok/s в 14.6 ГБ за счёт однопроходного ядра `MACaQSwitchLinear` WHT + `gatherQuantizedMM` (ускорение MoE-ядра в `3.41×`, `+82%` end-to-end на Gemma 4 26B-A4B, [#628](https://github.com/goodmai/basalt/pull/628)); `Qwen3.5-9B-PV-MACaQ-4bit` достигает `43.2 tok/s` по p50 (`+52%` скорости, `−33%` RAM по сравнению с 8-bit) при точном паритете `4/5` на Ladder; `Gemma-4-31B-it-PV-MACaQ-3.2bpw` умещает плотную 31B-модель в `15.1 ГБ` (`14.1 tok/s`, Ladder `4/5` + Lunar `4/6`) на Mac с 24 ГБ, где 4-bit падает с OOM; Qwen3-4B декодирует на 78–89 tok/s; Ternary Bonsai 2 27B помещается в 7.4 ГБ и выдаёт 20.9–21.5 tok/s, 27.1 со своей MTP-головой и 31–38 с prompt lookup на коде ([#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)); `.gguf`-чекпоинты GSQ-RCO / `i-quant` выполняются напрямую на специализированных Metal-ядрах ([#433](https://github.com/goodmai/basalt/pull/433), [#515](https://github.com/goodmai/basalt/pull/515)).
- **Модели больше объёма RAM с нулевым swap.** `Qwen3.6-35B-A3B-8bit` (37.7 ГБ) декодирует на 8.77 tok/s на Mac с 24 ГБ и 0 swap через пакетный POSIX AIO и упреждающую подкачку экспертов следующего слоя с SSD через `LIO_NOWAIT` (`--expert-streaming`, [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478), [#516](https://github.com/goodmai/basalt/pull/516)), в связке с UMA-осведомлённой иерархией KV-кэша L0/L1/L2 (`TieredKVHierarchyCore`) и адаптивным гетерогенным планированием ([#627](https://github.com/goodmai/basalt/pull/627)).
- **Спекулятивное декодирование без потери точности.** Prompt lookup даёт ускорение 1.84–2.66× на правках кода (Bonsai 2 27B, [#239](https://github.com/goodmai/basalt/pull/239)) и 3.45–3.62× (Gemma 4 12B, [#265](https://github.com/goodmai/basalt/pull/265)) с неизменным жадным выводом; `SpecTr` OT + `Residual Race`, `EAGLE-3`, `SmallM` (`M=2..16`), `Fast-dLLM` + `DFlash` и `DiffuCD` расширяют декодирование без потерь за счёт мульти-черновиков и блочной диффузии ([#593](https://github.com/goodmai/basalt/pull/593), [#603](https://github.com/goodmai/basalt/pull/603)).
- **Следующий ход стоит только новых токенов.** Префикс-дерево кэша промптов работает даже для гибридных моделей: на Ornith-1.5-9B с агентным промптом на 12.6k токенов второй ход занимает около 1 с вместо 36 с ([#82](https://github.com/goodmai/basalt/pull/82)).
- **Длинный контекст в меньшей памяти.** Квантование KV с точными зонами (`4|8|k8v4`, а также 2D-балансировка `KVarN`, несмещённый 1-bit `QJL`, `ChanMix`/`ThinKV`, кросс-языковой перенос `HeadKV` RBO, центроидный `RQ-KV`, `turbo3|turbo4|turbo8v4` и парные RoPE-ключи `polar4`, [#472](https://github.com/goodmai/basalt/pull/472), [#590](https://github.com/goodmai/basalt/pull/590), [#602](https://github.com/goodmai/basalt/pull/602)): пик памяти на 16k-промпте — 1.03 ГБ вместо 3.11 ГБ на Qwen3-4B ([#159](https://github.com/goodmai/basalt/pull/159)).
- **Видео, нейросетевой TTS, полнодуплексная речь, музыка и синтез 2K-изображений на MLX.** Пофазная 24 ГБ UMA Video Studio запускает `LTX-2.5 AV-DiT`, `Kandinsky 6.0 CrossDiT`, `LingBot-Video MoE` и `Wan 2.2` с каузальным 3D-VAE, brick-tiled STA Metal-ядрами и мультиплексированием `.mp4` через `AVAssetWriter` ([#608](https://github.com/goodmai/basalt/pull/608)–[#611](https://github.com/goodmai/basalt/pull/611)); дуплексные кодеки Mimi/SNAC `LFM2.5-Audio`, `SmartTurn` + `Sortformer`, `MossFormer2`, `Soprano-80M`, `Qwen3ForcedAligner`, `Zyphra/ZONOS3-Mini-hf` (`44.1 kHz`, 43 языка, 2.24 ГБ, RTF 0.228–0.263, [#527](https://github.com/goodmai/basalt/pull/527)) и `Kokoro-82M` обеспечивают `/v1/audio/*` ([#604](https://github.com/goodmai/basalt/pull/604)); 4-этапная композиция `FLUX.3`, апскейлинг `SeedVR2`, музыка `ACE-Step 1.5`, `Z-Image-Turbo`, `Iris-3B` и `SANA-1.5` работают за `/v1/images/*` и `/v1/music/*` ([#542](https://github.com/goodmai/basalt/pull/542), [#605](https://github.com/goodmai/basalt/pull/605)).
- **Полный локальный стек + Swift SDK.** Вызов инструментов, вывод по JSON-schema, regex и грамматике (XGrammar), Meta `Muse-Glimmer-30B` ([#534](https://github.com/goodmai/basalt/pull/534)), `Ripple-65B`, `Nemotron-3.5-Lightning` ([#522](https://github.com/goodmai/basalt/pull/522)), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)), `/v1/embeddings`, `/v1/rerank`, Swift SDK `BasaltFoundationModels` ([#606](https://github.com/goodmai/basalt/pull/606)), загрузка через Xet в `swift-huggingface` ([#587](https://github.com/goodmai/basalt/pull/587)), выбор адаптеров LoRA/PHM/OFT на уровне запроса и упорядоченные уровни резервного фолбэка LAN-перед-облаком ([#480](https://github.com/goodmai/basalt/pull/480)).
- **PV-MACaQ 3-bit: 27B помещается и летает на 24 ГБ.** `Qwen3.8-27B-PV-MACaQ-3bit` проходит Official Ladder 5/5 (включая `lunar` 6/6) в 11.5–12.4 ГБ на 15.6–16.7 tok/s на M4 Pro против 4/5 ступеней, 15.3 ГБ и 11.6 tok/s у обычного 4-bit. `--speculative-lookup` теперь работает и на чекпоинтах Qwen3.5/3.8 с vision-башней (принятие драфта 43–51%, +17% на правках кода), а `--kv-bits kvarn|kvarn-qjl` (2-bit KV) и `--attention-backend moba` подключены к `basalt serve` и проходят 4k passkey ([#632](https://github.com/goodmai/basalt/issues/632), [#634](https://github.com/goodmai/basalt/issues/634)). Чекпоинты: [`goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX), [`goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX).

Подробнее: [REST API](docs/i18n/ru/rest-api.md) · [интеграции](docs/i18n/ru/integrations.md) · [команды и флаги](docs/i18n/ru/commands.md) · [модели](docs/i18n/ru/models.md) · [бенчмарки](docs/i18n/ru/benchmarks.md)

## Обучение и квантование: универсальный конвертер «любой-в-любой», PV-MACaQ, распределённое обучение и QAT

- **Универсальный конвертер квантования any-to-any (`basalt models convert`).** Конвертация между `MACaQ` (`affine` / `mxfp4` / `moe` 2.8 bpw с гессианом, взвешенным по роутеру, [#537](https://github.com/goodmai/basalt/pull/537), [#599](https://github.com/goodmai/basalt/pull/599)), `FlatQuant` и `LIRQuant` ([#588](https://github.com/goodmai/basalt/pull/588), [#594](https://github.com/goodmai/basalt/pull/594)), двухосевой нормализацией `SINQ` / `A-SINQ` / `EAQuant` ([#561](https://github.com/goodmai/basalt/pull/561)), 2-битной решёткой `QuIP# E8P` и `TurboBoA` Babai–LLL ([#589](https://github.com/goodmai/basalt/pull/589)), битовым решётчатым кодом `QTIP` (`2.0–3.0 bpw`, [#494](https://github.com/goodmai/basalt/pull/494)), `SVD-LLM V2` / `OBR` ([#592](https://github.com/goodmai/basalt/pull/592)), `.gguf` i-quants и `MACaQ-R`, а также предметно-ориентированной доводкой `QAT STE` и постквантовательным аудитом качества, справедливости и утечек (`QuantizationAuditCore`, [#623](https://github.com/goodmai/basalt/pull/623)) — всё единообразно подключено в CLI, REST `/v1/models/convert*` и окне macOS `ConvertModelSheet` ([#546](https://github.com/goodmai/basalt/issues/546)).
- **Постквантовательная доводка PV-Tuning и PV-MACaQ.** Команда `basalt models convert --to macaq:affine:4:64 --pv-tune` чередует непрерывное обновление масштабов/смещений с дискретным перебором переворотов по гессиану на блоках Уолша–Адамара: на `Qwen3.5-9B` переход `8-bit → 4-bit PV-MACaQ` снижает потребление RAM на 33% (`9.97 → 6.68 ГБ`) и ускоряет декодирование на `+52%` (`27.3 → 41.6 tok/s`, p50 `43.2`) при точном паритете `4/5` на Ladder (`valKL = 0.0044`); на 2–3-битных чекпоинтах PV-Tuning сокращает потери перплексии на 8.7–29% менее чем за 3 минуты ([#486](https://github.com/goodmai/basalt/pull/486), [#545](https://github.com/goodmai/basalt/pull/545)).
- **Дистилляция, пост-обучение и распределённый запуск на нескольких Mac.** `basalt models distill` дистиллирует учителей OpenAI/Anthropic в адаптеры `lora`, `phm`, `oft` или квантованные `AdaLoRA (LR/40)` ([#501](https://github.com/goodmai/basalt/pull/501), [#508](https://github.com/goodmai/basalt/pull/508), [#624](https://github.com/goodmai/basalt/pull/624)); двухэтапный `SFT → DPO`, zero-copy `ADPO`/`cADPO`, `EMA-QLoRA`, `Self-Appraisal` и анти-чит-аудит `RLVR` удерживают точность выравнивания предпочтений ([#601](https://github.com/goodmai/basalt/pull/601), [#624](https://github.com/goodmai/basalt/pull/624)); обучение на кластере из нескольких Mac поддерживает аутентифицированный транспорт, `DDP + ZeRO-2`, `DiLoCo` + `PowerSGD`, `Zero-Bubble ZB-H1` и `Expert-Parallel MoE` ([#595](https://github.com/goodmai/basalt/pull/595)–[#598](https://github.com/goodmai/basalt/pull/598)).
- **Обучение прямо во время обслуживания с контролем качества.** Выделенный по тайм-слайсам MCP `distill`, строгий `train_experts` только по верифицированным данным, `train-jepa` и побитовый откат при деградации отложенной KL-дивергенции или перплексии.

Подробнее: [команды обучения](docs/i18n/ru/commands.md) · [самообучение, дистилляция и контроль качества](docs/i18n/ru/research-tools.md#self-training-train_experts)

## Compose: декларативный `basalt.yaml v2.2`, бизнес-управление и 117 аппаратных профилей

- **docker-compose, но для моделей (`basalt.yaml v2.2`).** Локальные и облачные модели, планировщик дедупликации RAM, профили специалистов, режимы памяти `resident` / `on-demand` / `sequential`, 5-компонентная маршрутизация `BusinessOrchestrator`, семантический кэш `LANTERN`, шлюз ручного подтверждения `HITL` (`/v1/mcp/approvals`) и нейро-символическое ядро знаний `LAAS-CNRS` (`SymbolicKnowledgeGraph`) — в одном декларативном файле ([#607](https://github.com/goodmai/basalt/pull/607), [#625](https://github.com/goodmai/basalt/pull/625), [#626](https://github.com/goodmai/basalt/pull/626)).
- **117 встроенных профилей для любого Mac (`16 ГБ` — `256 ГБ`).** Девять уровней объединённой памяти (`16gb`, `24gb`, `32-36gb`, `48gb`, `64gb`, `96gb`, `128gb`, `192gb`, `256gb`) × 13 архетипов рабочих нагрузок запускаются одной командой (`basalt compose up 24gb-voice-zonos`, [#528](https://github.com/goodmai/basalt/pull/528)).
- **Отклоняет то, что не поместится, обновляется на лету.** `basalt compose plan` оценивает размеры моделей с Hub ещё до скачивания и не допускает переполнения; `compose diff` / `compose up` аккуратно применяют горячие diff-обновления; `compose bench` прогоняет декларативные наборы бенчмарков и экспортирует бандлы для лидерборда ([#505](https://github.com/goodmai/basalt/pull/505)).
- **Локальные судьи, калиброванная маршрутизация и облачные бюджеты.** `OpenJev-Flash-9B-4bit` достигает точности сортировки 98.9% и сокращает расход токенов последующих LLM на 77% ([#465](https://github.com/goodmai/basalt/pull/465)), откалиброванный по температуре `CalibratedRouterGate` + `Gemini Decision 1` чисто направляют сложные запросы в пределах дневных лимитов в USD ([#418](https://github.com/goodmai/basalt/pull/418), [#622](https://github.com/goodmai/basalt/pull/622)), а распознанный текст документов и аудио по умолчанию никогда не покидает ваш Mac.

Подробнее: [справочник и замеры compose](docs/i18n/ru/compose.md) · [пресеты пайплайнов и режимы памяти](docs/i18n/ru/pipelines.md) · [облачные решения с Gemini](docs/i18n/ru/cloud-decisions.md)

## Исследования: вскройте модель прямо во время работы

- **Рентген модели, по-токенная атрибуция и эпистемический шлюз Белнапа.** `model_describe` (выводит манифесты модулей `MACaQ` / `FlatQuant` / `LIRQuant`, [#630](https://github.com/goodmai/basalt/pull/630)), `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run`, `trace_tokens` (с автоматической обработкой BOS у Gemma, [#630](https://github.com/goodmai/basalt/pull/630)), по-токенные logprobs и риск галлюцинаций `/v1/xray/tokens` ([#613](https://github.com/goodmai/basalt/pull/613)), атрибуция голов `Score d'influence` ([#622](https://github.com/goodmai/basalt/pull/622)), спектры Марченко–Пастура, четырёхзначные состояния свидетельств Белнапа `SIXTEEN_3` ([#500](https://github.com/goodmai/basalt/pull/500)), комплексные MVN-зонды фазоров ([#509](https://github.com/goodmai/basalt/pull/509)), дрейф Фишера–Рао ([#503](https://github.com/goodmai/basalt/pull/503)) и OpenMetrics на `/api/v1/metrics` ([#458](https://github.com/goodmai/basalt/pull/458)).
- **Обратимая хирургия и механистические схемы.** С флагом `serve --research-tools`: `intervene` (включая условную абляцию, патчинг групп GQA, когнитивное нуль-пространство и защиту от анти-стиринга, [#600](https://github.com/goodmai/basalt/pull/600)), разложение признаков `MAO-SNMF`, `weights_delta`, живые `tensor_pack`/`tensor_unpack`, `activation_contrast`, `steer_search`/`steer_commit`, `experiment_compare` и `introspect`. Каждый эксперимент — это отменяемый оверлей с TTL и журналом; веса на диске не изменяются никогда.
- **Бенчмарки, которые исполняют код модели.** Пятиступенчатая лестница задач, завершающаяся миссией Земля–Луна, запускает написанный моделью код на скрытых входных данных, а ARC-AGI оценивается по официальным правилам pass@2.

Подробнее: [инструменты исследования](docs/i18n/ru/research-tools.md) · [научные основы (95+ статей)](docs/i18n/ru/research-foundations.md) · [бенчмарки](docs/i18n/ru/benchmarks.md) · [флаги serve](docs/i18n/ru/commands.md)

## Basalt for Mac: macOS 27 Liquid Glass в одном нажатии или клике

- **Нативное приложение SwiftUI поверх управляемого движка.** Создано для macOS 27 Liquid Glass ([#479](https://github.com/goodmai/basalt/pull/479)) на английском и русском языках ([#456](https://github.com/goodmai/basalt/pull/456)): тёплое открытие окна за 257 мс, 0 пробуждений в простое, 7.4 мс p95 на потоковый токен в переписке из 2 000 сообщений и задержка быстрого вызова `⌥Space` всего 40 мс ([#463](https://github.com/goodmai/basalt/pull/463)).
- **Compose Video Studio, 2K Image Studio и UI универсального конвертера.** Генерируйте и воспроизводите A/V-видео `.mp4` на Mac с 24 ГБ ([#611](https://github.com/goodmai/basalt/pull/611)), редактируйте 2K RGBA-изображения со слайдером «До / После» ([#519](https://github.com/goodmai/basalt/pull/519)), конвертируйте модели и проводите их QAT-аудит прямо в `ConvertModelSheet` ([#623](https://github.com/goodmai/basalt/pull/623)), просматривайте и публикуйте модели на Hugging Face с SSE-телеметрией 10 Гц и загрузкой через Xet ([#587](https://github.com/goodmai/basalt/pull/587)).
- **Профили Compose (`⌥⌘C`), полнодуплексный голос и запуск в 1 клик по `basalt://`.** Автоматически подбирает лучшие профили под объём RAM вашего Mac (`16–256 GB`, [#528](https://github.com/goodmai/basalt/pull/528)), обеспечивает непрерывный голосовой диалог через `SpeechAnalyzer` / `LFM2.5-Audio` / `ZONOS3-Mini` ([#604](https://github.com/goodmai/basalt/pull/604)), поддерживает ссылки `basalt://run?model=...`, быстрый запрос по `⌥Space`, Быстрые команды (Shortcuts), Spotlight и нотаризованную упаковку в `.dmg`.


**Basalt.app: запущенный Compose и чат с инспектором** (macOS 27 Liquid Glass, тёмная и светлая темы):

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-compose-dark.png"><img src="docs/assets/screenshots/app-compose-light.png" alt="Basalt.app — Compose-профиль активен на движке" width="49%"></picture></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-chat-dark.png"><img src="docs/assets/screenshots/app-chat-light.png" alt="Basalt.app — чат с открытым инспектором" width="49%"></picture></p>

Подробнее: [Basalt for Mac](docs/i18n/ru/mac-app.md)

---

## Быстрый старт

Требования: macOS 27+, Xcode 27.0+ / Swift 6.4 с Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), Apple Silicon M1–M4, от 2 до 38 ГБ на диске в зависимости от модели. Подробные шаги, сборка из исходников и лаунчер `./basalt-claude` описаны в [docs/i18n/ru/install.md](docs/i18n/ru/install.md).

**Без Xcode и сборки — готовый бинарник для Apple Silicon в две строки** (macOS 27+, Apple Silicon; `mlx.metallib` внутри архива; первый запуск скачает модель на 12 ГБ, для лёгкого старта возьмите `goodmai/gemma-4-12B-it-qat-4bit`):

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

Стриминг MoE-модели на 37.7 ГБ на Mac с 24 ГБ или запуск полноценной мультимодельной системы Compose:

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 117 profiles across 16–256 GB Macs
```

Приложение для Mac из клона репозитория (при первом запуске автоматически регистрирует ссылки `basalt://` для запуска в 1 клик):

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

Локальные модели на 4B–12B — не прямая замена передовым облачным моделям; они нужны для приватной, автономной, бесплатной агентной работы и гибридных локально-облачных систем. Команда `basalt fit` подскажет, что реально сможет запустить ваша машина. А в таблице моделей честно отмечено и то, что *сломалось*.

**Поставьте звезду**, если хотите видеть больше инструментов для MLX на Swift, и откройте issue со своим Mac и скоростью tok/s — [PR с добавлением строки в таблицу бенчмарков](docs/i18n/ru/benchmarks.md#benchmarks) приветствуются.

---

## Документация

> Переводы документации находятся в `docs/i18n/`: [Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/) · [한국어](docs/i18n/ko/) · [Bahasa Indonesia](docs/i18n/id/).

| Тема | Документ |
|---|---|
| Установка, требования, сборка из исходников, `./basalt-claude` | [docs/i18n/ru/install.md](docs/i18n/ru/install.md) |
| Команды CLI, флаги `serve`, кэш промптов, CacheBlend, PV-Tuning | [docs/i18n/ru/commands.md](docs/i18n/ru/commands.md) |
| REST API: безопасность, управление моделями, OpenAI, Anthropic, изображения, SSE, OpenMetrics, Swagger | [docs/i18n/ru/rest-api.md](docs/i18n/ru/rest-api.md) |
| Claude Code, OpenCode, OpenAI SDK, MCP для Claude Desktop и Cursor | [docs/i18n/ru/integrations.md](docs/i18n/ru/integrations.md) |
| Проверенные модели, стек `goodmai` на Hub и скачивание | [docs/i18n/ru/models.md](docs/i18n/ru/models.md) |
| Обзор возможностей, лестница бенчмарков, ARC-AGI, происхождение замеров | [docs/i18n/ru/benchmarks.md](docs/i18n/ru/benchmarks.md) |
| Пресеты пайплайнов и режимы памяти | [docs/i18n/ru/pipelines.md](docs/i18n/ru/pipelines.md) |
| Облачные решения: Gemini Decision 1, мультимодальные медиа, учитель калибровки | [docs/i18n/ru/cloud-decisions.md](docs/i18n/ru/cloud-decisions.md) |
| Справочник и замеры `basalt compose` | [docs/i18n/ru/compose.md](docs/i18n/ru/compose.md) |
| Basalt for Mac и диплинки `basalt://` | [docs/i18n/ru/mac-app.md](docs/i18n/ru/mac-app.md) |
| Рентген модели, телеметрия, эксперименты, самообучение, дистилляция | [docs/i18n/ru/research-tools.md](docs/i18n/ru/research-tools.md) |
| Математические и научные основы (85+ статей и благодарности) | [docs/i18n/ru/research-foundations.md](docs/i18n/ru/research-foundations.md) |
| Структура проекта, архитектурные заметки, скрипты, зависимости | [docs/i18n/ru/architecture.md](docs/i18n/ru/architecture.md) |
| Паттерны архитектуры и модули (на русском) | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| Облако: настройка OpenRouter, оптимизация затрат, сравнение моделей | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| Работа над basalt с AI-агентами | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| Дорожная карта, политика конфиденциальности | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| Заметки по агентным CLI, чеклисты, спецификации (на русском) | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| Участие в разработке, список изменений | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## Лицензия

[MIT](LICENSE) © 2026 goodmai

Веса моделей **не** покрываются этой лицензией — у каждой модели на HuggingFace действуют собственные условия.
