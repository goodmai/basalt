<p align="center"><img src="images/logo.svg" alt="логотип basalt" width="128"></p>

# basalt — теперь ваш Mac и есть AI-лаборатория.

> Запускайте, обучайте, квантуйте, собирайте в системы и вскрывайте открытые LLM на Apple Silicon. Один бинарник на Swift 6.4 говорит на **OpenAI API, Anthropic API и MCP**, а сверху — нативное приложение для macOS 27 Liquid Glass. Никакого Python в рантайме. Никакого облака, пока вы сами о нём не попросите.

[English](README.md) · [Русский](README.ru.md)

[![Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/🤗%20Модели-goodmai%20(35%2B)-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![Запустить через Basalt](https://img.shields.io/badge/▶%20Запустить%20через%20Basalt-1--клик-0A84FF.svg)](https://huggingface.co/spaces/goodmai/run-with-basalt)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20·%20Apple%20Silicon-black.svg)](docs/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/benchmarks.md#benchmarks)

<p align="center"><img src="images/models-hive.svg" alt="Открытые модели, которые запускает basalt: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Запускает модели с открытыми весами на Apple Silicon. Сверены по логитам: Gemma 4, Qwen3.8 и LFM2.5 (таблица ниже); замерены и собраны в <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a>: Ornith, Bonsai, Qwen3.6-35B-A3B, DeepSeek-OCR-2, OpenJev и Laya (см. <a href="docs/models.md#verified-models">таблицу моделей</a>); gpt-oss и Llama совместимы с MLX. Логотипы — товарные знаки их владельцев и показаны только для обозначения совместимости. Модели Anthropic закрыты, поэтому их нет в «улье»; <i>API</i> Anthropic поддержан, чтобы Claude Code мог работать с локальной моделью.</sub></p>

**Пять вещей в одном репозитории:** локальный сервер инференса, с которым ваши инструменты уже умеют говорить; дистилляция LoRA и пост-квантовое дообучение весов; docker-compose для многомодельных систем; рентген-микроскоп для того, что происходит внутри модели на ходу; и нативное приложение для Mac, которое держит всё это на расстоянии одной клавиши или одного клика.

**Сверяется с эталонными логитами при каждом изменении** пути инференса: [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) повторяет записанные промпты и падает, если сдвинулся жадный токен или логит. Эталонные логиты по семействам qwen3, qwen3_5, gemma4, lfm2 и nemotron_h, dense и MoE, лежат в [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits).

| Модель с золотыми тестами | Класс | Фикстура |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | скользящее окно + полное внимание, MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | гибрид линейного и полного внимания, dense | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | гибрид свёрток и внимания, MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### Запуск в 1 клик с [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) (35+ проверенных моделей)

На карточке каждой модели в [`goodmai`](https://huggingface.co/goodmai) стоит кнопка **Запустить через Basalt**, которая открывает `Basalt.app` по ссылке `basalt://run?model=...` или выдаёт готовую команду `basalt serve`:

| Модель (`goodmai/*`) | Роль и память на Mac 24 GB | Запуск |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT на каждый день: чат и зрение (~7,6 GB) | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE (`A4B`), золотая фикстура (~14,5 GB) | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO нативный i-quant (11,8 GB, PPL 4,17, 300/300 System One) | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 35B MoE (37,7 GB) со стримингом с SSD: 8,77 tok/s, 0 swap | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B тернарная в 7,4 GB, 27,1 tok/s с `--mtp` | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B гибрид свёрток и внимания MoE, золотая фикстура (~12,6 GB) | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | OCR со зрительным энкодером и тайлингом 768 (`/v1/pipeline`) | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | 9B локальный судья System One (98,9% точность, −77% токенов LLM) | [![Запустить через Basalt](https://img.shields.io/badge/▶_Запустить_через_Basalt-0A84FF)](https://goodmai-run-with-basalt.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide) |

---

## Инференс: 27B в 7,4 GB или 37,7 GB MoE с нулевым свопом — за привычными API

- **Claude Code на вашем GPU одной строкой.** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — ваш настоящий API-ключ не используется. OpenAI SDK, OpenCode, Continue, Cline и aider подключаются через OpenAI API, Claude Desktop и Cursor — через MCP (с точечным включением групп и инструментов `--mcp-enable` / `--mcp-disable`, [#488](https://github.com/goodmai/basalt/pull/488)).
- **Быстро на ноутбуке.** На M4 Pro с 24 GB: Qwen3-4B декодирует 78–89 tok/s; Ternary Bonsai 2 27B помещается в 7,4 GB и выдаёт 20,9–21,5 tok/s, 27,1 со своей MTP-головой и 31–38 с prompt lookup на коде ([#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)); `.gguf` чекпойнты GSQ-RCO / `i-quant` исполняются напрямую на собственных Metal-ядрах ([#433](https://github.com/goodmai/basalt/pull/433)).
- **Модели больше объёма RAM — без единого байта свопа.** `Qwen3.6-35B-A3B-8bit` (37,7 GB) генерирует 8,77 tok/s на Mac с 24 GB при 0 swap за счёт пакетного POSIX AIO и упреждающей подгрузки следующего слоя `LIO_NOWAIT` (`--experimental-expert-streaming`, [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478)).
- **Спекулятивное декодирование без потери точности.** Prompt lookup даёт 1,84–2,66× на правках кода (Bonsai 2 27B, [#239](https://github.com/goodmai/basalt/pull/239)) и 3,45–3,62× (Gemma 4 12B, [#265](https://github.com/goodmai/basalt/pull/265)), а жадный вывод не меняется.
- **Следующий ход стоит только новых токенов.** Префиксное дерево кэша промптов, даже для гибридных моделей: на Ornith-1.5-9B с агентским промптом на 12,6k токенов второй ход занимает около 1 с вместо примерно 36 с ([#82](https://github.com/goodmai/basalt/pull/82)).
- **Длинный контекст в меньшей памяти.** Квантование KV с точными зонами (`4|8|k8v4`, плюс экспериментальные `turbo3|turbo4|turbo8v4` и полярные RoPE-ключи `polar4`, [#472](https://github.com/goodmai/basalt/pull/472), [#473](https://github.com/goodmai/basalt/pull/473)): промпт на 16k токенов на пике занимает 1,03 GB вместо 3,11 GB на Qwen3-4B ([#159](https://github.com/goodmai/basalt/pull/159)).
- **Целый локальный стек.** Вызовы инструментов, вывод по JSON-схеме, regex и грамматике (XGrammar), изображения и `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)), `/v1/embeddings`, `/v1/rerank`, LoRA-адаптеры на каждый запрос и каскадный fallback «сначала LAN-узлы, затем облако» ([#480](https://github.com/goodmai/basalt/pull/480)).

Подробнее: [REST API](docs/rest-api.md) · [интеграции](docs/integrations.md) · [команды и флаги](docs/commands.md) · [модели](docs/models.md) · [бенчмарки](docs/benchmarks.md)

## Обучение и квантование: дистилляция у сильной модели и спасение sub-3-bit весов

- **Дистиллируйте фронтирную модель в LoRA.** `basalt models distill` отправляет ваши промпты учителю за API в формате OpenAI или Anthropic, ограничивает траты через `--budget-usd`, может оставлять только проверенные ответы и сохраняет чекпойнты, так что прерванный прогон продолжается (#393). `--on-policy` обучает на собственных сэмплах ученика с потокенным обратным KL от учителя на basalt.
- **Пост-квантовое дообучение PV-Tuning.** `basalt models convert --experimental-pv-tune` чередует обновление непрерывных шкал/смещений и дискретный поиск целочисленных кодов прямо на Apple Silicon, сокращая потери перплексии на 2–3 битах на 8,7–29% быстрее чем за 3 минуты ([#486](https://github.com/goodmai/basalt/pull/486)).
- **Решётчатое квантование QTIP (bitshift trellis).** `basalt models convert --experimental-qtip --bits 2.5` укладывает `Qwen3.8-27B` в 10,00 GB и обходит 3-битный MLX affine и PV-Tuning по перплексии на 16k (4,621 против 4,702 / 4,652), а на 2,0 bpw (8,48 GB) срезает 2-битную перплексию на 72,1% (5,345 против 19,146, [#494](https://github.com/goodmai/basalt/pull/494)).
- **Обучение прямо во время работы сервера.** MCP-инструмент `distill` работает квантами: модель занята по одному шагу оптимизатора, и `serve` продолжает отвечать.
- **Самообучение, которое не учит непроверенному.** `train_experts` собирает ответы на проверяемые задачи, оставляет только проверенные, обучает LoRA и показывает точность на отложенной выборке до и после.
- **Гейт качества, прежде чем что-то останется.** Адаптер остаётся, только если KL и перплексия на отложенных промптах в пределах лимитов; проваленный гейт откатывает его бит в бит.
- **JEPA и конвертация.** `basalt models train-jepa` обучает JEPA-адаптер и проектор на многоракурсных данных; `basalt models convert` импортирует `.gguf` i-quant'ы, вливает LoRA, переквантовывает или экспортирует HF safetensors.

Подробнее: [команды обучения](docs/commands.md) · [самообучение, дистилляция и гейт качества](docs/research-tools.md#self-training-train_experts)

## Compose: один YAML-файл на целую систему моделей

- **docker-compose, только для моделей.** Локальные и облачные модели, режим памяти каждой (`resident`, `on-demand` с авто-выгрузкой `idle_unload` [#481](https://github.com/goodmai/basalt/pull/481), `sequential`) и пайплайн «воспринять → решить → ответить» — OCR, речь, каскад решений System One и чат-модель — в одном файле.
- **Отказывает в том, что не поместится.** `basalt compose plan` складывает байты (включая невлитые Jev LoRA-адаптеры [#462](https://github.com/goodmai/basalt/pull/462)), берёт размеры ещё не скачанных моделей из Hugging Face API и отказывает системе, которая не помещается, ещё до загрузки. `compose pull` заранее скачивает и проверяет все модели.
- **Меняйте на ходу.** `compose diff` показывает, что изменится; затем `compose up` на работающей системе перезагружает только изменившиеся модели, экспериментальные флаги или настройки MCP ([#489](https://github.com/goodmai/basalt/pull/489)) и подменяет пайплайны между запросами. `compose ps` показывает состояние и расходы, `compose down` останавливает систему.
- **Локальные судьи и облачный бюджет.** `OpenJev-Flash-9B-4bit` помещается рядом с Gemma 4 12B в 15,3 GB на 24 GB Mac, давая 98,9% точности триажа и экономя 77% токенов большой модели ([#465](https://github.com/goodmai/basalt/pull/465)), а Laya отвечает на Neural Engine за ~11 мс. Маршруты ответов отправляют сложный код в облако, пока хватает дневного лимита: 100% верных меток за $0,035 на 1000 запросов, p95 4,8 с против 10,0 с только локально ([#418](https://github.com/goodmai/basalt/pull/418)).
- **Приватно по умолчанию.** Ключи берутся только из переменных окружения, а текст, прочитанный из ваших документов или аудио, никогда не уходит в облачную модель.
- **Gemini Decision 1 от Google как ступень решений.** `cloud:gemini/gemini-decision-1` отвечает на вопросы «да/нет», выбора и оценки по тексту, изображениям, аудио, видео и PDF за один проход, перед локальным decider ([#449](https://github.com/goodmai/basalt/pull/449)). Это ещё и калибровочный учитель для локальных моделей решений (#448).

Подробнее: [справочник compose и замеры](docs/compose.md) · [пресеты пайплайнов и режимы памяти](docs/pipelines.md) · [облачные решения с Gemini](docs/cloud-decisions.md)

## Исследования: вскройте модель прямо на ходу

- **Model X-ray и живая телеметрия.** `model_describe`, `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run` и `telemetry_query` показывают анатомию слоёв, спектры Марченко — Пастура, дерево кэша промптов, logit lens по головам, маршрутизацию и hit-rate экспертов MoE, гистограммы уверенности токенов и OpenMetrics на `/api/v1/metrics` ([#458](https://github.com/goodmai/basalt/pull/458), [#459](https://github.com/goodmai/basalt/pull/459)).
- **Обратимые эксперименты.** С `serve --research-tools`: стиринг и абляция голов или экспертов (`intervene`), дельты весов, живая переупаковка слоя в 2–8 бит, mxfp4, nf4 или троичный формат, и `experiment_compare` с KL и перплексией. Каждый эксперимент — отменяемый оверлей с TTL и журналом; файлы модели на диске никогда не перезаписываются.
- **Найдите схему за поведением.** `trace_tokens` и `activation_contrast` ранжируют головы и экспертов по d Коэна и AUROC, после чего условная абляция срабатывает только на тех токенах, что её вызывают (#405). `steer_search` and `steer_commit` превращают контрастные направления в адаптеры с гейтом.
- **Модель изучает саму себя.** `introspect` отдаёт инструменты X-ray обслуживаемой модели и записывает её гипотезы, тесты и результаты.
- **Статьи и гибридный GPU+ANE счёт внутри движка.** CacheBlend (arXiv:2405.16444), головы чанков EPIC (arXiv:2410.15332), вытеснение KV по L2-норме (arXiv:2406.11430), DuoAttention, Token Recycling, вращение Уолша — Адамара и расщепление dense-MLP между GPU и Neural Engine на prefill ([#468](https://github.com/goodmai/basalt/pull/468)) — каждое включается отдельно, с замерами рядом с флагом.
- **Бенчмарки, которые запускают код модели.** Пятиступенчатая лестница, заканчивающаяся полётом с Земли на Луну, исполняет собственный код модели на невиданных входах, а ARC-AGI идёт по официальным правилам pass@2.

Подробнее: [исследовательские инструменты](docs/research-tools.md) · [бенчмарки](docs/benchmarks.md) · [флаги serve](docs/commands.md)

## Basalt для Mac: macOS 27 Liquid Glass — одна клавиша или один клик

- **Нативное SwiftUI-приложение поверх движка под присмотром.** Поверхности macOS 27 Liquid Glass ([#479](https://github.com/goodmai/basalt/pull/479)), английская и русская локализация ([#456](https://github.com/goodmai/basalt/pull/456)), жёсткие бюджеты отзывчивости: 257 мс тёплое окно, 0 холостых пробуждений CPU, 7,4 мс p95 на токен в чате на 2000 сообщений и 40 мс на вызов `⌥Space` ([#463](https://github.com/goodmai/basalt/pull/463)).
- **Запуск в 1 клик с Hugging Face (`basalt://`).** Нажатие **Запустить через Basalt** на любой карточке [`goodmai`](https://huggingface.co/goodmai) открывает `Basalt.app` по ссылке `basalt://run?model=...` и переключает движок на выбранную модель или роль пайплайна.
- **Спрашивайте откуда угодно.** `⌥Space` открывает плавающую панель поверх любого приложения, даже полноэкранного, без разрешения Accessibility. Разговоры хранятся в SwiftData, с системным промптом на каждый разговор.
- **Библиотека моделей, которая знает ваш Mac.** Окно Models показывает кэш Hugging Face с вердиктом «влезет ли» для этой машины, загружает модель одним кликом и докачивает прерванные загрузки.
- **Бросьте PDF, картинку или запись.** Вложения проходят через пайплайн «воспринять → решить → ответить»; инспектор показывает прочитанный движком текст, токены, время до первого токена, скорость и каждую стадию.
- **Часть macOS и нотаризованная поставка.** Shortcuts и Spotlight (Ask Basalt, Summarize File, Transcribe Audio, Switch Model), Службы → Ask Basalt, файлы на иконку в Dock, уведомления, меню-бар, Monitor с состоянием compose и облачных трат, а также подпись, нотаризация `.dmg` и Homebrew cask ([#457](https://github.com/goodmai/basalt/pull/457)).

Подробнее: [Basalt для Mac](docs/mac-app.md)

---

## Быстрый старт

Требования: macOS 27+, Xcode 27.0+ / Swift 6.4 с Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), Apple Silicon M1–M4, 2–38 GB диска в зависимости от модели. Полные шаги, сборка из исходников и лаунчер `./basalt-claude` — в [docs/install.md](docs/install.md).

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # собирает CLI и сервер из исходников

basalt onboard                              # подбирает модель под ваш Mac и скачивает её
basalt serve --model goodmai/gemma-4-12B-it-qat-4bit --rest

# в другом терминале: Claude Code на вашем собственном GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Стриминг 37,7 GB MoE-модели на Mac с 24 GB или запуск многомодельной системы `compose`:

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --experimental-expert-streaming --rest

basalt compose plan -f docs/examples/compose/document-qa-cloud.yaml
basalt compose up voice-agent               # встроенные: document-qa, voice-agent
```

Приложение для Mac из исходников (схема `basalt://` для кнопок на Hugging Face регистрируется автоматически при первом открытии):

```bash
./scripts/bundle_app.swift                  # release-сборка, Metal-ядра, затем .build/Basalt.app
open .build/Basalt.app
```

Локальные модели на 4B–12B не заменяют фронтирную модель один к одному — они для приватной, офлайн и бесплатной агентной работы и для гибридных локально-облачных систем. `basalt fit` скажет, что ваша машина реально потянет. В таблице моделей указано и то, что *сломалось*.

**Поставьте звезду**, если хотите больше MLX-нативных инструментов на Swift, и откройте issue со своим Mac и своими tok/s — [PR с новой строкой в таблице бенчмарков](docs/benchmarks.md#benchmarks) приветствуются.

---

## Документация

Подробные документы пока на английском, кроме отмеченных.

| Тема | Документ |
|---|---|
| Установка, требования, сборка из исходников, `./basalt-claude` | [docs/install.md](docs/install.md) |
| Команды CLI, флаги `serve`, кэш промптов, CacheBlend, PV-Tuning | [docs/commands.md](docs/commands.md) |
| REST API: безопасность, управление моделями, OpenAI, Anthropic, изображения, SSE, OpenMetrics, Swagger | [docs/rest-api.md](docs/rest-api.md) |
| Claude Code, OpenCode, OpenAI SDK, MCP для Claude Desktop и Cursor | [docs/integrations.md](docs/integrations.md) |
| Проверенные модели, стек `goodmai` на HF и скачивание | [docs/models.md](docs/models.md) |
| Возможности кратко, лестница бенчмарков, ARC-AGI, источники цифр | [docs/benchmarks.md](docs/benchmarks.md) |
| Пресеты пайплайнов и режимы памяти | [docs/pipelines.md](docs/pipelines.md) |
| Облачные решения: Gemini Decision 1, мультимодальные данные, калибровочный учитель | [docs/cloud-decisions.md](docs/cloud-decisions.md) |
| Справочник `basalt compose` и замеры | [docs/compose.md](docs/compose.md) |
| Basalt для Mac и ссылки `basalt://` | [docs/mac-app.md](docs/mac-app.md) |
| Model X-ray, телеметрия, эксперименты, самообучение, дистилляция | [docs/research-tools.md](docs/research-tools.md) |
| Структура проекта, архитектура, скрипты, зависимости | [docs/architecture.md](docs/architecture.md) |
| Архитектурные паттерны и модули (на русском) | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| Облако: настройка OpenRouter, оптимизация расходов, сравнение моделей | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| Работа над basalt с AI-агентами | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| Дорожная карта, политика конфиденциальности | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| Заметки об агентных CLI, чек-листы, спецификации (на русском) | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| Участие в разработке, история изменений | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## Лицензия

[MIT](LICENSE) © 2026 goodmai

Веса моделей **не** покрываются этой лицензией — у каждой модели на HuggingFace свои условия.
