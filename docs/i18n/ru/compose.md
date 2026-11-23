# basalt compose

> Languages: [English](../../compose.md) · **Русский** · [中文](../zh/compose.md) · [Español](../es/compose.md) · [Français](../fr/compose.md) · [日本語](../ja/compose.md) · [Deutsch](../de/compose.md) · [한국어](../ko/compose.md) · [Bahasa Indonesia](../id/compose.md)

`basalt compose` обслуживает систему моделей, описанную в одном YAML-файле. В файле перечисляются локальные и облачные модели и способ удержания каждой из них в памяти. В нём также описываются пайплайны, маршрутизирующие запрос между ними. Каждый пайплайн сначала выполняет восприятие (OCR, речь), затем принятие решения (System One), а затем ответ (чат-модель). Это смесь моделей в стиле `docker-compose` для моделей.

```bash
basalt compose plan -f basalt.yaml        # validate and show the compute plan on this Mac
basalt compose up -f basalt.yaml          # serve it; run it again with a changed file to apply the diff
basalt compose ps                         # what is loaded, when on-demand models unload, cloud spend
basalt compose down <name>                # stop it
basalt compose up document-qa             # legacy bundled files: document-qa, voice-agent, zonos-voice
basalt compose up 24gb-voice-zonos        # 84 hardware-tiered profiles: <tier>-<archetype>
```

`compose up` запускает `basalt serve` с флагами, в которые разворачивается файл; эти флаги выводятся при старте. Сервер предоставляет те же интерфейсы REST и MCP. Флаги после `--` передаются в `serve`, например `basalt compose up -f basalt.yaml -- --port 8090`.

Сервер сначала привязывает порт, а затем загружает модели. Пока они загружаются, `compose ps` показывает их как `loading`, `GET /v1/health/ready` отвечает кодом 503, а запись о запуске уже существует, поэтому повторный вызов `compose up` находит эту систему вместо запуска новой. После загрузки моделей каждая из них проходит стартовую проверку:
- чат-модель и каждая локальная модель ответа генерируют один токен;
- каждая локальная модель принятия решений отвечает на один вопрос;
- файлы модели по требованию (`on-demand`) проверяются без её загрузки.

Модель, не прошедшая проверку, отображается в `compose ps` со статусом `failed` и указанием причины. `/v1/health/ready` продолжает отвечать 503, пока `compose up` не заменит эту модель. Модель, которую `compose up` загружает или перезагружает, проходит такую же проверку. С флагом `--mcp` модели, как и раньше, загружаются до старта сервера.

## Файл

```yaml
version: 1
name: support-desk            # the system's name, used by ps, down and up-as-diff
memory: auto                  # resident | sequential | auto
budget: { usd_per_day: 2 }    # optional cap on all cloud spend

models:
  chat:     { source: mlx-community/gemma-4-12B-it-qat-4bit }
  big:      { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, idle_unload: 10m }
  ocr:      { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 2m }
  asr:      { source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX, keep: sequential }
  jev:      { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
  frontier: { source: "cloud:openrouter/anthropic/claude-haiku-5.5", api_key_env: OPENROUTER_API_KEY, budget: { usd_per_day: 1 } }

pipeline:                     # the default pipeline: /v1/pipeline and MCP basalt_pipeline
  perceive:
    documents: [ocr, apple-vision]     # the first available reads pdf and image attachments
    audio: [asr, speech-analyzer]
  decide:
    cascade: [jev, chat]               # up to four stages, cloud and local in any order, the last one local
    escalate_below: 0.6
    abstain_below: 0.75                # questions below this confidence go to the answer model
  answer:
    routes:                            # the first matching rule wins
      - when: { decide: { route: code }, cost_left: true }
        model: frontier
        fallback: chat
      - when: { prompt_tokens: { over: 8000 } }
        model: big
      - model: chat                    # the last rule has no when and names a local model

pipelines:                    # more pipelines, served at /v1/pipelines/<name>
  triage:
    decide: { cascade: [chat] }
    answer: chat
```

### Модели
- **`source`:** репозиторий Hugging Face, локальный путь (`./`, `/` или `~`), `cloud:openrouter/<model>` или `cloud:gemini/<model>` для моделей принятия решений Google, таких как `gemini-decision-1` (только для каскадов `decide`, ключ обычно находится в `GEMINI_API_KEY`).
- **`keep`** применяется к локальным моделям:
  - `resident` используется по умолчанию и остаётся загруженной;
  - `on-demand` загружается при первом запросе, которому нужна модель, и выгружается после `idle_unload`. Длительность задаётся в виде `30s`, `5m` или `1h`, значение по умолчанию — 120 s;
  - `sequential` загружает модель восприятия только на время своего этапа. В пайплайне по умолчанию она также может выгружать чат-модель на время работы модели восприятия, если обе не помещаются вместе.
- **Облачные модели:**
  - Укажите имя переменной окружения, содержащей ключ, в `api_key_env`. Ключ, записанный прямо в файле, отклоняется, как и интерполяция `${…}`.
  - `budget: { usd_per_day }` ограничивает расходы одной модели.
  - `data: no-perceived` используется по умолчанию: текст, прочитанный из документов или аудио, никогда не попадает в облачную модель.
  - `data: media` (только для моделей принятия решений `cloud:gemini`) отправляет сами прикреплённые файлы пайплайна (PDF, изображения, аудио в формате WAV) и распознанный из них текст в эту модель для принятия решения. Используйте этот параметр только для документов, которым разрешено покидать этот Mac.

### Память и план
`compose plan` суммирует резидентные байты каждой локальной модели. Модель `on-demand` учитывается как резидентная, поскольку она может быть загружена в любой момент.

- `memory: resident` требует, чтобы сумма помещалась в рабочий набор памяти.
- `memory: sequential` требует только самый крупный этап плюс 2 GB.
- `memory: auto` выбирает `resident`, если всё помещается.

Система, которая не помещается, отклоняется с указанием цифр до начала загрузки чего-либо. Для нескачанной модели сообщается, что её размер неизвестен.

### Принятие решений (decide)
- **Порядок:** `cascade` перечисляет модели принятия решений по порядку. Каждый этап отвечает на вопросы, в которых уверен (не менее `escalate_below`), а остальные передаёт дальше.
- **Этапы в любом порядке:** до четырёх этапов, облачных и локальных в любом порядке. Пример: сначала небольшая локальная модель-судья, затем облачная модель и последней — более крупная локальная модель: `[laya, jev, chat]`.
  - Последний этап — локальный.
  - Каждый локальный этап удерживается как движок принятия решений: чат-модель, считываемая как `self`, либо модель принятия решений, например пакет Laya Core AI или пакет OpenJev. Выделенная локальная модель принятия решений может задать `keep: on-demand` (и `idle_unload`), чтобы загружаться при первом дошедшем до неё решении и выгружаться при простое.
  - Каждый пайплайн указывает собственные этапы. Модель принятия решений, указанная в нескольких пайплайнах, загружается один раз.
  - Модель принятия решений только принимает решения; только чат-модель и отвечает, и принимает решения.
- **Калибровка:** модель принятия решений со считыванием берёт свои температуры из файла `basalt_readout.json` в своей папке (`{"temperature": …, "noul_temperature": …, "noul_bias": …}`), если он есть. В противном случае она использует известные значения для своего имени, а иначе — никаких. В #448 планируется подгонять эти файлы относительно Gemini Decision 1.
- **Размеры:** `compose plan` оценивает размер пакета головы OpenJev как базовую модель плюс адаптер, так как LoRA применяется в неслитом виде поверх базовой модели в том виде, как она хранится, включая 4-bit. Собственный план `serve` делает то же самое.
  - Измерено на OpenJev-0.6B: 100 размеченных состояний, 300 вопросов, M4 Pro, коммит 066da0a. Эталоном является слитый путь `float32`, который совпадает с PyTorch в пределах 1e-4 (#406).

    | Базовая модель, LoRA не слита | Точность | Совпадение лучшего ответа с float32 | Активная память | Медиана на состояние |
    |---|---|---|---|---|
    | слитый float32 (эталон) | 176/300 | — | 2,273 MB | 185 ms |
    | bf16, как указано в пакете (по умолчанию) | 175/300 | 288/300 | 1,141 MB | 181 ms |
    | 8-bit | 176/300 | 279/300 | 608 MB | 184 ms |
    | 4-bit | 149/300 | 215/300 | 324 MB | 185 ms |

  - Базовая модель 4-bit обходится этой модели на 0.6B в 9 пунктов, тогда как 8-bit сохраняет точность при четверти памяти. `serve` и `compose` загружают ту базовую модель, которая указана в пакете, поэтому квантованная база доступна только напрямую через `JevEvaluator`.
  - Повторить запуск: `BASALT_E2E=1 BASALT_E2E_JEV_MODEL=<bundle> BASALT_E2E_JEV_BASE=<bf16 base> BASALT_E2E_JEV_BASE_8BIT=… BASALT_E2E_JEV_BASE_4BIT=… swift test --filter JevRealModelTests/unmergedArms`.
- **Судья 9B рядом с Gemma 4 12B на 24 GB:** измерено с `openjev/OpenJev-Flash-9B-MLX` (8-bit, 9.5 GB) и `openjev/OpenJev-Flash-9B-MLX-4bit` (5.9 GB). На эскалации отвечает Gemma 4 12B (`gemma-4-12B-it-qat-4bit`). Набор — 60 размеченных состояний, 180 решений, на M4 Pro 24 GB на коммите 8294bf7.
  - Используйте 4-bit судью. Рядом с Gemma она удерживает 15.3 GB активной памяти (пик 15.9 GB) из 24 GB.
  - 8-bit судья удерживает около 21 GB вместе с чат-моделью и головой 0.6B. Сама по себе она решает нормально (92.8% без воздержания, 1.85 s), но как только Gemma декодирует эскалации, KV-кэш выталкивает Mac в своп. Ветка, рассчитанная на две минуты, выполнялась более 24 минут и была остановлена.

    | Решатель рядом с Gemma 4 12B (4-bit судья) | Точность | Отправлено в LLM | Сэкономлено токенов LLM | Decide p50 |
    |---|---|---|---|---|
    | Только LLM | 97.8% | 180 | — | — |
    | OpenJev-Flash-9B 4-bit, без воздержания | 96.7% | 0 | 100% | 2.0 s |
    | OpenJev-Flash-9B 4-bit, воздержание ниже 0.6 | 98.9% | 20 | 88.7% | 2.2 s |
    | OpenJev-Flash-9B 4-bit, воздержание ниже 0.75 | 98.9% | 41 | 76.7% | 2.4 s |
    | OpenJev-Flash-9B 4-bit, воздержание ниже 0.9 | 97.8% | 74 | 58.5% | 2.5 s |
    | считывание самой Gemma (`self`), воздержание ниже 0.75 | 99.4% | 1 | 99.4% | 1.9 s |
    | голова OpenJev-0.6B (прогон с 8-bit судьёй), воздержание ниже 0.75 | 82.8% | 86 | 48% | 0.26 s |

  - Принятие решений во время декодирования длинного ответа Gemma снижает скорость декодирования с 21 до 3 tok/s (−86%). Общее время составляет 341 s против 202 s при последовательном выполнении. Поэтому GPU-решатели разделяются шлюзом между генерациями.
  - Повторить запуск: `BASALT_E2E=1 BASALT_E2E_BUDGET_CHAT_MODEL=<chat> BASALT_E2E_BUDGET_FLASH=<judge dir> swift test --filter SystemOneBudgetBenchmarkTests`.
- **Что получает облако:** запросы с документами или аудио пропускают облачные этапы и поступают только на локальные этапы в их порядке.
- **Пороги:** вопросы, уверенность по которым ниже `abstain_below` или два лучших варианта которых ближе друг к другу, чем `min_margin`, переходят к модели ответа.
- **Эпистемический шлюз (version 2):** `epistemic` классифицирует `(support, refutation, needs_tool)` до декодирования единого токена ответа: `enabled`, `support_threshold`, `refutation_threshold`, `tool_threshold`, `on_no_evidence` (`abstain` или `escalate`), `on_conflict` (`report` или `escalate`) и необязательный `abstain_message`.

### Маршруты ответа
- **Сопоставление:** блок `when` правила срабатывает, когда выполняются все условия. Условия:
  - `modality`: `documents`, `audio` или `text`;
  - `decide`: идентификатор вопроса к метке, от System One;
  - `prompt_tokens: { over: N }`: оценивается как 4 байта на токен;
  - `escalated`: этап `decide` выполнил эскалацию или вопрос ушёл в модель ответа;
  - `cost_left`: у модели правила всё ещё есть бюджет.
- **Порядок попыток:** сначала `model` правила, затем его `fallback`, затем локальная модель пайплайна по умолчанию.
- **Пропуски:** облачная модель пропускается для распознанного текста, при отсутствии бюджета, после отклонённого ключа или во время действия ограничения частоты запросов. Каждый пропуск указывается в квитанции.

### Синтез речи — speak (`#493`, `#525`)
Пайплайн может объявить необязательный этап `speak` для синтеза аудио (`Kokoro-82M`, `Qwen3-TTS`, `Fun-CosyVoice3`, `Chatterbox`, `Zyphra/ZONOS3-Mini-hf`) с детерминированным фолбэком GPU → Neural Engine → системный голос:

```yaml
pipeline:
  answer: chat
  speak:
    chain: [tts_zonos, tts_ane, apple-speech]
    voice: Savannah
    language: en_us
    reference_audio: ./speaker.wav
    on_gpu_busy: fallback_ane
    sentence_streaming: true
    format: wav
    exaggeration: 0.5
    max_ttfa_ms: 80
```

- **`speak`:** принимает одно имя (`speak: apple-speech`), список фолбэков (`speak: [tts_zonos, tts_ane, apple-speech]`) или полное отображение параметров.
- **`chain`:** упорядоченные локальные TTS-модели, опционально завершающиеся системным синтезатором нулевого веса `apple-speech`.
- **`voice`:** идентификатор голоса по умолчанию (`Savannah`, `alloy`, `af_heart` или идентификатор `AVSpeechSynthesisVoice`).
- **`language`:** код языка или локали по умолчанию (`en_us`, `ru`, `de`, `cmn` и т. д.).
- **`reference_audio`:** необязательный локальный эталонный клип `.wav` / `.pcm` / `.caf` / `.aiff` / `.flac` для zero-shot клонирования голоса.
- **`on_gpu_busy`:** `keep`, `fallback_ane` или `fallback_apple_speech`, когда Metal занят декодированием.
- **`sentence_streaming`:** потоковая передача фрагментов предложений в TTS в полном дуплексе по мере поступления токенов.
- **`format`:** выходной контейнер или кодировка сэмплов (`wav`, `pcm`, `pcm_24k`, `mp3`, `opus`, `aac`, `flac`).
- **`exaggeration`:** управление выразительностью просодии в диапазоне `0.0...2.0`, где поддерживается.
- **`max_ttfa_ms`:** максимальный бюджет задержки до первого аудиосигнала перед переходом по цепочке фолбэков.

### Генерация изображений — imagine (`#517`, `#539`)
Пайплайн может объявить необязательный этап `imagine` для синтеза или редактирования изображений на Apple Silicon через до четырёх явных стадий (переписчик `upsample_via` -> текстовый/визуальный энкодер `share_encoder` -> DiT-генератор `chain` -> латентный VAE-декодер `decoder_via`), а также пропуск шагов TeaCache и отбор лучшего из N вариантов с помощью VLM-судьи:

```yaml
pipeline:
  answer: chat
  imagine:
    chain: [z_turbo, sana_fast]
    upsample_via: chat
    share_encoder: encoder
    decoder_via: vae
    steps: 8
    size: 1024x1024
    format: png
    teacache_threshold: 0.15
    preview_every_steps: 2
    candidates: 3
    judge_via: vlm_judge
    on_gpu_busy: fallback
```

- **`imagine`:** принимает одно имя генератора (`imagine: z_turbo`), список фолбэков (`imagine: [z_turbo, sana_fast]`) или полное отображение параметров.
- **`chain`:** упорядоченные локальные синтезаторы изображений (`FLUX.2/3-klein`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`, `Ming-Image`, `Bonsai-4B`, `Step1X-Edit`, `SonicDiffusion`).
- **`upsample_via`:** локальный слот LLM/VLM-переписчика, используемый для развёртывания кратких промптов в подробные описания сцен перед синтезом (при нацеливании на `FLUX.3` использует соглашение BFL для полных описаний сцен).
- **`share_encoder`:** локальный слот LLM/VLM (такой как `Qwen3-VL-4B-Instruct` в 4-bit, 8-bit или BF16), чьи обусловливающие эмбеддинги управляют синтезатором изображений.
- **`decoder_via`:** необязательный локальный слот латентного VAE-декодера (например, дистиллированный `black-forest-labs/FLUX.2-small-decoder` или полный `unsloth/FLUX.2-VAE` / `Felldude/FLUX.2-HDR-VAE`).
- **`steps`:** количество шагов шумоподавления по умолчанию в диапазоне `1...100`.
- **`size`:** выходное разрешение по умолчанию `<width>x<height>` (кратно 16, например `1024x1024` или `2048x2048`).
- **`format`:** формат выходного изображения (`png`, `webp` или `jpeg`).
- **`teacache_threshold`:** порог остаточной L1-нормы (`>= 0`) для пропуска внутренних шагов TeaCache.
- **`preview_every_steps`:** выдаёт промежуточные кадры предпросмотра TAEF каждые N шагов при SSE-стриминге (`>= 0`).
- **`candidates`:** количество изображений-кандидатов с разными seed (`1...64`), синтезируемых для выбора лучшего из N.
- **`judge_via`:** локальный слот VLM/LLM, оценивающий `candidates` относительно промпта и возвращающий победивший артефакт.
- **`on_gpu_busy`:** `queue` или `fallback`, когда Metal занят декодированием.

Встроенные 4-стадийные пресеты `FLUX.3` (`basalt compose up <preset>`): `flux3-16gb-fast`, `flux3-24gb-balanced`, `flux3-36gb-moe-vision`, `flux3-64gb-studio-base` и `flux3-128gb-reference`.

### Версия 2: переменные и профили
`version: 2` добавляет два блока, чтобы один файл обслуживал несколько Mac. Файл версии 1 продолжает работать; использование любого из этих блоков в нём отклоняется с ошибкой `"needs version: 2"`.

```yaml
version: 2
name: desk
variables:
  chat_repo: mlx-community/gemma-4-12B-it-qat-4bit
profiles:
  small:
    when: { memory_gb: { under: 20 } }
    variables: { chat_repo: mlx-community/gemma-4-E4B-it-4bit }
  big:
    when: { memory_gb: { over: 48 }, chip: Max }
models:
  chat:  { source: "${chat_repo}" }
  large: { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, profiles: [big] }
```

- **`variables`:** имена в нижнем регистре с однострочными простыми значениями. `${name}` заменяется в любом месте файла. Необъявленная подстановка `${…}` отклоняется, поэтому переменные окружения по-прежнему никогда не попадают в файл; секреты передаются только через `api_key_env`.
- **`profiles`:** профиль включён, когда его условие `when` выполняется на данном Mac: `memory_gb` с `under` и `over` (объединённая память в GB) и `chip` (регистронезависимая часть названия чипа, например `M4 Pro` или `Max`). Профиль без `when` включается только при указании через `--profile`.
  - Значения `variables` включённого профиля переопределяют переменные файла.
  - Модель или именованный пайплайн с `profiles: [...]` существуют только пока включён хотя бы один из них.
- **Переопределения:** `--set name=value` переопределяет объявленную переменную, а `--profile name` включает ровно указанные профили для `config`, `plan` и `up`. `compose up` передаёт их серверу, а `up-as-diff` отправляет их вместе с файлом.
- **Пути:** пути к моделям `./` или `../` разрешаются относительно папки файла compose.
- **`basalt compose config -f <file>`** выводит файл в том виде, как он разрешается на данном Mac, с включёнными профилями. **`basalt compose schema`** выводит JSON Schema формата для редакторов.

### Experimental и MCP (`#485`)
Файл compose может объявлять верхнеуровневые секции `experimental` и `mcp` (обе поддерживаются в версиях 1 и 2) вместо передачи длинного списка флагов после `--`:

```yaml
experimental:
  zero_copy_body: true
  smallm_kernel: true
  speech_analyzer: true
  image_fallback: ocr
  ane_vision_tower: ./tower-bundle
  telemetry_confidence_every: 4
  adaptive_thinking_budget: true
  samplers: true
  mtp_draft_head: true
  expert_streaming: true
  expert_cache_gb: 2.5
  kv_head_split: 0.5            # or quest_top_k: 64 (mutually exclusive)

mcp:
  enabled: true
  rest: true
  research_tools: false
  enable: [surgery, intervene]
  disable: [switch_model]
  image_dirs: [./images]
```

- **`experimental`:** разворачивается в соответствующие флаги `serve` (`zero_copy_body`, `smallm_kernel`, `speech_analyzer`, `image_fallback`, `ane_vision_tower`, `telemetry_confidence_every`, `adaptive_thinking_budget`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k`), проверяемые с точными номерами строк YAML (`telemetry_confidence_every` >= 0, `image_fallback` `ocr`, `ane_vision_tower`, содержащий `metadata.json` и `tower.aimodel`, `expert_cache_gb` > 0 при `expert_streaming: true`, `kv_head_split` в `0.05...1`, `quest_top_k` в `1...4096`). Относительные пути (`./`, `../`, `~`) разрешаются относительно папки файла compose.
- **`mcp`:** настраивает stdio-транспорт MCP (`enabled`, `rest`, `research_tools`, `enable`, `disable`, `image_dirs`), используя те же селекторы групп и инструментов, что и `--mcp-enable` / `--mcp-disable` (`serving`, `gemma`, `media`, `harness`, `xray`, `surgery`, `research`, `all` или отдельные имена инструментов; `disable` всегда побеждает).
- **Горячая перезагрузка против перезапуска:** `speech_analyzer`, `image_fallback`, `telemetry_confidence_every` и `adaptive_thinking_budget` обновляются на лету при выполнении `compose up` (`POST /v1/compose/apply`, выводятся под `experimental:`). Флаги, привязанные к движку и транспорту (`zero_copy_body`, `smallm_kernel`, `ane_vision_tower`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k` и `mcp`), в `compose diff` сообщают `needs a restart` и отклоняются при живом `apply`.

### Бенчмарки и отправка в таблицы лидеров (версия 2, `#492`)
Файл compose с `version: 2` может объявлять верхнеуровневый блок `benchmarks` с общими `defaults` и именованными `suites` (либо под `suites:`, либо в виде сокращённых ключей наборов прямо под `benchmarks:`):

```yaml
benchmarks:
  defaults:
    seed: 42
    temperature: 0.0
    gpu_lock: true
    output_dir: ./bench-out
  suites:
    smoke:
      - adapter: basalt:throughput
        target: model:chat
        params: { iterations: 2, max_tokens: 64 }
        gates: { min_decode_tps: 15, max_peak_gb: 20, max_p50_ms: 5000 }
    frontier:
      profiles: [big]
      budget: { usd_max: 5.0, max_minutes: 30 }
      tasks:
        - adapter: arc-agi-2
          dataset: ./arc-eval.json
          target: pipeline
          gates: { min_accuracy: 0.25, max_ppl: 12.0, max_ece: 0.08 }
          submit:
            target: arc-prize
            api_key_env: ARC_PRIZE_API_KEY
            endpoint: https://arcprize.org/api/v1/submissions
            subset: public-eval
            split: test
```

- **`defaults`:** необязательные `seed`, `temperature`, `gpu_lock` и `output_dir`.
- **`suites`:** каждый набор представляет собой либо сокращённый список задач (`smoke: [...]`), либо `{ profiles, budget: { usd_max, max_minutes }, tasks: [...] }`.
- **`tasks`:** каждая задача указывает `adapter` (`basalt:throughput`, `basalt:perplexity`, `basalt:system-one`, `basalt:passkey`, `basalt:mom`, `basalt:harness`, `arc-agi`, `arc-agi-2`, `terminal-bench`, `swe-bench`, `lm-eval`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `evalplus`), необязательный `dataset`, `target` (`pipeline`, `pipeline:<name>` или `model:<slot>`), скалярные `params`, регрессионные `gates` (`min_decode_tps`, `max_peak_gb`, `min_accuracy`, `max_p50_ms`, `max_ppl`, `max_ece`) и необязательный `submit` (`target`, `api_key_env`, `endpoint`, `subset`, `split`).
- **Команды:** `basalt compose plan` выводит разрешённые наборы; `basalt compose bench -f basalt.yaml --suite smoke --submit-dry-run` запускает набор и экспортирует верифицированные пакеты для таблиц лидеров; `basalt bench list|compare|submit` просматривает адаптеры, сравнивает артефакты и экспортирует dry-run пакеты для всех десяти целей (`arc-prize`, `terminal-bench`, `swe-bench`, `hf-leaderboard`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `weave-otel`).

### Сбои облачных моделей и расходы
- Отклонённый ключ (HTTP 401/403) отключает эту облачную модель до перезапуска сервера с одной строкой в логе об этом.
- Ограничение частоты запросов (429) ставит её на паузу на время `Retry-After` или на 60 s, если заголовок отсутствует.
- Пока облачная модель отключена или на паузе, отвечает следующий этап или фолбэк.
- Дневные расходы по каждой модели и в целом сохраняются в `~/.basalt/cloud-spend.json`, поэтому перезапуск не сбрасывает лимит. Дни отсчитываются по UTC.

## Работа с запущенными системами
- **`compose pull`:** скачивает каждую локальную модель из файла в кэш Hugging Face и проверяет каждый файл, выводя прогресс каждую десятую долю. После `pull` команда `compose plan` имеет точные размеры, а `compose up` загружается без ожидания скачивания.
- **Размеры в `compose plan`:** модель в кэше измеряется на диске. Нескачанная модель оценивается по списку файлов из Hugging Face API. Если размер неизвестен ни одному из них, `plan` отказывает и называет модель: выполните `compose pull` или передайте `--assume-size <GB>`, чтобы учитывать её в этом размере. `plan` больше не считает неизвестную модель за 0 байт.
- **`compose diff`:** показывает, что `compose up` изменит в запущенной системе с тем же именем, не изменяя её (`POST /v1/compose/diff`). Перечисляет загрузки, выгрузки, ключи и бюджеты, необходимость перезапуска или отказ: отсутствующий ключ либо систему, которая не помещается. Завершается с ненулевым кодом при необходимости перезапуска или отказе.
- **События:** `GET /v1/compose/events` — это поток server-sent events того, что делает система. Виды событий: `loading`, `ready`, `registered` (по требованию), `failed` (стартовая проверка), `applied`, `refused` и `up`. Новый клиент сначала получает последние 100 событий. `compose up --wait` выводит их во время ожидания.
- **`compose up` на запущенной системе:** если система с таким же `name` работает, `compose up` отправляет ей файл (`POST /v1/compose/apply`) и применяет разницу.
  - Пока система всё ещё загружается или применяет другое изменение, она отвечает 503. `compose up` ждёт и отправляет файл снова, до `--wait-timeout` секунд (по умолчанию 1800).
  - С флагом `--wait` команда `compose up` затем ждёт, пока `GET /v1/health/ready` не ответит 200, выводит модели, которых ждёт, и завершается с ненулевым кодом, если система не готова в пределах таймаута.
  - Загружает добавленные модели, перезагружает изменённые и выгружает удалённые.
  - Переключает пайплайны между запросами.
  - Сначала проверяет новую систему по памяти данного Mac, как это делает `compose plan`, с учётом моделей каждого пайплайна. Система, которая не помещается, отклоняется, и ничего не меняется.
  - Когда новая система помещается, но старые и новые модели не помещаются рядом друг с другом, она выгружает заменяемые модели до загрузки их преемников. Запрос, которому в это окно требуется одна из них, ждёт загрузки или завершается ошибкой. Если загрузка затем не удаётся, она загружает старые модели обратно.
  - Выводит `load`, `reload`, `unload` и `keep` для каждой модели, `budget` для каждого изменённого облачного бюджета (`total` для бюджета файла) и `key` для каждой облачной модели, чей `api_key_env` изменился.
  - Бюджет или переменная ключа облачной модели меняются без перезагрузки. Новые лимиты применяются сразу, а модель, отключённая из-за отклонённого ключа, снова включается при изменении её переменной ключа.
  - Другая чат-модель загружается на место обслуживающей модели и выводится как `chat`. Запросы ждут, пока она загружается. Если более поздний шаг применения не удаётся, старая чат-модель загружается обратно.
  - Новая локальная модель принятия решений пайплайна загружается как любая добавленная модель. Модель пайплайна по умолчанию также выводится как `decide`. Пайплайны и `/v1/systemone` используют её со следующего запроса. Модель, которая начинает или перестаёт принимать решения, перезагружается в своей новой роли.
- **Изменения, требующие перезапуска:** другой режим памяти, модель восприятия `sequential`, флаги запуска движка/HTTP в `experimental` или изменения в `mcp`. Выполните `compose down`, а затем `compose up`.
- **`compose ps`:** перечисляет каждую запущенную систему (`GET /v1/compose`) с её пайплайнами и для каждой модели:
  - её состояние, одно из `loading`, `ready`, `unloading`, `unloaded`, `idle-unloaded` (по требованию, сейчас не загружена), `decision` (удерживается движком принятия решений), `cloud`, `no-key`, `over-budget`, `rate-limited`, `refused` и `failed`;
  - для модели по требованию — когда она использовалась последний раз и когда выгрузится;
  - для облачной модели — её расходы за сегодня относительно лимита и отказ, если он применим.
- **Состояние (Health):** `GET /v1/health/live` отвечает 200, пока сервер отвечает по HTTP. `GET /v1/health/ready` отвечает 200, как только каждая модель может обслуживать запросы. До этого он отвечает 503 и перечисляет модели, всё ещё находящиеся в ожидании, такие как `chat (loading)`. Простаивающая модель по требованию, облачная модель сверх бюджета и облачная модель с ограничением частоты запросов считаются способными обслуживать; каскад или маршрутизатор проходят мимо них.
- **Облачные ключи:** `compose up` отказывается запускаться, когда `api_key_env` облачной модели не задан, и называет переменную. `up-as-diff` вместо этого проверяет окружение запущенного сервера.
- **`compose down`:** отправляет серверу SIGINT. Он завершает текущие запросы и выгружается. Если процесс всё ещё работает через 30 s, `down` сообщает об этом и завершается с ненулевым кодом.
- **Где хранится состояние:** pid запущенной системы, порт и время запуска процесса находятся в `~/.basalt/compose/<name>.json`. Запись, чей pid теперь принадлежит другому процессу из-за повторного использования pid, считается не запущенной.
- **API-ключ:** когда у сервера есть API-ключ, задайте `BASALT_API_KEY` для `ps` и `up-as-diff`.

## Адаптеры предметной области (`version: 2`)
Файл `version: 2` может объявлять `adapters:`, обученные на пресетах Hugging Face NVIDIA Nemotron (`nemotron:post-v2`, `nemotron:post-v1`, `nemotron:code`, `nemotron:math`, `nemotron:crossthink`, `nemotron:personas`), датасетах `hf:<org>/<repo>` или локальных файлах `.jsonl`, и маршрутизировать запросы между ними на одной резидентной модели `base` без перезагрузки весов:

```yaml
adapters:
  code_oft:
    base: chat
    method: oft              # lora | phm | oft
    rank: 16
    scale: 10.0
    hypercomplex_dim: 8
    factor_rank: 1
    shared_basis: true
    block_size: 16
    parameterization: cayley # cayley | matrixExp
    block_shared: false
    layers: [0, 1, 2, 3]
    output: ./adapters/code_oft
    iterations: 200
    batch_size: 2
    learning_rate: 0.0001
    max_seq_len: 2048
    checkpoint_every: 50
    corpus:
      source: nemotron:code
      subset: default
      split: split_0
      max_rows: 512
      min_pass_rate: 0.9
      include_reasoning: true
      seed: 42
    gate:
      max_kl: 0.2
      max_ppl_rise: 0.05
      held_out: ["Explain what a hash table is in two sentences."]
```

Запустите `basalt compose train -f basalt.yaml` (с необязательными `--adapter`, `--force`, `--resume`), чтобы обучить объявленные адаптеры на диск, а затем выберите `adapter:` в `pipeline.answer` или любой записи `pipeline.answer.routes`.

## Квитанции (Receipts)
Каждый ответ `/v1/pipeline` и каждая запись в журнале сессии несут квитанцию (`receipt`). По каждому этапу она указывает модель, миллисекунды и токены. Этапы `decide` также несут свою стоимость и все облачные этапы, которые были пропущены. Этапы `answer` несут сработавшее правило (`route`), стоимость и пропущенные модели. `swap_ms` — это время, затраченное на загрузку моделей `on-demand` и `sequential`.

## Встроенные профили по уровням оборудования (`#526`)

`basalt compose` и `Basalt.app` (`Compose Profiles`, `⌥⌘C`, а также **Settings → Engine → Bundled Profile**) поставляются с **84 калиброванными под оборудование профилями** по 7 уровням объединённой памяти Apple Silicon (`16gb`, `24gb`, `32-36gb` [псевдонимы `32gb` и `36gb`], `48gb`, `64gb`, `128gb`, `256gb` [псевдонимы `192gb` и `236gb`]) × 12 архетипов рабочих нагрузок:

| Архетип | Основной | Фокус |
|---|---|---|
| `daily-chat` | ★ | Повседневный чат, восприятие документов Apple Vision, локальный голосовой ввод |
| `coding-agent` | ★ | Спекулятивное ядро `small-M`, специализированная маршрутизация кода, инструменты MCP |
| `voice-zonos` | ★ | Нейронный TTS Zyphra `ZONOS3-Mini` 44.1 kHz + фолбэк на `Kokoro-82M` на ANE |
| `multimodal-desk` | ★ | OCR, восприятие аудио, сортировка System One, чат и голос ZONOS3 |
| `reasoning-lab` | | Адаптивные бюджеты размышлений + каскад System One для математики и логики |
| `document-ocr` | | Нейронный OCR (`PaddleOCR-VL-1.5`) + четырёхзначный эпистемический шлюз Белнапа |
| `voice-duplex` | | Полнодуплексная транскрибация и потоковый синтез с TTFA менее 80 ms |
| `creative-studio` | | Нативный синтез изображений MLX (`Z-Image-Turbo` / `FLUX`) с TeaCache |
| `rag-knowledge` | | Обоснованный поиск по длинному контексту со шлюзом конфликтов и воздержания |
| `hybrid-cloud` | | Локальный приватный чат с облачной эскалацией, ограниченной `$1/day` |
| `moe-frontier` | | Разреженные MoE (`Flint 40B-A2.5B`, `Nemotron-3.5`, `Qwen3.8`) + стриминг экспертов |
| `research-xray` | | Model X-Ray, хирургия весов и дымовые бенчмарки `basalt:throughput` |

При запуске `Basalt.app` определяет физическую объединённую память (`ComposeMemoryTier.detect`) и закрепляет 4 отмеченных звёздочкой основных профиля для данного Mac в разделе **Recommended for this Mac**, сохраняя все 84 профиля доступными для поиска и выбора. Любой профиль запускается идентично из CLI (`basalt compose plan 24gb-voice-zonos`, `basalt compose up 36gb-moe-frontier`, `basalt compose up 236gb-multimodal-desk`).

## Примеры
- `docs/examples/compose/any-mac.yaml`: один файл версии 2 для Mac с 16, 24 и 64 GB. Чат-модель поступает из профиля, выбранного по памяти, а профиль 64 GB добавляет модель 27B по требованию и пайплайн `long`.
- `docs/examples/compose/mac-16gb.yaml`: локальная Ornith 9B, OCR по требованию, длинные промпты в Claude Haiku в пределах $0.50 в день.
- `docs/examples/compose/mac-24gb.yaml`: Gemma 4 12B, OCR по требованию, код в Haiku, второй пайплайн `support`, $1 в день суммарно.
- `docs/examples/compose/mac-64gb.yaml`: Gemma 4 26B-A4B с Qwen3.8-27B и Nemotron Transcribe по требованию, без облачных ответов.
- `docs/examples/compose/document-qa-cloud.yaml`, `mom.yaml` и `too-big.yaml`: файлы, стоящие за замерами ниже.

Больше замеров приведено в #412.

## Замеры

```yaml
version: 1
name: document-qa-cloud
models:
  chat:  { source: mlx-community/gemma-4-12B-it-qat-4bit }
  ocr:   { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 30s }
  jev:   { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
pipeline:
  perceive: { documents: [ocr, apple-vision] }
  decide:   { cascade: [jev, chat], escalate_below: 0.6 }
  answer: chat
```

```bash
basalt compose plan -f docs/examples/compose/document-qa-cloud.yaml
basalt compose up -f docs/examples/compose/document-qa-cloud.yaml -- --port 8080
basalt compose up voice-agent            # bundled: document-qa, voice-agent
```

```yaml
pipeline:
  decide: { cascade: [jev, chat], escalate_below: 0.6 }
  answer:
    routes:
      - when: { decide: { route: code }, cost_left: true }
        model: haiku          # cloud:openrouter/anthropic/claude-haiku-5.5, budget: { usd_per_day: 0.5 }
        fallback: chat
      - model: chat
```

Смешанная нагрузка из 55 запросов через `/v1/pipeline` (30 решений, 10 по коду, 10 чат, 5 документов), M4 Pro, релизная сборка (`scripts/compose_mom_bench.swift`, `docs/examples/compose/mom.yaml`):

| Конфигурация | Верные метки | Облако на 1000 запросов | p50 | p95 | Время локального этапа |
|---|---|---|---|---|---|
| только локально (Gemma 4 12B решает и отвечает) | 55/55 | $0 | 0.92 s | 10.0 s | 176 s |
| сначала облако (Jev решает, Haiku отвечает на весь текст) | 55/55 | $0.046 | 1.25 s | 4.8 s | 24 s |
| MoM (Jev решает, код в Haiku пока есть бюджет, остальное Gemma) | 55/55 | $0.035 | 1.21 s | 4.8 s | 53 s |

Измерено с `compose up` на M4 Pro с 24 GB, релизная сборка:

| Файл | Запрос | Квитанция |
|---|---|---|
| document-qa-cloud | документ (первый) | OCR загружен по требованию (489 ms), решение локально, ответ: 6.1 s |
| | документ (повторно) | OCR уже загружен (0 ms): 5.0 s |
| | после 40 s простоя | OCR выгружен, объём памяти 12 → 11 GB |
| | текстовый вопрос | решение принято Jev в облаке: 0.6 s |
| | документ | OCR перезагружен (267 ms): 5.3 s |
| voice-agent (встроенный) | аудио | чат выгружен на время Nemotron Transcribe (своп 6.9 s): 17.3 s |
| too-big (пример) | `compose plan` | отказ: одному этапу нужно 20.7 GB плюс 2 GB при рабочем наборе 19.1 GB |


## Расширения схемы `version: "2.2"`

Compose `version: "2.2"` (совместимая с `version: 2`) добавляет декларативную конфигурацию для блочно-диффузионных LLM, спекулятивных драфтеров `DFlash`, дуплексной речи в речь, общих текстовых энкодеров VLM/DiT, 1-шагового апскейлинга `SeedVR2`, музыкальной DiT-модели `ACE-Step`, каузальных 3D-VAE видео-DiT, обучения предпочтениям по сегментам и кластеров из нескольких Mac по Thunderbolt/LAN:

- Помодельные `speculative` (`mode`, `drafter`, `block_size`, `verify`) и `diffusion_text` (`dual_kv_cache`, `confidence_threshold`, `contrastive_alpha`).
- Расширения `imagine`: `share_text_encoder_with`, `kv_cache_references`, `rgba` и `upscale` (`model`, `steps`, `target_resolution`).
- Корневой `voice` (`mode`, `sts_model`, `denoise`, `turn_detection` с `model` и `diarization`, `tts_fallback`, `aligner`).
- Корневые `music` (`model`, `steps`, `duration_seconds`) и `video` (`model`, `vae_tiling`, `sta_sparse`, `fps`, `frames`).
- Корневые `train` (`method`, `segments`, `ema_decay`) и `cluster` (`topology`, `peers`, `compression`).
- Корпоративная маршрутизация, кэширование и управление (`v0.12`): 5-компонентная маршрутизация `BusinessOrchestratorCore`, семантический кэш `LANTERN` (`SemanticCacheCore` / `SemanticCacheActor`, `#625`), шлюз одобрения `HITL` (`/v1/mcp/approvals`) + `SymbolicKnowledgeGraph` от LAAS-CNRS (`#626`) и учитывающая UMA иерархия L0/L1/L2 `TieredKVHierarchyCore` + `AdaptiveHeterogeneousPlanner` (`#627`).
