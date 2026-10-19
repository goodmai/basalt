<p align="center"><img src="images/logo.svg" alt="basalt logo" width="128"></p>

# basalt — превратите Mac в локальный LLM-сервер. Подключите к нему Claude Code.

> Gemma 4, Qwen 3 и любые MLX-модели на GPU Apple Silicon — через **OpenAI API, Anthropic API и MCP**. Один Swift-бинарник. Без Python в рантайме. Без облака. Без счетов за API.

[English](README.md) · [Русский](README.ru.md)

[![Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

[![Platform](https://img.shields.io/badge/platform-macOS%2015%2B%20·%20Apple%20Silicon-black.svg)](#требования)
[![Swift 6](https://img.shields.io/badge/Swift-6-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](#установка-через-homebrew)
[![Ladder](https://img.shields.io/badge/лестница-6%20задач-black.svg)](#бенчмарки)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](#бенчмарки)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (актор, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

Два транспорта работают через один экземпляр актора — **MCP stdio** для интеграции с IDE (Claude Desktop, Cursor) и **REST HTTP** для агентных сценариев и в роли бэкенда Claude Code.


<p align="center"><img src="images/models-hive.svg" alt="Открытые модели, которые запускает basalt: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Запускает open-weight модели на Apple Silicon. Проверено здесь: Gemma, Qwen, Ornith (см. <a href="#проверенные-модели">таблицу</a>); gpt-oss, DeepSeek и Llama совместимы с MLX, но здесь пока не прогонялись. Логотипы принадлежат их владельцам и показаны только для обозначения совместимости. Модели Anthropic закрытые, поэтому их в сотах нет; поддерживается именно <i>API</i> Anthropic, чтобы Claude Code работал с локальной моделью.</sub></p>

## Зачем basalt

- **Один сервер, два транспорта.** REST отвечает по OpenAI *и* Anthropic (`/v1/messages`), MCP stdio подключается к Claude Desktop и Cursor, а все маршруты генерации умеют стримить по SSE. Все они делят один актор модели — ничего не грузится дважды.
- **Claude Code на локальной модели в две строки.** Задайте `ANTHROPIC_BASE_URL` и запустите `claude` — ваш настоящий API-ключ не затрагивается. См. [интеграцию с Claude Code](#интеграция-с-claude-code).
- **Быстро на ноутбуке.** На Mac с 24 ГБ: ~110 ток/с на Gemma 4 E2B, ~92 ток/с на Qwen3.5-4B, ~37 ток/с на 9B, ~12 ток/с на 27B.
- **Чистый Swift.** Только SwiftPM — без Node, Python и Docker в рантайме.
- **Честные цифры.** В таблице моделей есть и то, что *не заработало* (мусорный вывод, OOM). Лестница бенчмарков запускает код самой модели на входах, которых не было в промпте, — правдоподобный неверный ответ не проходит.

## Демо за 60 секунд

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt          # сборка из исходников, нужен Xcode 26.4+

basalt onboard                              # подберёт модель под ваш Mac и скачает её
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# в другом терминале: Claude Code на вашем собственном GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Локальные модели 4B–9B — не замена топовым облачным: они для приватной, офлайн и бесплатной агентной работы и экспериментов. `basalt fit` покажет, что реально потянет ваша машина.

**Поставьте звезду**, если хотите больше MLX-инструментов на Swift, и напишите в issue свой Mac и ток/с — PR со строкой в таблицу бенчмарков приветствуются.

---

## Требования

| | |
|---|---|
| **macOS** | 15+ (Sequoia) |
| **Xcode / Swift** | 26.4+ / Swift 6.3 |
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` — в Xcode 26 компилятор Metal ставится отдельно, без него ядра MLX не собрать |
| **Железо** | Apple Silicon M1–M4, Unified Memory |
| **Диск** | 2–30 ГБ в зависимости от модели |

---

## Быстрый старт

### Установка через Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

URL в `tap` указывается явно, потому что репозиторий называется не
`homebrew-basalt`. Бутылки нет: формула собирает из исходников, поэтому нужен
Xcode 26.4+ и Metal toolchain — в Xcode 26 он качается отдельно:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Формула проверяет его наличие до сборки и останавливается с этой командой, а не
заваливает вас ошибками компиляции по каждому ядру.

Сборка делает две вещи, и важны обе:

1. `swift build -c release` — сам сервер.
2. `scripts/build_metal.swift` — компилирует Metal-ядра MLX и кладёт
   `mlx.metallib` рядом с бинарём в `libexec`. SwiftPM-сборка mlx-swift не
   содержит metallib, а запасной путь поиска у MLX считается от рабочей
   директории — без этого шага `basalt` работает только из папки, где лежит
   библиотека. Этот шаг и занимает основное время установки.

```bash
brew install --HEAD goodmai/basalt/basalt   # собрать main, а не последний тег
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### Сборка из исходников

```bash
git clone https://github.com/goodmai/basalt
cd basalt

# Сборка
swift build -c release

# Компиляция Metal-ядер MLX. Нужна один раз на каждый checkout: SwiftPM-сборка
# mlx-swift не содержит metallib, и без него первый же вызов инференса падает с
# "Failed to load the default metallib". Скрипт кладёт библиотеку рядом с
# бинарём, поэтому он работает из любой директории.
./scripts/build_metal.swift

# Интерактивный чат
.build/release/basalt chat --model mlx-community/Qwen3.5-4B-4bit

# REST-сервер на :8080 (совместим с OpenAI и Anthropic)
.build/release/basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# MCP stdio сервер (для Claude Desktop / Cursor)
.build/release/basalt serve --model mlx-community/gemma-4-e4b-it-4bit --mcp
```

### Лаунчер одной командой: `./basalt-claude`

В репозитории лежит самодостаточный launcher: он собирает сервер (если нужно), ждёт готовности модели и открывает Claude Code — все переменные окружения живут только внутри этой сессии:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, порт 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # свой порт
./basalt-claude -- --model haiku                            # передать --model haiku в claude
```

`basalt-claude` жёстко ставит `ANTHROPIC_API_KEY=local`, а не наследует переменную, — так настоящий ключ никогда не уходит на локальный сервер и не попадает в payload `--settings`. Другие терминалы не затрагиваются.

---

## Команды

`basalt` без подкоманды запускает `chat`.

| Команда | Что делает |
|---|---|
| `basalt onboard` | мастер первого запуска: подбирает модель под машину и качает её |
| `basalt fit` | читает железо и ранжирует каталог моделей под него |
| `basalt chat --model <id>` | интерактивный чат в терминале |
| `basalt serve --model <id> --rest` | REST-сервер на :8080 (совместим с OpenAI и Anthropic) |
| `basalt serve --model <id> --mcp` | MCP stdio сервер для Claude Desktop / Cursor |
| `basalt serve --model <id> --stats` | телеметрия в stderr каждые 5 с (`--stats-interval`); те же цифры в `GET /api/v1/stats` и в MCP `status` |
| `basalt models list --author <org>` | посмотреть модели автора на HuggingFace |
| `basalt models download <repo-id>` | скачать в общий кэш HF |
| `basalt models info <repo-id>` | размер, квантизация, окно контекста |
| `basalt models cache` | что лежит на диске и сколько занимает |
| `basalt models check <repo-id>` | влезет ли модель в память (ненулевой код выхода, если нет) |
| `basalt models inspect <repo-id>` | проверить, что модель скачана полностью, и измерить её реальную битность |
| `basalt cloud configure` | облачный фолбэк OpenRouter для того, что не влезает локально |
| `basalt cloud cost` | сколько уже потрачено на облако |

Внутри `basalt chat`:

| Ввод | Действие |
|---|---|
| `/clear` | очистить диалог и экран |
| `/color`, `/theme` | переключить цветовую тему терминала |
| `exit`, `quit` | выйти (без слеша) |

Флаги `serve`, которые стоит знать:

| Флаг | Зачем нужен |
|---|---|
| `--reasoning-effort none` | не дать reasoning-модели потратить весь бюджет внутри `<think>` |
| `--reasoning-effort xhigh\|medium\|low` | бюджет размышлений для семейства Qwen |
| `--quant 4bit` | выбрать подпапку с квантизацией в репозиториях, где их несколько |
| `--kv-bits 4\|8` | квантовать KV-кэш — покупает контекст на машине с тесной памятью; выключает переиспользование кэша промпта |
| `--prompt-cache-mb N` | память под префиксное дерево кэша промптов (по умолчанию — половина того, что рекомендованный Metal рабочий набор GPU оставляет после весов; пишется в лог при загрузке; `0` выключает), см. ниже |
| `--cache-blend` | приближённо переиспользовать закэшированные куски промпта в любом порядке (CacheBlend); `--cache-blend-ratio R` — доля пересчитываемых токенов (по умолчанию 0.5), см. ниже |
| `--min-p`, `--top-k`, `--seed` | сэмплирование; `--seed` делает негреди-прогон воспроизводимым; `min_p` запроса перекрывает `--min-p` |
| `--max-tokens` | потолок генерации по умолчанию (2048–128000) |
| `--dry-run` | проверить, влезает ли модель в память, и выйти без загрузки |
| `--port`, `--host` | по умолчанию 127.0.0.1:8080; любой другой хост требует API-ключ (`BASALT_API_KEY` или `--auto-generate-key`) |
| `--rate-limit N` | лимит запросов в минуту на весь сервер, сверх него 429 — по умолчанию выключен на loopback и 120 вне его; `0` выключает; также `rate_limit` в `~/.gem/config.json` |

**Переиспользование кэша промпта.** Сервер хранит KV-кэш модели для недавних промптов в префиксном дереве: промпты, начинающиеся с одних и тех же токенов, делят одну сохранённую копию этого начала, и запрос префиллит только то, что идёт после самого длинного сохранённого префикса. Поэтому следующий ход диалога стоит только своих новых токенов (результат инструмента, дописанный к длинной истории агента, а не вся история заново; на `/v1/messages` блоки инструментов модели пока не передаются, см. #73), а второй диалог с тем же системным промптом и инструментами — субагент Claude Code, другой тред Codex — начинает с общей части, а не префиллит её снова. Сравниваются токены, поэтому идентификатор диалога не нужен (`prompt_cache_key`, `metadata.user_id` и `x-session-id` игнорируются); запросы заголовков и прочие служебные кэш не трогают. В пределах `--prompt-cache-mb` давно не использованная ветка урезается с конца (ветка со снимком гибридной модели, см. ниже, удаляется целиком: её рекуррентное состояние верно только в её конце), так что неактивный диалог дольше всего хранит системный промпт и инструменты, а узел, до которого переиспользует идущий запрос, не выгружают никогда (у гибридной модели это узел со снимком; у запроса своя копия данных, так что выгрузка остальной ветки на него не влияет); кроме того, в дереве не больше 4096 веток, сверх этого выгружается давно не использованная свободная. Usage показывает реально переиспользованное: `prompt_tokens_details.cached_tokens` (Chat Completions), `input_tokens_details.cached_tokens` (Responses), `cache_read_input_tokens` (Anthropic, где `input_tokens` считает только префилленный остаток). Гибридные модели внимание/SSM вроде Qwen3.5/3.6 и Ornith-1.5 тоже переиспользуют предыдущий ход диалога: их рекуррентное состояние нельзя откатить, поэтому при префилле промпта сервер сохраняет его снимок в конце текста запроса, до закрывающих токенов шаблона чата, и следующий ход, продолжающий этот текст, восстанавливает снимок и префиллит только то, что идёт дальше (Ornith-1.5-9B, промпт агента на 12,6 тыс. токенов: второй ход занимает около 1 с вместо примерно 36 с). Переиспользованный ход может разойтись с прогоном без кэша после первых токенов: префилл нарезан на чанки иначе, и результаты в bf16 немного сдвигаются (замер на Ornith: восстановленное состояние побитово совпадает с префиллом с нуля с теми же разрезами, а от однофазного префилла логиты последней позиции отличаются не больше чем на 1,56 при одинаковых топ-5). Переиспользование возможно только от такого снимка, поэтому второй диалог с тем же системным промптом префиллится целиком. Слои со скользящим окном вроде Gemma 3/4 и `--kv-bits` всегда префиллят промпт целиком.

**CacheBlend (по флагу, приближённо).** Префиксное дерево помогает, только если промпт начинается так же; агент, вставивший те же файлы в другом порядке, префиллит их все заново. `--cache-blend` хранит ещё и KV каждого куска промпта — кусок кончается после каждой пустой строки, поэтому файлы и сообщения, разделённые пустыми строками, остаются своими кусками, — и переиспользует кусок, куда бы он ни попал в следующем промпте (ключи хранятся до RoPE и поворачиваются на новую позицию). Куски считались рядом с другим текстом, поэтому их KV устарел: как в CacheBlend (arXiv:2405.16444), первый слой пересчитывается для всех токенов, на втором выбираются токены с наибольшим отклонением KV, и только они — доля `--cache-blend-ratio` от переиспользованных (по умолчанию 0.5) плюс все токены, которых нет в хранилище, — пересчитываются на следующих слоях. Результат приближённый. Замер на Qwen3-4B-4bit (M4 Pro): шесть файлов по ~850 токенов в новом порядке и новый вопрос (5 044 токена из кусков); жадный ответ полного префилла (50 токенов) оценивается с teacher forcing — средний log-prob эталонных токенов и совпадение top-1 по позициям:

| кэш | log-prob | top-1 | время до первого токена |
|---|---|---|---|
| полный префилл | −0.098 | 0.99 | 8.8 с |
| доля 0 | −0.218 | 0.93 | 0.6 с |
| доля 0.15 (как в статье) | −0.209 | 0.93 | 2.0 с |
| **доля 0.5 (по умолчанию)** | −0.127 | 0.96 | 5.7 с |
| доля 1 | −0.098 | 0.99 | 10.7 с |

На этой модели 0.15 из статьи почти не лучше отсутствия пересчёта, поэтому по умолчанию 0.5 (качество близко к полному префиллу, ~1.5× быстрее); `--cache-blend-ratio 0` меняет качество на ~14×. Только плотные модели семейств Llama и Qwen3; остальные идут точным префиксным путём. С флагом бюджет кэша промптов делится пополам между деревом и хранилищем кусков. `cached_tokens` по-прежнему считает только точные токены из дерева; смешанные токены входят в `prompt_tokens` и пишутся в лог (`[CacheBlend] Blended N prompt tokens: R from cached chunks, X of them recomputed`), а в дерево попадает только точное начало смешанного промпта. По умолчанию выключено, потому что меняет ответы; `--prompt-cache-mb 0` выключает и его.

Приватность: все клиенты делят одно префиксное дерево, и `cached_tokens` показывает, сколько начальных токенов промпта совпало с прежними промптами, так что клиент может прощупать, что прислал другой. На общем сервере, где пользователи не должны видеть промпты друг друга, запускайте с `--prompt-cache-mb 0`; с `--cache-blend` ещё и время до первого токена выдаёт, какие куски прислал другой клиент.

---

## Проверенные модели

Модели скачиваются с HuggingFace и кэшируются в `~/.cache/huggingface/hub/`.

```bash
basalt models download mlx-community/Qwen3.5-4B-4bit
basalt models download ornith-ai/Ornith-1.5-9B-MLX-4bit
```

| Модель | Параметры | RAM | Статус на Mac с 24 ГБ |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 ГБ | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 ГБ | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 ГБ | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 ГБ | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 ГБ | ✅ ~37 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 ГБ | ✅ ~48 TPS (reasoning — см. заметку) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 ГБ | ✅ на 24 ГБ не проверялась |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 ГБ | ✅ ~12 TPS (abliterated) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 ГБ | ✅ ~14 TPS (Dynamic Quant) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 ГБ | ✅ ~8 TPS (NVFP4) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (base BF16) | 27B | ~54 ГБ | ❌ нужно 64 ГБ+ |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 ГБ | ❌ мусор на выходе |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 ГБ | ❌ мусор на выходе |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 ГБ | ❌ OOM (нужно 32 ГБ+) |

> **Про Mac с 24 ГБ:**
> - Неквантованные 27B/35B (~54 ГБ весов) не влезают в физическую память. Берите MLX 4-bit или Dynamic Quant — они укладываются в ~15 ГБ.
> - Модели тяжелее 10 ГБ зажимают KV-кэш на длинных диалогах. KV-кэш занимает `2 · слои · kv_heads · head_dim · 2 Б` на токен (fp16), геометрия берётся из `config.json` модели; у гибридных моделей считаются только слои полного внимания плюс фиксированное рекуррентное состояние, слои со скользящим окном перестают расти на размере окна, `--kv-bits 4|8` хранит `bits + 0,5` бита на элемент (`--kv-bits` не действует на слои со скользящим окном). Qwen3-4B: 144 КиБ на токен, 4,5 ГиБ на 32 тыс. токенов; Ornith-1.5-9B (полное внимание в 8 из 32 слоёв): 32 КиБ на токен, 1,05 ГиБ на 32 тыс. `serve --dry-run`, `models check`, `fit` (сетка квантований и оценка скорости декода; колонка RAM и уровень fit по-прежнему считают только веса) и бюджет токенов по памяти считают по этой цифре (потолок = max(--max-tokens, бюджет по памяти); бюджет может только поднять его выше --max-tokens (см. #85)); без скачанного конфига — по оценке от числа параметров, близкой для плотных 7B–32B, но в разы неточной для маленьких, гибридных и MoE-моделей. Это оценки, а не замеры: рекуррентное состояние слоёв Mamba (Jamba, NemotronH) не учитывается, только их слои внимания; MLA-модели (DeepSeek V2/V3) считаются по развёрнутым K и V на голову, которые хранит MLX, а не по сжатому латенту.

> **Про reasoning-модели (Ornith 1.5, Qwen3.x):** шаблон чата открывает `<think>` на каждом ходу. На задачах, где модель уходит в длинные размышления, весь бюджет токенов может уйти в них — ответ не успеет начаться. Лечится флагом `--reasoning-effort none`.

---

## REST API

Базовый URL: `http://127.0.0.1:8080` — **аутентификация не требуется**.

### Управление моделями

```bash
# Список локально закэшированных моделей (формат OpenAI)
curl http://127.0.0.1:8080/v1/models

# Текущая модель и готовность
curl http://127.0.0.1:8080/v1/models/current

# Горячая замена модели на лету (блокируется до загрузки)
curl -s http://127.0.0.1:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-31b-it-4bit"}'
```

`GET /v1/models` возвращает ID с префиксом `claude-local/` — иначе автоподбор моделей в Claude Code их не подхватит. Оригинальный repo ID с HuggingFace лежит в поле `display_name`.

> Обращайтесь по `127.0.0.1`, а не по `localhost`: сервер слушает только IPv4, а `localhost` во многих рантаймах сначала резолвится в `::1`.

### Сырая генерация

```bash
curl -s http://127.0.0.1:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Объясни квантовую запутанность.", "maxTokens": 256}'
```

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `prompt` | string | обязательное | Входной текст |
| `maxTokens` | int | 8192 | Максимум токенов генерации |
| `temperature` | float | 0.7 | Температура сэмплирования (0–2) |
| `topP` | float | 0.9 | Nucleus sampling |
| `minP` | float | сервер | Min-P 0–1: отбрасывает токены с вероятностью ниже `minP × max p`. Перекрывает `--min-p` и профиль модели; `0` выключает |
| `dynamicTemperature` | bool | false | Динамическая температура по энтропии (EDT): на каждом токене `T = T0·0.8^(1.5/H) + 0.5·R` в пределах 0.05–2 (у более холодного запроса нижняя граница — его собственная температура; `top_p`/`min_p` режут исходное распределение), где `H` — энтропия распределения следующего токена, `R ∈ [0,1]` — индекс повторов. На уверенных токенах почти greedy, при начале цикла теплее. При `temperature: 0` не действует |

`/v1/chat/completions`, `/v1/responses` и `/v1/messages` принимают те же поля как `min_p` и `dynamic_temperature`; MCP `gemma_generate` — как `minP` и `dynamicTemperature`.

### Совместимость с OpenAI

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "messages": [{"role": "user", "content": "Привет"}],
    "stream": true
  }'
```

Поддерживаются потоковый SSE, системные промпты и многоходовые диалоги. Если в поле `"model"` передать HuggingFace ID, произойдёт горячая замена модели.

### Совместимость с Anthropic

```bash
curl -s http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Привет"}]
  }'
```

Реализована полная последовательность SSE-событий Anthropic (`message_start`, `content_block_start`, `content_block_delta`, `message_stop`). Принимается и строка, и массив блоков в `content`.

### SSE-стриминг (нативный API)

```bash
curl -N http://127.0.0.1:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Привет", "maxTokens": 512}'
```

Каждое событие — строка `data: {...}`, поток завершается `data: [DONE]`. Ошибка посреди потока приходит как `event: error` с `data: {"error": ...}`.

### Swagger UI

Интерактивная документация — `http://127.0.0.1:8080/swagger`.

---

## Интеграция с Claude Code

### Вариант A — переменные окружения (на сессию терминала)

```bash
# Запускаем Basalt
.build/release/basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# В другом терминале — переменные видит только этот процесс claude
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` уходит как `Authorization: Bearer local` (а не `x-api-key`), поэтому настоящий `ANTHROPIC_API_KEY` не задействуется.

### Вариант B — функция в `~/.zshrc`

```bash
function claude-local() {
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080          \
  ANTHROPIC_AUTH_TOKEN=local                        \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
  ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
  ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
  claude "$@"
}

basalt-claude                   # алиас sonnet → Qwen 4B
claude-local --model haiku     # алиас haiku → Gemma 4B (самая быстрая)
claude-local --model opus      # алиас opus → Gemma 31B (самая сильная)
```

### Вариант C — лаунчер `./basalt-claude`

Поднимает сервер и Claude Code одной командой (см. «Быстрый старт»).

### Обнаружение моделей

Claude Code (v2.1.126+) при старте дёргает `GET /v1/models` и добавляет модели в пикер `/model` — но только если ID начинается с `claude` или `anthropic`. Basalt отдаёт их в форме `claude-local/<hf-id>`, поэтому всё подхватывается само.

---

## Интеграция с OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) — терминальный агент для кода. Basalt отдаёт нативно совместимый OpenAI API (`/v1/chat/completions`, `/v1/models`), так что подключается напрямую, без облака и ключей.

### 1. Запустить Basalt

```bash
# Крупная модель (abliterated / MoE)
.build/release/basalt serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Или быстрая кодовая
.build/release/basalt serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Подключить OpenCode

Переменными окружения:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

Или конфигом (`~/.config/opencode/config.json`):

```json
{
  "provider": "openai",
  "base_url": "http://127.0.0.1:8080/v1",
  "api_key": "local",
  "model": "basalt",
  "temperature": 0.7,
  "max_tokens": 16384
}
```

---

## Скачивание моделей

Basalt умеет искать, показывать и скачивать модели любого автора на Hugging Face:

```bash
# Список моделей по автору
basalt models list --author ornith-ai
basalt models list --author Ex0bit
basalt models list --author mlx-community

# Поиск с фильтром
basalt models list --author Ex0bit --search PRISM

# Скачать конкретную модель
basalt models download ornith-ai/Ornith-1.5-9B-MLX-4bit
basalt models download AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit

# Интерактивный выбор по автору
basalt models download --author Ex0bit
```

---

## MCP (Claude Desktop / Cursor)

Добавьте в конфиг MCP (`~/.config/claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "basalt": {
      "command": "/path/to/basalt",
      "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--mcp"]
    }
  }
}
```

Доступные MCP-инструменты:

| Инструмент | Описание |
|---|---|
| `gemma_generate` | Генерация текста — `prompt`, `maxTokens`, `temperature`, `topP`, `minP`, `dynamicTemperature` |
| `gemma_status` | Готовность сервера, версия, текущая модель |
| `playwright_screenshot` | Скриншот страницы через Playwright |
| `gemma_add_knowledge` | Подмешать свой контекст в сессию |

---

## Структура проекта

```
Sources/                    — один SwiftPM-таргет на каталог; зависимости направлены только вниз
  BasaltCore/               — DTO, ошибки, конфиг, логгер, профайлеры, калькуляторы, HF-кэш, оркестратор
    Chat/                   — ModelProfile: всё, что отличается между семействами моделей
    Config/                 — ServerConfig
  BasaltInference/          — движок инференса MLX, ContextDegradationProfiler, RepetitionGuard (→ Core)
  BasaltCloud/              — облачный фолбэк OpenRouter (CostTracker, ModelRouter)          (→ Core)
  BasaltServer/             — REST (Hummingbird 2.x, middleware) и MCP JSON-RPC поверх stdio  (→ Core)
  BasaltUI/                 — терминальный UI (Markdown, спиннер, прогресс-бар, diff, таблицы) (→ Core)
  BasaltApp/                — SwiftUI REST-клиент, в CLI не линкуется (запуск — #55)        (→ Core)
  BasaltCLI/                — подкоманды и корневая CLI-команда (App/)                       (→ все выше)
  BasaltBin/                — тонкая обёртка-исполняемый файл (продукт `Basalt`, ставится как `basalt`)
  PerformanceBenchmark/     — отдельный CLI для бенчмарков

Tests/BasaltTests/          — юнит- и интеграционные тесты
Formula/basalt.rb             — формула Homebrew
basalt-claude               — лаунчер (сборка + сервер + claude)
scripts/                    — скрипты сборки и обслуживания
docs/                       — расширенная документация
```

SwiftUI/AppKit может импортировать только `BasaltApp`; все таргеты, которые линкует CLI, остаются headless (проверяет `ModuleBoundaryTests`).

---

## Заметки по архитектуре

**Слой чата** — всё, что различается между семействами моделей, живёт за протоколом `ModelProfile` (`Sources/BasaltCore/Chat/ModelProfile.swift`): переменные шаблона чата, маркеры thinking, дефолтные лимиты. Профиль выбирается по `model_type` из `config.json` модели. Добавить семейство — это одна новая конформность и одна строка в реестре.

**Динамическая квантизация и MoE** — `MLXInferenceEngine` читает `config.json`, определяет 2/3/4/8-битную квантизацию и переключается между Dense и Sparse MoE (оптимизированные ядра `SwitchGLU`).

**Изоляция через актор** — `ModelOrchestratorActor` это Swift 6 актор. Все вызовы инференса сериализуются (FIFO) без явных блокировок. MCP и REST делят один экземпляр.

**Горячая замена модели** — `switchModel(to:)` резолвит repo ID в локальном кэше, выгружает текущую модель (`container = nil` + `MLX.GPU.clearCache()`) и грузит новые веса. Актор дожидается завершения запросов в полёте.

**Защита по таймауту** — каждый `generate` и `generateStream` обёрнут в пятиминутный таймаут с кооперативной отменой Task, чтобы зависшая модель не блокировала сервер.

**Стриминг** — тело ответа Hummingbird 2 в виде `AsyncStream<ByteBuffer>` используется для обоих форматов SSE: OpenAI (`data: {...}`) и Anthropic (`event: content_block_delta`).

**Вырезание think-блоков** — в потоковом пути `generateStream` работает конечный автомат, который подавляет `<think>…</think>` до отправки клиенту. В непотоковом пути (`"stream": false`) фильтра нет — размышления приходят внутри `content`.

**Трансляция ID** — `ModelsController` переводит между HuggingFace ID (`mlx-community/Qwen3.5-4B-4bit`) и формой `claude-local/mlx-community--Qwen3.5-4B-4bit`, которую требует фильтр обнаружения моделей в Claude Code. В запросах принимаются обе формы.

---

## Бенчмарки

Два стенда, оба работают против запущенного сервера и дают числа, которые можно
воспроизвести у себя.

**Лестница до Луны** — `benchmarks/ladder/run.py`. Пять ступеней, каждая строго
тяжелее предыдущей, финал — полная миссия Земля → Луна. Оценивается не то, как
ответ выглядит: код модели исполняется на входах, которых не было в промпте,
поэтому правдоподобный, но неверный ответ всё равно падает.

| Ступень | Задача | Что реально проверяется |
|---|---|---|
| 1 | Биквадрат, вещественные корни | замкнутая форма плюс краевые случаи: пусто, кратный, ноль |
| 2 | Биквадрат, комплексные корни | работа с ветвями — все четыре корня, с кратностью |
| 3 | Коэффициенты Фурье | численное интегрирование: пила, меандр, косинус |
| 4 | Напор насоса | единицы (мм против м, л/с против м³/с) + Colebrook без замкнутой формы |
| 5 | **Миссия Земля → Луна** | скорость истечения из `sqrt(2ηQ)`, Циолковский по ступеням, задача четырёх тел, торможение у Луны, мягкая посадка — все константы даны в промпте, из справочника брать нечего |

Верхняя ступень оценивается по шести инвариантам, а не по одному ответу: модель,
которая верно взяла энергетику, но не довела интегрирование траектории, всё
равно получает число. Её промпт лежит в `benchmarks/ladder/lunar_prompt.md`,
эталонные значения в `lunar_task.md` модели не показываются.

`--warmup` добавляет перед лестницей три дешёвых дымовых теста (алгебра,
Фибоначчи, перевод), которые обязана брать любая instruct-модель. В счёт
лестницы они не идут.

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`, официальные правила
pass@2. Корпус не вендорится: сначала положите его в
`benchmarks/arc-agi/dataset/` (см. `benchmarks/arc-agi/ARC_AGI.md`).

### Результаты

Замеры на Mac с 24 ГБ unified memory. `thinking` — это блок `<think>` из шаблона
чата: у reasoning-моделей включён по умолчанию, отключается флагом
`--reasoning-effort none`.

| Модель | Разминка | Лестница | Верхняя ступень | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking выкл) | 3/3 | 1/4 ² | не запускалась | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking вкл) | 2/3 | 0/4 ² | не запускалась | не запускался | ~43 |

Ornith 9B по ступеням, thinking выключен:

| Ступень | Итог | Детали |
|---|---|---|
| 1 · биквадрат вещественный | ✅ | корни верные, включая кратный и пустой случай |
| 2 · биквадрат комплексный | ❌ | для `x⁴ + 1` вернула 2 корня вместо 4 |
| 3 · Фурье | ❌ | `a0 = 10.86` для `f(x) = x` вместо 0 — неверно считает интеграл, формула правильная |
| 4 · напор насоса | ❌ | `v = 0.0298` м/с вместо 1.79 — не перевёл мм в метры |
| 5 · лунная миссия | — | не запускалась |

Разминка, thinking выключен: алгебра ✅ (163 tok, 4 с), Фибоначчи ✅ (757 tok,
16 с), перевод ✅ (303 tok, 7 с — немецкий идиоматичный, французский пересказан,
а не переведён).

² Ступени 1 и 2 замерялись до того, как лестницу разделили на отдельные
промпты: один промпт просил обе функции, вещественная половина оказалась верной,
комплексная — нет. Ступень 5 на этой модели ещё не запускалась.

¹ Колонка ARC-AGI — неполный прогон: три задачи из training (`007bbfb7`,
`00d62c1b`, `017c7c7b`), ни одна не решена по pass@2, 330 с суммарно. Оставшиеся
две не запускались, так что это дымовой тест, а не оценка.

Сами валидаторы тоже проверяются — без модели и GPU:

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 проверок в обе стороны
```

С **включённым** thinking все три тяжёлые задачи и перевод закончились
`finish_reason: length` — бюджет ушёл в блок `<think>`, до ответа дело не дошло.
Это отказ по бюджету, а не потолок способностей, поэтому в таблице два режима
разведены.

### Как воспроизвести

```bash
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

Добавить модель в таблицу — это один прогон каждого стенда. PR с новыми
строками приветствуются.

---

## Зависимости

Всё — пакеты SwiftPM; ни CocoaPods, ни Node, ни Python в рантайме.
`Package.resolved` фиксирует точные ревизии.

| Пакет | Для чего |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | слой массивов и NN на Metal |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | архитектуры моделей и загрузка — `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | токенизаторы и шаблоны чата |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | HTTP-сервер |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | CLI |
| [Rainbow](https://github.com/onevcat/Rainbow) | цвет и вёрстка в терминале |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | рендер Markdown и подсветка синтаксиса |

`MLXVLM` здесь не опционален: Gemma 4 и чекпойнты Ornith объявляют
`*ForConditionalGeneration`, а эти архитектуры зарегистрированы именно там, а не
в `MLXLLM`.

Обновления приходят PR-ами от Dependabot — еженедельно для SwiftPM, ежемесячно
для GitHub Actions. mlx-swift развивается быстро, и нужная архитектура модели
часто оказывается в одном релизе от нас — отсюда недельный ритм.

---

## Как поучаствовать

Issues и pull request'ы приветствуются. Перед PR:

```bash
swift build          # должно собираться
swift test           # должно оставаться зелёным
```

Никогда не коммитьте креды, токены и состояние `*.sqlite3` — `.gitignore` закрывает обычных подозреваемых, а релизный workflow прогоняет скан секретов по дереву.

---

## Лицензия

[MIT](LICENSE) © 2026 goodmai

Веса моделей этой лицензией **не** покрываются — у каждой модели на HuggingFace свои условия.
