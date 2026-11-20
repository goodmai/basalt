# Установка

> Languages: [English](../../install.md) · **Русский** · [中文](../zh/install.md) · [Español](../es/install.md) · [Français](../fr/install.md) · [日本語](../ja/install.md) · [Deutsch](../de/install.md)

## Демо за 60 секунд

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Локальные модели 4B–9B не заменяют frontier-модели напрямую — они предназначены для приватной, автономной, бесплатной агентной работы и экспериментов. Команда `basalt fit` покажет, что именно может запустить ваш компьютер.

## Требования

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` — Xcode поставляет компилятор Metal отдельно, и без него ядра MLX не соберутся |
| **Оборудование** | Apple Silicon M1–M4, Unified Memory |
| **Диск** | 2–30 ГБ в зависимости от модели |

---

## Быстрый старт

### Установка через Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

Явный URL для tap нужен потому, что репозиторий называется не
`homebrew-basalt`. В Homebrew 7+ команда `brew trust goodmai/basalt` авторизует
сторонний tap. Готовых bottle-пакетов нет: формула собирается из исходников,
поэтому должны быть установлены Xcode 27.0+ (macOS 27+) и Metal toolchain —
отдельный загружаемый компонент:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Формула проверяет его наличие перед сборкой и при отсутствии останавливается,
выводя эту команду, вместо того чтобы засыпать вас ошибками компиляции для каждого
ядра.

Сборка выполняет два действия, и оба важны:

1. `swift build -c release` — сам сервер.
2. `scripts/build_metal.swift` — компилирует Metal-ядра MLX и помещает
   `mlx.metallib` рядом с бинарным файлом в `libexec`. Сборка mlx-swift через
   SwiftPM не поставляет полный metallib, а резервный поиск MLX привязан к
   текущему рабочему каталогу, поэтому без этого шага `basalt` будет запускаться
   только из каталога, содержащего библиотеку. Учтите, что этот шаг занимает
   большую часть времени установки.

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### Сборка из исходников

```bash
git clone https://github.com/goodmai/basalt
cd basalt

# Build
swift build -c release

# Compile the MLX Metal kernels. Required once per checkout: a SwiftPM build of
# mlx-swift ships no complete metallib, and without it the first inference call
# fails with "Failed to load the default metallib". The script places it next to
# the built binary, so the binary works from any directory.
./scripts/build_metal.swift
BIN="$(swift build -c release --show-bin-path)/Basalt"

# Interactive chat
"$BIN" chat --model mlx-community/Qwen3.5-4B-4bit

# REST server on :8080 (OpenAI + Anthropic compatible)
"$BIN" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# MCP stdio server (for Claude Desktop / Cursor)
"$BIN" serve --model mlx-community/gemma-4-e4b-it-4bit --mcp
```

### Запуск одной командой: `./basalt-claude`

В репозиторий включён автономный Swift-лаунчер, который собирает сервер (при необходимости), ожидает готовности и запускает Claude Code — всё с локальными переменными окружения, ограниченными этой сессией:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` жёстко задаёт `ANTHROPIC_API_KEY=local` вместо наследования переменной из окружения, поэтому настоящий ключ никогда не передаётся локальному серверу и не записывается в полезную нагрузку `--settings`. Другие терминалы при этом не затрагиваются.
