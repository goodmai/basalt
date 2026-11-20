# Instalación

> Languages: [English](../../install.md) · [Русский](../ru/install.md) · [中文](../zh/install.md) · **Español** · [Français](../fr/install.md) · [日本語](../ja/install.md) · [Deutsch](../de/install.md)

## Binario precompilado (dos líneas, sin Xcode)

Bastan macOS 27+ y Apple Silicon. El archivo trae `basalt`, `mlx.metallib` y los paquetes de recursos juntos; mantenlos en una sola carpeta.

```bash
curl -fsSL https://github.com/goodmai/basalt/releases/latest/download/basalt-macos-arm64.tar.gz | tar -xz
./basalt-macos-arm64/basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --rest
```

## Demostración en 60 segundos

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Los modelos locales de 4B–9B no son un sustituto directo de un modelo de frontera: están pensados para trabajo agéntico privado, sin conexión y sin coste, así como para experimentar. `basalt fit` te indica qué puede ejecutar realmente tu equipo.

## Requisitos

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Cadena de herramientas de Metal** | `xcodebuild -downloadComponent MetalToolchain` — Xcode distribuye el compilador de Metal por separado y los kernels de MLX no se pueden compilar sin él |
| **Hardware** | Apple Silicon M1–M4, memoria unificada |
| **Disco** | 2–30 GB según el modelo |

---

## Inicio rápido

### Instalar con Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

El tap necesita la URL explícita porque el repositorio no se llama
`homebrew-basalt`. En Homebrew 7+, `brew trust goodmai/basalt` autoriza el
tap de terceros. No hay bottle: la fórmula se compila desde el código fuente, por lo que Xcode
27.0+ (macOS 27+) debe estar instalado, junto con la cadena de herramientas de Metal — una
descarga independiente:

```bash
xcodebuild -downloadComponent MetalToolchain
```

La fórmula comprueba si está presente antes de compilar y se detiene mostrando ese comando si
falta, en lugar de abrumarte con errores de compilación por cada kernel.

La compilación realiza dos cosas, y ambas son importantes:

1. `swift build -c release` — el servidor en sí.
2. `scripts/build_metal.swift` — compila los kernels de Metal de MLX y coloca
   `mlx.metallib` junto al binario en `libexec`. Una compilación con SwiftPM de mlx-swift
   no incluye un metallib completo, y la búsqueda de respaldo de MLX es relativa al directorio
   de trabajo, por lo que sin este paso `basalt` solo funciona desde el directorio que contiene
   la biblioteca. Ten en cuenta que este paso dominará el tiempo de instalación.

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### Compilar desde el código fuente

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

### Lanzador con un solo comando: `./basalt-claude`

El repositorio incluye un lanzador autónomo en Swift que compila el servidor (si es necesario), espera a que esté listo y abre Claude Code, todo ello con variables de entorno locales limitadas a esa sesión:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` fija `ANTHROPIC_API_KEY=local` en lugar de heredarlo, de modo que nunca se reenvía una clave real al servidor local ni se escribe en la carga útil de `--settings`. Las demás terminales no se ven afectadas.
