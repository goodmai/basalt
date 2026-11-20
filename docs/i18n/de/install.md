# Installation

> Languages: [English](../../install.md) · [Русский](../ru/install.md) · [中文](../zh/install.md) · [Español](../es/install.md) · [Français](../fr/install.md) · [日本語](../ja/install.md) · **Deutsch**

## Vorkompiliertes Binary (zwei Zeilen, ohne Xcode)

macOS 27+ und Apple Silicon genügen. Das Archiv enthält `basalt`, `mlx.metallib` und die Ressourcen-Bundles nebeneinander; lasst sie in einem Ordner.

```bash
curl -fsSL https://github.com/goodmai/basalt/releases/latest/download/basalt-macos-arm64.tar.gz | tar -xz
./basalt-macos-arm64/basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --rest
```

## 60-Sekunden-Demo

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Lokale 4B–9B-Modelle sind kein direkter Ersatz für ein Frontier-Modell — sie sind für private, kostenlose Agenten-Workflows offline sowie zum Experimentieren gedacht. `basalt fit` zeigt Ihnen, was auf Ihrem Rechner tatsächlich läuft.

## Voraussetzungen

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Metal-Toolchain** | `xcodebuild -downloadComponent MetalToolchain` — Xcode liefert den Metal-Compiler separat aus, und ohne ihn lassen sich die Kernel von MLX nicht bauen |
| **Hardware** | Apple Silicon M1–M4, Unified Memory |
| **Speicherplatz** | 2–30 GB je nach Modell |

---

## Schnellstart

### Installation mit Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

Der Tap benötigt die explizite URL, da das Repository nicht
`homebrew-basalt` heißt. Ab Homebrew 7+ autorisiert `brew trust goodmai/basalt`
den Drittanbieter-Tap. Es gibt kein vorgefertigtes Bottle: Die Formula baut aus dem Quellcode,
daher müssen Xcode 27.0+ (macOS 27+) sowie die Metal-Toolchain installiert sein — ein
separater Download:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Die Formula prüft vor dem Build, ob sie vorhanden ist, und bricht bei Fehlen mit diesem
Befehl ab, statt Sie mit Compiler-Fehlern für jeden einzelnen Kernel zu überfluten.

Der Build erledigt zwei Dinge, und beide sind entscheidend:

1. `swift build -c release` — der Server selbst.
2. `scripts/build_metal.swift` — kompiliert die Metal-Kernel von MLX und legt
   `mlx.metallib` neben das Binary in `libexec`. Ein SwiftPM-Build von mlx-swift
   liefert keine vollständige metallib mit, und die Fallback-Suche von MLX erfolgt relativ
   zum aktuellen Arbeitsverzeichnis; ohne diesen Schritt läuft `basalt` daher nur aus dem
   Verzeichnis heraus, in dem sich die Bibliothek befindet. Rechnen Sie damit, dass dieser
   Schritt den Großteil der Installationszeit in Anspruch nimmt.

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### Aus dem Quellcode bauen

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

### Ein-Befehl-Launcher: `./basalt-claude`

Das Repository enthält einen eigenständigen Swift-Launcher, der bei Bedarf den Server baut, auf dessen Bereitschaft wartet und Claude Code öffnet — alles mit lokalen Umgebungsvariablen, die auf diese eine Sitzung beschränkt sind:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` setzt `ANTHROPIC_API_KEY=local`, statt den Wert zu übernehmen, sodass niemals ein echter Schlüssel an den lokalen Server weitergeleitet oder in die `--settings`-Nutzdaten geschrieben wird. Andere Terminals bleiben unberührt.
