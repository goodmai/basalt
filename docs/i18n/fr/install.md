# Installation

> Languages: [English](../../install.md) · [Русский](../ru/install.md) · [中文](../zh/install.md) · [Español](../es/install.md) · **Français** · [日本語](../ja/install.md) · [Deutsch](../de/install.md)

## Démo en 60 secondes

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Les modèles locaux de 4B à 9B ne remplacent pas directement un modèle de pointe — ils sont conçus pour des workflows agentiques privés, hors ligne et sans coût, ainsi que pour l'expérimentation. `basalt fit` vous indique ce que votre machine est réellement capable d'exécuter.

## Prérequis

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Chaîne d'outils Metal** | `xcodebuild -downloadComponent MetalToolchain` — Xcode distribue le compilateur Metal séparément, et les noyaux MLX ne peuvent pas être compilés sans lui |
| **Matériel** | Apple Silicon M1–M4, mémoire unifiée |
| **Disque** | 2–30 Go selon le modèle |

---

## Démarrage rapide

### Installer avec Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

Le tap nécessite l'URL explicite car le dépôt ne s'appelle pas
`homebrew-basalt`. Sur Homebrew 7+, `brew trust goodmai/basalt` autorise le
tap tiers. Il n'y a pas de bottle précompilée : la formule compile depuis les sources, donc Xcode
27.0+ (macOS 27+) doit être installé, ainsi que la chaîne d'outils Metal — un
téléchargement distinct :

```bash
xcodebuild -downloadComponent MetalToolchain
```

La formule vérifie sa présence avant la compilation et s'arrête en affichant cette commande si elle est
absente, plutôt que de vous noyer sous des erreurs de compilation noyau par noyau.

La compilation effectue deux opérations, toutes deux essentielles :

1. `swift build -c release` — le serveur lui-même.
2. `scripts/build_metal.swift` — compile les noyaux Metal de MLX et place
   `mlx.metallib` à côté du binaire dans `libexec`. Une compilation SwiftPM de mlx-swift
   ne fournit pas de metallib complète, et la recherche de secours de MLX est relative au répertoire
   de travail ; sans cette étape, `basalt` ne fonctionne que depuis le répertoire contenant
   la bibliothèque. Attendez-vous à ce que cette étape représente la majeure partie du temps d'installation.

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### Compiler depuis les sources

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

### Lanceur en une commande : `./basalt-claude`

Le dépôt inclut un lanceur Swift autonome qui compile le serveur (si nécessaire), attend qu'il soit prêt et ouvre Claude Code — le tout avec des variables d'environnement locales limitées à cette session :

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` fixe `ANTHROPIC_API_KEY=local` au lieu d'en hériter, afin qu'une vraie clé ne soit jamais transmise au serveur local ni écrite dans la charge utile `--settings`. Les autres terminaux restent intacts.
