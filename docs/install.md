# Install

## 60-second demo

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Local 4B–9B models are not a drop-in replacement for a frontier model — they are for private, offline, zero-cost agentic work and for tinkering. `basalt fit` tells you what your machine can actually run.

## Requirements

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` — Xcode ships the Metal compiler separately, and MLX's kernels cannot be built without it |
| **Hardware** | Apple Silicon M1–M4, Unified Memory |
| **Disk** | 2–30 GB depending on model |

---

## Quick Start

### Install with Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

The tap needs the explicit URL because the repository is not named
`homebrew-basalt`. On Homebrew 7+, `brew trust goodmai/basalt` authorizes the
third-party tap. There is no bottle: the formula builds from source, so Xcode
27.0+ (macOS 27+) must be installed, along with the Metal toolchain — a
separate download:

```bash
xcodebuild -downloadComponent MetalToolchain
```

The formula checks for it before building and stops with that command if it is
missing, rather than burying you in per-kernel compiler errors.

The build does two things, and both matter:

1. `swift build -c release` — the server itself.
2. `scripts/build_metal.swift` — compiles MLX's Metal kernels and puts
   `mlx.metallib` next to the binary in `libexec`. A SwiftPM build of mlx-swift
   ships no complete metallib, and MLX's fallback lookup is relative to the working
   directory, so without this step `basalt` only runs from the directory holding
   the library. Expect this step to dominate the install time.

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### Build from source

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

### One-command launcher: `./basalt-claude`

The repo includes a self-contained Swift launcher that builds the server (if needed), waits for readiness, and opens Claude Code — all with local env vars scoped to that session:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` pins `ANTHROPIC_API_KEY=local` rather than inheriting it, so a real key is never forwarded to the local server or written into the `--settings` payload. Other terminals are untouched.
