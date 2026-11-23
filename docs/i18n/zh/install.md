# 安装

> Languages: [English](../../install.md) · [Русский](../ru/install.md) · **中文** · [Español](../es/install.md) · [Français](../fr/install.md) · [日本語](../ja/install.md) · [Deutsch](../de/install.md) · [한국어](../ko/install.md) · [Bahasa Indonesia](../id/install.md)

## 预编译二进制（两行命令，无需 Xcode）

只需 macOS 27+ 与 Apple Silicon。压缩包内 `basalt`、`mlx.metallib` 与资源包并排放置，请保持在同一目录。

```bash
curl -fsSL https://github.com/goodmai/basalt/releases/latest/download/basalt-macos-arm64.tar.gz | tar -xz
./basalt-macos-arm64/basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --rest
```

## 60 秒快速体验

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

本地 4B–9B 模型并不能直接替代前沿大模型——它们适用于私密、离线、零成本的智能体工作流以及实验探索。运行 `basalt fit` 可以查看你的设备实际能够运行的模型。

## 系统要求

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Metal 工具链** | `xcodebuild -downloadComponent MetalToolchain` —— Xcode 单独分发 Metal 编译器，缺少它时无法编译 MLX 的内核 |
| **硬件** | Apple Silicon M1–M4，统一内存 |
| **磁盘空间** | 2–30 GB（取决于所选模型） |

---

## 快速开始

### 通过 Homebrew 安装

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

由于仓库名称不是 `homebrew-basalt`，因此 `tap` 命令需要显式指定完整 URL。在 Homebrew 7+ 上，需要通过 `brew trust goodmai/basalt` 授权该第三方 tap。此 formula 不提供预编译 bottle，会从源码构建，因此必须已安装 Xcode 27.0+（macOS 27+）以及需单独下载的 Metal 工具链：

```bash
xcodebuild -downloadComponent MetalToolchain
```

formula 会在构建前检查该工具链是否存在；如果缺失，会直接提示上述命令并中止构建，而不会让你淹没在逐个内核的编译报错中。

构建过程包含两个关键步骤，缺一不可：

1. `swift build -c release` —— 编译服务端程序本身。
2. `scripts/build_metal.swift` —— 编译 MLX 的 Metal 内核，并将 `mlx.metallib` 放置在 `libexec` 中可执行文件旁。通过 SwiftPM 构建的 mlx-swift 不会附带完整的 metallib，而 MLX 的回退查找路径是相对于当前工作目录的；如果没有这一步，`basalt` 就只能在包含该库文件的目录下运行。安装过程中的大部分时间通常会花在这一步。

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### 从源码构建

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

### 一键启动器：`./basalt-claude`

仓库中自带一个独立的 Swift 启动脚本，它会在需要时自动构建服务端、等待服务就绪，然后打开 Claude Code——所有本地环境变量仅作用于当前会话：

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` 会显式固定 `ANTHROPIC_API_KEY=local` 而非继承现有环境变量，因此真实的密钥绝不会被转发到本地服务端，也不会写入 `--settings` 负载中。其他终端会话完全不受影响。
