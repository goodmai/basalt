# インストール

> Languages: [English](../../install.md) · [Русский](../ru/install.md) · [中文](../zh/install.md) · [Español](../es/install.md) · [Français](../fr/install.md) · **日本語** · [Deutsch](../de/install.md)

## 60秒デモ

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

ローカルの 4B〜9B モデルはフロンティアモデルの完全な代替ではありません。プライバシー保護、オフライン動作、コストゼロのエージェント処理や試行錯誤に向いています。`basalt fit` を使えば、お使いのマシンで実際に動作するモデルを確認できます。

## 動作要件

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Metal ツールチェーン** | `xcodebuild -downloadComponent MetalToolchain` — Xcode は Metal コンパイラを別配布しており、これがないと MLX のカーネルをビルドできません |
| **ハードウェア** | Apple Silicon M1–M4、ユニファイドメモリ |
| **ディスク** | モデルに応じて 2〜30 GB |

---

## クイックスタート

### Homebrew でインストール

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

リポジトリ名が `homebrew-basalt` ではないため、tap には明示的な URL の指定が必要です。Homebrew 7+ では、`brew trust goodmai/basalt` でサードパーティ tap を信頼・許可します。bottle は用意されていないため、formula はソースからビルドされます。そのため Xcode 27.0+（macOS 27+）と、別途ダウンロードが必要な Metal ツールチェーンがインストールされている必要があります。

```bash
xcodebuild -downloadComponent MetalToolchain
```

formula はビルド前にこのコンポーネントの有無を確認し、見つからない場合はカーネルごとのコンパイルエラーを大量に出すのではなく、このコマンドを表示して停止します。

ビルドでは次の 2 つの処理が行われ、どちらも重要です：

1. `swift build -c release` — サーバー本体をビルドします。
2. `scripts/build_metal.swift` — MLX の Metal カーネルをコンパイルし、`mlx.metallib` を `libexec` 内のバイナリの隣に配置します。mlx-swift の SwiftPM ビルドには完全な metallib が同梱されておらず、MLX のフォールバック探索は作業ディレクトリからの相対パスで行われるため、この手順がないと `basalt` はライブラリを含むディレクトリからしか実行できません。インストール時間の大部分はこのステップが占めます。

```bash
brew install --HEAD goodmai/basalt/basalt   # build main instead of the last tag
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### ソースからビルド

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

### ワンコマンドランチャー: `./basalt-claude`

リポジトリには、必要に応じてサーバーをビルドし、起動完了を待機してから Claude Code を開く自己完結型の Swift ランチャーが含まれています。ローカル環境変数はすべてそのセッション内のみにスコープされます：

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # custom port
./basalt-claude -- --model haiku                            # pass --model haiku to claude
```

`basalt-claude` は環境変数を継承せず `ANTHROPIC_API_KEY=local` を固定で設定するため、実際の API キーがローカルサーバーに転送されたり `--settings` ペイロードに書き込まれたりすることはありません。他のターミナルには影響しません。
