# 설치

> Languages: [English](../../install.md) · [Русский](../ru/install.md) · [中文](../zh/install.md) · [Español](../es/install.md) · [Français](../fr/install.md) · [日本語](../ja/install.md) · [Deutsch](../de/install.md) · **한국어** · [Bahasa Indonesia](../id/install.md)

## 사전 빌드 바이너리 (2줄, Xcode 불필요)

macOS 27+와 Apple Silicon만 있으면 충분합니다. 아카이브에는 `basalt`, `mlx.metallib`, 리소스 번들이 나란히 들어 있으므로 같은 폴더에 두세요.

```bash
curl -fsSL https://github.com/goodmai/basalt/releases/latest/download/basalt-macos-arm64.tar.gz | tar -xz
./basalt-macos-arm64/basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --rest
```

## 60초 데모

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

로컬 4B–9B 모델은 프런티어 모델의 완전한 대체재가 아니라, 프라이빗·오프라인·비용 0 에이전트 작업과 실험을 위한 것입니다. `basalt fit`으로 현재 Mac에서 실제로 실행 가능한 모델을 확인할 수 있습니다.

## 요구 사항

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Metal 툴체인** | `xcodebuild -downloadComponent MetalToolchain` — Xcode는 Metal 컴파일러를 별도로 배포하며, 없으면 MLX 커널을 빌드할 수 없습니다 |
| **하드웨어** | Apple Silicon M1–M4, 통합 메모리 (Unified Memory) |
| **디스크** | 모델에 따라 2–30 GB |

---

## 빠른 시작

### Homebrew로 설치

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

저장소 이름이 `homebrew-basalt`가 아니므로 명시적 URL이 필요합니다. Homebrew 7+에서는 `brew trust goodmai/basalt`로 서드파티 탭을 승인하세요. 미리 빌드된 bottle은 없으며 소스에서 빌드하므로 Xcode 27.0+(macOS 27+)와 별도 Metal 툴체인이 필요합니다:

```bash
xcodebuild -downloadComponent MetalToolchain
```

빌드는 두 단계로 진행됩니다:

1. `swift build -c release` — 서버 바이너리 빌드.
2. `scripts/build_metal.swift` — MLX Metal 커널을 컴파일해 `libexec`의 바이너리 옆에 `mlx.metallib`를 배치합니다.

```bash
brew install --HEAD goodmai/basalt/basalt   # 마지막 태그 대신 main 빌드
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### 소스에서 빌드

```bash
git clone https://github.com/goodmai/basalt
cd basalt

swift build -c release
./scripts/build_metal.swift
BIN="$(swift build -c release --show-bin-path)/Basalt"

# 대화형 채팅
"$BIN" chat --model mlx-community/Qwen3.5-4B-4bit

# :8080 REST 서버 (OpenAI + Anthropic 호환)
"$BIN" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# MCP stdio 서버 (Claude Desktop / Cursor용)
"$BIN" serve --model mlx-community/gemma-4-e4b-it-4bit --mcp
```

### 원커맨드 런처: `./basalt-claude`

저장소에 포함된 Swift 런처가 필요 시 서버를 빌드하고 준비 완료를 기다린 뒤, 해당 세션에만 적용되는 로컬 환경 변수로 Claude Code를 실행합니다:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, 포트 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # 커스텀 포트
./basalt-claude -- --model haiku                            # claude에 --model haiku 전달
```

`basalt-claude`는 실제 키를 상속하지 않고 `ANTHROPIC_API_KEY=local`로 고정하므로 실제 API 키가 로컬 서버나 `--settings`로 유출되지 않습니다.
