# Instalasi

> Languages: [English](../../install.md) · [Русский](../ru/install.md) · [中文](../zh/install.md) · [Español](../es/install.md) · [Français](../fr/install.md) · [日本語](../ja/install.md) · [Deutsch](../de/install.md) · [한국어](../ko/install.md) · **Bahasa Indonesia**

## Biner siap pakai (dua baris, tanpa Xcode)

Cukup macOS 27+ dan Apple Silicon. Arsip berisi `basalt`, `mlx.metallib`, dan bundel sumber daya secara berdampingan; simpan semuanya dalam satu folder.

```bash
curl -fsSL https://github.com/goodmai/basalt/releases/latest/download/basalt-macos-arm64.tar.gz | tar -xz
./basalt-macos-arm64/basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --rest
```

## Demo 60 detik

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds from source, needs Xcode 27.0+ (macOS 27+)

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model mlx-community/Qwen3.5-4B-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

Model lokal 4B–9B bukan pengganti langsung model frontier — melainkan untuk pekerjaan agen privat, luring, bebas biaya, dan eksperimen. `basalt fit` menunjukkan apa yang benar-benar bisa dijalankan Mac Anda.

## Persyaratan

| | |
|---|---|
| **macOS** | 27+ |
| **Xcode / Swift** | 27.0+ / Swift 6.4 |
| **Metal toolchain** | `xcodebuild -downloadComponent MetalToolchain` — Xcode mendistribusikan kompilator Metal secara terpisah dan kernel MLX membutuhkannya |
| **Perangkat keras** | Apple Silicon M1–M4, Unified Memory |
| **Disk** | 2–30 GB tergantung model |

---

## Mulai Cepat

### Instal dengan Homebrew

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew install goodmai/basalt/basalt
```

URL eksplisit diperlukan karena repositori tidak bernama `homebrew-basalt`. Di Homebrew 7+, jalankan `brew trust goodmai/basalt` untuk mengizinkan tap pihak ketiga. Formula dibangun dari sumber sehingga membutuhkan Xcode 27.0+ (macOS 27+) dan Metal toolchain:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Proses build melakukan dua hal penting:

1. `swift build -c release` — membangun server itu sendiri.
2. `scripts/build_metal.swift` — mengompilasi kernel Metal MLX dan menempatkan `mlx.metallib` di samping biner pada `libexec`.

```bash
brew install --HEAD goodmai/basalt/basalt   # bangun dari main alih-alih tag terakhir
brew upgrade goodmai/basalt/basalt
brew uninstall basalt && brew untap goodmai/basalt
```

### Bangun dari sumber

```bash
git clone https://github.com/goodmai/basalt
cd basalt

swift build -c release
./scripts/build_metal.swift
BIN="$(swift build -c release --show-bin-path)/Basalt"

# Obrolan interaktif
"$BIN" chat --model mlx-community/Qwen3.5-4B-4bit

# Server REST di :8080 (kompatibel OpenAI + Anthropic)
"$BIN" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# Server stdio MCP (untuk Claude Desktop / Cursor)
"$BIN" serve --model mlx-community/gemma-4-e4b-it-4bit --mcp
```

### Peluncur satu perintah: `./basalt-claude`

Repositori menyertakan peluncur Swift mandiri yang membangun server (bila perlu), menunggu siap, lalu membuka Claude Code dengan variabel lingkungan lokal khusus sesi tersebut:

```bash
chmod +x ./basalt-claude

./basalt-claude                                             # Qwen 4B, port 8080
./basalt-claude --model mlx-community/gemma-4-31b-it-4bit  # Gemma 4 31B
./basalt-claude --port 8081                                 # port kustom
./basalt-claude -- --model haiku                            # teruskan --model haiku ke claude
```

`basalt-claude` mengunci `ANTHROPIC_API_KEY=local` alih-alih mewarisinya, sehingga kunci API asli Anda tidak pernah dikirim ke server lokal.
