<p align="center"><img src="images/logo.svg" alt="basalt logo" width="128"></p>

# basalt — あなたの Mac がそのまま AI ラボになる。

> Apple Silicon 上で、オープンウェイト LLM、ニューラル音声、動画、OCR、画像モデルを推論サービング、学習、量子化、構成、解析できます。単一の Swift 6.4 バイナリが **OpenAI API、Anthropic API、MCP** に対応し、その上にネイティブの macOS 27 Liquid Glass アプリを備えています。実行時に Python は不要。明示的に指定しない限りクラウド通信も一切行いません。

[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md) · [Español](README.es.md) · [Français](README.fr.md) · **日本語** · [Deutsch](README.de.md) · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.11.0](https://img.shields.io/badge/release-v0.11.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.11.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2855%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/i18n/ja/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/i18n/ja/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/i18n/ja/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/i18n/ja/benchmarks.md#benchmarks)

<p align="center"><img src="images/models-hive.svg" alt="basalt で動作するオープンモデル: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Apple Silicon 上でオープンウェイトモデルを実行します。ここでロジット検証済みのモデル: Gemma 4、Qwen3.8、LFM2.5（下の表）。<a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a> で実測・パッケージ済みのモデル: Ornith、Bonsai、Qwen3.6-35B-A3B、DeepSeek-OCR-2、OpenJev、Laya（<a href="docs/i18n/ja/models.md#verified-models">モデル一覧表</a>を参照）。gpt-oss と Llama は MLX 互換です。各ロゴはそれぞれの所有者の商標であり、互換性を示す目的でのみ掲載しています。Anthropic のモデルはクローズドウェイトのため一覧には含まれていませんが、Claude Code がローカルモデルと通信できるよう Anthropic <i>API</i> に対応しています。</sub></p>

**1 つのリポジトリに 5 つの機能を集約:** 既存ツールがそのまま接続できるローカル推論サーバー、LoRA 蒸留および量子化後チューニング、マルチモデル構成のための docker-compose 相当のオーケストレーション、モデル内部を可視化するライブ X-ray 顕微鏡、そしてワンキー操作で呼び出せるネイティブ Mac アプリを提供します。

**推論パスの変更ごとに参照ロジットと照合:** [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) は記録済みプロンプトを再生し、greedy トークンまたはロジットが 1 つでもずれるとテストが失敗します。dense および MoE 向けの qwen3、qwen3_5、gemma4、lfm2、nemotron_h 各ファミリーの参照ロジットは [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits) に格納されています。

| Golden テスト済みモデル | クラス | フィクスチャ |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | スライディングウィンドウ + フルアテンション、MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | ハイブリッド線形/フルアテンション、dense | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | ハイブリッド畳み込み + アテンション、MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) からワンクリック起動（検証済みモデル 55 種以上）

[`goodmai`](https://huggingface.co/goodmai) 配下の各モデルカードには **Run with Basalt** ボタン（[`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en)）があり、`basalt://run?model=...` 経由で `Basalt.app` を起動するか、正確な `basalt serve` / `basalt compose` CLI コマンドを表示します:

| モデル / スタック | 用途と 24 GB Mac でのメモリ使用量 | 起動 |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT 普段使い向けチャット & ビジョン（約 7.6 GB、Ladder 4/5 + Lunar 6/6） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE（`A4B`）、Golden 検証済み（14.9 GB、56–63 tok/s、Ladder 4/5） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 24 GB Mac 向け 31B dense PV-MACaQ 3.2 bpw（15.1 GB、14.1 tok/s、Ladder 4/5 + Lunar 4/6） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B dense PV-MACaQ 4-bit（15.3 GB、p50 11.6 tok/s、PPL 1.23、MSE −19.5%、Ladder 4/5） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE（`A3B`）3-bit MACaQ（14.6 GB、`#628` スイッチ WHT により 64–75 tok/s） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw トレリス量子化（10.0 GB、16k PPL 4.621） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO ネイティブ i-quant（11.8 GB、PPL 4.17、System One 300/300） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 37.7 GB の 35B MoE を SSD から 8.77 tok/s、スワップ 0 でストリーミング実行 | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 7.4 GB の 27B Hadamard 三値モデル、`--mtp` 有効時 27.1 tok/s | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B ハイブリッド畳み込み + アテンション MoE、Golden 検証済み（約 12.6 GB） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | 43 言語対応 44.1 kHz ニューラル TTS（2.24 GB、RTF 0.23–0.26） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | 公式 768 タイリング対応ビジョンエンコーダ OCR（`/v1/pipeline`） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | 9B ローカル System One 判定器（精度 98.9%、LLM トークン −77%） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/i18n/ja/compose.md#bundled-hardware-tiered-profiles-526) | 推奨 24 GB Compose スタック（Chat + ZONOS3 + Kokoro ANE） | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

---

## 推論: 40B MoE を 67 tok/s、27B を 7.4 GB、37.7 GB MoE をスワップ 0 で — 使い慣れた API の裏側で高速動作

- **1 行で Claude Code を自分の GPU 上で実行。** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — 本番の API キーには一切触れません。OpenAI SDK、OpenCode、Continue、Cline、aider は OpenAI API 経由で接続でき、Claude Desktop と Cursor は MCP 経由で接続できます（ツール単位・グループ単位の `--mcp-enable` / `--mcp-disable` に対応、[#488](https://github.com/goodmai/basalt/pull/488)）。
- **ノート PC でも高速。** M4 Pro 24 GB 環境で: Recursive `Flint-40B-A2.5B`（`oQ3e`）が 17.61 GB で 67.02 tok/s のデコード速度を達成（[#522](https://github.com/goodmai/basalt/pull/522)）。`Ornith-1.5-35B-A3B-MACaQ-3bit` はシングルパス `MACaQSwitchLinear` WHT + `gatherQuantizedMM` カーネル（MoE カーネル `3.41×` 高速化、Gemma 4 26B-A4B のエンドツーエンド `+82%`、[#628](https://github.com/goodmai/basalt/pull/628)）により 14.6 GB で 64–75 tok/s で動作。`Qwen3.5-9B-PV-MACaQ-4bit` は 8-bit と完全な `4/5` Ladder 一致を保ったまま p50 `43.2 tok/s`（8-bit 比 `+52%` 高速化、`−33%` RAM）に到達。`Gemma-4-31B-it-PV-MACaQ-3.2bpw` は 4-bit では OOM となる 24 GB Mac に 31B dense を `15.1 GB` で収容し（`14.1 tok/s`、Ladder `4/5` + Lunar `4/6`）動作。Qwen3-4B は 78–89 tok/s。Ternary Bonsai 2 27B は 7.4 GB に収まり 20.9–21.5 tok/s、MTP ヘッド併用で 27.1 tok/s、コード編集時のプロンプトルックアップで 31–38 tok/s（[#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)）。`.gguf` GSQ-RCO / `i-quant` チェックポイントはカスタム Metal カーネル上で直接実行できます（[#433](https://github.com/goodmai/basalt/pull/433), [#515](https://github.com/goodmai/basalt/pull/515)）。
- **RAM を超えるモデルをスワップ 0 で実行。** `Qwen3.6-35B-A3B-8bit`（37.7 GB）はバッチ化 POSIX AIO と `LIO_NOWAIT` 次層 SSD エキスパート先読み（`--expert-streaming`、[#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478), [#516](https://github.com/goodmai/basalt/pull/516)）、さらに UMA 対応 L0/L1/L2 階層化 KV キャッシュ（`TieredKVHierarchyCore`）および適応型ヘテロジニアスプランニング（[#627](https://github.com/goodmai/basalt/pull/627)）により、24 GB Mac 上でスワップ 0 のまま 8.77 tok/s でデコードします。
- **出力の厳密さを保つ投機的デコーディング。** プロンプトルックアップにより、greedy 出力を一切変えずにコード編集で 1.84–2.66×（Bonsai 2 27B、[#239](https://github.com/goodmai/basalt/pull/239)）、3.45–3.62×（Gemma 4 12B、[#265](https://github.com/goodmai/basalt/pull/265)）の高速化を実現。`SpecTr` OT + `Residual Race`、`EAGLE-3`、`SmallM`（`M=2..16`）、`Fast-dLLM` + `DFlash`、`DiffuCD` がロスレス・マルチドラフトおよびブロック拡散デコーディングを拡張します（[#593](https://github.com/goodmai/basalt/pull/593), [#603](https://github.com/goodmai/basalt/pull/603)）。
- **次のターンは新規トークン分だけのコストに。** ハイブリッドモデルにも対応したプレフィックスツリー・プロンプトキャッシュを搭載。Ornith-1.5-9B で 12.6k トークンのエージェントプロンプトを使った場合、2 ターン目は約 36 秒ではなく約 1 秒で完了します（[#82](https://github.com/goodmai/basalt/pull/82)）。
- **より少ないメモリでロングコンテキストを実現。** 厳密ゾーン指定付き KV 量子化（`4|8|k8v4` に加え、`KVarN` 2D バランシング、不偏 1-bit `QJL`、`ChanMix`/`ThinKV`、クロスリンガル `HeadKV` RBO 転移、セントロイド `RQ-KV`、`turbo3|turbo4|turbo8v4`、RoPE ペアキー用 `polar4`、[#472](https://github.com/goodmai/basalt/pull/472), [#590](https://github.com/goodmai/basalt/pull/590), [#602](https://github.com/goodmai/basalt/pull/602)）により、Qwen3-4B で 16k プロンプトのピークメモリが 3.11 GB から 1.03 GB に削減されます（[#159](https://github.com/goodmai/basalt/pull/159)）。
- **動画、ニューラル TTS、全二重音声、音楽、2K MLX 画像合成。** フェーズ分割 24 GB UMA Video Studio が `LTX-2.5 AV-DiT`、`Kandinsky 6.0 CrossDiT`、`LingBot-Video MoE`、`Wan 2.2` を因果 3D-VAE、ブリックタイリング STA Metal カーネル、`AVAssetWriter` `.mp4` 多重化で実行（[#608](https://github.com/goodmai/basalt/pull/608)–[#611](https://github.com/goodmai/basalt/pull/611)）。`/v1/audio/*` は `LFM2.5-Audio` Mimi/SNAC 全二重コーデック、`SmartTurn` + `Sortformer`、`MossFormer2`、`Soprano-80M`、`Qwen3ForcedAligner`、`Zyphra/ZONOS3-Mini-hf`（`44.1 kHz`、43 言語、2.24 GB、RTF 0.228–0.263、[#527](https://github.com/goodmai/basalt/pull/527)）、`Kokoro-82M` に対応（[#604](https://github.com/goodmai/basalt/pull/604)）。`/v1/images/*` および `/v1/music/*` は `FLUX.3` 4 段階コンポーズ、`SeedVR2` アップスケール、`ACE-Step 1.5` 音楽生成、`Z-Image-Turbo`、`Iris-3B`、`SANA-1.5` を動かします（[#542](https://github.com/goodmai/basalt/pull/542), [#605](https://github.com/goodmai/basalt/pull/605)）。
- **包括的なローカルスタック + Swift SDK。** ツール呼び出し、JSON スキーマ・正規表現・文法制約出力（XGrammar）、Meta `Muse-Glimmer-30B`（[#534](https://github.com/goodmai/basalt/pull/534)）、`Ripple-65B`、`Nemotron-3.5-Lightning`（[#522](https://github.com/goodmai/basalt/pull/522)）、`DeepSeek-OCR-2`（[#476](https://github.com/goodmai/basalt/pull/476)）、`/v1/embeddings`、`/v1/rerank`、`BasaltFoundationModels` Swift SDK（[#606](https://github.com/goodmai/basalt/pull/606)）、`swift-huggingface` Xet ダウンロード（[#587](https://github.com/goodmai/basalt/pull/587)）、リクエストごとの LoRA/PHM/OFT アダプタ切り替え、クラウドより LAN を優先する順序付きフォールバックティア（[#480](https://github.com/goodmai/basalt/pull/480)）を備えています。

詳細: [REST API](docs/i18n/ja/rest-api.md) · [外部ツール連携](docs/i18n/ja/integrations.md) · [コマンドとフラグ](docs/i18n/ja/commands.md) · [モデル一覧](docs/i18n/ja/models.md) · [ベンチマーク](docs/i18n/ja/benchmarks.md)

## 学習と量子化: ユニバーサル any-to-any 変換、PV-MACaQ、分散学習 & QAT

- **ユニバーサル any-to-any 量子化コンバータ（`basalt models convert`）。** `MACaQ`（`affine` / `mxfp4` / `moe` 2.8 bpw ルータ重み付き Hessian、[#537](https://github.com/goodmai/basalt/pull/537), [#599](https://github.com/goodmai/basalt/pull/599)）、`FlatQuant` & `LIRQuant`（[#588](https://github.com/goodmai/basalt/pull/588), [#594](https://github.com/goodmai/basalt/pull/594)）、`SINQ` / `A-SINQ` / `EAQuant` 双軸正規化（[#561](https://github.com/goodmai/basalt/pull/561)）、`QuIP# E8P` 2-bit 格子 & `TurboBoA` Babai–LLL（[#589](https://github.com/goodmai/basalt/pull/589)）、`QTIP` ビットシフト・トレリス（`2.0–3.0 bpw`、[#494](https://github.com/goodmai/basalt/pull/494)）、`SVD-LLM V2` / `OBR`（[#592](https://github.com/goodmai/basalt/pull/592)）、`.gguf` i-quant & `MACaQ-R` を相互変換可能。さらにドメイン対応 `QAT STE` リファインメントと量子化後品質・公平性・リーク監査（`QuantizationAuditCore`、[#623](https://github.com/goodmai/basalt/pull/623)）を備え、CLI、REST `/v1/models/convert*`、macOS `ConvertModelSheet` のすべてに同一仕様で組み込まれています（[#546](https://github.com/goodmai/basalt/issues/546)）。
- **量子化後 PV-Tuning & PV-MACaQ。** `basalt models convert --to macaq:affine:4:64 --pv-tune` は、連続値スケール/バイアス更新と Walsh–Hadamard ブロック上の離散 Hessian 反転リファインメントを交互に実行します。`Qwen3.5-9B` では `8-bit → 4-bit PV-MACaQ` により RAM を 33% 削減（`9.97 → 6.68 GB`）、デコード速度を `+52%` 向上（`27.3 → 41.6 tok/s`、p50 `43.2`）させつつ `4/5` Ladder の完全一致を維持（`valKL = 0.0044`）。2–3-bit チェックポイントでは 3 分未満でパープレキシティ劣化を 8.7–29% 回復します（[#486](https://github.com/goodmai/basalt/pull/486), [#545](https://github.com/goodmai/basalt/pull/545)）。
- **Mac クラスタ間での蒸留、事後学習、分散学習。** `basalt models distill` は OpenAI/Anthropic 教師モデルから `lora`、`phm`、`oft`、または量子化 `AdaLoRA (LR/40)` アダプタへ蒸留します（[#501](https://github.com/goodmai/basalt/pull/501), [#508](https://github.com/goodmai/basalt/pull/508), [#624](https://github.com/goodmai/basalt/pull/624)）。2 段階 `SFT → DPO`、ゼロコピー `ADPO`/`cADPO`、`EMA-QLoRA`、`Self-Appraisal`、およびショートカット対策 `RLVR` 監査が選好アライメントの精度を高く維持します（[#601](https://github.com/goodmai/basalt/pull/601), [#624](https://github.com/goodmai/basalt/pull/624)）。マルチ Mac クラスタ学習は認証付きトランスポート、`DDP + ZeRO-2`、`DiLoCo` + `PowerSGD`、`Zero-Bubble ZB-H1`、`Expert-Parallel MoE` に対応しています（[#595](https://github.com/goodmai/basalt/pull/595)–[#598](https://github.com/goodmai/basalt/pull/598)）。
- **品質ゲート付きでサービング中に学習。** タイムスライス方式の MCP `distill`、検証済みデータのみを使う `train_experts`、`train-jepa`、そしてホールドアウト KL またはパープレキシティが悪化した場合のビット単位ロールバックを備えています。

詳細: [学習コマンド](docs/i18n/ja/commands.md) · [自己学習、蒸留と品質ゲート](docs/i18n/ja/research-tools.md#self-training-train_experts)

## Compose: 宣言的 `basalt.yaml v2.2`、ビジネスガバナンス & 84 種のハードウェアプロファイル

- **モデルのための docker-compose（`basalt.yaml v2.2`）。** ローカルおよびクラウドモデル、RAM 重複排除プランナ、スペシャリストプロファイル、`resident` / `on-demand` / `sequential` メモリモード、`BusinessOrchestrator` 5 コンポーネントルーティング、`LANTERN` セマンティックキャッシュ、`HITL` 承認ゲート（`/v1/mcp/approvals`）、`LAAS-CNRS` ニューロシンボリック知識コア（`SymbolicKnowledgeGraph`）を 1 つの宣言的ファイルで定義できます（[#607](https://github.com/goodmai/basalt/pull/607), [#625](https://github.com/goodmai/basalt/pull/625), [#626](https://github.com/goodmai/basalt/pull/626)）。
- **あらゆる Mac（`16 GB`〜`256 GB`）向け 84 種の組み込みプロファイル。** 7 つのユニファイドメモリ階層（`16gb`、`24gb`、`32-36gb`、`48gb`、`64gb`、`128gb`、`256gb`）× 12 のワークロード類型を 1 コマンドで起動できます（`basalt compose up 24gb-voice-zonos`、[#528](https://github.com/goodmai/basalt/pull/528)）。
- **収まらない構成は事前拒否、差分はライブ適用。** `basalt compose plan` はダウンロード前に Hub からモデルサイズを算出してメモリ超過を拒否し、`compose diff` / `compose up` は差分をクリーンにホットリロードします。`compose bench` は宣言的ベンチマークスイートを実行し、リーダーボードバンドルをエクスポートします（[#505](https://github.com/goodmai/basalt/pull/505)）。
- **ローカル判定器、キャリブレーション済みルーティング & クラウド予算管理。** `OpenJev-Flash-9B-4bit` は 98.9% のトリアージ精度を達成しながら後段 LLM のトークン消費を 77% 削減します（[#465](https://github.com/goodmai/basalt/pull/465)）。温度スケーリング付き `CalibratedRouterGate` + `Gemini Decision 1` が日次 USD 上限内で高難度クエリを適切にルーティングし（[#418](https://github.com/goodmai/basalt/pull/418), [#622](https://github.com/goodmai/basalt/pull/622)）、知覚された文書や音声テキストはデフォルトで Mac の外に出ません。

詳細: [compose リファレンスと実測値](docs/i18n/ja/compose.md) · [パイプラインプリセットとメモリモード](docs/i18n/ja/pipelines.md) · [Gemini によるクラウド判断](docs/i18n/ja/cloud-decisions.md)

## Research: 実行中のモデルを解剖する

- **Model X-ray、トークン単位寄与度 & Belnap 認識論的ゲーティング。** `model_describe`（`MACaQ` / `FlatQuant` / `LIRQuant` モジュールマニフェストを報告、[#630](https://github.com/goodmai/basalt/pull/630)）、`model_tensors`、`model_tensor`、`kv_inspect`、`trace_run`、`trace_tokens`（Gemma BOS 自動処理付き、[#630](https://github.com/goodmai/basalt/pull/630)）、`/v1/xray/tokens` のトークン単位対数確率およびハルシネーションリスク（[#613](https://github.com/goodmai/basalt/pull/613)）、`Score d'influence` ヘッド寄与度（[#622](https://github.com/goodmai/basalt/pull/622)）、Marchenko–Pastur スペクトル、4 値 Belnap `SIXTEEN_3` 証拠状態（[#500](https://github.com/goodmai/basalt/pull/500)）、複素 MVN フェーザプローブ（[#509](https://github.com/goodmai/basalt/pull/509)）、Fisher–Rao ドリフト（[#503](https://github.com/goodmai/basalt/pull/503)）、`/api/v1/metrics` の OpenMetrics（[#458](https://github.com/goodmai/basalt/pull/458)）を提供します。
- **可逆サージェリー & 機構的回路解析。** `serve --research-tools` により、`intervene`（条件付きアブレーション、GQA グループパッチング、認知ヌルスペース & アンチステアリングガードを含む、[#600](https://github.com/goodmai/basalt/pull/600)）、`MAO-SNMF` 特徴分解、`weights_delta`、ライブ `tensor_pack`/`tensor_unpack`、`activation_contrast`、`steer_search`/`steer_commit`、`experiment_compare`、`introspect` が利用可能です。すべての実験は TTL とジャーナルを持つ取り消し可能なオーバーレイとして適用され、ディスク上の重みは一切変更されません。
- **モデルが書いたコードを実際に実行するベンチマーク。** 地球から月へのミッションを最終段とする 5 段のラダーは、未知の入力に対してモデル自身のコードを実際に実行して採点します。ARC-AGI は公式の pass@2 ルールで評価します。

詳細: [リサーチツール](docs/i18n/ja/research-tools.md) · [研究基盤（95 本以上の論文）](docs/i18n/ja/research-foundations.md) · [ベンチマーク](docs/i18n/ja/benchmarks.md) · [serve フラグ](docs/i18n/ja/commands.md)

## Basalt for Mac: macOS 27 Liquid Glass をワンキーまたはワンクリックで

- **監視付きエンジン上のネイティブ SwiftUI アプリ。** macOS 27 Liquid Glass 向けに構築され（[#479](https://github.com/goodmai/basalt/pull/479)）、英語とロシア語に対応（[#456](https://github.com/goodmai/basalt/pull/456)）: ウォームウィンドウ起動 257 ms、アイドルウェイクアップ 0 回、2,000 メッセージのスレッドでもストリーミングトークンあたり p95 7.4 ms、`⌥Space` クイック質問オーバーヘッド 40 ms を実現しています（[#463](https://github.com/goodmai/basalt/pull/463)）。
- **Compose Video Studio、2K Image Studio & ユニバーサルコンバータ UI。** 24 GB Mac 上で A/V `.mp4` 動画を生成・再生し（[#611](https://github.com/goodmai/basalt/pull/611)）、Before/After スライダー付きで 2K RGBA 画像を編集（[#519](https://github.com/goodmai/basalt/pull/519)）、`ConvertModelSheet` からモデルの変換と QAT 監査を直接実行（[#623](https://github.com/goodmai/basalt/pull/623)）、10 Hz SSE テレメトリと Xet ダウンロードで Hugging Face を閲覧・公開できます（[#587](https://github.com/goodmai/basalt/pull/587)）。
- **Compose Profiles（`⌥⌘C`）、全二重音声 & `basalt://` ワンクリック起動。** Mac の RAM 階層（`16–256 GB`、[#528](https://github.com/goodmai/basalt/pull/528)）に応じた上位プロファイルを自動推奨し、`SpeechAnalyzer` / `LFM2.5-Audio` / `ZONOS3-Mini` によるハンズフリー音声ストリーミング（[#604](https://github.com/goodmai/basalt/pull/604)）、さらに `basalt://run?model=...` リンク、`⌥Space` クイック質問、Shortcuts、Spotlight、公証済み `.dmg` パッケージングに対応しています。

詳細: [Basalt for Mac](docs/i18n/ja/mac-app.md)

---

## クイックスタート

動作要件: macOS 27+、Xcode 27.0+ / Swift 6.4（Metal ツールチェーン `xcodebuild -downloadComponent MetalToolchain` を含む）、Apple Silicon M1–M4、モデルに応じて 2–38 GB のディスク空き容量。詳細な手順、ソースからのビルド、および `./basalt-claude` ランチャーについては [docs/i18n/ja/install.md](docs/i18n/ja/install.md) を参照してください。

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds CLI & server from source

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model goodmai/gemma-4-12B-it-qat-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

24 GB Mac で 37.7 GB の MoE モデルをストリーミング実行する、あるいはマルチモデル Compose システムを起動する:

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 84 profiles across 16–256 GB Macs
```

チェックアウトから Mac アプリを起動する（初回起動時に `basalt://` ワンクリックリンクが自動登録されます）:

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

ローカルの 4B–12B モデルはフロンティアモデルの単純な置き換えではありません。プライベート、オフライン、コストゼロのエージェント作業や、ローカル/クラウドのハイブリッドシステムのための選択肢です。`basalt fit` はお使いのマシンで実際に何が動作するかを判定します。モデル一覧表では、何が *動作しなかったか* も併せて記載しています。

Swift 製 MLX ネイティブツールを増やしたい方は **Star** を押してください。また、お使いの Mac の機種と tok/s を添えて Issue を立ててください — [ベンチマーク表への行追加 PR](docs/i18n/ja/benchmarks.md#benchmarks) も歓迎します。

---

## ドキュメント

> 翻訳済みドキュメントは `docs/i18n/` にあります: [Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/)。

| トピック | ドキュメント |
|---|---|
| インストール、動作要件、ソースビルド、`./basalt-claude` | [docs/i18n/ja/install.md](docs/i18n/ja/install.md) |
| CLI コマンド、`serve` フラグ、プロンプトキャッシュ、CacheBlend、PV-Tuning | [docs/i18n/ja/commands.md](docs/i18n/ja/commands.md) |
| REST API: セキュリティ、モデル管理、OpenAI、Anthropic、画像、SSE、OpenMetrics、Swagger | [docs/i18n/ja/rest-api.md](docs/i18n/ja/rest-api.md) |
| Claude Code、OpenCode、OpenAI SDK、Claude Desktop・Cursor 向け MCP | [docs/i18n/ja/integrations.md](docs/i18n/ja/integrations.md) |
| 検証済みモデル、`goodmai` Hub スタックとダウンロード | [docs/i18n/ja/models.md](docs/i18n/ja/models.md) |
| 機能一覧、ベンチマークラダー、ARC-AGI、実測値の出典 | [docs/i18n/ja/benchmarks.md](docs/i18n/ja/benchmarks.md) |
| パイプラインプリセットとメモリモード | [docs/i18n/ja/pipelines.md](docs/i18n/ja/pipelines.md) |
| クラウド判断: Gemini Decision 1、マルチモーダルメディア、キャリブレーション教師 | [docs/i18n/ja/cloud-decisions.md](docs/i18n/ja/cloud-decisions.md) |
| `basalt compose` リファレンスと実測値 | [docs/i18n/ja/compose.md](docs/i18n/ja/compose.md) |
| Basalt for Mac と `basalt://` ディープリンク | [docs/i18n/ja/mac-app.md](docs/i18n/ja/mac-app.md) |
| Model X-ray、テレメトリ、実験、自己学習、蒸留 | [docs/i18n/ja/research-tools.md](docs/i18n/ja/research-tools.md) |
| 数学的・科学的研究基盤（85 本以上の論文と謝辞） | [docs/i18n/ja/research-foundations.md](docs/i18n/ja/research-foundations.md) |
| プロジェクト構成、アーキテクチャ解説、スクリプト、依存関係 | [docs/i18n/ja/architecture.md](docs/i18n/ja/architecture.md) |
| アーキテクチャパターンとモジュール（ロシア語） | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| クラウド: OpenRouter 設定、コスト最適化、モデル比較 | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| AI エージェントを用いた basalt 開発 | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| ロードマップ、プライバシーポリシー | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| エージェント CLI に関する研究ノート、チェックリスト、仕様（ロシア語） | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| コントリビュート、変更履歴 | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## ライセンス

[MIT](LICENSE) © 2026 goodmai

モデルの重みは本ライセンスの対象**外**です — HuggingFace 上の各モデルにはそれぞれ独自のライセンス条件が適用されます。
