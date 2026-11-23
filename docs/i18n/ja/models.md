# モデル

> Languages: [English](../../models.md) · [Русский](../ru/models.md) · [中文](../zh/models.md) · [Español](../es/models.md) · [Français](../fr/models.md) · **日本語** · [Deutsch](../de/models.md) · [한국어](../ko/models.md) · [Bahasa Indonesia](../id/models.md)

## 検証済みモデル

モデルは HuggingFace からダウンロードされ、`~/.cache/huggingface/hub/` にキャッシュされます。42 種類以上の検証済みチャット、MoE、QTIP、PV-Tuned、GSQ-RCO、OCR、ニューラル TTS、埋め込み、リランク、System One 判定チェックポイントが、ワンクリックの **Run with Basalt**（[`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en) → `basalt://run?model=...`）ボタン付きで [`goodmai` Hugging Face コレクション](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) に公開されています。

```bash
basalt models download goodmai/gemma-4-12B-it-qat-4bit
basalt models download goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native
```

### 0.12 向け実測値

M4 Pro、24 GB、release ビルド（`620641c` / `82a99d4`）、2026 年 11 月（`--reasoning-effort none --kv-bits 4`）。

| モデル | 用途 / アーキテクチャ | 実効メモリ | デコード tok/s | Ladder / 備考 |
|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` | 31B dense ハイブリッド（4-bit attn/embed + 3-bit affine PV-MACaQ MLP） | 15.1 GB（ピーク 18.09 GB） | 11.6–13.5（p50 `14.07`、p95 `14.42`） | **warmup 3/3、4/5 段 + Lunar 4/6**、`probe_ppl = 2.39`、ルックアップ採択率 50.6%。4-bit では OOM する 24 GB Mac に収まる（`--prompt-cache-mb 0 --wired-limit`） |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` | 27B dense ハイブリッド（線形 48 + フル attn 16、496 個の MACaQ 4-bit g64 モジュール） | 15.3 GB（ピーク 17.45 GB） | 9.4–11.4（p50 `11.59`、p95 `11.86`） | **warmup 3/3、4/5 段**、`probe_ppl = 1.23`、`SQNR 20.61 → 21.59 dB`（`MSE −19.5%`）、64 層最大ドリフト `5.53%`（`#630`） |
| `Qwen3.5-9B-PV-MACaQ-4bit`（`Qwen3.5-9B-8bit` から変換） | 9B ハイブリッド（線形 24 + フル attn 8、200 個の MACaQ 4-bit g64 モジュール） | 6.68 GB（8-bit 比 −33%） | 35.6–41.6（p50 `43.22`、8-bit 比 +52%） | **warmup 3/3、4/5 段**（8-bit 参照 `24.7–27.3 tok/s` と 100% 一致）、`valKL = 0.0044`、`probe_ppl = 1.28`、最大層ドリフト `3.37%` |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` | 26B MoE（`A4B`、128 エキスパート、top-8、g32 affine MACaQ + 8-bit 共有 MLP） | 16.75 GB | 49.2–54.8（`#628` スイッチ WHT で +82%） | **warmup 3/3、3/5 段**（`biquad-complex`、`fourier`、`pump`）。カーネル `3.41×` 高速化（`2.06 → 0.60 ms`、`#628`） |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` | 35B MoE（`A3B`、120 個の `MACaQSwitchLinear` 層） | 14.64 GB | 64.0–74.7（8k 時 68.7） | **warmup 3/3、1/5 段**、`probe_ppl = 1.46`、シングルパス・スイッチ WHT（`#628`） |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` | 26B MoE（`A4B`、QAT 4-bit） | 14.92 GB | 56.0–63.4 | **warmup 3/3、4/5 段**（`pump` ✅、`lunar 5/6`） |
| `goodmai/gemma-4-12B-it-qat-4bit` | 12B dense QAT 4-bit | 10.51 GB | 18.8–22.2 | **warmup 3/3、4/5 段**（`lunar` ✅ 6/6） |
| 動画 / 音声 / 音楽（`LTX-2.5 AV-DiT`、`Kandinsky 6.0`、`LingBot-Video`、`Wan 2.2`、`LFM2.5-Audio`、`FLUX.3`、`SeedVR2`、`ACE-Step 1.5`） | フェーズ分割 24 GB UMA Video Studio、全二重 Mimi/SNAC 音声 & 4 段階画像/音楽 | フェーズ分割時 ≤ 18 GB | 3D-VAE + STA Metal カーネル | `#542`、`#604`、`#605`、`#608`–`#611` で検証済み |

### 0.11 向け実測値

M4 Pro、24 GB、release ビルド、2026 年 11 月。

| モデル | 用途 / アーキテクチャ | 実効メモリ | スループット / レイテンシ | 備考 |
|---|---|---|---|---|
| `Recursive/Flint-40B-A2.5B`（`flint-oQ3e`） | 40B-A2.5B ハイブリッド スライディング/フル MoE（`512` ルーティング、top-10 + 共有 1） | 17.61 GB | 67.02 tok/s デコード · 273.5 tok/s プリフィル | ネイティブ `oQ3e`（`groupSize = 32`）ローダ（#518, #522）、24 GB 上でスワップ 0 |
| `Zyphra/ZONOS3-Mini-hf` | 1.6B Llama + 32 コードブック FSQ マルチヘッド DAC TTS（`44.1 kHz`） | 2.24 GB | RTF 0.228–0.263（実時間の 3.8–4.4×） | 43 言語、27 話者、クロスリンガル `embedding` + CFG `1.5`（#525, #527） |
| `hexgrad/Kokoro-82M` | 82M ニューラル TTS（ANE + MLX） | 約 0.35 GB | TTFA 80 ms 未満 | `pipeline.speak` における文ストリーミング用フォールバック段（#493, #513） |
| `Tongyi-MAI/Z-Image-Turbo` | DiT 画像合成 | 約 6.4 GB | 4–8 ステップ（`1024×1024`） | TeaCache と共有テキストエンコーダを備えた `pipeline.imagine`（#517, #519） |
| `Ricursive/Ripple-65B-4bit` | 65B ゲーテッドアテンション スパース MoE（`128` エキスパート、top-8） | 約 35 GB（64 GB+ 常駐、または SSD ストリーミング） | アーキテクチャ検証済み | NoPE / 部分 RoPE（#520, #522） |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | 30B-A3B Mamba-2 + 潜在ルーティング MoE（`lr_moe`） | 約 17 GB（4-bit） | アーキテクチャ検証済み | インターリーブ GQA + Mamba-2（#521, #522） |
| `meta-models/Muse-Glimmer-30B`（`mlx-community/Muse-Glimmer-30B-4bit`） | 30B ハイブリッド SWA（`2048`）+ NoPE フルアテンション + ViT-G/14 VLM | 約 18.1 GB（4-bit）· 約 19.8 GB（`oQ4e`） | アーキテクチャと HF シャード検証済み | Onyx `to=self` 推論 + ATEM ツール呼び出し + GGUF 脱中心化ノルム取り込み（#533） |

### 0.8 向け実測値

M4 Pro、24 GB、68 W 電源アダプタ接続、release ビルド、greedy デコーディング、2026 年 11 月。

| モデル | パラメータ数 | 実効メモリ | デコード、tok/s | 備考 |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | 約 2.2 GB | 78–89 | dense。プロンプトルックアップおよび全 KV モードが利用可能 |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B 三値 | 7.4 GB | 通常 20.9–21.5 · コードで `--mtp` 時 27.1 · コードで `--speculative-lookup` 時 31–38 | ハイブリッド（Gated DeltaNet）。ミラー先 `goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | 約 11.8 GB | 15–19 | ハイブリッド。Golden テスト済み。ミラー先 `goodmai/Qwen3.8-27B-3bit-MLX-TextOnly` |
| `goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native` | 27B i-quant | 約 12.4 GB | 5.2 | ネイティブ GSQ-RCO `IQ3_S` 取り込み（#433）、WikiText-2 PPL 4.169、System One 300/300 |
| `goodmai/Qwen3.6-35B-A3B-8bit` | 35B MoE（37.7 GB） | 約 13.9 GB | 8.77 | `--experimental-expert-streaming`（#478）、バッチ化 AIO + `LIO_NOWAIT` により 24 GB 上でスワップ 0 |
| `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly` | 27B | 未計測 | 18–22 | 一部プロンプト（8 件中 3 件）でループ発生。`RepetitionGuard` が停止 |

| 埋め込み / リランクモデル | タスク | 実測値 |
|---|---|---|
| `mlx-community/bge-small-en-v1.5-bf16` | 埋め込み（384 次元） | 229–299 texts/s、バッチ 32 × 約 240 トークン |
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | 埋め込み（1024 次元） | 20 texts/s、同条件バッチ |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | リランク | 既知ペアを正しく順位付け |
| `jinaai/jina-reranker-v3-mlx` | リランク | 既知ペアを正しく順位付け |

### 過去の報告（0.8 向けの再計測なし）

| モデル | パラメータ数 | RAM | 24GB Mac での状態 |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 GB | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 GB | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 GB | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 GB | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 GB | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 GB | ✅ ~12 TPS（アブリテレート版） |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 GB | ✅ ~14 TPS（動的量子化） |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 GB | ✅ ~8 TPS（NVFP4） |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 GB | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 GB | ✅ ~48 TPS（推論モード — 注記参照） |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ 24 GB 超では未検証 |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 GB | ✅ ~76 TPS（常時推論思考） |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 GB | ✅ ~40 TPS（常時推論思考） |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated`（ベース BF16） | 27B | 約 54 GB | ❌ 64GB 以上が必要（24GB では 4-bit MLX 版を使用） |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ 意味不明な出力 |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ 意味不明な出力 |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM（32GB 以上が必要） |

> **24GB RAM Mac に関する注意:** 
> - 非量子化の 27B/35B モデル（重み約 54 GB）は物理メモリを超過します。MLX 4-bit / Dynamic Quant（DQ）量子化済みの重み（例: `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`、`Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`）を使用すれば、約 15 GB の RAM に余裕をもって収まります。
> - 10GB を超えるモデルは、マルチターンの重いコンテキスト下で KV キャッシュサイズを圧迫します。KV キャッシュは 1 トークンあたり `2 · layers · kv_heads · head_dim · 2 B`（fp16）を消費し、値はモデルの `config.json` から読み取られます。ハイブリッドモデルはフルアテンション層と固定リカレント状態のみを計上し、スライディングウィンドウ層はウィンドウ長で上限が止まり、`--kv-bits 4|8` は要素あたり `bits + 0.5` ビットを格納します（`--kv-bits` はスライディングウィンドウ層には適用されません）。Qwen3-4B: 1 トークンあたり 144 KiB、32k トークンで 4.5 GiB。Ornith-1.5-9B（32 層中 8 層がフルアテンション）: 1 トークンあたり 32 KiB、32k で 1.05 GiB。`serve --dry-run`、`models check`、`fit`（量子化グリッドとデコード速度推定値 — RAM 列と適合レベルは引き続き重みのみを計上）、およびメモリ由来のトークン予算はこの数値を用います（上限 = min(要求 max_tokens, --max-tokens, KV 空き容量 − プロンプトトークン数)。ここで KV 空き容量は Metal 推奨ワーキングセットから読み込み済み重みとプロンプトキャッシュ予算を引いた値であり、プロンプトはモデルのトークナイザでカウントされます。まったく収まらないプロンプトは生成前に 413 で拒否されます）。ダウンロード済み config がない場合はパラメータ数ベースの推定にフォールバックします（dense 7B–32B モデルでは近い値になりますが、小型・ハイブリッド・MoE モデルでは数倍ずれます）。なお、これらは実測ではなく推定値です。Mamba 層（Jamba、NemotronH）のリカレント状態は計上されず、アテンション層のみが計算対象となります。MLA モデル（DeepSeek V2/V3）は圧縮潜在表現ではなく、MLX がキャッシュする展開済みヘッド単位の K と V として計上されます。

## モデルのダウンロード（mlx-community、huihui-ai、Ex0bit）

Basalt は Hugging Face 上の任意の作成者・組織のモデルを参照、検査、ダウンロードできます:

```bash
# List models by author
basalt models list --author Ex0bit
basalt models list --author huihui-ai
basalt models list --author mlx-community

# Search with filters
basalt models list --author Ex0bit --search PRISM
basalt models list --search Huihui-Qwen3.8

# Download specific models
basalt models download AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit
basalt models download Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX
basalt models download Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4

# Interactive picker with custom author
basalt models download --author Ex0bit
```
