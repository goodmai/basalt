# basalt compose

> Languages: [English](../../compose.md) · [Русский](../ru/compose.md) · [中文](../zh/compose.md) · [Español](../es/compose.md) · [Français](../fr/compose.md) · **日本語** · [Deutsch](../de/compose.md)

`basalt compose` は、1つの YAML ファイルに記述されたモデルシステムを提供します。このファイルには、ローカルモデルとクラウドモデルの一覧、およびそれぞれをメモリに保持する方法を記述します。また、それらの間でリクエストをルーティングするパイプラインも定義します。各パイプラインは perceive（OCR、音声認識）、decide（System One）、answer（チャットモデル）の順に処理を行います。これは、モデル向けの docker-compose ファイルのようなスタイルで構成された、モデルの組み合わせ（mixture of models）です。

```bash
basalt compose plan -f basalt.yaml        # validate and show the compute plan on this Mac
basalt compose up -f basalt.yaml          # serve it; run it again with a changed file to apply the diff
basalt compose ps                         # what is loaded, when on-demand models unload, cloud spend
basalt compose down <name>                # stop it
basalt compose up document-qa             # legacy bundled files: document-qa, voice-agent, zonos-voice
basalt compose up 24gb-voice-zonos        # 84 hardware-tiered profiles: <tier>-<archetype>
```

`compose up` は、設定ファイルから展開されたフラグを使って `basalt serve` を起動し、起動時にそれらのフラグを出力します。提供する REST および MCP のサーフェスは同じです。`--` 以降のフラグは serve に渡されます（例：`basalt compose up -f basalt.yaml -- --port 8090`）。

サーバーはまずポートをバインドし、その後にモデルをロードします。ロード中、`compose ps` ではモデルが `loading` と表示され、`GET /v1/health/ready` は 503 を返し、実行レコードはすでに存在するため、2回目の `compose up` は新しいプロセスを起動する代わりに既存のシステムを検出します。モデルのロードが完了すると、各モデルに対して起動時プローブ（startup probe）が実行されます。
- チャットモデルおよび各ローカル回答モデルは 1 トークンを生成します。
- 各ローカル判定モデルは 1 つの質問に回答します。
- オンデマンドモデルは、ロードせずにファイルのみが検証されます。

プローブに失敗したモデルは、`compose ps` で理由とともに `failed` と表示されます。`compose up` によってそのモデルが置き換えられるまで、`/v1/health/ready` は 503 を返し続けます。`compose up` によってロードまたはリロードされたモデルにも同じプローブが実行されます。`--mcp` を指定した場合は、これまでと同様にサーバー起動前にモデルがロードされます。

## 設定ファイル

```yaml
version: 1
name: support-desk            # the system's name, used by ps, down and up-as-diff
memory: auto                  # resident | sequential | auto
budget: { usd_per_day: 2 }    # optional cap on all cloud spend

models:
  chat:     { source: mlx-community/gemma-4-12B-it-qat-4bit }
  big:      { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, idle_unload: 10m }
  ocr:      { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 2m }
  asr:      { source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX, keep: sequential }
  jev:      { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
  frontier: { source: "cloud:openrouter/anthropic/claude-haiku-5.5", api_key_env: OPENROUTER_API_KEY, budget: { usd_per_day: 1 } }

pipeline:                     # the default pipeline: /v1/pipeline and MCP basalt_pipeline
  perceive:
    documents: [ocr, apple-vision]     # the first available reads pdf and image attachments
    audio: [asr, speech-analyzer]
  decide:
    cascade: [jev, chat]               # up to four stages, cloud and local in any order, the last one local
    escalate_below: 0.6
    abstain_below: 0.75                # questions below this confidence go to the answer model
  answer:
    routes:                            # the first matching rule wins
      - when: { decide: { route: code }, cost_left: true }
        model: frontier
        fallback: chat
      - when: { prompt_tokens: { over: 8000 } }
        model: big
      - model: chat                    # the last rule has no when and names a local model

pipelines:                    # more pipelines, served at /v1/pipelines/<name>
  triage:
    decide: { cascade: [chat] }
    answer: chat
```

### モデル
- **`source`:** Hugging Face リポジトリ、ローカルパス（`./`、`/`、または `~`）、`cloud:openrouter/<model>`、あるいは `gemini-decision-1` などの Google の判定モデル向けの `cloud:gemini/<model>`（decide カスケード専用。API キーは通常 `GEMINI_API_KEY` に設定します）。
- **`keep`** はローカルモデルに適用されます：
  - `resident` はデフォルトで、常時メモリに保持されます。
  - `on-demand` はそのモデルを必要とする最初のリクエスト時にロードされ、`idle_unload` 経過後にアンロードされます。期間は `30s`、`5m`、`1h` のように指定し、デフォルトは 120 秒です。
  - `sequential` は perceiver（知覚モデル）をそのステージの間だけロードします。デフォルトパイプラインでは、両方が同時にメモリに収まらない場合、perceiver の実行前後にチャットモデルを一時的にアンロードすることもできます。
- **クラウドモデル:**
  - キーを保持する環境変数名を `api_key_env` で指定します。ファイル内に直接キーを記述することや、`${…}` による展開は拒否されます。
  - `budget: { usd_per_day }` は、1つのモデルの 1 日あたりの支出上限を設定します。
  - `data: no-perceived` がデフォルトであり、ドキュメントや音声から読み取られたテキストがクラウドモデルに送信されることはありません。
  - `data: media` は `cloud:gemini` 判定モデル専用で、パイプラインに添付されたファイル自体（PDF、画像、WAV 形式の音声）とそれらの知覚済みテキストを判定のためにそのモデルへ送信します。この Mac の外部へ送信しても問題ないドキュメントにのみ使用してください。

### メモリとプラン
`compose plan` は、すべてのローカルモデルの常駐バイト数（resident bytes）を合計します。オンデマンドモデルはいつでもロードされ得るため、常駐モデルとして計算に含まれます。

- `memory: resident` は、合計サイズがワーキングセットに収まる必要があります。
- `memory: sequential` は、最大のステージに 2 GB を加えたサイズのみを必要とします。
- `memory: auto` は、収まる場合に resident を選択します。

収まらないシステムは、モデルをロードする前に数値を示して拒否されます。ダウンロードされていないモデルはサイズ不明として報告されます。

### Decide（判定）
- **順序:** `cascade` には判定モデルを順番に並べます。各ステージは、確信度が高い（少なくとも `escalate_below` 以上の）質問に回答し、残りを次のステージへ渡します。
- **任意の順序でのステージ構成:** クラウドとローカルを任意の順序で最大 4 ステージまで構成できます。たとえば、最初に小規模なローカル判定器、次にクラウドモデル、最後に大規模なローカルモデルを置く `[laya, jev, chat]` のような構成が可能です。
  - 最後のステージはローカルモデルです。
  - すべてのローカルステージは判定エンジンとして保持されます。すなわち、`self` として読み出されるチャットモデル、または Laya Core AI バンドルや OpenJev バンドルなどの判定モデルです。専用のローカル判定モデルでは `keep: on-demand`（および `idle_unload`）を設定でき、その場合、最初に判定が到達した時点でロードされ、アイドル時にアンロードされます。
  - 各パイプラインは独自のステージを指定します。複数のパイプラインで参照される判定モデルは 1 回だけロードされます。
  - 判定モデルは判定のみを行います。回答と判定の両方を行うのはチャットモデルだけです。
- **キャリブレーション:** リードアウト型判定モデル（readout decision model）は、フォルダ内に `basalt_readout.json`（`{"temperature": …, "noul_temperature": …, "noul_bias": …}`）があればそこから温度パラメータを取得します。ない場合はそのモデル名に対して既知の値を、それもなければデフォルトなしを使用します。#448 ではこれらのファイルを Gemini Decision 1 に合わせてフィッティングすることを計画しています。
- **サイズ計算:** `compose plan` は、OpenJev ヘッドバンドルのサイズを「ベースモデル + アダプター」として計算します。これは 4-bit を含め、LoRA が保存されているそのままのベースモデルに対して未マージ（unmerged）で適用されるためです。serve 側のプランも同様です。
  - OpenJev-0.6B での実測：ラベル付き状態 100 件、質問 300 件、M4 Pro、コミット 066da0a。参照値はマージ済み float32 パスであり、PyTorch と 1e-4 以内で一致します（#406）。

    | ベース（LoRA 未マージ） | 精度 | float32 との最上位回答一致率 | アクティブメモリ | 状態あたり中央値 |
    |---|---|---|---|---|
    | マージ済み float32（参照値） | 176/300 | — | 2,273 MB | 185 ms |
    | bf16（バンドル指定・デフォルト） | 175/300 | 288/300 | 1,141 MB | 181 ms |
    | 8-bit | 176/300 | 279/300 | 608 MB | 184 ms |
    | 4-bit | 149/300 | 215/300 | 324 MB | 185 ms |

  - この 0.6B モデルでは 4-bit ベースにすると精度が 9 ポイント低下する一方、8-bit では 4 分の 1 のメモリで精度を維持します。serve と compose はバンドルが指定するベースをロードするため、量子化ベースは `JevEvaluator` 経由でのみ直接利用できます。
  - 再実行するには `BASALT_E2E=1 BASALT_E2E_JEV_MODEL=<bundle> BASALT_E2E_JEV_BASE=<bf16 base> BASALT_E2E_JEV_BASE_8BIT=… BASALT_E2E_JEV_BASE_4BIT=… swift test --filter JevRealModelTests/unmergedArms` を実行してください。
- **24 GB 環境における Gemma 4 12B と 9B 判定器の併用:** `openjev/OpenJev-Flash-9B-MLX`（8-bit、9.5 GB）および `openjev/OpenJev-Flash-9B-MLX-4bit`（5.9 GB）で実測しました。エスカレーション先には Gemma 4 12B（`gemma-4-12B-it-qat-4bit`）を使用しています。データセットはラベル付き状態 60 件、判定 180 件、M4 Pro 24 GB（コミット 8294bf7）です。
  - 4-bit 版の判定器を使用してください。Gemma と併用しても、24 GB 中アクティブメモリは 15.3 GB（ピーク 15.9 GB）に収まります。
  - 8-bit 版の判定器は、チャットモデルおよび 0.6B ヘッドと合わせると約 21 GB を占有します。単独での判定は問題なく動作しますが（棄権なしで 92.8%、1.85 s）、Gemma がエスカレーションされた質問のデコードを開始すると KV キャッシュによって Mac がスワップに陥ります。本来 2 分の計測アームが 24 分以上経過しても終わらず、中断しました。

    | Gemma 4 12B と併用した判定器（4-bit 判定器） | 精度 | LLM への送信数 | LLM トークン削減率 | 判定 p50 |
    |---|---|---|---|---|
    | LLM のみ | 97.8% | 180 | — | — |
    | OpenJev-Flash-9B 4-bit、棄権なし | 96.7% | 0 | 100% | 2.0 s |
    | OpenJev-Flash-9B 4-bit、0.6 未満で棄権 | 98.9% | 20 | 88.7% | 2.2 s |
    | OpenJev-Flash-9B 4-bit、0.75 未満で棄権 | 98.9% | 41 | 76.7% | 2.4 s |
    | OpenJev-Flash-9B 4-bit、0.9 未満で棄権 | 97.8% | 74 | 58.5% | 2.5 s |
    | Gemma 自身のリードアウト（`self`）、0.75 未満で棄権 | 99.4% | 1 | 99.4% | 1.9 s |
    | OpenJev-0.6B ヘッド（8-bit 判定器実行）、0.75 未満で棄権 | 82.8% | 86 | 48% | 0.26 s |

  - Gemma が長い回答をデコードしている最中に判定を同時実行すると、デコード速度は 21 tok/s から 3 tok/s（−86%）に低下します。経過時間は、順次実行時の 202 s に対して 341 s かかります。そのため、GPU 判定器は生成処理の間でゲート制御（排他制御）されます。
  - 再実行するには `BASALT_E2E=1 BASALT_E2E_BUDGET_CHAT_MODEL=<chat> BASALT_E2E_BUDGET_FLASH=<judge dir> swift test --filter SystemOneBudgetBenchmarkTests` を実行してください。
- **クラウドに送信される内容:** ドキュメントや音声を含むリクエストはクラウドステージをスキップし、ローカルステージのみを順に通ります。
- **しきい値:** 確信度が `abstain_below` 未満の質問、または上位 2 つの選択肢の差が `min_margin` 未満の質問は回答モデルへ送られます。
- **認識論的ゲート（version 2）:** `epistemic` は回答トークンを 1 つもデコードする前に `(support, refutation, needs_tool)` を分類します：`enabled`、`support_threshold`、`refutation_threshold`、`tool_threshold`、`on_no_evidence`（`abstain` または `escalate`）、`on_conflict`（`report` または `escalate`）、およびオプションの `abstain_message` を設定できます。

### Answer routes（回答ルート）
- **マッチング:** ルールの `when` は、すべての条件が成立したときにマッチします。条件は以下のとおりです：
  - `modality`: documents、audio、または text
  - `decide`: System One から得られる質問 ID とラベル
  - `prompt_tokens: { over: N }`: 1 トークンあたり 4 バイトとして見積もられます
  - `escalated`: decide ステージがエスカレートしたか、質問が回答モデルへ送られた場合
  - `cost_left`: ルールのモデルにまだ予算が残っている場合
- **試行順序:** ルールの `model` を最初に試し、次に `fallback`、その次にパイプラインのデフォルトローカルモデルを試します。
- **スキップ:** クラウドモデルは、知覚済みテキスト（perceived text）が含まれる場合、予算不足の場合、API キーが拒否された後、またはレート制限中の場合にスキップされます。各スキップはレシートに記録されます。

### Speak（`#493`、`#525`）
パイプラインではオプションの `speak` ステージを宣言して音声合成（`Kokoro-82M`、`Qwen3-TTS`、`Fun-CosyVoice3`、`Chatterbox`、`Zyphra/ZONOS3-Mini-hf`）を行えます。GPU → Neural Engine → システム音声への決定論的フォールバックを備えています：

```yaml
pipeline:
  answer: chat
  speak:
    chain: [tts_zonos, tts_ane, apple-speech]
    voice: Savannah
    language: en_us
    reference_audio: ./speaker.wav
    on_gpu_busy: fallback_ane
    sentence_streaming: true
    format: wav
    exaggeration: 0.5
    max_ttfa_ms: 80
```

- **`speak`:** 単一のモデル名（`speak: apple-speech`）、フォールバックリスト（`speak: [tts_zonos, tts_ane, apple-speech]`）、または完全なマッピングを受け付けます。
- **`chain`:** 順序付けられたローカル TTS モデルのリスト。末尾にはオプションで重みゼロの `apple-speech` システム音声合成を指定できます。
- **`voice`:** デフォルトの音声識別子（`Savannah`、`alloy`、`af_heart`、または `AVSpeechSynthesisVoice` 識別子）。
- **`language`:** デフォルトの言語またはロケールコード（`en_us`、`ru`、`de`、`cmn` など）。
- **`reference_audio`:** ゼロショット音声クローニング用のオプションのローカル参照 `.wav` / `.pcm` / `.caf` / `.aiff` / `.flac` クリップ。
- **`on_gpu_busy`:** Metal がデコードでビジー状態のときの挙動（`keep`、`fallback_ane`、または `fallback_apple_speech`）。
- **`sentence_streaming`:** トークンが到着するにつれて、文単位のチャンクを全二重で TTS にストリーミングします。
- **`format`:** 出力コンテナまたはサンプルエンコーディング（`wav`、`pcm`、`pcm_24k`、`mp3`、`opus`、`aac`、`flac`）。
- **`exaggeration`:** 対応モデルにおける `0.0...2.0` の範囲の表現力（プロソディ）制御。
- **`max_ttfa_ms`:** フォールバックチェーンにルーティングするまでの最大初回音声出力レイテンシ（TTFA）予算。

### Imagine（`#517`、`#539`）
パイプラインではオプションの `imagine` ステージを宣言して、Apple Silicon 上で最大 4 つの明示的なステージ（`upsample_via` リライター -> `share_encoder` テキスト/ビジョンエンコーダー -> `chain` DiT ペインター -> `decoder_via` 潜在 VAE デコーダー）にまたがる画像合成・編集を行えます。さらに TeaCache によるステップスキップや best-of-N VLM ジャッジもサポートします：

```yaml
pipeline:
  answer: chat
  imagine:
    chain: [z_turbo, sana_fast]
    upsample_via: chat
    share_encoder: encoder
    decoder_via: vae
    steps: 8
    size: 1024x1024
    format: png
    teacache_threshold: 0.15
    preview_every_steps: 2
    candidates: 3
    judge_via: vlm_judge
    on_gpu_busy: fallback
```

- **`imagine`:** 単一のジェネレーター名（`imagine: z_turbo`）、フォールバックリスト（`imagine: [z_turbo, sana_fast]`）、または完全なマッピングを受け付けます。
- **`chain`:** 順序付けられたローカル画像合成モデル（`FLUX.2/3-klein`、`Z-Image-Turbo`、`Iris-3B`、`SANA-1.5`、`Ming-Image`、`Bonsai-4B`、`Step1X-Edit`、`SonicDiffusion`）。
- **`upsample_via`:** 合成前に簡潔なプロンプトを詳細なシーンキャプションへ拡張するためのローカル LLM/VLM リライタースロット（`FLUX.3` をターゲットにする場合は BFL のフルシーンキャプションプロンプト規約を使用します）。
- **`share_encoder`:** 画像合成モデルを駆動する条件付け埋め込みを生成するローカル LLM/VLM スロット（4-bit、8-bit、または BF16 の `Qwen3-VL-4B-Instruct` など）。
- **`decoder_via`:** オプションのローカル潜在 VAE デコーダースロット（蒸留版 `black-forest-labs/FLUX.2-small-decoder` やフル版 `unsloth/FLUX.2-VAE` / `Felldude/FLUX.2-HDR-VAE` など）。
- **`steps`:** `1...100` の範囲でのデフォルトのデノイズステップ数。
- **`size`:** デフォルトの出力解像度 `<width>x<height>`（16 の倍数。例：`1024x1024` や `2048x2048`）。
- **`format`:** 出力画像フォーマット（`png`、`webp`、または `jpeg`）。
- **`teacache_threshold`:** TeaCache による中間ステップスキップのための残差 L1 しきい値（`>= 0`）。
- **`preview_every_steps`:** SSE ストリーミング中に N ステップごとに中間の TAEF プレビューフレームを出力します（`>= 0`）。
- **`candidates`:** best-of-N 選択のためにシードを変えて合成する候補画像の数（`1...64`）。
- **`judge_via`:** プロンプトに対して `candidates` を採点し、最も優れた成果物を返すローカル VLM/LLM スロット。
- **`on_gpu_busy`:** Metal がデコードでビジー状態のときの挙動（`queue` または `fallback`）。

バンドルされた 4 ステージ `FLUX.3` プリセット（`basalt compose up <preset>`）：`flux3-16gb-fast`、`flux3-24gb-balanced`、`flux3-36gb-moe-vision`、`flux3-64gb-studio-base`、および `flux3-128gb-reference`。

### Version 2: 変数とプロファイル
`version: 2` では 2 つのブロックが追加され、1 つのファイルで複数の Mac に対応できるようになります。version 1 のファイルも引き続き動作しますが、version 1 ファイル内でこれらのブロックを使用すると「needs version: 2」というエラーで拒否されます。

```yaml
version: 2
name: desk
variables:
  chat_repo: mlx-community/gemma-4-12B-it-qat-4bit
profiles:
  small:
    when: { memory_gb: { under: 20 } }
    variables: { chat_repo: mlx-community/gemma-4-E4B-it-4bit }
  big:
    when: { memory_gb: { over: 48 }, chip: Max }
models:
  chat:  { source: "${chat_repo}" }
  large: { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, profiles: [big] }
```

- **`variables`:** 小文字の名前と 1 行のプレーンな値のペアです。ファイル内のあらゆる箇所で `${name}` が置換されます。宣言されていない `${…}` は拒否されるため、環境変数がファイルに紛れ込むことはなく、シークレットは必ず `api_key_env` 経由でのみ渡されます。
- **`profiles`:** この Mac 上で `when` 条件が成立するとき、そのプロファイルが有効になります。条件には `memory_gb`（`under` および `over`、GB 単位のユニファイドメモリ）と `chip`（`M4 Pro` や `Max` などのチップ名に対する大文字小文字を区別しない部分一致）を指定できます。`when` を持たないプロファイルは、`--profile` で明示的に指定された場合のみ有効になります。
  - 有効なプロファイルの `variables` は、ファイル全体の変数を上書きします。
  - `profiles: [...]` を持つモデルや名前付きパイプラインは、指定されたプロファイルのいずれかが有効な間だけ存在します。
- **上書き指定:** `config`、`plan`、`up` の各コマンドでは、`--set name=value` で宣言済みの変数を上書きし、`--profile name` で指定したプロファイルのみを有効にできます。`compose up` はこれらをサーバーに引き継ぎ、up-as-diff でもファイルと一緒に送信します。
- **パス:** `./` または `../` で始まるモデルパスは、compose ファイルのあるフォルダからの相対パスとして解決されます。
- **`basalt compose config -f <file>`** は、この Mac 上で有効なプロファイルを反映して解決されたファイルを出力します。**`basalt compose schema`** は、エディタ向けにこのフォーマットの JSON Schema を出力します。

### Experimental と MCP（`#485`）
compose ファイルでは、`--` の後ろに長いフラグ列を渡す代わりに、トップレベルの `experimental` セクションおよび `mcp` セクションを宣言できます（version 1 と 2 の両方でサポート）：

```yaml
experimental:
  zero_copy_body: true
  smallm_kernel: true
  speech_analyzer: true
  image_fallback: ocr
  ane_vision_tower: ./tower-bundle
  telemetry_confidence_every: 4
  adaptive_thinking_budget: true
  samplers: true
  mtp_draft_head: true
  expert_streaming: true
  expert_cache_gb: 2.5
  kv_head_split: 0.5            # or quest_top_k: 64 (mutually exclusive)

mcp:
  enabled: true
  rest: true
  research_tools: false
  enable: [surgery, intervene]
  disable: [switch_model]
  image_dirs: [./images]
```

- **`experimental`:** 対応する `serve` フラグ（`zero_copy_body`、`smallm_kernel`、`speech_analyzer`、`image_fallback`、`ane_vision_tower`、`telemetry_confidence_every`、`adaptive_thinking_budget`、`samplers`、`mtp_draft_head`、`expert_streaming`、`expert_cache_gb`、`kv_head_split`、`quest_top_k`）に変換され、正確な YAML 行番号付きで検証されます（`telemetry_confidence_every` >= 0、`image_fallback` は `ocr`、`ane_vision_tower` は `metadata.json` と `tower.aimodel` を含むこと、`expert_cache_gb` > 0 かつ `expert_streaming: true`、`kv_head_split` は `0.05...1` の範囲、`quest_top_k` は `1...4096` の範囲）。相対パス（`./`、`../`、`~`）は compose ファイルのフォルダを基準に解決されます。
- **`mcp`:** `--mcp-enable` / `--mcp-disable` と同じグループおよびツールセレクタ（`serving`、`gemma`、`media`、`harness`、`xray`、`surgery`、`research`、`all`、または個別のツール名。常に `disable` が優先されます）を使用して、MCP stdio トランスポート（`enabled`、`rest`、`research_tools`、`enable`、`disable`、`image_dirs`）を設定します。
- **ホットリロードと再起動:** `speech_analyzer`、`image_fallback`、`telemetry_confidence_every`、`adaptive_thinking_budget` は `compose up` 時にライブ更新されます（`POST /v1/compose/apply`、`experimental:` 配下に出力）。エンジンおよびトランスポートに紐づくフラグ（`zero_copy_body`、`smallm_kernel`、`ane_vision_tower`、`samplers`、`mtp_draft_head`、`expert_streaming`、`expert_cache_gb`、`kv_head_split`、`quest_top_k`、および `mcp`）は `compose diff` で `needs a restart` と報告され、ライブの `apply` では拒否されます。

### ベンチマークとリーダーボード提出（version 2、`#492`）
`version: 2` の compose ファイルでは、共有の `defaults` と名前付きの `suites`（`suites:` の配下、または `benchmarks:` 直下の短縮スイートキー）を持つトップレベルの `benchmarks` ブロックを宣言できます：

```yaml
benchmarks:
  defaults:
    seed: 42
    temperature: 0.0
    gpu_lock: true
    output_dir: ./bench-out
  suites:
    smoke:
      - adapter: basalt:throughput
        target: model:chat
        params: { iterations: 2, max_tokens: 64 }
        gates: { min_decode_tps: 15, max_peak_gb: 20, max_p50_ms: 5000 }
    frontier:
      profiles: [big]
      budget: { usd_max: 5.0, max_minutes: 30 }
      tasks:
        - adapter: arc-agi-2
          dataset: ./arc-eval.json
          target: pipeline
          gates: { min_accuracy: 0.25, max_ppl: 12.0, max_ece: 0.08 }
          submit:
            target: arc-prize
            api_key_env: ARC_PRIZE_API_KEY
            endpoint: https://arcprize.org/api/v1/submissions
            subset: public-eval
            split: test
```

- **`defaults`:** オプションの `seed`、`temperature`、`gpu_lock`、および `output_dir`。
- **`suites`:** 各スイートは短縮形式のタスクリスト（`smoke: [...]`）または `{ profiles, budget: { usd_max, max_minutes }, tasks: [...] }` の形式で記述します。
- **`tasks`:** 各タスクには `adapter`（`basalt:throughput`、`basalt:perplexity`、`basalt:system-one`、`basalt:passkey`、`basalt:mom`、`basalt:harness`、`arc-agi`、`arc-agi-2`、`terminal-bench`、`swe-bench`、`lm-eval`、`livecodebench`、`bfcl`、`inspect-ai`、`tau-bench`、`gaia`、`evalplus`）、オプションの `dataset`、`target`（`pipeline`、`pipeline:<name>`、または `model:<slot>`）、スカラー値の `params`、リグレッション用の `gates`（`min_decode_tps`、`max_peak_gb`、`min_accuracy`、`max_p50_ms`、`max_ppl`、`max_ece`）、およびオプションの `submit`（`target`、`api_key_env`、`endpoint`、`subset`、`split`）を指定します。
- **コマンド:** `basalt compose plan` は解決されたスイートを表示します。`basalt compose bench -f basalt.yaml --suite smoke --submit-dry-run` はスイートを実行し、検証済みのリーダーボード用バンドルをエクスポートします。`basalt bench list|compare|submit` は、10 種類すべてのターゲット（`arc-prize`、`terminal-bench`、`swe-bench`、`hf-leaderboard`、`livecodebench`、`bfcl`、`inspect-ai`、`tau-bench`、`gaia`、`weave-otel`）に対してアダプターの検査、成果物の差分比較、および dry-run バンドルのエクスポートを行います。

### クラウド障害と支出管理
- キーが拒否された場合（HTTP 401/403）、そのクラウドモデルはサーバーが再起動されるまで無効化され、その旨を示すログが 1 行出力されます。
- レート制限（429）が発生した場合、`Retry-After` ヘッダーの秒数、またはヘッダーがない場合は 60 秒間一時停止します。
- クラウドモデルが無効化または一時停止されている間は、次のステージまたはフォールバックモデルが回答します。
- モデルごとおよび合計の当日支出額は `~/.basalt/cloud-spend.json` に保存されるため、再起動しても上限はリセットされません。日付は UTC 基準です。

## システムの実行
- **`compose pull`:** ファイル内のすべてのローカルモデルを Hugging Face キャッシュにダウンロードし、各ファイルを検証して 10% ごとに進捗を表示します。pull の完了後は、`compose plan` で正確なサイズが得られ、`compose up` ではダウンロードを待たずにロードできます。
- **`compose plan` におけるサイズ計算:** キャッシュ済みのモデルはディスク上で実測されます。未ダウンロードのモデルは Hugging Face API のファイル一覧からサイズが算出されます。どちらからもサイズが取得できない場合、`plan` はそのモデル名を挙げて拒否します。この場合は `compose pull` を実行するか、`--assume-size <GB>` を渡してそのサイズとして計算させてください。`plan` はサイズ不明のモデルを 0 バイトとして扱うことはなくなりました。
- **`compose diff`:** 同じ名前で実行中のシステムに対して、`compose up` を実行した場合に変更される内容を、実際には変更を加えずに表示します（`POST /v1/compose/diff`）。ロード、アンロード、キーと予算の変更、必要な再起動、または拒否理由（キーの欠落やメモリに収まらないシステムなど）を一覧表示します。再起動が必要または拒否される場合は非ゼロの終了コードで終了します。
- **イベント:** `GET /v1/compose/events` は、システムの動作を通知する Server-Sent Events（SSE）ストリームです。イベント種別は `loading`、`ready`、`registered`（オンデマンド）、`failed`（起動時プローブ）、`applied`、`refused`、および `up` です。新規接続クライアントにはまず直近 100 件のイベントが送信されます。`compose up --wait` は待機中にこれらのイベントを表示します。
- **実行中システムに対する `compose up`:** 同じ `name` のシステムがすでに実行中の場合、`compose up` はそのシステムへファイルを送信し（`POST /v1/compose/apply`）、差分を適用します。
  - システムがまだロード中または別の変更を適用中の場合は 503 を返します。`compose up` は待機して最大 `--wait-timeout` 秒（デフォルト 1800）までファイルを再送信します。
  - `--wait` を指定すると、`compose up` は `GET /v1/health/ready` が 200 を返すまで待機し、待機中のモデルを表示して、タイムアウト内に準備完了にならなければ非ゼロで終了します。
  - 追加されたモデルをロードし、変更されたモデルをリロードし、削除されたモデルをアンロードします。
  - リクエストの合間にパイプラインを切り替えます。
  - `compose plan` と同様に、すべてのパイプラインのモデルを含めて新しいシステムがこの Mac のメモリに収まるかを最初に確認します。収まらないシステムは拒否され、何も変更されません。
  - 新しいシステム全体は収まるものの、新旧のモデルを同時に並べて保持できない場合は、後継モデルをロードする前に置き換え対象のモデルをアンロードします。その間にそれらのモデルのいずれかを必要とするリクエストは、ロード完了を待つか失敗します。その後のロードが失敗した場合は、古いモデルをロードし直します。
  - モデルごとに `load`、`reload`、`unload`、`keep` を出力し、変更されたクラウド予算ごとに `budget`（ファイル全体の予算は `total`）、`api_key_env` が変更されたクラウドモデルごとに `key` を出力します。
  - クラウドモデルの予算やキーの環境変数は、リロードなしで変更されます。新しい制限は即座に適用され、キーの拒否によって無効化されていたモデルも、キー変数が変更されると再び有効になります。
  - 別のチャットモデルが指定された場合は、サービング中のモデルの位置にロードされ、`chat` として表示されます。ロード中はリクエストが待機します。その後の適用ステップが失敗した場合は、元のチャットモデルがロードし直されます。
  - パイプラインの新しいローカル判定モデルは、他の追加モデルと同様にロードされます。デフォルトパイプラインの判定モデルは `decide` としても表示されます。パイプラインおよび `/v1/systemone` は次のリクエストからそれを使用します。判定を開始または停止するモデルは、新しいロールでリロードされます。
- **再起動が必要な変更:** メモリモードの変更、sequential perceiver の変更、`experimental` 配下のエンジン/HTTP 起動フラグ、または `mcp` 配下の変更には再起動が必要です。`compose down` を実行してから `compose up` を実行してください。
- **`compose ps`:** 実行中の各システム（`GET /v1/compose`）を、そのパイプラインおよびモデルごとの情報とともに一覧表示します：
  - 状態は `loading`、`ready`、`unloading`、`unloaded`、`idle-unloaded`（オンデマンド、現在は未ロード）、`decision`（判定エンジンによって保持）、`cloud`、`no-key`、`over-budget`、`rate-limited`、`refused`、`failed` のいずれかです。
  - オンデマンドモデルについては、最終使用時刻とアンロード予定時刻が表示されます。
  - クラウドモデルについては、上限に対する当日の支出額と、該当する場合は拒否理由が表示されます。
- **ヘルスチェック:** `GET /v1/health/live` はサーバーが HTTP に応答している間 200 を返します。`GET /v1/health/ready` はすべてのモデルがサービング可能になった時点で 200 を返します。それまでは 503 を返し、`chat (loading)` のように待機中のモデルを一覧表示します。アイドル状態のオンデマンドモデル、予算超過のクラウドモデル、レート制限中のクラウドモデルはサービング可能とみなされ、カスケードやルーターはそれらをスキップして次の候補へ進みます。
- **クラウドキー:** クラウドモデルの `api_key_env` が設定されていない場合、`compose up` は起動を拒否し、変数名を提示します。up-as-diff の場合は、代わりに実行中サーバーの環境変数が確認されます。
- **`compose down`:** サーバーに SIGINT を送信します。サーバーは処理中のリクエストを完了してからアンロードを行います。30 秒経過してもプロセスがまだ実行中の場合、`down` はその旨を表示して非ゼロで終了します。
- **状態の保存場所:** 実行中システムの PID、ポート、プロセス開始時刻は `~/.basalt/compose/<name>.json` に保存されます。PID が再利用されたために現在別のプロセスに属しているレコードは、実行中ではないとみなされます。
- **API キー:** サーバーに API キーが設定されている場合、`ps` や up-as-diff を実行する際は `BASALT_API_KEY` を設定してください。

## ドメインアダプター（`version: 2`）
`version: 2` のファイルでは、Hugging Face の NVIDIA Nemotron プリセット（`nemotron:post-v2`、`nemotron:post-v1`、`nemotron:code`、`nemotron:math`、`nemotron:crossthink`、`nemotron:personas`）、`hf:<org>/<repo>` データセット、またはローカルの `.jsonl` ファイルで学習した `adapters:` を宣言し、重みをリロードすることなく単一の常駐 `base` モデル上でリクエストを振り分けることができます：

```yaml
adapters:
  code_oft:
    base: chat
    method: oft              # lora | phm | oft
    rank: 16
    scale: 10.0
    hypercomplex_dim: 8
    factor_rank: 1
    shared_basis: true
    block_size: 16
    parameterization: cayley # cayley | matrixExp
    block_shared: false
    layers: [0, 1, 2, 3]
    output: ./adapters/code_oft
    iterations: 200
    batch_size: 2
    learning_rate: 0.0001
    max_seq_len: 2048
    checkpoint_every: 50
    corpus:
      source: nemotron:code
      subset: default
      split: split_0
      max_rows: 512
      min_pass_rate: 0.9
      include_reasoning: true
      seed: 42
    gate:
      max_kl: 0.2
      max_ppl_rise: 0.05
      held_out: ["Explain what a hash table is in two sentences."]
```

`basalt compose train -f basalt.yaml`（オプションで `--adapter`、`--force`、`--resume` を指定可能）を実行して宣言されたアダプターをディスク上に学習し、その後 `pipeline.answer` または `pipeline.answer.routes` の任意のエントリーで `adapter:` を選択します。

## レシート（Receipts）
`/v1/pipeline` の各レスポンスおよび各セッションログレコードにはレシート（receipt）が含まれます。ステージごとに使用モデル、ミリ秒数、トークン数が記録されます。decide ステージにはコストとスキップされたクラウドステージも記録されます。answer ステージには発火したルール（`route`）、コスト、およびスキップされたモデルが記録されます。`swap_ms` はオンデマンドモデルおよび sequential モデルのロードに費やされた時間です。

## バンドルされたハードウェア階層別プロファイル（`#526`）

`basalt compose` と `Basalt.app`（`Compose Profiles`、`⌥⌘C`、および **Settings → Engine → Bundled Profile**）には、Apple Silicon の 7 つのユニファイドメモリ階層（`16gb`、`24gb`、`32-36gb` [エイリアス `32gb` および `36gb`]、`48gb`、`64gb`、`128gb`、`256gb` [エイリアス `192gb` および `236gb`]）× 12 のワークロードアーキタイプにわたる、**84 種類のハードウェア調整済みプロファイル**が同梱されています：

| アーキタイプ | 主要 | 主な用途 |
|---|---|---|
| `daily-chat` | ★ | 日常のチャット、Apple Vision によるドキュメント認識、ローカル音声入力 |
| `coding-agent` | ★ | 投機的 small-M カーネル、コード特化ルーティング、MCP ツール |
| `voice-zonos` | ★ | Zyphra `ZONOS3-Mini` 44.1 kHz ニューラル TTS + `Kokoro-82M` ANE フォールバック |
| `multimodal-desk` | ★ | OCR、音声認識、System One トリアージ、チャット、および ZONOS3 音声合成 |
| `reasoning-lab` | | 適応的思考予算 + 数学・論理向けの System One カスケード |
| `document-ocr` | | ニューラル OCR（`PaddleOCR-VL-1.5`）+ 4値 Belnap 認識論的ゲート |
| `voice-duplex` | | 全二重文字起こしおよび TTFA 80 ms 未満のストリーミング音声合成 |
| `creative-studio` | | TeaCache 対応のネイティブ MLX 画像合成（`Z-Image-Turbo` / `FLUX`） |
| `rag-knowledge` | | 矛盾検出・棄権ゲートを備えたグラウンディング済み長文コンテキスト検索 |
| `hybrid-cloud` | | ローカルでのプライバシー優先チャット + 日額上限 `$1/day` のクラウドエスカレーション |
| `moe-frontier` | | スパース MoE（`Flint 40B-A2.5B`、`Nemotron-3.5`、`Qwen3.8`）+ エキスパートストリーミング |
| `research-xray` | | Model X-Ray、重みサージェリー、および `basalt:throughput` スモークベンチマーク |

起動時に `Basalt.app` は物理ユニファイドメモリを検出し（`ComposeMemoryTier.detect`）、この Mac 向けの星印付き主要 4 プロファイルを **Recommended for this Mac** にピン留めしつつ、84 個すべてのプロファイルを検索・選択できる状態に保ちます。すべてのプロファイルは CLI からもまったく同じように実行できます（`basalt compose plan 24gb-voice-zonos`、`basalt compose up 36gb-moe-frontier`、`basalt compose up 236gb-multimodal-desk`）。

## サンプル
- `docs/examples/compose/any-mac.yaml`: 16、24、64 GB の各 Mac に対応する 1 つの version 2 ファイル。チャットモデルはメモリに応じて選択されるプロファイルから決まり、64 GB プロファイルでは 27B モデルをオンデマンドで追加し、`long` パイプラインも有効になります。
- `docs/examples/compose/mac-16gb.yaml`: ローカルの Ornith 9B、オンデマンドの OCR、1 日 $0.50 以内で長いプロンプトを Claude Haiku へ送信する構成。
- `docs/examples/compose/mac-24gb.yaml`: Gemma 4 12B、オンデマンドの OCR、コードを Haiku へ送る設定、2 つ目の `support` パイプライン、合計 1 日 $1 の構成。
- `docs/examples/compose/mac-64gb.yaml`: Gemma 4 26B-A4B に加えて Qwen3.8-27B と Nemotron Transcribe をオンデマンドで使用し、クラウド回答を使用しない構成。
- `docs/examples/compose/document-qa-cloud.yaml`、`mom.yaml`、および `too-big.yaml`: 以下の実測値で使用されたファイル。

さらなる実測値は #412 に記載されています。

## 実測値

```yaml
version: 1
name: document-qa-cloud
models:
  chat:  { source: mlx-community/gemma-4-12B-it-qat-4bit }
  ocr:   { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 30s }
  jev:   { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
pipeline:
  perceive: { documents: [ocr, apple-vision] }
  decide:   { cascade: [jev, chat], escalate_below: 0.6 }
  answer: chat
```

```bash
basalt compose plan -f docs/examples/compose/document-qa-cloud.yaml
basalt compose up -f docs/examples/compose/document-qa-cloud.yaml -- --port 8080
basalt compose up voice-agent            # bundled: document-qa, voice-agent
```

```yaml
pipeline:
  decide: { cascade: [jev, chat], escalate_below: 0.6 }
  answer:
    routes:
      - when: { decide: { route: code }, cost_left: true }
        model: haiku          # cloud:openrouter/anthropic/claude-haiku-5.5, budget: { usd_per_day: 0.5 }
        fallback: chat
      - model: chat
```

`/v1/pipeline` 経由での 55 リクエストの混合ワークロード（判定 30 件、コード 10 件、チャット 10 件、ドキュメント 5 件）、M4 Pro、リリースビルド（`scripts/compose_mom_bench.swift`、`docs/examples/compose/mom.yaml`）：

| 構成 | 正解ラベル数 | 1000 リクエストあたりのクラウド費用 | p50 | p95 | ローカルステージ時間 |
|---|---|---|---|---|---|
| ローカルのみ（Gemma 4 12B が判定と回答を実行） | 55/55 | $0 | 0.92 s | 10.0 s | 176 s |
| クラウド優先（Jev が判定、Haiku がすべてのテキストに回答） | 55/55 | $0.046 | 1.25 s | 4.8 s | 24 s |
| MoM（Jev が判定、予算内のコードは Haiku、その他は Gemma） | 55/55 | $0.035 | 1.21 s | 4.8 s | 53 s |

24 GB の M4 Pro、リリースビルドにおける `compose up` での実測値：

| ファイル | リクエスト | レシート |
|---|---|---|
| document-qa-cloud | ドキュメント（初回） | OCR をオンデマンドでロード（489 ms）、ローカルで判定、回答：6.1 s |
| | ドキュメント（2回目） | OCR はすでにロード済み（0 ms）：5.0 s |
| | 40 秒アイドル後 | OCR をアンロード、メモリフットプリント 12 → 11 GB |
| | テキスト入力の質問 | クラウド上の Jev により判定：0.6 s |
| | ドキュメント | OCR を再ロード（267 ms）：5.3 s |
| voice-agent（バンドル） | 音声 | Nemotron Transcribe の前後に chat をアンロード（スワップ 6.9 s）：17.3 s |
| too-big（サンプル） | `compose plan` | 拒否：1 つのステージに 20.7 GB プラス 2 GB が必要（ワーキングセット 19.1 GB に対して超過） |


## スキーマ `version: "2.2"` 拡張

Compose `version: "2.2"`（`version: 2` と互換）では、ブロック拡散 LLM、`DFlash` 投機的ドラフター、全二重音声対音声（speech-to-speech）、共有 VLM/DiT テキストエンコーダー、1 ステップ `SeedVR2` アップスケーリング、`ACE-Step` 音楽 DiT、因果的 3D-VAE 動画 DiT、セグメント単位の選好学習、および Thunderbolt/LAN マルチ Mac クラスタ向けの宣言的設定が追加されています：

- モデルごとの `speculative`（`mode`、`drafter`、`block_size`、`verify`）および `diffusion_text`（`dual_kv_cache`、`confidence_threshold`、`contrastive_alpha`）。
- `imagine` 拡張：`share_text_encoder_with`、`kv_cache_references`、`rgba`、および `upscale`（`model`、`steps`、`target_resolution`）。
- ルートの `voice`（`mode`、`sts_model`、`denoise`、`model` と `diarization` を持つ `turn_detection`、`tts_fallback`、`aligner`）。
- ルートの `music`（`model`、`steps`、`duration_seconds`）および `video`（`model`、`vae_tiling`、`sta_sparse`、`fps`、`frames`）。
- ルートの `train`（`method`、`segments`、`ema_decay`）および `cluster`（`topology`、`peers`、`compression`）。
- エンタープライズ向けルーティング・キャッシュ・ガバナンス（`v0.12`）：`BusinessOrchestratorCore` 5 コンポーネントルーティング、`LANTERN` セマンティックキャッシュ（`SemanticCacheCore` / `SemanticCacheActor`、`#625`）、`HITL` 承認ゲート（`/v1/mcp/approvals`）+ LAAS-CNRS `SymbolicKnowledgeGraph`（`#626`）、および UMA 対応の L0/L1/L2 `TieredKVHierarchyCore` + `AdaptiveHeterogeneousPlanner`（`#627`）。
