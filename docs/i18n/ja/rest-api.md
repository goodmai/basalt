# REST API

> Languages: [English](../../rest-api.md) · [Русский](../ru/rest-api.md) · [中文](../zh/rest-api.md) · [Español](../es/rest-api.md) · [Français](../fr/rest-api.md) · **日本語** · [Deutsch](../de/rest-api.md) · [한국어](../ko/rest-api.md) · [Bahasa Indonesia](../id/rest-api.md)

ベース URL：`http://127.0.0.1:8080` — **認証不要**。

## セキュリティ

ループバック上では任意のローカルプロセスがキーなしで API を呼び出せますが、ブラウザ内の Web ページからは呼び出せません：

- **Host。** ループバックにバインドされている場合、サーバーは `Host` が `localhost`、`127.0.0.1`、または `[::1]`（ポート任意）のリクエストにのみ応答し、それ以外は 403 を返します。これにより、攻撃者のドメインが 127.0.0.1 に再解決されてページから応答を読み取られる DNS リバインディングを防ぎます。特定のアドレスにバインドした場合はそのアドレスも許可されます。`0.0.0.0`/`::` にバインドした場合は任意の `Host` が許可されますが、そこでは API キーが必須となります。
- **Origin。** `Origin` ヘッダーを含むリクエストは、サーバー自身のループバックオリジン（リッスンしているポート上の `http(s)://localhost|127.0.0.1|[::1]`；ポート省略時は http が 80、https が 443）、または `--allow-origin` / `allowed_origins` で指定されたオリジン（`scheme://host[:port]` 形式、パスなし、末尾スラッシュなし、`*` なし；デフォルトポートは一致とみなされるため `https://app.example:443` は `https://app.example` と等価）でない限り拒否されます（403）。他のローカルポートは信頼されません。許可しなければ開発サーバー（`localhost:3000`）や Jupyter 上のページがモデルをロードして生成を実行できてしまうため、そうした UI には `--allow-origin http://localhost:3000` が必要です。`Origin` を送信しないクライアント（curl、SDK、Claude Code、Codex）には影響しません。
- **Content-Type。** すべての `POST` は `Content-Type: application/json`（`charset` パラメータは可）を送信する必要があり、そうでない場合は 415 になります。`-H "Content-Type: application/json"` なしの `curl -d` はフォーム型を送信するため拒否されます。
- **モデル ID。** REST の `model` 値は Hugging Face リポジトリ ID です。`/v1/models/load` は `org/name` のみを受け付けます。生成リクエストの `model` においてパス形式（`/`、`.`、`~` で始まるもの、または `..` を含むもの、`claude-local/..--x` のような隠蔽形式も含む；前後の空白は無視）は 400 になりますが、リポジトリ ID ではないその他の値（`basalt`、`qwen/qwen3-8b-instruct:free`）はエイリアスとして扱われ、ロード済みのモデルが応答し、切り替えは行われません。ローカルパスは `basalt serve --model-path` および MCP では引き続き利用可能です。ロードに失敗しても現在ロードされているモデルはそのまま維持され、エラーボディにファイルシステムパスが含まれることはありません（サーバーログにのみ出力されます）。

## モデル管理

```bash
# List all locally cached models (OpenAI-compatible format)
curl http://localhost:8080/v1/models

# Currently loaded model + readiness status
curl http://localhost:8080/v1/models/current

# Hot-swap to a different model at runtime (blocks until loaded)
curl -s http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-31b-it-4bit"}'
```

`GET /v1/models` は、Claude Code の自動モデルピッカーに追加されるように `claude-local/` プレフィックス付きの ID を返します。`display_name` フィールドには元の HuggingFace リポジトリ ID が格納されます。

## 生の生成（Raw generation）

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement.", "maxTokens": 256}'
```

| フィールド | 型 | デフォルト | 説明 |
|---|---|---|---|
| `prompt` | string | required | 入力テキスト |
| `maxTokens` | int | 8192 | 生成する最大トークン数 |
| `temperature` | float | model | サンプリング温度（0–2）。未指定時はモデルの `generation_config.json` を使用し、それもなければ 0.7 |
| `topP` | float | model | Nucleus サンプリング（0–1）。未指定時は `generation_config.json` を使用し、それもなければ 0.9 |
| `topK` | int | model | 確率の高い上位 k 個のトークンを保持。未指定時は `--top-k`、次に `generation_config.json` を使用し、それもなければオフ |
| `presencePenalty` | float | off | −2…2、直近 1024 個の生成済みトークンに含まれる各トークンから減算されます（プロンプトにはペナルティがかかりません） |
| `repetitionPenalty` | float | `--repetition-penalty` | (0, 2]、直近トークンに対する乗算ペナルティ |
| `thinkingBudget` | int | off | 回答が `maxTokens` 内に収まるようサーバーが思考終了を強制するまでの推論トークン数（Qwen: "Considering the limited time by the user…" + `</think>`）。`maxTokens` から遷移トークンと回答用予約分 min(512, maxTokens/4) を引いた値に収まるよう引き下げられます。予約分差し引き後に 64 トークン未満しか残らない `maxTokens` は 400 になります。思考がオフの場合、モデルに推論プロトコルがない場合、またはプロトコルが遷移を宣言していない非 Qwen モデルの場合は無視されます |
| `minP` | float | server | Min-P 0–1：`minP × max p` 未満のトークンを除外します。`--min-p` とモデルプロファイルを上書きし、`0` でオフになります |
| `dynamicTemperature` | bool | false | エントロピーベースの動的温度（EDT）：トークンごとに `T = T0·0.8^(1.5/H) + 0.5·R` を計算し、0.05–2 にクランプします（より低温のリクエストはそのリクエスト自身の温度を下限として維持し、`top_p`/`min_p` は元の分布をカットします）。ここで `H` は次トークン分布のエントロピー、`R ∈ [0,1]` は反復指標です。確信度の高いトークンではほぼグリーディになり、ループが始まると温度が上がります。`temperature: 0` では効果がありません |
| `speculativeLookup` | bool | `--speculative-lookup` | このリクエストに対するプロンプトルックアップ投機的デコード：プロンプトまたは出力内の n-gram 一致からコピーした最大 16 トークンを 1 回のフォワードパスで検証します。ドラフトが却下された後は 1、2、4… 最大 64 トークンまでドラフト作成を一時停止します。`serve --speculative-recycling` 併用時は、n-gram が一致しない場合に 2 回連続して記録されたモデル自身の argmax 後続トークンから最大 6 トークンをドラフトします（Token Recycling、#38；独自の一時停止カウンタあり）。アテンションのみのモデル、dense リカレントハイブリッド（Qwen3.5/Ornith、Nemotron-H：ドラフト却下時にリカレント状態を復元し受理済みトークンを再再生）、スライディングウィンドウモデル（Gemma 4：各検証ラウンドがステージングされるため、却下されたドラフトがウィンドウのリングを上書きしない）で、`--kv-bits` 未使用かつテキストのみ入力の場合に動作します。エキスパートルーティングを伴うリカレントハイブリッド（LFM2.5 MoE）では無視されます。Gemma 4 26B-A4B（MoE）ではコード編集で 1.97× の高速化が得られる一方、散文や推論では 3–6% 低下します |

`/v1/chat/completions`、`/v1/responses`、`/v1/messages` は同じ 2 つのフィールドを `min_p` および `dynamic_temperature` として受け付け、MCP の `gemma_generate` は `minP` および `dynamicTemperature` として受け付けます。OpenAI エンドポイントは `top_k`、`presence_penalty`、`repetition_penalty`、`thinking_budget` を受け付け、`/v1/messages` は `top_k` と思考予算としての `thinking: {"type": "enabled", "budget_tokens": N}`（1 ≤ N < `max_tokens`、それ以外は 400）を受け付けます。これら 3 つはいずれも `speculative_lookup`（boolean 以外は 400）を受け付け、MCP の `gemma_generate` は `speculativeLookup`（boolean 以外は -32602）を受け付けます。Usage は `completion_tokens_details.reasoning_tokens`（chat；`stream_options.include_usage` でストリーム出力）および `output_tokens_details.reasoning_tokens`（Responses）を報告します。モデルが推論しなかった場合は `0` となり、強制された遷移も推論トークンとして計上されます。

## OpenAI 互換

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

SSE ストリーミング、システムプロンプト、およびマルチターン会話をサポートします。`"model"` フィールドに HuggingFace モデル ID を送信すると、自動的にホットスワップがトリガーされます。

推論モデルの思考内容は、OpenRouter / Ollama の名前である `message.reasoning`（ストリーム時は `delta.reasoning`）として返され、さらに vLLM、DeepSeek、llama.cpp、LM Studio のクライアントが参照する `reasoning_content` としても同じテキストが返されます。どちらのフィールドを伴って返送されたアシスタントターンも受け付けられますが、思考内容自体はプロンプトには戻されません。

## Anthropic 互換

```bash
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

完全な Anthropic SSE イベントシーケンス（`message_start`、`content_block_start`、`content_block_delta`、`message_stop`）を実装しています。プレーン文字列形式とブロック配列形式の両方のコンテンツを受け付けます。

`thinking` ブロックの `signature` は空になります（basalt には署名用の鍵がないためです）。これらは basalt 間ではそのまま往復できますが、本物の Anthropic API はそれらを再送した会話を拒否するため、同じ履歴をそちらへ送る前に削除してください。

**制約付き出力。** `response_format` は、任意の推論ブロックの後に XGrammar を用いて回答を制約します：
- `{"type": "json_object"}` または `{"type": "json_schema", "json_schema": {"schema": …}}`；
- `{"type": "regex", "regex": "[A-Z]{3}-\\d{4}"}`；
- `{"type": "grammar", "grammar": "root ::= …"}`、ルート規則 `root` を持つ XGrammar の EBNF/GBNF 方言；
- `{"type": "structural_tag", "format": …}`、XGrammar の structural-tag JSON（`sequence`、`const_string`、`regex`、`json_schema`、`tag`、…）。

vLLM の `guided_regex` および `guided_grammar` リクエストフィールドも、一度に 1 つずつ動作します。MCP の `generate` ツールは `regex` または `grammar` を受け付けます。64 KB を超えるソース、または XGrammar がコンパイルできないソースは、形式名を明示した 400 エラーになります。

**プリフィル（Prefill）。** Anthropic API、vLLM の `continue_final_message`、llama.cpp と同様に、`/v1/messages`、`/v1/chat/completions`、`/v1/responses` でアシスタントメッセージで終わる会話を送ると、新しいターンとして応答する代わりにそのメッセージの続きを生成します。それ以前のターンは思考オフの状態で生成プロンプトまでレンダリングされ、アシスタントのテキストが閉じられずにその後に続き、レスポンスにはモデルがその後に書いた部分のみが含まれます（`{"name":` をプリフィルすると応答は ` "Whiskers", "age": 3}` となります）。`response_format` または強制ツールが指定されている場合、プリフィルは JSON ドキュメントの先頭となり（スキーマの開始として成立しないプリフィルは 400）、強制ツールの引数にもそれが含まれます。思考ブロック（`<think>…`）を開くプリフィルは要求された思考を維持し、推論として継続します。末尾に複数のアシスタントメッセージが続く場合は 1 つのプリフィルとして結合され、空のものは破棄されます。ツール呼び出しを伴うアシスタントメッセージは依然として完了済みターンとして扱われます。画像を含むプロンプトは継続できません（400）。プリフィル内の制御トークン（`<|im_end|>`、`<|im_start|>`、…）は会話の他の部分と同様に無害化されるため、プリフィルからターンを偽造することはできません。推論を開閉する思考マーカーのみが有効なまま維持され、テンプレート自体がすでに `<think>` を開く場合（Ornith、Qwen3.5+）はプリフィル側の先頭の `<think>` が二重にならず削除されます。`--cloud-fallback` 使用時、すべてのプロバイダーが末尾のアシスタントメッセージの継続に対応しているわけではないため、プリフィルされたリクエストは常にローカルに留まります。システムターン内の `/no_think` 行で思考をオフにする Nemotron Nano v2 では、プリフィルされたリクエストは同じ履歴上の思考オンのリクエストとは異なるシステムターンをレンダリングするため、両者はその手前までのプロンプトキャッシュのみを共有します。

## 画像入力

ビジョンモデル（Gemma 4、Qwen3.5-VL、… — MLXVLM がロードできる任意のモデル）は、`/v1/chat/completions`、`/v1/responses`、`/v1/messages`、および MCP の `generate` ツールで画像を受け付けます：

- **OpenAI：** コンテンツパート `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}`（`/v1/responses`：`{"type": "input_image", "image_url": "data:…"}`）。
- **Anthropic：** コンテンツブロック `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "…"}}`。
- **MCP：** `images: [{"data": "<base64>", "mime": "image/png"}]`、または `serve --image-dir <dir>`（複数回指定可）で指定したディレクトリ内のファイルに対する `[{"path": "/abs/file.png"}]`。パスはシンボリックリンク解決後に検証され、`--image-dir` が未指定の場合はすべてのパスが拒否されます。

PNG、JPEG、WebP、HEIC に対応し、1 リクエストあたり最大 8 枚、1 枚あたりエンコード後バイト数で最大 20 MB、リクエスト内の全画像合計で最大 32 MB です。いずれの上限を超過した場合も上限名を明示した 400 エラーとなります。リモート URL（`image_url` 内の `http(s)://`、`source.type: "url"`）は 400 で拒否されます — basalt はそれらを取得しないため、`data:` URI として送信してください。ビジョンモデルではないモデルに画像を送信すると、暗黙に無視されることはなく 400（`<model> does not take images`）となり、画像を含むリクエストは決して `--cloud-fallback` へ送られません。プロンプトキャッシュは各画像の SHA-256 をキーとするため、同じ画像を同じ順序で送ればキャッシュされたプレフィックスが再利用され、異なる画像では一切再利用されません。セッションログにはダイジェストとバイト数のみが記録され、画像バイト自体は記録されません。ツール結果内の画像はモデルには引き続き `[image omitted]` として渡されます。

## SSE ストリーミング（ネイティブ API）

```bash
curl -N http://localhost:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "maxTokens": 512}'
```

各イベントは `data: {...}` 行となり、ストリームは `data: [DONE]` で終了します。ストリーム途中で失敗が発生した場合は `event: error` と `data: {"error": ...}` が送信されます。

## テレメトリと Prometheus

- **`GET /api/v1/stats`** はテレメトリのスナップショットを JSON として返します。
  - 直近 1 分、5 分、1 時間のリクエスト数、トークン数、TTFT および tok/s の p50/p95、プロンプトキャッシュヒット率、ドラフト受理率を含む `windows` も返します。
  - 判断モデルごとのリクエスト数、回答数、棄権数、エスカレーション数、クラウドコスト、レイテンシ、確信度十分位数ごとの回答数を含む `decisions` も返します。
- `--experimental-expert-streaming` 指定時は、ストリーミングされたエキスパートキャッシュのヒット数・ミス数・エビクション数・読み込みバイト数、および MoE 層ごとのロード以降のルーティング済みエキスパート情報（回数、ユニーク数、上位 8 個とその割合）を含む `experts` も返します。
- `--telemetry-confidence-every N` を指定すると、N 回ごとの生成でトークンごとの最上位トークン確率、エントロピー、上位 2 候補マージンを記録します。明示的に要求したリクエストは常に記録されます。`confidence` は、最低の最上位トークン確率、平均確率、最小マージンの各十分位数ごと、および平均エントロピーの 0.5 nat 刻みビンごとの生成回数を集計します。計測対象の各トークンで GPU 同期が 1 回発生するため、デフォルトの 0 では要求されたリクエストのみを計測します。
- **`GET /api/v1/metrics`** は同じデータを Prometheus または Grafana 向けの OpenMetrics テキストとして返します。
  - メトリクス名は `basalt_` で始まります。ウィンドウや分位点は、判断モデルと同様にラベルになります。
  - 他のすべてのエンドポイントと同様にサーバーの API キー認証を通るため、スクレイパーに bearer トークンを設定してください。

```yaml
scrape_configs:
  - job_name: basalt
    metrics_path: /api/v1/metrics
    authorization: { credentials_file: /etc/prometheus/basalt.token }
    static_configs: [{ targets: ["127.0.0.1:8080"] }]
```

テレメトリには回数、タイミング、ヒストグラムのみが保持され、プロンプトや回答のテキストは一切含まれません（#450）。

## マルチモーダル、コンバーター、X-Ray、ガバナンス用エンドポイント（`v0.12`）

- **Video Studio (`/v1/videos/generations`, `/v1/videos/{id}`)** — `AVAssetWriter` でマルチプレクスした `.mp4` を返すフェーズ分割 24 GB UMA 生成（`LTX-2.5 AV-DiT`、`Kandinsky 6.0`、`LingBot-Video MoE`、`Wan 2.2`）（`#608`–`#611`）。
- **Audio & Music (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/music/generations`)** — `ZONOS3-Mini`、`Kokoro-82M`、`LFM2.5-Audio` Mimi/SNAC デュプレックスコーデック、`SmartTurn`/`Sortformer`、`MossFormer2`、`Soprano-80M`、`Qwen3ForcedAligner`、`ACE-Step 1.5`（`#604`、`#605`）。
- **Images (`/v1/images/generations`, `/v1/images/edits`)** — `FLUX.3`、`Z-Image-Turbo`、`Iris-3B`、`SANA-1.5`、`SeedVR2` アップスケール、RGBA サポート（`#542`、`#605`）。
- **Universal Converter (`/v1/models/convert`, `/v1/models/convert/jobs/{id}`)** — 非同期 any-to-any 量子化（`MACaQ`、`FlatQuant`、`LIRQuant`、`SINQ`、`QTIP`、`E8P`、`PV-Tuning`、`QAT STE` + `QuantizationAuditCore`、`#546`、`#623`）。
- **Token Attribution & Hallucination Risk (`POST /v1/xray/tokens`)** — トークンごとの logprobs、top-k エントロピー、層/ヘッド/MoE アトリビューションおよびハルシネーションリスクシグナル（`#613`、`#614`）。
- **HITL Governance (`/v1/mcp/approvals`)** — 影響の大きい MCP ツール呼び出しに対する Human-in-the-loop 承認キュー（`HITLGovernanceCore` および `SymbolicKnowledgeGraph` と連携、`#626`）。

## Swagger UI

インタラクティブな API ドキュメントは `http://localhost:8080/swagger` で参照できます。
