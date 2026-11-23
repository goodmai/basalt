# 連携・インテグレーション

> Languages: [English](../../integrations.md) · [Русский](../ru/integrations.md) · [中文](../zh/integrations.md) · [Español](../es/integrations.md) · [Français](../fr/integrations.md) · **日本語** · [Deutsch](../de/integrations.md) · [한국어](../ko/integrations.md) · [Bahasa Indonesia](../id/integrations.md)

## お使いのツールから利用する

basalt は 3 つのクライアントプロトコルに同時に対応しているため、お使いのツールがすでにサポートしているプロトコルをそのまま選択できます。

| クライアント | エンドポイント | セットアップ |
|---|---|---|
| 任意の OpenAI クライアント（Python/JS SDK、Continue、Cline、aider） | `http://localhost:8080/v1` | ループバック上でベース URL と任意の API キーを指定 |
| OpenCode | `/v1/chat/completions` | [OpenCode 連携](#opencode-連携)を参照 |
| ターミナル上の Claude Code | `/v1/messages`（Anthropic） | [Claude Code 連携](#claude-code-連携)を参照 |
| Claude Desktop、Cursor | MCP stdio | [MCP 連携](#mcp-連携claude-desktop--cursor)を参照 |

1 つのサーバーを起動して、必要な機能を追加するだけで、これらすべてから同時に利用できます：

```bash
basalt serve --rest --model Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed --mtp \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rerank-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

公式の OpenAI Python SDK が変更なしで動作します（`openai` 3.19.2 で動作確認済み）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(model="basalt", messages=[{"role": "user", "content": "Hello"}])
vectors = client.embeddings.create(model="basalt", input=["first text", "second text"])
```

リランクは Cohere/Jina 形式を使用します：

```bash
curl -s http://localhost:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 1}'
```

## Claude Code 連携

### 方法 A — 環境変数（ターミナルセッションごと）

```bash
# Start Basalt
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` は（`x-api-key` ではなく）`Authorization: Bearer local` として送信されるため、実際の `ANTHROPIC_API_KEY` に触れることはありません。

### 方法 B — シェル関数（`~/.zshrc` に追加）

```bash
function claude-local() {
  ANTHROPIC_BASE_URL=http://localhost:8080          \
  ANTHROPIC_AUTH_TOKEN=local                        \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
  ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
  ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
  claude "$@"
}

claude-local                   # sonnet alias → Qwen 4B
claude-local --model haiku     # haiku alias → Gemma 4B (fastest)
claude-local --model opus      # opus alias → Gemma 31B (strongest)
```

### 方法 C — `./basalt-claude` ランチャー

1 つのコマンドでサーバーと Claude Code を起動します（[インストールのランチャー解説](install.md#one-command-launcher-basalt-claude)を参照）。

### ツール呼び出し

Claude Code のツールは `/v1/messages` 上で動作します。送信された `tools` はモデルのチャットテンプレートに渡され、モデルの呼び出しは `tool_use` ブロックとして返され（`content_block_start` / `input_json_delta` / `content_block_stop`、`stop_reason: "tool_use"` としてストリーム配信）、履歴の `tool_use` / `tool_result` ブロックはプロンプトへレンダリングされるため、モデルはツールが返した結果を参照できます。`/v1/chat/completions` も OpenAI 形式（`tools`、`message.tool_calls` / `delta.tool_calls`、`finish_reason: "tool_calls"`、role `tool` メッセージ）で同様の動作を行い、`/v1/responses` もすでに対応しています。履歴内のツール結果は `tool_use_id` / `tool_call_id` によって対応する呼び出しと突き合わされるため、並列呼び出しでも名前が保持されます。ツール結果内の画像は `[image omitted]` として表示されます。`tool_choice: none` はプロンプトにツールを残しつつ呼び出しを一切通さず、ツール名を指定した場合はすべてのツールをプロンプトに残した上でその 1 つだけを通します（`tools` にない名前を指定すると 400）。`any` / `required` は強制されず `auto` として動作します（#42 を参照）。`parallel_tool_calls: false` / `disable_parallel_tool_use: true` は最初の呼び出し後に生成を終了します。JSON オブジェクトではないツール入力は `{}` として渡され、元の引数は目に見える注記として添付されます。モデルが修復不能なほど破損させた呼び出しや、提供されていないツールへの呼び出しは、テキスト内に `[tool call … rejected: …]` という注記として現れます。レガシーな Chat Completions の `functions` / `function_call` フィールドは無視されます。`mlx-community/Qwen3-4B-4bit` 上で Claude Code 2.1 を用いて、`Bash` でディレクトリを一覧表示し、`Read` / `Edit` でファイルを読み取って編集する動作を確認済みです。

### モデルディスカバリー

Claude Code（v2.1.126+）は起動時に `GET /v1/models` をクエリして返されたモデルを `/model` ピッカーに追加しますが、ID が `claude` または `anthropic` で始まる場合に限られます。Basalt はディスカバリーが自動的に機能するよう `claude-local/<hf-id>` 形式で ID を返します。

---

## OpenCode 連携

[OpenCode](https://github.com/opencode-ai/opencode) はターミナルネイティブの AI コーディングエージェントです。Basalt は、クラウドへの依存や API キーなしで OpenCode に直接接続できるネイティブの OpenAI 互換 API（`/v1/chat/completions`、`/v1/models`）と Anthropic 互換 API を提供します。

### 1. Basalt サーバーを起動する

お好みのローカルモデル（例：Huihui Qwen3.8 Abliterated、Ex0bit MYTHOS 26B、または Qwen3.5 4B）を指定して Basalt を起動します：

```bash
# High capability (Abliterated / MoE)
"$(swift build -c release --show-bin-path)/Basalt" serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. OpenCode を接続する

#### 方法 A — ターミナルの環境変数

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### 方法 B — OpenCode 設定ファイル（`~/.config/opencode/config.json` または `opencode.json`）

```json
{
  "provider": "openai",
  "base_url": "http://localhost:8080/v1",
  "api_key": "local",
  "model": "basalt",
  "temperature": 0.7,
  "max_tokens": 16384
}
```

#### 方法 C — シェルエイリアス（`~/.zshrc` に追加）

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

## MCP 連携（Claude Desktop / Cursor）

MCP 設定ファイル（`~/.config/claude/claude_desktop_config.json`）に追加します：

```json
{
  "mcpServers": {
    "basalt": {
      "command": "/path/to/basalt",
      "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--mcp"]
    }
  }
}
```

利用可能な MCP ツール：

| ツール | 説明 |
|---|---|
| `gemma_generate` | テキスト生成 — `prompt`、`maxTokens`、`temperature`、`topP`、`minP`、`dynamicTemperature`、`images`（[画像入力](rest-api.md#画像入力)を参照） |
| `gemma_status` | サーバーの準備状態、バージョン、現在のモデルを返します |
| `status` | エンジンの健全性、ロード済みモデル、メモリスナップショット、テレメトリを返します — 確認対象の `modelId` を任意指定可能 |
| `check_hardware` | Apple Silicon のリソース情報を取得します：チップ、ユニファイドメモリ、GPU、ディスク |
| `inspect_model` | `modelId`（リポジトリ ID またはローカルパス） — ウェイトシャードを検証し、MLX/Safetensors ウェイトの実際のビット幅を検出します |
| `switch_model` | ローカルキャッシュ内の別モデルへ切り替えます — `modelId` |
| `gemma_add_knowledge` | Harness Experience Library（`~/.basalt/harness-library`）に知見を保存します |
| `harness_validate` | 作業ディレクトリまたは `BASALT_HARNESS_ROOTS` 配下の Claude Code / opencode / AGENTS.md ハーネスを検証します |
| `harness_optimize` | ロード済みモデル上で 1 回の Harness² propose–probe–compose パスを実行します（`basalt harness optimize` でも実行可能） |
| `harness_library_query` | Experience Library を検索します |
| `model_describe`、`model_tensors`、`model_tensor`、`kv_inspect`、`trace_run` | Model X-ray（読み取り専用、バックグラウンド優先度）：レイヤー構造、パラメータ一覧および境界付きテンソル統計、プロンプトキャッシュツリー、1 つのプロンプトに対するキャプチャ実行（#328） |
| `intervene`、`weights_delta`、`overlays_list`、`overlay_undo`、`experiment_compare` | サンドボックス実験（`serve --research-tools` 指定時のみ）：リサーチ呼び出しにのみ適用され元に戻せるオーバーレイとしてのステアリングおよびヘッド/エキスパートのアブレーション。ウェイトデルタはライブモデルを変更するため、有効な間はすべての応答に変更済みマークが付き、プロンプトキャッシュの再利用はオフになります。[research-tools.md](research-tools.md) を参照してください（#328） |
