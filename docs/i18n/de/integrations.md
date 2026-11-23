# Integrationen

> Languages: [English](../../integrations.md) · [Русский](../ru/integrations.md) · [中文](../zh/integrations.md) · [Español](../es/integrations.md) · [Français](../fr/integrations.md) · [日本語](../ja/integrations.md) · **Deutsch** · [한국어](../ko/integrations.md) · [Bahasa Indonesia](../id/integrations.md)

## Nutzung aus Ihren Tools

Basalt spricht drei Client-Protokolle gleichzeitig, sodass Sie genau das wählen können, was Ihr Tool bereits nutzt.

| Client | Endpunkt | Einrichtung |
|---|---|---|
| Beliebiger OpenAI-Client (Python/JS-SDK, Continue, Cline, aider) | `http://localhost:8080/v1` | Basis-URL + beliebiger API-Key auf Loopback |
| OpenCode | `/v1/chat/completions` | siehe [OpenCode-Integration](#opencode-integration) |
| Claude Code im Terminal | `/v1/messages` (Anthropic) | siehe [Claude-Code-Integration](#claude-code-integration) |
| Claude Desktop, Cursor | MCP stdio | siehe [MCP-Integration](#mcp-integration-claude-desktop--cursor) |

Starten Sie einen einzigen Server für alle, mit den gewünschten Extras:

```bash
basalt serve --rest --model Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed --mtp \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rerank-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

Das offizielle OpenAI-Python-SDK funktioniert unverändert (geprüft mit `openai` 3.19.2):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(model="basalt", messages=[{"role": "user", "content": "Hello"}])
vectors = client.embeddings.create(model="basalt", input=["first text", "second text"])
```

Rerank verwendet das Cohere-/Jina-Format:

```bash
curl -s http://localhost:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 1}'
```

## Claude-Code-Integration

### Option A — Umgebungsvariablen (pro Terminalsitzung)

```bash
# Start Basalt
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` wird als `Authorization: Bearer local` gesendet (nicht als `x-api-key`), sodass Ihr echter `ANTHROPIC_API_KEY` niemals angerührt wird.

### Option B — Shell-Funktion (in `~/.zshrc` einfügen)

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

### Option C — `./basalt-claude`-Launcher

Startet den Server und Claude Code in einem Befehl (siehe [den Launcher unter Installation](install.md#ein-befehl-launcher-basalt-claude)).

### Tool-Aufrufe (Tool Calling)

Die Tools von Claude Code funktionieren über `/v1/messages`: Die gesendeten `tools` erreichen das Chat-Template des Modells, die Aufrufe des Modells kommen als `tool_use`-Blöcke zurück (gestreamt als `content_block_start` / `input_json_delta` / `content_block_stop`, `stop_reason: "tool_use"`), und die `tool_use`- / `tool_result`-Blöcke des Verlaufs werden in den Prompt gerendert, sodass das Modell sieht, was seine Tools zurückgegeben haben. `/v1/chat/completions` erledigt dasselbe im OpenAI-Format (`tools`, `message.tool_calls` / `delta.tool_calls`, `finish_reason: "tool_calls"`, Nachrichten mit Rolle `tool`), genau wie `/v1/responses` bereits zuvor. Tool-Ergebnisse im Verlauf werden ihren Aufrufen über `tool_use_id` / `tool_call_id` zugeordnet, sodass parallele Aufrufe ihre Namen behalten; Bilder in einem Tool-Ergebnis erscheinen als `[image omitted]`. `tool_choice: none` belässt die Tools im Prompt und lässt keinen Aufruf durch; ein namentlich genanntes Tool belässt alle Tools im Prompt und lässt nur dieses eine durch (die Angabe eines Tools, das nicht in `tools` steht, liefert 400); `any` / `required` werden nicht erzwungen und verhalten sich wie `auto` (siehe #42); `parallel_tool_calls: false` / `disable_parallel_tool_use: true` beenden die Generierung nach dem ersten Aufruf. Ein Tool-Input, der kein JSON-Objekt ist, wird als `{}` mit den ursprünglichen Argumenten in einem sichtbaren Hinweis ausgegeben. Ein vom Modell irreparabel verunstalteter Aufruf oder ein Aufruf eines nicht angebotenen Tools erscheint als sichtbarer Hinweis `[tool call … rejected: …]` im Text. Die veralteten Chat-Completions-Felder `functions` / `function_call` werden ignoriert. Geprüft mit Claude Code 2.1 auf `mlx-community/Qwen3-4B-4bit`: Verzeichnis mit `Bash` auflisten sowie eine Datei mit `Read` / `Edit` lesen und anschließend bearbeiten.

### Modellerkennung

Claude Code (v2.1.126+) fragt beim Start `GET /v1/models` ab und fügt zurückgegebene Modelle zur `/model`-Auswahl hinzu — aber nur, wenn die ID mit `claude` oder `anthropic` beginnt. Basalt liefert IDs in der Form `claude-local/<hf-id>`, damit die Erkennung automatisch funktioniert.

---

## OpenCode-Integration

[OpenCode](https://github.com/opencode-ai/opencode) ist ein terminal-nativer KI-Programmieragent. Basalt stellt eine native OpenAI-kompatible API (`/v1/chat/completions`, `/v1/models`) sowie eine Anthropic-kompatible API bereit, die sich ohne Cloud-Abhängigkeiten oder API-Keys direkt mit OpenCode verbindet.

### 1. Basalt-Server starten

Starten Sie Basalt mit Ihrem gewünschten lokalen Modell (z. B. Huihui Qwen3.8 Abliterated, Ex0bit MYTHOS 26B oder Qwen3.5 4B):

```bash
# High capability (Abliterated / MoE)
"$(swift build -c release --show-bin-path)/Basalt" serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. OpenCode verbinden

#### Option A — Terminal-Umgebungsvariablen

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### Option B — OpenCode-Konfigurationsdatei (`~/.config/opencode/config.json` oder `opencode.json`)

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

#### Option C — Shell-Alias (in `~/.zshrc` einfügen)

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

## MCP-Integration (Claude Desktop / Cursor)

Fügen Sie dies zu Ihrer MCP-Konfiguration (`~/.config/claude/claude_desktop_config.json`) hinzu:

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

Verfügbare MCP-Tools:

| Tool | Beschreibung |
|---|---|
| `gemma_generate` | Text generieren — `prompt`, `maxTokens`, `temperature`, `topP`, `minP`, `dynamicTemperature`, `images` (siehe [Bildeingabe](rest-api.md#bildeingabe)) |
| `gemma_status` | Gibt Server-Bereitschaft, Version und aktuelles Modell zurück |
| `status` | Engine-Zustand, geladenes Modell, Speicher-Snapshot und Telemetrie — optional `modelId` zur Prüfung |
| `check_hardware` | Apple-Silicon-Ressourcen: Chip, Unified Memory, GPU, Festplatte |
| `inspect_model` | `modelId` (Repo-ID oder lokaler Pfad) — Gewichts-Shards verifizieren und die reale Bitbreite von MLX-/Safetensors-Gewichten erkennen |
| `switch_model` | Zu einem anderen Modell aus dem lokalen Cache wechseln — `modelId` |
| `gemma_add_knowledge` | Eine Erkenntnis in der Harness Experience Library (`~/.basalt/harness-library`) speichern |
| `harness_validate` | Einen Claude-Code- / opencode- / AGENTS.md-Harness unter dem Arbeitsverzeichnis oder `BASALT_HARNESS_ROOTS` validieren |
| `harness_optimize` | Ein Harness²-Propose–Probe–Compose-Durchlauf auf dem geladenen Modell (auch `basalt harness optimize`) |
| `harness_library_query` | Die Experience Library durchsuchen |
| `model_describe`, `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run` | Model X-Ray (schreibgeschützt, Hintergrundpriorität): Layer-Anatomie, Parameterliste und begrenzte Tensor-Statistiken, der Präfixbaum des Prompt-Caches, ein Mitschnitt-Lauf eines Prompts (#328) |
| `intervene`, `weights_delta`, `overlays_list`, `overlay_undo`, `experiment_compare` | Isolierte Experimente, nur mit `serve --research-tools`: Steering und Head-/Expert-Ablation als rückgängig machbare Overlays, die nur auf Research-Aufrufe angewendet werden; Weight-Deltas verändern das laufende Modell, sodass währenddessen jede Antwort als modifiziert markiert wird und die Prompt-Cache-Wiederverwendung abgeschaltet ist; siehe [research-tools.md](research-tools.md) (#328) |
