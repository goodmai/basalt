# REST-API

> Languages: [English](../../rest-api.md) · [Русский](../ru/rest-api.md) · [中文](../zh/rest-api.md) · [Español](../es/rest-api.md) · [Français](../fr/rest-api.md) · [日本語](../ja/rest-api.md) · **Deutsch** · [한국어](../ko/rest-api.md) · [Bahasa Indonesia](../id/rest-api.md)

Basis-URL: `http://127.0.0.1:8080` — **keine Authentifizierung erforderlich**.

## Sicherheit

Auf Loopback darf jeder lokale Prozess die API ohne Key aufrufen, eine Webseite in Ihrem Browser jedoch nicht:

- **Host.** An Loopback gebunden beantwortet der Server nur Anfragen, deren `Host` `localhost`, `127.0.0.1` oder `[::1]` (beliebiger Port) ist; alles andere erhält 403. Dies verhindert DNS-Rebinding, bei dem die Domain eines Angreifers neu auf 127.0.0.1 aufgelöst wird und seine Seite Ihre Antworten mitliest. Bei Bindung an eine bestimmte Adresse wird diese Adresse ebenfalls akzeptiert. Bei Bindung an `0.0.0.0`/`::` wird jeder `Host` akzeptiert: Dort ist der API-Key zwingend erforderlich.
- **Origin.** Eine Anfrage mit einem `Origin`-Header wird abgelehnt (403), es sei denn, es handelt sich um den eigenen Loopback-Origin des Servers (`http(s)://localhost|127.0.0.1|[::1]` auf dem Port, auf dem der Server lauscht; kein Port bedeutet 80 für http, 443 für https) oder er ist über `--allow-origin` / `allowed_origins` aufgeführt (`scheme://host[:port]`, kein Pfad, kein abschließender Schrägstrich, kein `*`; Standard-Ports werden abgeglichen, sodass `https://app.example:443` gleich `https://app.example` ist). Andere lokale Ports sind nicht vertrauenswürdig: Eine Seite auf einem Dev-Server (`localhost:3000`) oder Jupyter könnte sonst Modelle laden und Generierungen ausführen, daher benötigt eine solche UI `--allow-origin http://localhost:3000`. Clients, die keinen `Origin` senden (curl, SDKs, Claude Code, Codex), sind nicht betroffen.
- **Content-Type.** Jedes `POST` muss `Content-Type: application/json` senden (ein `charset`-Parameter ist zulässig), sonst 415. `curl -d` ohne `-H "Content-Type: application/json"` sendet einen Form-Typ und wird abgelehnt.
- **Modell-IDs.** REST-`model`-Werte sind Hugging-Face-Repo-IDs: `/v1/models/load` akzeptiert nur `org/name`; im `model`-Feld einer Generierungsanfrage führt eine Pfadform (beginnend mit `/`, `.` oder `~` oder mit `..`, auch versteckt in `claude-local/..--x`; umgebende Leerzeichen werden ignoriert) zu 400, während jeder andere Wert, der keine Repo-ID ist (`basalt`, `qwen/qwen3-8b-instruct:free`), ein Alias ist: das geladene Modell antwortet und nichts wird umgeschaltet. Lokale Pfade bleiben für `basalt serve --model-path` und MCP verfügbar. Ein fehlgeschlagener Ladevorgang lässt das aktuelle Modell geladen, und Fehler-Bodys enthalten niemals Dateisystempfade (diese stehen im Server-Log).

## Modellverwaltung

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

`GET /v1/models` liefert IDs mit dem Präfix `claude-local/`, damit die automatische Modellauswahl von Claude Code sie hinzufügt. Das Feld `display_name` enthält die ursprüngliche HuggingFace-Repo-ID.

## Rohe Generierung

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement.", "maxTokens": 256}'
```

| Feld | Typ | Standard | Beschreibung |
|---|---|---|---|
| `prompt` | string | erforderlich | Eingabetext |
| `maxTokens` | int | 8192 | Maximale Anzahl zu generierender Tokens |
| `temperature` | float | Modell | Sampling-Temperatur (0–2); ungesetzt übernimmt die `generation_config.json` des Modells, sonst 0.7 |
| `topP` | float | Modell | Nucleus-Sampling (0–1); ungesetzt übernimmt `generation_config.json`, sonst 0.9 |
| `topK` | int | Modell | Behält die k wahrscheinlichsten Tokens; ungesetzt übernimmt `--top-k`, dann `generation_config.json`, sonst aus |
| `presencePenalty` | float | aus | −2…2, wird von jedem Token unter den letzten 1024 generierten Tokens abgezogen (der Prompt wird nicht bestraft) |
| `repetitionPenalty` | float | `--repetition-penalty` | (0, 2], multiplikative Strafe auf kürzliche Tokens |
| `thinkingBudget` | int | aus | Reasoning-Tokens, bevor der Server das Ende des Denkens erzwingt (Qwen: „Considering the limited time by the user…“ + `</think>`), damit die Antwort in `maxTokens` passt; wird herabgesetzt, um in `maxTokens` abzüglich des Übergangs und einer Antwortreserve von min(512, maxTokens/4) zu passen; ein `maxTokens`, das nach der Reserve weniger als 64 Tokens übrig lässt, liefert 400; wird ignoriert, wenn Thinking aus ist, das Modell kein Reasoning-Protokoll hat oder es ein Nicht-Qwen-Modell ist, dessen Protokoll keinen Übergang deklariert |
| `minP` | float | Server | Min-P 0–1: verwirft Tokens unter `minP × max p`. Überschreibt `--min-p` und das Modellprofil; `0` schaltet es ab |
| `dynamicTemperature` | bool | false | Entropiebasierte dynamische Temperatur (EDT): pro Token `T = T0·0.8^(1.5/H) + 0.5·R`, begrenzt auf 0.05–2 (eine kältere Anfrage behält ihre eigene Temperatur als Untergrenze; `top_p`/`min_p` schneiden die ursprüngliche Verteilung zu), wobei `H` die Entropie der Verteilung für das nächste Token und `R ∈ [0,1]` ein Wiederholungsindex ist. Nahezu greedy bei sicheren Tokens, wärmer, wenn eine Schleife beginnt. Keine Wirkung bei `temperature: 0` |
| `speculativeLookup` | bool | `--speculative-lookup` | Prompt-Lookup-Speculative-Decoding für diese Anfrage: bis zu 16 Tokens, die aus einem n-Gramm-Treffer im Prompt oder in der Ausgabe kopiert werden, werden in einem Forward-Pass verifiziert; nach einem verworfenen Entwurf pausiert das Drafting für 1, 2, 4… bis zu 64 Tokens. Mit `serve --speculative-recycling` entwirft es, wenn kein n-Gramm passt, bis zu 6 Tokens aus den zweimal hintereinander aufgezeichneten eigenen Argmax-Nachfolgern des Modells (Token Recycling, #38), mit eigener Pause. Nur reine Attention-Modelle und dichte rekurrente Hybriden (Qwen3.5/Ornith, Nemotron-H: ein verworfener Entwurf stellt den rekurrenten Zustand wieder her und spielt die akzeptierten Tokens erneut ab), Sliding-Window-Modelle (Gemma 4: jede Verifikationsrunde wird gestaged, sodass ein verworfener Entwurf nie den Ring des Fensters überschreibt), ohne `--kv-bits` und nur mit reiner Texteingabe; rekurrente Hybriden mit Expert-Routing (LFM2.5 MoE) werden ignoriert. Auf Gemma 4 26B-A4B (MoE) beschleunigt Lookup Code-Bearbeitungen um 1.97× und kostet bei Prosa und Reasoning 3–6% |

`/v1/chat/completions`, `/v1/responses` und `/v1/messages` akzeptieren dieselben beiden Felder als `min_p` und `dynamic_temperature`; MCP `gemma_generate` als `minP` und `dynamicTemperature`. Die OpenAI-Endpunkte nehmen `top_k`, `presence_penalty`, `repetition_penalty` und `thinking_budget`; `/v1/messages` nimmt `top_k` und `thinking: {"type": "enabled", "budget_tokens": N}` als Thinking-Budget (1 ≤ N < `max_tokens`, sonst 400). Alle drei akzeptieren `speculative_lookup` (ein Nicht-Boolean liefert 400); MCP `gemma_generate` akzeptiert `speculativeLookup` (ein Nicht-Boolean liefert -32602). Usage meldet `completion_tokens_details.reasoning_tokens` (Chat; gestreamt mit `stream_options.include_usage`) und `output_tokens_details.reasoning_tokens` (Responses): `0`, wenn das Modell kein Reasoning durchgeführt hat, und der erzwungene Übergang zählt als Reasoning.

## OpenAI-kompatibel

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Unterstützt SSE-Streaming, System-Prompts und mehrstufige Konversationen. Das Senden einer HuggingFace-Modell-ID im Feld `"model"` löst automatisch einen Hot-Swap aus.

Die Gedankengänge eines Reasoning-Modells kommen als `message.reasoning` zurück (Stream: `delta.reasoning`), dem Namen von OpenRouter / Ollama, und derselbe Text noch einmal als `reasoning_content`, dem Namen, den vLLM-, DeepSeek-, llama.cpp- und LM-Studio-Clients lesen. Zurückgesendete Assistant-Turns mit einem der beiden Felder werden akzeptiert; das Reasoning selbst wird nicht wieder in den Prompt eingefügt.

## Anthropic-kompatibel

```bash
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Implementiert die vollständige Anthropic-SSE-Event-Sequenz (`message_start`, `content_block_start`, `content_block_delta`, `message_stop`). Akzeptiert Content-Formate als einfachen String sowie als Block-Array.

`thinking`-Blöcke tragen eine leere `signature`: Basalt besitzt keinen Key, um sie zu signieren. Sie funktionieren im Round-Trip durch Basalt, aber die echte Anthropic-API lehnt eine Konversation ab, die sie erneut abspielt; entfernen Sie sie daher, bevor Sie denselben Verlauf dorthin senden.

**Beschränkte Ausgabe (Constrained Output).** `response_format` schränkt die Antwort nach einem eventuellen Reasoning-Block über XGrammar ein:
- `{"type": "json_object"}` oder `{"type": "json_schema", "json_schema": {"schema": …}}`;
- `{"type": "regex", "regex": "[A-Z]{3}-\\d{4}"}`;
- `{"type": "grammar", "grammar": "root ::= …"}`, EBNF/GBNF-Dialekt von XGrammar mit Startregel `root`;
- `{"type": "structural_tag", "format": …}`, das Structural-Tag-JSON von XGrammar (`sequence`, `const_string`, `regex`, `json_schema`, `tag`, …).

Die vLLM-Anfragefelder `guided_regex` und `guided_grammar` funktionieren ebenfalls, jeweils eines gleichzeitig. Das MCP-Tool `generate` akzeptiert `regex` oder `grammar`. Eine Quelle über 64 KB oder eine, die XGrammar nicht kompiliert, liefert ein 400, das das Format benennt.

**Prefill.** Eine Konversation, die mit einer Assistant-Nachricht endet, setzt diese Nachricht fort, statt sie in einem neuen Turn zu beantworten — auf `/v1/messages`, `/v1/chat/completions` und `/v1/responses`, wie bei der Anthropic-API, `continue_final_message` in vLLM und llama.cpp. Die früheren Turns werden bis zum Generierungs-Prompt mit ausgeschaltetem Thinking gerendert, der Assistant-Text folgt ungeschlossen darauf, und die Antwort enthält nur das, was das Modell danach schreibt: bei Prefill `{"name":` lautet die Antwort ` "Whiskers", "age": 3}`. Bei `response_format` oder einem erzwungenen Tool bildet der Prefill den Anfang des JSON-Dokuments (ein Prefill, mit dem das Schema nicht beginnen kann, liefert 400), und die Argumente eines erzwungenen Tools schließen ihn ein. Ein Prefill, der einen Thinking-Block öffnet (`<think>…`), behält das angeforderte Thinking bei und setzt als Reasoning fort. Mehrere abschließende Assistant-Nachrichten werden zu einem einzigen Prefill zusammengefügt; eine leere wird verworfen; eine Assistant-Nachricht mit Tool-Calls bleibt ein abgeschlossener Turn. Ein Prompt mit Bildern kann nicht fortgesetzt werden (400). Kontroll-Tokens im Prefill (`<|im_end|>`, `<|im_start|>`, …) werden wie in der übrigen Konversation neutralisiert, sodass ein Prefill keinen Turn fälschen kann; nur die Thinking-Marker, die sein Reasoning öffnen und schließen, bleiben echt, und bei einem Template, das `<think>` bereits selbst öffnet (Ornith, Qwen3.5+), wird ein führendes `<think>` des Prefills entfernt statt verdoppelt. Bei `--cloud-fallback` bleibt eine geprefillte Anfrage immer lokal, da nicht jeder Anbieter eine abschließende Assistant-Nachricht fortsetzt. Bei Nemotron Nano v2, das Thinking über eine `/no_think`-Zeile im System-Turn abschaltet, rendert eine geprefillte Anfrage einen anderen System-Turn als eine Anfrage mit aktiviertem Thinking auf demselben Verlauf, sodass beide nur den Prompt-Cache davor teilen.

## Bildeingabe

Ein Vision-Modell (Gemma 4, Qwen3.5-VL, … — jedes Modell, das MLXVLM lädt) akzeptiert Bilder auf `/v1/chat/completions`, `/v1/responses`, `/v1/messages` und dem MCP-Tool `generate`:

- **OpenAI:** ein Content-Part `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` (`/v1/responses`: `{"type": "input_image", "image_url": "data:…"}`).
- **Anthropic:** ein Content-Block `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "…"}}`.
- **MCP:** `images: [{"data": "<base64>", "mime": "image/png"}]` oder `[{"path": "/abs/file.png"}]` für eine Datei innerhalb eines Verzeichnisses, das mit `serve --image-dir <dir>` (wiederholbar) angegeben wurde. Der Pfad wird nach dem Auflösen von Symlinks geprüft; ohne `--image-dir` wird jeder Pfad abgelehnt.

PNG, JPEG, WebP und HEIC, höchstens 8 Bilder pro Anfrage, 20 MB kodierte Bytes pro Bild und 32 MB für alle Bilder einer Anfrage; jede Überschreitung führt zu 400 mit Benennung des Limits. Remote-URLs (`http(s)://` in `image_url`, `source.type: "url"`) werden mit 400 abgelehnt — Basalt ruft sie nicht ab; senden Sie das Bild als `data:`-URI. Ein Bild, das an ein Modell gesendet wird, das kein Vision-Modell ist, liefert 400 (`<model> does not take images`) und wird niemals stillschweigend verworfen; zudem geht eine Anfrage mit Bildern niemals an `--cloud-fallback`. Der Prompt-Cache wird nach dem SHA-256 jedes Bildes geschlüsselt: dieselben Bilder in derselben Reihenfolge verwenden das zwischengespeicherte Präfix wieder, andere Bilder verwenden nichts wieder. Sitzungsprotokolle erfassen Digests und Byte-Zahlen, niemals Bildbytes. Bilder innerhalb von Tool-Ergebnissen werden dem Modell weiterhin als `[image omitted]` gezeigt.

## SSE-Streaming (native API)

```bash
curl -N http://localhost:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "maxTokens": 512}'
```

Jedes Event ist eine `data: {...}`-Zeile; der Stream endet mit `data: [DONE]`. Ein Fehler mitten im Stream trifft als `event: error` mit `data: {"error": ...}` ein.

## Telemetrie und Prometheus

- **`GET /api/v1/stats`** gibt den Telemetrie-Snapshot als JSON zurück.
  - Er enthält außerdem `windows`: die vergangenen 1 Minute, 5 Minuten und 1 Stunde mit Anfragen, Tokens, p50/p95 Time-to-First-Token und tok/s, der Trefferquote des Prompt-Caches sowie der Draft-Akzeptanzrate.
  - Er enthält außerdem `decisions`: pro Entscheidungsmodell Anfragen, Antworten, Enthaltungen, Eskalationen, Cloud-Kosten, Latenz und Antworten pro Konfidenzdezil.
- Mit `--experimental-expert-streaming` enthält er außerdem `experts`: Treffer, Fehltreffer, Verdrängungen und gelesene Bytes des gestreamten Experten-Caches sowie pro MoE-Layer die seit dem Laden gerouteten Experten (Anzahl, eindeutige Experten, die acht meistgenutzten und ihr Anteil).
- Mit `--telemetry-confidence-every N` zeichnet jede N-te Generierung pro Token die Wahrscheinlichkeit des Top-Tokens, die Entropie und die Marge zwischen den Top zwei auf. Anfragen, die dies anfordern, werden immer aufgezeichnet. `confidence` zählt dann Generierungen pro Dezil der niedrigsten Top-Token-Wahrscheinlichkeit, der mittleren Wahrscheinlichkeit und der kleinsten Marge sowie pro Halbnat-Bin der mittleren Entropie. Jedes gemessene Token kostet eine GPU-Synchronisation, daher misst der Standardwert 0 nur Anfragen, die es anfordern.
- **`GET /api/v1/metrics`** gibt dieselben Daten als OpenMetrics-Text für Prometheus oder Grafana zurück.
  - Metriknamen beginnen mit `basalt_`. Fenster und Quantil sind Labels, ebenso das Entscheidungsmodell.
  - Der Zugriff läuft wie bei jedem anderen Endpunkt über den API-Key des Servers: Geben Sie dem Scraper das Bearer-Token.

```yaml
scrape_configs:
  - job_name: basalt
    metrics_path: /api/v1/metrics
    authorization: { credentials_file: /etc/prometheus/basalt.token }
    static_configs: [{ targets: ["127.0.0.1:8080"] }]
```

Die Telemetrie speichert ausschließlich Zähler, Zeitmessungen und Histogramme: keinen Prompt- oder Antworttext (#450).

## Multimodal-, Konverter-, X-Ray- & Governance-Endpunkte (`v0.12`)

- **Video Studio (`/v1/videos/generations`, `/v1/videos/{id}`)** — Gestaffelte 24-GB-UMA-Generierung (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video MoE`, `Wan 2.2`) mit Rückgabe von `.mp4`, gemuxt über `AVAssetWriter` (`#608`–`#611`).
- **Audio & Musik (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/music/generations`)** — `ZONOS3-Mini`, `Kokoro-82M`, `LFM2.5-Audio` Mimi/SNAC-Duplex-Codecs, `SmartTurn`/`Sortformer`, `MossFormer2`, `Soprano-80M`, `Qwen3ForcedAligner`, `ACE-Step 1.5` (`#604`, `#605`).
- **Bilder (`/v1/images/generations`, `/v1/images/edits`)** — `FLUX.3`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`, `SeedVR2`-Upscaling, RGBA-Unterstützung (`#542`, `#605`).
- **Universeller Konverter (`/v1/models/convert`, `/v1/models/convert/jobs/{id}`)** — Asynchrone Any-to-Any-Quantisierung (`MACaQ`, `FlatQuant`, `LIRQuant`, `SINQ`, `QTIP`, `E8P`, `PV-Tuning`, `QAT STE` + `QuantizationAuditCore`, `#546`, `#623`).
- **Token-Attribution & Halluzinationsrisiko (`POST /v1/xray/tokens`)** — Logprobs pro Token, Top-k-Entropie, Layer-/Head-/MoE-Attribution und Signale für das Halluzinationsrisiko (`#613`, `#614`).
- **HITL-Governance (`/v1/mcp/approvals`)** — Human-in-the-Loop-Freigabewarteschlange für folgenreiche MCP-Tool-Aufrufe, gekoppelt mit `HITLGovernanceCore` und `SymbolicKnowledgeGraph` (`#626`).

## Swagger-UI

Besuchen Sie `http://localhost:8080/swagger` für die interaktive API-Dokumentation.
