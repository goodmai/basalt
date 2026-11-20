# basalt compose

> Languages: [English](../../compose.md) · [Русский](../ru/compose.md) · [中文](../zh/compose.md) · [Español](../es/compose.md) · [Français](../fr/compose.md) · [日本語](../ja/compose.md) · **Deutsch**

`basalt compose` stellt ein in einer einzigen YAML-Datei beschriebenes Modellsystem bereit. Die Datei listet lokale und Cloud-Modelle sowie deren Speicherhaltung auf. Außerdem beschreibt sie die Pipelines, die Anfragen zwischen diesen Modellen routen. Jede Pipeline durchläuft zunächst `perceive` (OCR, Sprache), dann `decide` (System One) und schließlich `answer` (ein Chat-Modell). Es handelt sich um ein Mixture-of-Models-Verfahren im Stil einer `docker-compose`-Datei für Modelle.

```bash
basalt compose plan -f basalt.yaml        # validate and show the compute plan on this Mac
basalt compose up -f basalt.yaml          # serve it; run it again with a changed file to apply the diff
basalt compose ps                         # what is loaded, when on-demand models unload, cloud spend
basalt compose down <name>                # stop it
basalt compose up document-qa             # legacy bundled files: document-qa, voice-agent, zonos-voice
basalt compose up 24gb-voice-zonos        # 84 hardware-tiered profiles: <tier>-<archetype>
```

`compose up` startet `basalt serve` mit den Flags, zu denen die Datei expandiert wird; diese werden beim Start ausgegeben. Es stellt dieselben REST- und MCP-Schnittstellen bereit. Flags nach `--` werden an `serve` weitergereicht, zum Beispiel `basalt compose up -f basalt.yaml -- --port 8090`.

Der Server bindet zuerst seinen Port und lädt die Modelle anschließend. Während des Ladens zeigt `compose ps` sie als `loading` an, `GET /v1/health/ready` antwortet mit 503, und der Laufzeiteintrag existiert bereits, sodass ein zweites `compose up` das bestehende System findet, anstatt ein weiteres zu starten. Sobald die Modelle geladen sind, erhält jedes eine Startprüfung (Startup Probe):
- das Chat-Modell und jedes lokale Antwortmodell generieren ein Token;
- jedes lokale Entscheidungsmodell beantwortet eine Frage;
- bei einem On-Demand-Modell werden die Dateien geprüft, ohne es zu laden.

Ein Modell, das seine Prüfung nicht besteht, wird in `compose ps` zusammen mit dem Grund als `failed` angezeigt. `/v1/health/ready` bleibt auf 503, bis ein `compose up` das Modell ersetzt. Ein Modell, das `compose up` neu lädt oder nachlädt, erhält dieselbe Prüfung. Mit `--mcp` werden die Modelle wie bisher geladen, bevor der Server startet.

## Die Datei

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

### Modelle
- **`source`:** ein Hugging-Face-Repository, ein lokaler Pfad (`./`, `/` oder `~`), `cloud:openrouter/<model>` oder `cloud:gemini/<model>` für Googles Entscheidungsmodelle wie `gemini-decision-1` (nur für `decide`-Kaskaden, der Schlüssel liegt üblicherweise in `GEMINI_API_KEY`).
- **`keep`** gilt für lokale Modelle:
  - `resident` ist der Standard und bleibt geladen;
  - `on-demand` lädt bei der ersten Anfrage, die das Modell benötigt, und entlädt nach `idle_unload`. Zeitangaben haben das Format `30s`, `5m` oder `1h`, der Standard beträgt 120 s;
  - `sequential` lädt einen Perceiver ausschließlich für seine Stufe. In der Standard-Pipeline kann es um den Perceiver herum auch das Chat-Modell entladen, wenn beide nicht gleichzeitig in den Speicher passen.
- **Cloud-Modelle:**
  - Benennen Sie die Umgebungsvariable, die den Schlüssel enthält, mit `api_key_env`. Ein direkt in die Datei geschriebener Schlüssel wird abgelehnt, ebenso wie `${…}`-Interpolation.
  - `budget: { usd_per_day }` begrenzt die Ausgaben eines einzelnen Modells.
  - `data: no-perceived` ist der Standard: aus Dokumenten oder Audio erkannter Text geht niemals an ein Cloud-Modell.
  - `data: media`, nur für `cloud:gemini`-Entscheidungsmodelle, sendet die angehängten Dateien einer Pipeline selbst (PDFs, Bilder, Audio als WAV) sowie deren erkannten Text für die Entscheidung an dieses Modell. Verwenden Sie dies nur für Dokumente, die diesen Mac verlassen dürfen.

### Speicher und der Plan
`compose plan` summiert die residenten Bytes jedes lokalen Modells. Dabei zählt ein `on-demand`-Modell als resident, da es jederzeit geladen werden kann.

- `memory: resident` erfordert, dass die Summe in das Working Set passt.
- `memory: sequential` benötigt nur die größte Stufe plus 2 GB.
- `memory: auto` wählt `resident`, wenn es passt.

Ein System, das nicht in den Speicher passt, wird unter Angabe der Zahlen abgelehnt, bevor irgendetwas geladen wird. Ein nicht heruntergeladenes Modell wird mit unbekannter Größe gemeldet.

### Decide
- **Reihenfolge:** `cascade` listet die Entscheidungsmodelle der Reihe nach auf. Jede Stufe beantwortet die Fragen, bei denen sie sicher genug ist (mindestens `escalate_below`), und reicht den Rest weiter.
- **Stufen in beliebiger Reihenfolge:** bis zu vier Stufen, Cloud und lokal in beliebiger Reihenfolge. Ein Beispiel ist zuerst ein kleines lokales Judge-Modell, danach ein Cloud-Modell und zuletzt ein größeres lokales Modell: `[laya, jev, chat]`.
  - Die letzte Stufe ist lokal.
  - Jede lokale Stufe wird als Entscheidungs-Engine gehalten: das Chat-Modell, ausgelesen als `self`, oder ein Entscheidungsmodell wie ein Laya-Core-AI-Bundle oder ein OpenJev-Bundle. Ein dediziertes lokales Entscheidungsmodell kann `keep: on-demand` (und `idle_unload`) setzen, sodass es bei der ersten Entscheidung, die es erreicht, geladen und im Leerlauf wieder entladen wird.
  - Jede Pipeline benennt ihre eigenen Stufen. Ein Entscheidungsmodell, das von mehreren Pipelines genannt wird, wird nur einmal geladen.
  - Ein Entscheidungsmodell trifft nur Entscheidungen; nur das Chat-Modell beantwortet und entscheidet gleichermaßen.
- **Kalibrierung:** Ein Readout-Entscheidungsmodell bezieht seine Temperaturen aus einer `basalt_readout.json` in seinem Ordner (`{"temperature": …, "noul_temperature": …, "noul_bias": …}`), sofern vorhanden. Andernfalls verwendet es die bekannten Werte für seinen Namen und sonst keine. #448 plant, diese Dateien an Gemini Decision 1 anzupassen.
- **Größen:** `compose plan` bemisst ein OpenJev-Head-Bundle als dessen Basismodell plus Adapter, da das LoRA unmerged auf der gespeicherten Basis angewendet wird, 4-Bit eingeschlossen. Der eigene Plan von `serve` verfährt genauso.
  - Gemessen auf OpenJev-0.6B: 100 gelabelte Zustände, 300 Fragen, M4 Pro, Commit 066da0a. Die Referenz ist der gemergte `float32`-Pfad, der innerhalb von 1e-4 mit PyTorch übereinstimmt (#406).

    | Basis, LoRA unmerged | Genauigkeit | Gleiche Top-Antwort wie float32 | Aktiver Speicher | Median pro Zustand |
    |---|---|---|---|---|
    | merged float32 (Referenz) | 176/300 | — | 2,273 MB | 185 ms |
    | bf16, wie im Bundle angegeben (Standard) | 175/300 | 288/300 | 1,141 MB | 181 ms |
    | 8-Bit | 176/300 | 279/300 | 608 MB | 184 ms |
    | 4-Bit | 149/300 | 215/300 | 324 MB | 185 ms |

  - Eine 4-Bit-Basis kostet dieses 0.6B-Modell 9 Punkte, während 8-Bit die Genauigkeit bei einem Viertel des Speichers beibehält. `serve` und `compose` laden die im Bundle genannte Basis, sodass eine quantisierte Basis nur direkt über `JevEvaluator` erreicht wird.
  - Erneut ausführen mit `BASALT_E2E=1 BASALT_E2E_JEV_MODEL=<bundle> BASALT_E2E_JEV_BASE=<bf16 base> BASALT_E2E_JEV_BASE_8BIT=… BASALT_E2E_JEV_BASE_4BIT=… swift test --filter JevRealModelTests/unmergedArms`.
- **Ein 9B-Judge neben Gemma 4 12B auf 24 GB:** gemessen mit `openjev/OpenJev-Flash-9B-MLX` (8-bit, 9.5 GB) und `openjev/OpenJev-Flash-9B-MLX-4bit` (5.9 GB). Gemma 4 12B (`gemma-4-12B-it-qat-4bit`) beantwortet die Eskalationen. Das Testset umfasst 60 gelabelte Zustände, 180 Entscheidungen auf einem M4 Pro 24 GB bei Commit 8294bf7.
  - Verwenden Sie den 4-Bit-Judge. Neben Gemma belegt er 15.3 GB aktiv (Spitze 15.9 GB) von 24 GB.
  - Der 8-Bit-Judge belegt zusammen mit dem Chat-Modell und einem 0.6B-Head etwa 21 GB. Für sich genommen entscheidet er gut (92.8% ohne Enthaltung, 1.85 s), aber sobald Gemma die Eskalationen dekodiert, drückt der KV-Cache den Mac in den Swap. Ein Zwei-Minuten-Lauf dauerte dann über 24 Minuten und wurde abgebrochen.

    | Entscheider neben Gemma 4 12B (4-Bit-Judge) | Genauigkeit | An das LLM gesendet | Gesparte LLM-Tokens | Decide p50 |
    |---|---|---|---|---|
    | Nur LLM | 97.8% | 180 | — | — |
    | OpenJev-Flash-9B 4-bit, keine Enthaltung | 96.7% | 0 | 100% | 2.0 s |
    | OpenJev-Flash-9B 4-bit, Enthaltung unter 0.6 | 98.9% | 20 | 88.7% | 2.2 s |
    | OpenJev-Flash-9B 4-bit, Enthaltung unter 0.75 | 98.9% | 41 | 76.7% | 2.4 s |
    | OpenJev-Flash-9B 4-bit, Enthaltung unter 0.9 | 97.8% | 74 | 58.5% | 2.5 s |
    | Readout von Gemma selbst (`self`), Enthaltung unter 0.75 | 99.4% | 1 | 99.4% | 1.9 s |
    | OpenJev-0.6B Head (8-Bit-Judge-Lauf), Enthaltung unter 0.75 | 82.8% | 86 | 48% | 0.26 s |

  - Das Entscheiden während Gemma eine lange Antwort dekodiert senkt das Dekodieren von 21 auf 3 tok/s (−86%). Die Gesamtlaufzeit beträgt 341 s gegenüber 202 s bei sequentieller Ausführung. Die GPU-Entscheider bleiben daher zwischen den Generierungen gegatet.
  - Erneut ausführen mit `BASALT_E2E=1 BASALT_E2E_BUDGET_CHAT_MODEL=<chat> BASALT_E2E_BUDGET_FLASH=<judge dir> swift test --filter SystemOneBudgetBenchmarkTests`.
- **Was die Cloud erhält:** Anfragen mit Dokumenten oder Audio überspringen die Cloud-Stufen und gehen in ihrer Reihenfolge nur an die lokalen Stufen.
- **Schwellenwerte:** Fragen, deren Konfidenz unter `abstain_below` liegt oder deren zwei beste Optionen näher beieinander liegen als `min_margin`, gehen an das Antwortmodell.
- **Epistemisches Gate (Version 2):** `epistemic` klassifiziert `(support, refutation, needs_tool)` noch vor dem Dekodieren eines einzigen Antwort-Tokens: `enabled`, `support_threshold`, `refutation_threshold`, `tool_threshold`, `on_no_evidence` (`abstain` oder `escalate`), `on_conflict` (`report` oder `escalate`) und optional `abstain_message`.

### Answer-Routen
- **Abgleich:** Das `when` einer Regel greift, wenn jede Bedingung erfüllt ist. Bedingungen:
  - `modality`: Dokumente, Audio oder Text;
  - `decide`: Frage-ID zu Label, von System One;
  - `prompt_tokens: { over: N }`: geschätzt mit 4 Bytes pro Token;
  - `escalated`: die `decide`-Stufe hat eskaliert, oder eine Frage ging an das Antwortmodell;
  - `cost_left`: das Modell der Regel verfügt noch über Budget.
- **Reihenfolge der Versuche:** zuerst `model` der Regel, dann ihr `fallback`, dann das lokale Standardmodell der Pipeline.
- **Überspringen:** Ein Cloud-Modell wird bei erkanntem Text, bei erschöpftem Budget, nach einem abgelehnten Schlüssel oder während eines Rate-Limits übersprungen. Jeder Übersprung wird im Beleg (Receipt) benannt.

### Speak (`#493`, `#525`)
Eine Pipeline kann eine optionale `speak`-Stufe deklarieren, um Audio zu synthetisieren (`Kokoro-82M`, `Qwen3-TTS`, `Fun-CosyVoice3`, `Chatterbox`, `Zyphra/ZONOS3-Mini-hf`), mit deterministischem Fallback GPU → Neural Engine → Systemstimme:

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

- **`speak`:** akzeptiert einen einzelnen Namen (`speak: apple-speech`), eine Fallback-Liste (`speak: [tts_zonos, tts_ane, apple-speech]`) oder die vollständige Zuordnung.
- **`chain`:** geordnete lokale TTS-Modelle, die optional mit dem gewichtslosen Systemsynthesizer `apple-speech` enden.
- **`voice`:** Standard-Stimmkennung (`Savannah`, `alloy`, `af_heart` oder eine `AVSpeechSynthesisVoice`-Kennung).
- **`language`:** Standard-Sprach- oder Locale-Code (`en_us`, `ru`, `de`, `cmn` usw.).
- **`reference_audio`:** optionaler lokaler Referenzclip im Format `.wav` / `.pcm` / `.caf` / `.aiff` / `.flac` für Zero-Shot-Stimmenklonen.
- **`on_gpu_busy`:** `keep`, `fallback_ane` oder `fallback_apple_speech`, wenn Metal mit dem Dekodieren ausgelastet ist.
- **`sentence_streaming`:** streamt Satz-Chunks im Vollduplex-Verfahren an TTS, sobald Tokens eintreffen.
- **`format`:** Ausgabe-Container oder Sample-Kodierung (`wav`, `pcm`, `pcm_24k`, `mp3`, `opus`, `aac`, `flac`).
- **`exaggeration`:** expressive Prosodiesteuerung im Bereich `0.0...2.0`, sofern unterstützt.
- **`max_ttfa_ms`:** maximales Latenzbudget bis zum ersten Audio (Time-to-First-Audio), bevor entlang der Fallback-Kette weitergeleitet wird.

### Imagine (`#517`, `#539`)
Eine Pipeline kann eine optionale `imagine`-Stufe deklarieren, um Bilder auf Apple Silicon über bis zu vier explizite Stufen zu synthetisieren oder zu bearbeiten (`upsample_via`-Rewriter -> `share_encoder`-Text-/Vision-Encoder -> `chain`-DiT-Painter -> `decoder_via`-Latent-VAE-Decoder), ergänzt um TeaCache-Schrittüberspringung und Best-of-N-VLM-Bewertung:

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

- **`imagine`:** akzeptiert einen einzelnen Generatornamen (`imagine: z_turbo`), eine Fallback-Liste (`imagine: [z_turbo, sana_fast]`) oder die vollständige Zuordnung.
- **`chain`:** geordnete lokale Bildsynthesizer (`FLUX.2/3-klein`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`, `Ming-Image`, `Bonsai-4B`, `Step1X-Edit`, `SonicDiffusion`).
- **`upsample_via`:** lokaler LLM-/VLM-Rewriter-Slot, um knappe Prompts vor der Synthese zu ausführlichen Szenenbeschreibungen zu erweitern (verwendet bei `FLUX.3` BFLs Konvention für vollständige Szenen-Captions).
- **`share_encoder`:** lokaler LLM-/VLM-Slot (wie `Qwen3-VL-4B-Instruct` in 4-Bit, 8-Bit oder BF16), dessen Conditioning-Embeddings den Bildsynthesizer steuern.
- **`decoder_via`:** optionaler lokaler Latent-VAE-Decoder-Slot (wie das destillierte `black-forest-labs/FLUX.2-small-decoder` oder das vollständige `unsloth/FLUX.2-VAE` / `Felldude/FLUX.2-HDR-VAE`).
- **`steps`:** Standardanzahl der Denoising-Schritte im Bereich `1...100`.
- **`size`:** Standard-Ausgabeauflösung `<width>x<height>` (Vielfache von 16, z. B. `1024x1024` oder `2048x2048`).
- **`format`:** Ausgabe-Bildformat (`png`, `webp` oder `jpeg`).
- **`teacache_threshold`:** Residual-L1-Schwellenwert (`>= 0`) für das Überspringen innerer Schritte mit TeaCache.
- **`preview_every_steps`:** gibt während des SSE-Streamings alle N Schritte TAEF-Zwischenvorschau-Frames aus (`>= 0`).
- **`candidates`:** Anzahl der mit Seeds erzeugten Kandidatenbilder (`1...64`) für die Best-of-N-Auswahl.
- **`judge_via`:** lokaler VLM-/LLM-Slot, der `candidates` anhand des Prompts bewertet und das Gewinner-Artefakt zurückgibt.
- **`on_gpu_busy`:** `queue` oder `fallback`, wenn Metal mit dem Dekodieren ausgelastet ist.

Mitgelieferte 4-stufige `FLUX.3`-Presets (`basalt compose up <preset>`): `flux3-16gb-fast`, `flux3-24gb-balanced`, `flux3-36gb-moe-vision`, `flux3-64gb-studio-base` und `flux3-128gb-reference`.

### Version 2: Variablen und Profile
`version: 2` fügt zwei Blöcke hinzu, damit eine einzige Datei mehrere Macs bedienen kann. Eine Datei mit Version 1 funktioniert weiterhin; wird einer der beiden Blöcke darin verwendet, wird sie mit `"needs version: 2"` abgelehnt.

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

- **`variables`:** kleingeschriebene Namen mit einzeiligen einfachen Werten. `${name}` wird überall in der Datei ersetzt. Ein nicht deklariertes `${…}` wird abgelehnt, sodass Umgebungsvariablen weiterhin nie in die Datei gelangen; Secrets kommen ausschließlich über `api_key_env`.
- **`profiles`:** Ein Profil ist aktiv, wenn sein `when` auf diesem Mac zutrifft: `memory_gb` mit `under` und `over` (Unified Memory in GB) sowie `chip` (ein Teil des Chipnamens ohne Beachtung der Groß-/Kleinschreibung, z. B. `M4 Pro` oder `Max`). Ein Profil ohne `when` ist nur aktiv, wenn es über `--profile` angegeben wird.
  - Die `variables` eines aktiven Profils überschreiben die der Datei.
  - Ein Modell oder eine benannte Pipeline mit `profiles: [...]` existiert nur, solange mindestens eines dieser Profile aktiv ist.
- **Überschreibungen:** `--set name=value` überschreibt eine deklarierte Variable und `--profile name` aktiviert genau die genannten Profile – für `config`, `plan` und `up`. `compose up` gibt sie an den Server weiter, und Up-as-Diff sendet sie mit der Datei mit.
- **Pfade:** Ein `./`- oder `../`-Modellpfad ist relativ zum Ordner der Compose-Datei.
- **`basalt compose config -f <file>`** gibt die Datei so aus, wie sie auf diesem Mac mit den aktiven Profilen aufgelöst wird. **`basalt compose schema`** gibt das JSON-Schema des Formats für Editoren aus.

### Experimental und MCP (`#485`)
Eine Compose-Datei kann Top-Level-Abschnitte `experimental` und `mcp` deklarieren (beide werden in Version 1 und 2 unterstützt), anstatt lange Flag-Ketten nach `--` zu übergeben:

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

- **`experimental`:** wird in die entsprechenden `serve`-Flags übersetzt (`zero_copy_body`, `smallm_kernel`, `speech_analyzer`, `image_fallback`, `ane_vision_tower`, `telemetry_confidence_every`, `adaptive_thinking_budget`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k`) und mit exakten YAML-Zeilennummern validiert (`telemetry_confidence_every` >= 0, `image_fallback` `ocr`, `ane_vision_tower` mit `metadata.json` und `tower.aimodel`, `expert_cache_gb` > 0 bei `expert_streaming: true`, `kv_head_split` in `0.05...1`, `quest_top_k` in `1...4096`). Relative Pfade (`./`, `../`, `~`) werden relativ zum Ordner der Compose-Datei aufgelöst.
- **`mcp`:** konfiguriert den MCP-stdio-Transport (`enabled`, `rest`, `research_tools`, `enable`, `disable`, `image_dirs`) mit denselben Gruppen- und Tool-Selektoren wie `--mcp-enable` / `--mcp-disable` (`serving`, `gemma`, `media`, `harness`, `xray`, `surgery`, `research`, `all` oder einzelne Tool-Namen; `disable` gewinnt immer).
- **Hot-Reload vs. Neustart:** `speech_analyzer`, `image_fallback`, `telemetry_confidence_every` und `adaptive_thinking_budget` aktualisieren sich bei `compose up` live (`POST /v1/compose/apply`, ausgegeben unter `experimental:`). Engine- und Transport-gebundene Flags (`zero_copy_body`, `smallm_kernel`, `ane_vision_tower`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k` und `mcp`) melden in `compose diff` `needs a restart` und werden bei einem Live-`apply` abgelehnt.

### Benchmarks und Leaderboard-Einreichungen (Version 2, `#492`)
Eine Compose-Datei mit `version: 2` kann einen Top-Level-Block `benchmarks` mit gemeinsamen `defaults` und benannten `suites` deklarieren (entweder unter `suites:` oder als Kurzform mit Suite-Schlüsseln direkt unter `benchmarks:`):

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

- **`defaults`:** optional `seed`, `temperature`, `gpu_lock` und `output_dir`.
- **`suites`:** Jede Suite ist entweder eine verkürzte Task-Liste (`smoke: [...]`) oder `{ profiles, budget: { usd_max, max_minutes }, tasks: [...] }`.
- **`tasks`:** Jeder Task benennt einen `adapter` (`basalt:throughput`, `basalt:perplexity`, `basalt:system-one`, `basalt:passkey`, `basalt:mom`, `basalt:harness`, `arc-agi`, `arc-agi-2`, `terminal-bench`, `swe-bench`, `lm-eval`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `evalplus`), optional `dataset`, `target` (`pipeline`, `pipeline:<name>` oder `model:<slot>`), skalare `params`, Regressions-`gates` (`min_decode_tps`, `max_peak_gb`, `min_accuracy`, `max_p50_ms`, `max_ppl`, `max_ece`) und optional `submit` (`target`, `api_key_env`, `endpoint`, `subset`, `split`).
- **Befehle:** `basalt compose plan` gibt aufgelöste Suites aus; `basalt compose bench -f basalt.yaml --suite smoke --submit-dry-run` führt die Suite aus und exportiert verifizierte Leaderboard-Bundles; `basalt bench list|compare|submit` inspiziert Adapter, vergleicht Artefakte und exportiert Dry-Run-Bundles über alle zehn Ziele (`arc-prize`, `terminal-bench`, `swe-bench`, `hf-leaderboard`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `weave-otel`).

### Cloud-Fehler und -Ausgaben
- Ein abgelehnter Schlüssel (HTTP 401/403) deaktiviert das betreffende Cloud-Modell bis zum Neustart des Servers, begleitet von einer entsprechenden Log-Zeile.
- Ein Rate-Limit (429) pausiert es für die Dauer von `Retry-After`, oder für 60 s, falls der Header fehlt.
- Solange ein Cloud-Modell deaktiviert oder pausiert ist, antwortet die nächste Stufe oder der Fallback.
- Die Tagesausgaben pro Modell und insgesamt werden in `~/.basalt/cloud-spend.json` gespeichert, sodass ein Neustart ein Limit nicht zurücksetzt. Tage gelten in UTC.

## Laufende Systeme
- **`compose pull`:** lädt jedes lokale Modell der Datei in den Hugging-Face-Cache herunter und verifiziert jede Datei, wobei der Fortschritt bei jedem Zehntel ausgegeben wird. Nach einem Pull kennt `compose plan` die exakten Größen und `compose up` lädt ohne Wartezeit auf Downloads.
- **Größen in `compose plan`:** Ein Modell im Cache wird auf der Festplatte gemessen. Ein noch nicht heruntergeladenes Modell wird anhand der Dateiliste der Hugging-Face-API bemessen. Kennt keine der beiden Quellen die Größe, lehnt `plan` ab und nennt das Modell: Führen Sie `compose pull` aus oder übergeben Sie `--assume-size <GB>`, um es mit dieser Größe anzusetzen. `plan` zählt ein unbekanntes Modell nicht mehr als 0 Bytes.
- **`compose diff`:** zeigt, was `compose up` am laufenden System mit demselben Namen ändern würde, ohne es zu verändern (`POST /v1/compose/diff`). Es listet Lade-, Entladevorgänge, Schlüssel und Budgets, einen erforderlichen Neustart oder eine Ablehnung auf: ein fehlender Schlüssel oder ein System, das nicht in den Speicher passt. Bei einem Neustart oder einer Ablehnung beendet es sich mit einem Exit-Code ungleich null.
- **Ereignisse:** `GET /v1/compose/events` ist ein Server-Sent-Events-Stream darüber, was das System gerade tut. Ereignisarten sind `loading`, `ready`, `registered` (bei Bedarf), `failed` (Startprüfung), `applied`, `refused` und `up`. Ein neuer Client erhält zuerst die letzten 100 Ereignisse. `compose up --wait` gibt sie während des Wartens aus.
- **`compose up` bei einem laufenden System:** Läuft bereits ein System mit demselben `name`, sendet `compose up` ihm die Datei (`POST /v1/compose/apply`) und wendet die Differenz an.
  - Solange das System noch lädt oder eine andere Änderung anwendet, antwortet es mit 503. `compose up` wartet und sendet die Datei erneut, bis zu `--wait-timeout` Sekunden lang (Standard: 1800).
  - Mit `--wait` wartet `compose up` anschließend, bis `GET /v1/health/ready` mit 200 antwortet, gibt die Modelle aus, auf die gewartet wird, und beendet sich mit einem Exit-Code ungleich null, falls das System innerhalb des Timeouts nicht bereit ist.
  - Es lädt hinzugefügte Modelle, lädt geänderte neu und entlädt entfernte.
  - Es tauscht die Pipelines zwischen Anfragen aus.
  - Es prüft das neue System zuerst gegen den Speicher dieses Macs, genau wie `compose plan`, wobei die Modelle jeder Pipeline mitgezählt werden. Ein System, das nicht passt, wird abgelehnt, und nichts ändert sich.
  - Wenn das neue System zwar passt, alte und neue Modelle aber nicht nebeneinander Platz finden, entlädt es die zu ersetzenden Modelle, bevor es deren Nachfolger lädt. Eine Anfrage, die in diesem Zeitfenster eines davon benötigt, wartet auf das Laden oder schlägt fehl. Schlägt ein Ladevorgang dann fehl, werden die alten Modelle wieder geladen.
  - Es gibt pro Modell `load`, `reload`, `unload` und `keep` aus, `budget` für jedes geänderte Cloud-Budget (`total` für das Gesamtbudget der Datei) und `key` für jedes Cloud-Modell, dessen `api_key_env` sich geändert hat.
  - Budget oder Schlüsselvariable eines Cloud-Modells ändern sich ohne Neuladen. Die neuen Limits gelten sofort, und ein durch einen abgelehnten Schlüssel deaktiviertes Modell wird wieder aktiviert, sobald sich seine Schlüsselvariable ändert.
  - Ein anderes Chat-Modell wird anstelle des aktiven Modells geladen und als `chat` ausgegeben. Anfragen warten, während es lädt. Schlägt ein späterer Schritt des `apply` fehl, wird das alte Chat-Modell wieder geladen.
  - Ein neues lokales Entscheidungsmodell einer Pipeline wird wie jedes hinzugefügte Modell geladen. Das der Standard-Pipeline wird zusätzlich als `decide` ausgegeben. Die Pipelines und `/v1/systemone` verwenden es ab der nächsten Anfrage. Ein Modell, das neu zu entscheiden beginnt oder damit aufhört, wird in seiner neuen Rolle neu geladen.
- **Änderungen, die einen Neustart erfordern:** ein anderer Speichermodus, ein sequentieller Perceiver, Engine-/HTTP-Startflags unter `experimental` oder Änderungen unter `mcp`. Führen Sie `compose down` und danach `compose up` aus.
- **`compose ps`:** listet jedes laufende System (`GET /v1/compose`) mit seinen Pipelines und pro Modell auf:
  - seinen Zustand, einen von `loading`, `ready`, `unloading`, `unloaded`, `idle-unloaded` (on-demand, aktuell nicht geladen), `decision` (von der Entscheidungs-Engine gehalten), `cloud`, `no-key`, `over-budget`, `rate-limited`, `refused` und `failed`;
  - bei einem On-Demand-Modell, wann es zuletzt verwendet wurde und wann es entladen wird;
  - bei einem Cloud-Modell die heutigen Ausgaben im Vergleich zum Limit sowie eine eventuelle Ablehnung.
- **Health:** `GET /v1/health/live` antwortet mit 200, solange der Server HTTP beantwortet. `GET /v1/health/ready` antwortet mit 200, sobald jedes Modell bedienen kann. Bis dahin antwortet es mit 503 und listet die noch wartenden Modelle auf, etwa `chat (loading)`. Ein ungenutztes On-Demand-Modell, ein Cloud-Modell mit überschrittenem Budget und ein Cloud-Modell im Rate-Limit gelten als betriebsbereit; Kaskade oder Router leiten an ihnen vorbei.
- **Cloud-Schlüssel:** `compose up` verweigert den Start, wenn `api_key_env` eines Cloud-Modells nicht gesetzt ist, und nennt die Variable. Up-as-Diff prüft stattdessen die Umgebung des laufenden Servers.
- **`compose down`:** sendet dem Server `SIGINT`. Er schließt laufende Anfragen ab und entlädt. Läuft der Prozess nach 30 s immer noch, meldet `down` dies und beendet sich mit einem Exit-Code ungleich null.
- **Wo der Zustand gespeichert ist:** PID, Port und Prozess-Startzeit des laufenden Systems liegen in `~/.basalt/compose/<name>.json`. Ein Eintrag, dessen PID inzwischen zu einem anderen Prozess gehört, weil die PID neu vergeben wurde, gilt als nicht laufend.
- **API-Schlüssel:** Wenn der Server einen API-Schlüssel verwendet, setzen Sie `BASALT_API_KEY` für `ps` und Up-as-Diff.

## Domain-Adapter (`version: 2`)
Eine `version: 2`-Datei kann `adapters:` deklarieren, die auf Hugging-Face-NVIDIA-Nemotron-Presets (`nemotron:post-v2`, `nemotron:post-v1`, `nemotron:code`, `nemotron:math`, `nemotron:crossthink`, `nemotron:personas`), `hf:<org>/<repo>`-Datasets oder lokalen `.jsonl`-Dateien trainiert werden, und Anfragen ohne Neuladen von Gewichten über ein einziges residentes `base`-Modell zwischen ihnen routen:

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

Führen Sie `basalt compose train -f basalt.yaml` aus (optional mit `--adapter`, `--force`, `--resume`), um deklarierte Adapter auf die Festplatte zu trainieren, und wählen Sie anschließend ein `adapter:` in `pipeline.answer` oder einem beliebigen Eintrag von `pipeline.answer.routes`.

## Belege (Receipts)
Jede `/v1/pipeline`-Antwort und jeder Sitzungs-Log-Eintrag enthält einen Beleg (Receipt). Pro Stufe gibt er Modell, Millisekunden und Tokens an. `decide`-Stufen enthalten zudem ihre Kosten und alle übersprungenen Cloud-Stufen. `answer`-Stufen enthalten die ausgelöste Regel (`route`), die Kosten und die übersprungenen Modelle. `swap_ms` ist die Zeit, die für das Laden von `on-demand`- und `sequential`-Modellen aufgewendet wurde.

## Mitgelieferte Hardware-Profile nach Speicherstufen (`#526`)

`basalt compose` und `Basalt.app` (`Compose-Profile`, `⌥⌘C`, sowie **Einstellungen → Engine → Mitgeliefertes Profil**) liefern **84 hardwarekalibrierte Profile** über 7 Apple-Silicon-Unified-Memory-Stufen (`16gb`, `24gb`, `32-36gb` [Aliase `32gb` und `36gb`], `48gb`, `64gb`, `128gb`, `256gb` [Aliase `192gb` und `236gb`]) × 12 Workload-Archetypen:

| Archetyp | Primär | Fokus |
|---|---|---|
| `daily-chat` | ★ | Alltags-Chat, Apple-Vision-Dokumentwahrnehmung, lokale Spracheingabe |
| `coding-agent` | ★ | Spekulativer Small-M-Kernel, code-spezialisiertes Routing, MCP-Tools |
| `voice-zonos` | ★ | Zyphra `ZONOS3-Mini` 44.1 kHz neuronales TTS + `Kokoro-82M` ANE-Fallback |
| `multimodal-desk` | ★ | OCR, Audio-Wahrnehmung, System-One-Triage, Chat und ZONOS3-Stimme |
| `reasoning-lab` | | Adaptive Thinking-Budgets + System-One-Kaskade für Mathematik und Logik |
| `document-ocr` | | Neuronales OCR (`PaddleOCR-VL-1.5`) + vierwertiges epistemisches Belnap-Gate |
| `voice-duplex` | | Vollduplex-Transkription und Streaming-Synthese mit unter 80 ms TTFA |
| `creative-studio` | | Native MLX-Bildsynthese (`Z-Image-Turbo` / `FLUX`) mit TeaCache |
| `rag-knowledge` | | Fundiertes Long-Context-Retrieval mit Konflikt- und Enthaltungs-Gating |
| `hybrid-cloud` | | Lokaler datenschutzorientierter Chat mit auf `$1/day` begrenzter Cloud-Eskalation |
| `moe-frontier` | | Sparse MoE (`Flint 40B-A2.5B`, `Nemotron-3.5`, `Qwen3.8`) + Expert-Streaming |
| `research-xray` | | Model X-Ray, Weight Surgery und `basalt:throughput`-Smoke-Benchmarks |

Beim Start erkennt `Basalt.app` den physischen Unified Memory (`ComposeMemoryTier.detect`) und heftet die 4 mit Stern markierten Primärprofile für diesen Mac unter **Für diesen Mac empfohlen** an, während alle 84 Profile durchsuchbar und auswählbar bleiben. Jedes Profil läuft identisch über die CLI (`basalt compose plan 24gb-voice-zonos`, `basalt compose up 36gb-moe-frontier`, `basalt compose up 236gb-multimodal-desk`).

## Beispiele
- `docs/examples/compose/any-mac.yaml`: eine `version: 2`-Datei für Macs mit 16, 24 und 64 GB. Das Chat-Modell stammt aus einem nach Speicher ausgewählten Profil, und das 64-GB-Profil fügt ein 27B-Modell `on-demand` sowie eine `long`-Pipeline hinzu.
- `docs/examples/compose/mac-16gb.yaml`: Ornith 9B lokal, OCR bei Bedarf, lange Prompts an Claude Haiku innerhalb von $0.50 pro Tag.
- `docs/examples/compose/mac-24gb.yaml`: Gemma 4 12B, OCR bei Bedarf, Code an Haiku, eine zweite `support`-Pipeline, $1 pro Tag insgesamt.
- `docs/examples/compose/mac-64gb.yaml`: Gemma 4 26B-A4B mit Qwen3.8-27B und Nemotron Transcribe bei Bedarf, keine Cloud-Antworten.
- `docs/examples/compose/document-qa-cloud.yaml`, `mom.yaml` und `too-big.yaml`: die Dateien hinter den unten stehenden Messungen.

Weitere Messungen finden sich in #412.

## Messwerte

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

Gemischter Workload aus 55 Anfragen über `/v1/pipeline` (30 Entscheidungen, 10 Code, 10 Chat, 5 Dokumente), M4 Pro, Release-Build (`scripts/compose_mom_bench.swift`, `docs/examples/compose/mom.yaml`):

| Setup | Richtige Labels | Cloud pro 1000 Anfragen | p50 | p95 | Lokale Stufenzeit |
|---|---|---|---|---|---|
| nur lokal (Gemma 4 12B entscheidet und antwortet) | 55/55 | $0 | 0.92 s | 10.0 s | 176 s |
| Cloud zuerst (Jev entscheidet, Haiku beantwortet allen Text) | 55/55 | $0.046 | 1.25 s | 4.8 s | 24 s |
| MoM (Jev entscheidet, Code an Haiku solange Budget reicht, der Rest an Gemma) | 55/55 | $0.035 | 1.21 s | 4.8 s | 53 s |

Gemessen mit `compose up` auf einem M4 Pro mit 24 GB, Release-Build:

| Datei | Anfrage | Beleg |
|---|---|---|
| document-qa-cloud | Dokument (erstes) | OCR bei Bedarf geladen (489 ms), lokal entschieden, beantwortet: 6.1 s |
| | Dokument (erneut) | OCR bereits geladen (0 ms): 5.0 s |
| | nach 40 s Leerlauf | OCR entladen, Speicherbedarf 12 → 11 GB |
| | eingetippte Frage | durch Jev in der Cloud entschieden: 0.6 s |
| | Dokument | OCR neu geladen (267 ms): 5.3 s |
| voice-agent (mitgeliefert) | Audio | Chat um Nemotron Transcribe herum entladen (Swap 6.9 s): 17.3 s |
| too-big (Beispiel) | `compose plan` | abgelehnt: eine Stufe benötigt 20.7 GB plus 2 GB gegenüber einem Working Set von 19.1 GB |


## Erweiterungen des Schemas `version: "2.2"`

Compose `version: "2.2"` (kompatibel mit `version: 2`) ergänzt deklarative Konfiguration für Block-Diffusions-LLMs, spekulative `DFlash`-Drafter, Duplex-Speech-to-Speech, gemeinsam genutzte VLM/DiT-Text-Encoder, 1-Schritt-`SeedVR2`-Upscaling, `ACE-Step`-Musik-DiT, kausale 3D-VAE-Video-DiTs, segmentweise Präferenztraining-Läufe sowie Thunderbolt-/LAN-Multi-Mac-Cluster:

- Pro Modell `speculative` (`mode`, `drafter`, `block_size`, `verify`) und `diffusion_text` (`dual_kv_cache`, `confidence_threshold`, `contrastive_alpha`).
- `imagine`-Erweiterungen: `share_text_encoder_with`, `kv_cache_references`, `rgba` und `upscale` (`model`, `steps`, `target_resolution`).
- Root `voice` (`mode`, `sts_model`, `denoise`, `turn_detection` mit `model` und `diarization`, `tts_fallback`, `aligner`).
- Root `music` (`model`, `steps`, `duration_seconds`) und `video` (`model`, `vae_tiling`, `sta_sparse`, `fps`, `frames`).
- Root `train` (`method`, `segments`, `ema_decay`) und `cluster` (`topology`, `peers`, `compression`).
- Enterprise-Routing, -Caching & -Governance (`v0.12`): 5-Komponenten-Routing mit `BusinessOrchestratorCore`, semantischer Cache `LANTERN` (`SemanticCacheCore` / `SemanticCacheActor`, `#625`), `HITL`-Freigabe-Gate (`/v1/mcp/approvals`) + LAAS-CNRS `SymbolicKnowledgeGraph` (`#626`) sowie UMA-bewusste L0/L1/L2-`TieredKVHierarchyCore` + `AdaptiveHeterogeneousPlanner` (`#627`).
