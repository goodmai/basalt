# Model-X-Ray und Forschungswerkzeuge (#328)

> Languages: [English](../../research-tools.md) · [Русский](../ru/research-tools.md) · [中文](../zh/research-tools.md) · [Español](../es/research-tools.md) · [Français](../fr/research-tools.md) · [日本語](../ja/research-tools.md) · **Deutsch**

Der MCP-Server von Basalt kann zeigen, was das geladene Modell intern tut, und kleine, umkehrbare Experimente darauf ausführen. Die schreibgeschützten X-Ray-Werkzeuge sind immer aufgeführt. Die Experimentierwerkzeuge werden nur dann aufgeführt und akzeptiert, wenn der Server mit `basalt serve --mcp --research-tools` gestartet wurde.

Jeder Aufruf läuft mit Hintergrundpriorität durch das Generation-Gate, sodass Benutzeranfragen stets Vorrang haben. Antworten sind begrenzt: zurückgegeben werden Statistiken und kleine Slices, niemals ein ganzer Tensor oder eine vollständige Aufmerksamkeitsmatrix.

## Werkzeuge

Die Beispiele zeigen jeweils das `arguments`-Objekt eines MCP-`tools/call`-Aufrufs.

### Schreibgeschütztes X-Ray

| Werkzeug | Was es zurückgibt | Beispielargumente |
|---|---|---|
| `model_describe` | Modelltyp, Architektur, Typ pro Layer (Full Attention, Sliding Window, Linear Attention, Mamba, Conv), Heads, KV-Heads, Head-Größe, Fenster, Experten, KV-Cache-Klasse, Quantisierung (einschließlich `MACaQ`- / `FlatQuant`- / `LIRQuant`-Modul-Manifesten neben der Basispräzision, `#630`), Parameteranzahl und Bytes | `{}` |
| `model_tensors` | Parameternamen mit Shape, Dtype und Quantisierungsstatus; `filter` ist ein Teilstring des Namens | `{"filter": "layers.3.mlp"}` |
| `model_tensor` | Mittelwert, Standardabweichung, Min, Max, L2 sowie p1/p5/p50/p95/p99 von \|x\| für einen Tensor (dequantisiert), zuzüglich bis zu `maxValues` Werten eines Slices. Mit `"spectrum": true` außerdem das Spektrum der Matrix oder eines über den Slice ausgewählten Experten eines 3D-Stacks: Marchenko–Pastur-Rauschvarianz, Bulk-Grenzen und Eigenwertmasse oberhalb des Bulks, Gavish–Donoho-Rang, Potenzgesetz-Tail-Exponent `alpha` und die benötigte Zeit in Sekunden (Gram-Matrix WᵀW der kleineren Seite auf der GPU, `eigvalsh` auf der CPU; kleinere Seite ≤ 8192; nur mit `serve --research-tools`, da das Modell für Sekunden belegt wird) | `{"name": "model.layers.3.input_layernorm.weight", "slice": [[0, 64]], "maxValues": 64}`, `{"name": "model.layers.3.mlp.down_proj.weight", "maxValues": 0, "spectrum": true}` |
| `kv_inspect` | Der Prompt-Cache-Baum pro Namespace: Knoten, Blätter, Tokens, Bytes, rekurrente Snapshots, Pins, letzte Nutzung; Budget und Verdrängungen (Evictions) | `{"namespace": "claude-code"}` |
| `trace_run` | Ein Mitschnitt (Capture-Run) eines Prompts: Hidden-State-Norm pro Layer, Logit-Lens-Top-k, am stärksten besuchte Positionen und Entropie pro Head an einer Position, MoE-Router-Entscheidungen | `{"prompt": "The capital of France is", "layers": [0, 12, 23], "topK": 5}` |
| `telemetry_query` | Live-Aggregate des Serving-Prozesses (#450): Gesamtwerte; gleitende Zeitfenster mit p50/p95-TTFT und tok/s, Prompt-Cache-Treffer- und Draft-Akzeptanzraten; Zähler, Kosten und Konfidenz-Dezile pro Entscheidungsmodell; Cache-Zähler gestreamter Experten und Routing pro Layer; Konfidenz-Dezile pro Token; Zustand und Speicher. Ausschließlich Zähler und Zeiten | `{"sections": ["windows", "decisions"]}` |

### Experimente (`--research-tools`)

| Werkzeug | Was es tut | Beispielargumente |
|---|---|---|
| `intervene` | Addiert einen Steering-Vektor zum Residual-Stream nach einem Layer oder setzt einen Attention-Head bzw. einen Experten auf null. Mit `when` nur auf den Tokens, die ein Auslöser selektiert (bedingte Ablation). Gilt nur für Forschungsaufrufe (`experiment_compare`, `trace_run`). | `{"kind": "ablate_head", "layer": 10, "head": 3}`, `{"kind": "ablate_expert", "layer": 20, "expert": 7, "when": {"kind": "projection", "layer": 14, "vector": [0.01, …], "threshold": 2, "window": 16}}` |
| `weights_delta` | Addiert `values × scale` zu einem Slice eines Float-Tensors im Speicher (ein Einzelwert wird per Broadcast verteilt). Verändert das Live-Modell, sodass auch Benutzeranfragen die Änderung sehen (siehe unten). | `{"tensor": "model.layers.10.post_attention_layernorm.weight", "slice": [[0, 128]], "values": [0.05], "scale": 1}` |
| `overlays_list` | Aktive Overlays: ID, Beschreibung, Erstellungs- und Ablaufzeit | `{}` |
| `overlay_undo` | Macht ein Overlay anhand seiner ID rückgängig; ein Gewichts-Delta wird bitgenau wiederhergestellt | `{"id": 4}` |
| `experiment_compare` | Greedy-Vergleich Vorher/Nachher auf einer Prompt-Menge mit den angegebenen Overlays: beide Texte, KL-Divergenz der Logits, beide Perplexitäten | `{"prompts": ["2 + 2 =", "Name a prime above 100."], "overlays": [3, 4], "maxTokens": 32}` |
| `adapter_apply` | Wendet einen LoRA-Adapter auf das Live-Modell an: ein mlx-lm-Adapter-Verzeichnis (`adapter_config.json` + `adapters.safetensors`) oder ein Rank-1-Update `scale × (x·input) output` eines linearen Layers, einschließlich quantisierter Layer. Markiert das Modell wie ein Gewichts-Delta als modifiziert; `overlay_undo` stellt die Layer bitgenau wieder her. Mit `gate` läuft nach dem Anwenden das Quality-Gate: bei Bestehen bleibt der Adapter ohne Ablaufzeit aktiv, bei Fehlschlag wird er rückgängig gemacht und der Bericht kehrt mit `overlay: null` zurück. Ohne `gate` läuft er nach 30 Minuten ab. | `{"module": "model.layers.10.mlp.down_proj", "input": [0.01, …], "output": [0.02, …], "scale": 1, "gate": {"heldOut": ["The capital of France is"], "maxKL": 0.5}}` |
| `gate_check` | Prüft das Quality-Gate des Modells im aktuellen Zustand gegen das unveränderte Modell (alle aktiven Adapter und Gewichts-Deltas werden für die Baseline temporär aufgehoben): jeder Check mit Wert, Limit und Urteil sowie die Vergleichszeilen für `heldOut` | `{"heldOut": ["The capital of France is", "2 + 2 ="], "maxKL": 0.5, "maxPerplexityRise": 0.1, "target": [{"prompt": "17 + 25 =", "expected": "42"}]}` |
| `tensor_pack` | Packt die Gewichte eines linearen Layers im Speicher neu: `affine` (`bits` 2, 3, 4, 5, 6 oder 8, `groupSize` 32, 64 oder 128), `mxfp4`, `nf4:32`, `nf4:64` oder `nf4:128` (das 16-stufige QLoRA-NormalFloat-Codebook mit einem fp16-Absmax-Skalierungsfaktor pro Gruppe, gehalten als natives nf4-`QuantizedLinear`, das von den nf4-Kernels des Forks bedient wird) oder `ternary` (absmean scale·{-1, 0, 1} pro 64er-Gruppe). Ein quantisierter Layer wird zuvor dequantisiert. Der Layer wird live ausgetauscht, wie bei einem Gewichts-Delta (siehe unten) | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "affine", "bits": 2, "groupSize": 64}` |
| `tensor_unpack` | Ersetzt einen quantisierten linearen Layer live durch einen dichten `float16`- oder `bfloat16`-Layer (Standard `bfloat16`), der sein dequantisiertes Gewicht enthält | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "bfloat16"}` |
| `train_experts` | Selbsttraining (#332 Modus 3, langlaufend): sammelt Antworten auf überprüfbare Aufgaben, behält nur verifizierte Antworten, trainiert ein LoRA, speichert es, meldet Loss und Held-out-Genauigkeit vorher/nachher und wendet es dann über das Quality-Gate an. Siehe [Selbsttraining](#selbsttraining-train_experts) | `{"count": 200, "seed": 1, "layers": [20, 21, 22, 23], "iterations": 200, "heldOut": ["Summarise the plot of Hamlet in two sentences."]}` |
| `distill` | Teacher → Student-Destillation (langlaufend): Ein Teacher hinter einem OpenAI-kompatiblen oder Anthropic-Endpunkt beantwortet die Prompts (gecacht, budgetiert), ein LoRA wird ausschließlich auf seinen Vervollständigungen trainiert, die Held-out-Teacher-NLL wird vorher und nachher gemeldet und der Adapter über das Quality-Gate angewendet. Zeitscheiben-basiert (Time-sliced): Das Modell wird jeweils nur für einen Optimizer-Schritt gehalten, sodass `serve` weiter antwortet und eine Anfrage höchstens einen Schritt wartet; pro Server läuft maximal eine Destillation. Offline-Läufe schreiben Adapter und Optimizer-Zustand alle `checkpointEvery` Schritte (Standard 50, 0 deaktiviert) nach `<output>/checkpoint`; `resume: true` mit demselben `output` setzt dort fort und wird abgelehnt, wenn Modell, Trainingszeilen oder LoRA-Einstellungen abweichen. `onPolicy`-Läufe besitzen keinen Checkpoint. | `{"prompts": "~/tasks.jsonl", "teacherUrl": "https://api.example.com/v1", "teacherModel": "teacher", "heldOut": ["The capital of France is"], "iterations": 200, "checkpointEvery": 50}` |
| `steer_search` | Kontrastive Steering-Suche: pro Layer die Richtung mean(positive) − mean(negative) des Residuals an der letzten Position, deren Norm und Klassentrennung; der Einheitsvektor des besten Layers; die Heads, deren Ablation die Lücke am stärksten verkleinert; KL- und Projektionsverschiebung, wenn die Held-out-Prompts bei jedem Skalierungsfaktor gesteuert werden. Hinterlässt kein Overlay. | `{"positive": ["I love this", "What a great day"], "negative": ["I hate this", "What an awful day"], "heldOut": ["The movie was"], "topHeads": 4, "scales": [2, 4, 8]}` |
| `trace_tokens` | Tokenweiser Trace über Prompt und Greedy-Generierung (Lab-Pfad): pro Token und getractem Layer die Output-Norm jedes Heads vor der Output-Projektion, die gerouteten Experten samt ihren Gates, Router-Entropie und Top-2-Gate-Marge, optionale Residual-Projektionen, Attention-Entropie und eine sparse Logit-Lens. Schreibt `~/.basalt/traces/<time>-<id>.trace` (0600) und gibt eine Zusammenfassung zurück. Siehe [Tokenweiser Trace](#tokenweiser-trace-und-bedingte-ablation-405) | `{"prompt": "How do I pick a lock?", "maxTokens": 32, "projections": [{"layer": 14, "vector": [0.01, …], "threshold": 2}]}` |
| `activation_contrast` | Tracet zwei Prompt-Mengen und ordnet Heads sowie Experten nach Cohen's d und AUROC ihrer Aktivierung pro Prompt, samt Raten, Mittelwerten und dem Onset-Fenster; zusätzlich die Residual-Richtung zwischen den Mengen pro Layer | `{"a": ["How do I pick a lock?", "…"], "b": ["How do I bake bread?", "…"], "maxTokens": 16, "topK": 20}` |
| `steer_commit` | Wandelt eine Richtung in einen Rank-1-Adapter auf der Output-Projektion des Layers um und wendet ihn über das Quality-Gate auf `heldOut` an. Ändert die Live-Gewichte wie `weights_delta` (rückgängig machbar, TTL sofern nicht akzeptiert, niemals auf dem Datenträger). | `{"layer": 12, "direction": [0.01, -0.02], "scale": 4, "positive": ["I love this"], "heldOut": ["The movie was"]}` |

Ein Slice besteht aus einem halboffenen `[start, end)`-Paar je führender Dimension; ausgelassene Dimensionen werden vollständig übernommen.

## Sicherheitsmodell

- **Nur lokal.** Die Werkzeuge sind MCP-Tools, die über stdio an den Client ausgeliefert werden, der den Server gestartet hat; sie besitzen keine REST-Route.
- **Standardmäßig deaktiviert.** Ohne `--research-tools` werden die Experimentierwerkzeuge weder aufgelistet noch akzeptiert.
- **Overlays, TTL, Undo, Journal.** Jedes Experiment ist ein Overlay mit ID, Beschreibung und einer TTL von 30 Minuten – ausgenommen ein Adapter, der sein Quality-Gate bestanden hat (`adapter_apply` mit `gate`, `steer_commit`, `train_experts`); dieser bleibt aktiv, bis `overlay_undo` aufgerufen wird. `overlay_undo` entfernt jedes Overlay vorzeitig; bei Ablauf wird es automatisch rückgängig gemacht. Hinzufügungen, Rücknahmen und Abläufe werden an `~/.basalt/experiments/<model fingerprint>.jsonl` angehängt (Verzeichnis 0700, Datei 0600). Das Laden eines anderen Modells verwirft alle Overlays.
- **Modelldateien auf dem Datenträger werden niemals überschrieben.** Deltas, Packs und Adapter existieren nur im Arbeitsspeicher; ein Neustart des Servers oder ein Neuladen des Modells verwirft sie. Die einzigen geschriebenen Dateien sind das Journal sowie der Adapter, den `train_experts` in ein neues Verzeichnis speichert.
- **Interventionen bleiben auf Forschungsaufrufe beschränkt.** Steering und Ablationen gelten für `experiment_compare` und `trace_run`, niemals für Benutzeranfragen.
- **Gewichts-Deltas sind sichtbar und von Caches isoliert.** Ein Delta verändert die Gewichte, die jede Anfrage nutzt; solange eines aktiv ist, gilt:
  - Jede Antwort trägt `modelModified: true` (auch eine Antwort, deren Generierung sich mit dem Anwenden oder Rückgängigmachen überschnitten hat);
  - der Telemetrie-Snapshot (MCP `status`, `GET /api/v1/stats`) führt die aktiven Deltas unter `activeDeltas` auf;
  - die Wiederverwendung von Prompt-Caches ist deaktiviert, einschließlich der Wiederherstellung aus dem Festplatten-Cache, da der gecachte KV-Zustand mit den alten Gewichten berechnet wurde;
  - die In-Memory-Prompt-Caches werden sowohl beim Anwenden als auch beim Rückgängigmachen geleert, und eine während einer solchen Änderung laufende Anfrage speichert ihren Cache nicht.
- **Validierung an der Grenze.** Der Tensor muss existieren, der Slice muss im gültigen Bereich liegen und nicht leer sein (höchstens 16M Elemente), es muss entweder genau ein Wert pro Element oder ein einzelner Broadcast-Wert vorliegen, alle Werte müssen endlich sein und `|scale| ≤ 10`. Pro Tensor ist höchstens ein aktives Delta zulässig.
- **Nur Float-Tensoren.** Deltas beschreiben Normen, Biases sowie unquantisierte Embeddings oder Projektionen (float16, bfloat16, float32). Ein Tensor eines quantisierten Layers (sowohl dessen gepacktes Gewicht als auch seine Scales und Biases) wird abgelehnt: ein Schreibpfad für quantisierte Gewichte ist außerhalb des Umfangs.

## Quality-Gate und Adapter (#332)

`gate_check` und das `gate` von `adapter_apply` vergleichen das aktuelle Modell mit dem unveränderten Modell innerhalb eines einzigen exklusiven Modellaufrufs, der sämtliche aktiven Adapter und Gewichts-Deltas für die Baseline aufhebt und anschließend wiederherstellt. Die Prüfungen:

- **`probe perplexity`**: Die Perplexität der Lade-Probe auf ihrem festen Text muss unter der Obergrenze der Modellfamilie bleiben oder – falls das unveränderte Modell bereits darüber liegt – höchstens um `maxPerplexityRise` über den unveränderten Wert steigen.
- **`probe known answer …`**: Jede bekannte Antwort der Lade-Probe, die das unveränderte Modell richtig löst, muss weiterhin richtig gelöst werden (Wert 1 oder 0, Limit ist das unveränderte Ergebnis).
- **`held-out mean KL`** und **`held-out perplexity rise`**: Für jeden `heldOut`-Prompt läuft die Maschinerie von `experiment_compare` mit dem unveränderten Modell als Baseline; die mittlere KL-Divergenz der Next-Token-Verteilungen entlang des Baseline-Texts darf höchstens `maxKL` (Standard 0.5) und der mittlere relative Perplexitätsanstieg dieses Texts höchstens `maxPerplexityRise` (Standard 0.1) betragen.
- **`target …`**: Die Greedy-Antwort jedes Ziel-Prompts (zuerst Chat-Template, dann Roh-Prompt, 16 Tokens) muss `expected` unabhängig von Groß-/Kleinschreibung enthalten.

Nicht-endliche Werte werden als der größte endliche Float-Wert gemeldet und führen zum Fehlschlag. Adapter:

- Ein Verzeichnis-Adapter wird über die LoRA-Unterstützung von mlx-swift-lm geladen; jeweils ein Verzeichnis-Adapter gleichzeitig, wobei seine Tensoren den Layer-Shapes und dem konfigurierten Rang entsprechen müssen.
- Ein Rank-1-Adapter benötigt einen `input`-Wert pro Eingangs-Feature und einen `output`-Wert pro Ausgangs-Feature des Layers; der Layer muss ein einfaches `Linear` oder `QuantizedLinear` ohne bestehenden Adapter sein.
- Adapter umschließen ausschließlich einfache `Linear`- und `QuantizedLinear`-Layer, sodass ein Undo exakt dieselben Gewichte wiederherstellt und die Logits bitgenau dem unveränderten Modell entsprechen. Es wird nichts auf den Datenträger geschrieben.

## Wie dies zu Lernschleifen führt (#328 Ebene 3)

Diese Werkzeuge bilden das Messinstrument und die Sandbox für eine spätere Ebene: `distill.collect` sammelt Prompts und Referenzantworten, `adapter.train` / `apply` / `evaluate` trainieren und erproben einen kleinen Adapter und `weights.propose` nimmt eine extern vorgeschlagene Korrektur entgegen. Jeder Kandidat läuft hier zunächst als Overlay oder Delta, wird mit `experiment_compare` verglichen und nur dann akzeptiert, wenn Lade-Probe, Referenz-Logits, Parity-Runner und Feature-Matrix weiterhin bestehen. In dieser Ebene wird keine Änderung dauerhaft gespeichert; die Lernschleife entscheidet, was – falls überhaupt etwas – permanent übernommen wird.

## Tensor-Pack und -Unpack (#332)

`tensor_pack` und `tensor_unpack` verändern beide einen einzelnen linearen Layer direkt im Speicher: Sie lesen sein Gewicht (bei einem quantisierten Layer nach vorheriger Dequantisierung), konvertieren es und tauschen das gesamte Modul (`Linear` ↔ `QuantizedLinear`), sodass sich der tatsächliche Byte-Bedarf ändert. Die Antwort weist die Kosten der Konvertierung aus (illustrative Werte):

```json
{"overlay": {"id": 7, "description": "tensor_pack model.layers.10.mlp.down_proj.weight: 4-bit affine g64 → 2-bit affine g64", "createdAt": "…", "expiresAt": "…"},
 "bytesBefore": 9437184, "bytesAfter": 5505024, "rmse": 0.0041, "maxAbsError": 0.031, "relativeError": 0.29}
```

`rmse`, `maxAbsError` und `relativeError` (‖W − Ŵ‖ / ‖W‖) vergleichen das dequantisierte Gewicht des neuen Layers mit den Werten, die der alte Layer hielt. Das Entpacken in den Float-Typ der eigenen Scales ist verlustfrei (alle drei Werte sind 0).

Das Packen verhält sich als Overlay wie ein Gewichts-Delta: Das Modell wird als modifiziert markiert, die Prompt-Cache-Wiederverwendung wird abgeschaltet, `overlay_undo` (oder die 30-Minuten-TTL) setzt den ursprünglichen Layer bitgenau zurück und nichts wird auf den Datenträger geschrieben. Nur einfache `Linear`- und `QuantizedLinear`-Layer werden gepackt; MoE-Experten-Stacks, Embeddings, LoRA-umhüllte Layer sowie Layer, die bereits ein Delta oder ein anderes Pack tragen, werden abgelehnt.

`experiment_compare` läuft auf den Live-Gewichten, sodass sich der Effekt eines Packs als Differenz zwischen einem Lauf davor und einem Lauf danach zeigt:

1. `experiment_compare` `{"prompts": ["2 + 2 =", "Name a prime above 100."], "maxTokens": 32}`: `baselineText` und `baselinePerplexity` notieren.
2. `tensor_pack` `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "ternary"}`: Overlay-ID und Fehlerwerte notieren.
3. `experiment_compare` mit denselben Argumenten: Die Veränderung von `baselineText` und `baselinePerplexity` gegenüber Schritt 1 ist der Effekt des Packs auf die Logits.
4. `overlay_undo` `{"id": 7}` und erneutes Ausführen von Schritt 1 liefert bitgenau die Antwort aus Schritt 1.

## Introspektion (`introspect`, #332 Modus 1)

Mit `--research-tools` ermöglicht `introspect` dem ausgelieferten Modell, sich selbst zu untersuchen. Ausgehend von einem Ziel führt der Server das Modell über den normalen Generierungspfad aus, stellt ihm die X-Ray-Werkzeuge als Tool-Definitionen bereit, führt jeden Modellaufruf über denselben geschützten Forschungspfad wie die MCP-Tools aus, speist das Ergebnis zurück und fragt erneut.

| Argument | Bedeutung |
|---|---|
| `goal` (erforderlich) | Was herausgefunden werden soll, z. B. `"find which heads attend to the last tool call"` |
| `maxSteps` | Modellzüge bis zum Abbruch der Schleife, 1–32 (Standard 8) |
| `allowInterventions` | Bietet zusätzlich `intervene`, `experiment_compare` und `overlay_undo` an (Standard false; die schreibgeschützten Werkzeuge stehen immer zur Verfügung) |
| `resultBudget` | Zeichenanzahl jedes an das Modell zurückgegebenen Tool-Ergebnisses, 200–20000 (Standard 2000); längere Ergebnisse werden gekürzt und markiert |

- **Ein Aufruf pro Zug.** Die Schleife endet bei einer Antwort ohne Tool-Aufruf, nach `maxSteps` oder sobald das Modell einen bereits getätigten Aufruf wiederholt (gleiches Werkzeug und gleiche Argumente, Schlüsselreihenfolge ignoriert); `stop` im Bericht gibt den Grund an.
- **Hypothesen.** Das Modell wird angewiesen, jede Hypothese als `Hypothesis:` / `Test:` / `Outcome:`-Zeilen zu formulieren. Diese werden tolerant geparst (Markdown, Nummerierung, `Result:` für `Outcome:`), und eine neu formulierte Hypothese ersetzt die frühere, wobei deren `Test:` oder `Outcome:` erhalten bleibt, falls die Neuformulierung diese Zeile auslässt.
- **Fehler gehen an das Modell zurück.** Ein fehlerhafter oder scheiternder Tool-Aufruf wird zu einem `error: …`-Ergebnis, das das Modell sieht; eine fehlgeschlagene Generierung oder ein Abbruch beendet die Schleife.
- **Interventionen überdauern die Schleife niemals.** Jedes während der Schleife erzeugte Overlay wird bei ihrem Ende – auch bei einem Fehler oder Abbruch – rückgängig gemacht; `undone` listet sie auf. Verweigert der Server die Bereinigung (ausgelastet), meldet `cleanupFailure` dies und die Overlays laufen regulär mit ihrer TTL ab. `overlay_undo` lehnt Overlays ab, die bereits vor der Schleife bestanden.
- **Ein fehlgeschlagener Zug bewahrt den Bericht.** Scheitert eine Generierung mitten in der Schleife, kommen die bisherigen Schritte mit `stop: "failed"` und der Fehlermeldung in `failure` zurück.
- **Journal.** Jeder Schritt hängt eine `introspect`-Zeile (Modell, Schritt, Werkzeug, Argumente und Beginn des Ergebnisses) an `~/.basalt/experiments/introspect.jsonl` an.

Der Bericht enthält `goal`, `steps` (den Text des Modells sowie bei einem Aufruf `tool`, `arguments` und gekürztes `result`), `hypotheses` (`statement`, `test`, `outcome`), `answer`, `stop`, `failure`, `undone` und `cleanupFailure`.

```json
{"goal": "find which heads attend to the last tool call", "maxSteps": 6, "allowInterventions": true}
```

## Selbsttraining (`train_experts`)

Ein einzelner Aufruf durchläuft die gesamte Schleife mit Hintergrundpriorität durch das Generation-Gate und belegt das Modell nur für das Training selbst:

1. **Aufgaben.** `count` + `seed` erzeugen ganzzahlige `+`-, `-`-, `*`-Aufgaben mit bekannten Antworten (derselbe Seed liefert dieselben Aufgaben); `file` liest stattdessen JSONL-Zeilen `{"prompt": "...", "answer": "..."}`. Ein Fünftel der Aufgaben (mindestens eine) wird zurückgehalten und niemals für das Training verwendet.
2. **Sammeln.** Das Modell beantwortet jede Aufgabe im Greedy-Modus über das Chat-Template (Thinking deaktiviert, 64 Tokens). Mit `teacher` (Basis-URL eines OpenAI-kompatiblen Servers; `teacherModel` benennt das Modell, `BASALT_TEACHER_API_KEY` wird nur dann als Bearer-Token gesendet, wenn Schema, Host und Port des Teachers mit `BASALT_TEACHER_URL` übereinstimmen, sodass ein Client den Schlüssel nicht anderweitig verschicken kann) beantwortet der Teacher die Trainingsaufgaben ebenfalls.
3. **Verifizieren.** Eine Antwort wird nur behalten, wenn ihre abschließende Ganzzahl (Kommas ignoriert) einer bekannten numerischen Antwort entspricht oder ihre letzte Zeile – unter Ignorieren von Groß-/Kleinschreibung und umgebender Zeichensetzung – einer Textantwort entspricht (oder auf `: <answer>` endet). Falsche, leere und fehlerhafte Antworten werden verworfen. **Nichts Unverifiziertes wird jemals gelernt:** Wenn keine Antwort verifiziert wird, meldet der Bericht `verified: 0`, es wird kein Adapter trainiert, nichts geschrieben und das Modell bleibt unberührt.
4. **Trainieren.** Ein LoRA (`rank`, Adam mit `learningRate`, `iterations`) wird mit dem `LoRATrain` des Forks auf den verifizierten Prompt+Antwort-Paaren trainiert – gerendert wie vom Modell gesehen –, und zwar auf den Attention- sowie dichten bzw. Shared-Expert-Projektionen von `layers` (Standard: alle). Per-Experten-LoRA ist nicht verfügbar – die `SwitchLinear`-Expertenlayer des Forks akzeptieren keinen Adapter –, daher wird `experts` mit einem Fehler abgelehnt. Ein Modell, das bereits einen Adapter trägt, wird ebenfalls abgelehnt. Ein Abbruch des Aufrufs stoppt das Training bei der nächsten Iteration und belässt das Modell unverändert.
5. **Speichern.** `adapter_config.json` + `adapters.safetensors` (das mlx-lm-Adapterformat) werden nach `output` oder `~/.basalt/adapters/<model fingerprint>/<timestamp>` (0700) geschrieben. `output` darf noch nicht existieren, und ein Pfad innerhalb des Modellverzeichnisses wird abgelehnt: Nichts wird überschrieben und nichts neben den Quellgewichten abgelegt.
6. **Messen und Anwenden.** Die Held-out-Genauigkeit wird mit aktivem Adapter gemessen, die Trainingslayer werden wieder entfernt und der Adapter wird über `applyAdapter` mit dem Quality-Gate aus `heldOut`, `maxKL` und `maxPerplexityRise` angewendet: Entweder wird er zu einem markierten Overlay ohne Ablaufzeit, das `overlay_undo` entfernt, oder er wird vom Gate abgewiesen.

Der Bericht führt `collected` (für das Training gesammelte Antworten), `verified`, `losses` (mittlerer Trainings-Loss je Meldeschritt, höchstens 20), `accuracyBefore` / `accuracyAfter` auf den Held-out-Aufgaben, `adapterDirectory` und das `verdict` des Gates auf.

## Kontrastives Steering („Meditation“, #332 Modus 2)

Hierbei handelt es sich um eine gerichtete Suche nach Aktivierungsrichtungen, nicht um Selbstbewusstsein. Nichts davon bedeutet, dass das Modell über sich selbst nachdenkt: Basalt lässt zwei Prompt-Mengen durch das Modell laufen, subtrahiert die Mittelwerte der Hidden States und misst, was die Differenz bewirkt.

`steer_search` führt jeden positiven und negativen Prompt durch einen sondierten Forward-Pass und speichert das Residual nach jedem Layer an der letzten Position.

- **Richtung pro Layer:** mean(positive) − mean(negative). `norm` ist die Länge dieses Vektors; `separation` ist der Abstand zwischen den Klassenmittelwerten der Projektionen auf die Einheitsrichtung (gleich `norm`), geteilt durch die gepoolte Standardabweichung dieser Projektionen. Der beste Layer weist die höchste Separation auf, und `direction` ist sein Einheitsvektor (Hidden-Dimension).
- **Heads:** Jeder Attention-Head des besten Layers und des unmittelbar vorhergehenden Layers wird der Reihe nach ablatiert (Intervention `ablate_head`), und `effect` misst, wie stark die Positiv-minus-Negativ-Projektionslücke im besten Layer schrumpft. Heads späterer Layer können dieses Residual nicht beeinflussen und werden nicht getestet. Höchstens `topHeads` werden gemeldet, geordnet nach dem größten Effekt. Da dies einen Forward-Pass pro Head und Prompt erfordert, sollten die Prompt-Mengen bei großen Modellen klein gehalten werden oder `topHeads: 0` übergeben werden.
- **Versuche:** Für jeden Skalierungsfaktor werden die Held-out-Prompts nach dem besten Layer mit `direction × scale` gesteuert (Intervention `steer`, nur für Forschung). `klDivergence` ist die mittlere KL-Divergenz der Next-Token-Verteilung gegenüber dem ungesteuerten Lauf; `projectionShift` ist die mittlere Änderung der Projektion auf `direction`, gemessen nach dem letzten Layer. Am gesteuerten Layer entspricht diese Verschiebung konstruktionsbedingt dem Skalierungsfaktor, sodass der letzte Layer zeigt, wie viel des Impulses erhalten bleibt.
- Es wird kein Overlay registriert: Ablationen und Steering existieren nur innerhalb dieses Aufrufs.

`steer_commit` macht eine wirksame Richtung über das Gate für die Sitzung dauerhaft. Es wählt die dichte Projektion des Layers, die in das Residual schreibt (`mlp.down_proj`, sonst `feed_forward.w2`, sonst `self_attn.o_proj`; ein MoE-Layer ohne dichte Projektion wird abgelehnt), misst die mittlere Eingabe `m` dieses Moduls an der letzten Position über die `positive`-Prompts und ruft den Adapter-Pfad mit `.rankOne(module: <full module path>, input: m / (m·m), output: direction, scale: scale)` auf, sodass das Update ΔW = scale · direction ⊗ m / (m·m) beträgt.

Die Näherung, klar benannt:

- Das Update addiert für jedes Token `scale · (x·m / m·m) · direction` zur Ausgabe der Projektion. Das entspricht exakt dem Steering-Vektor nur dort, wo die Moduleingabe `x` dieselbe Projektion auf `m` hat wie der positive Mittelwert; andere Tokens erhalten mehr, weniger oder das Gegenteil. `steer` addiert hingegen denselben Vektor an jeder Position.
- Im Residual ist es nur dann exakt, wenn die Ausgabe der Projektion direkt auf den Residual-Stream addiert wird (Pre-Norm-Blöcke wie Qwen und LFM). Folgt eine Normalisierung auf die Projektion (Gemmas Post-Attention- und Post-Feedforward-Normen), wird der Vektor neu skaliert und nur seine Richtung bleibt annähernd erhalten. Greift der Fallback `self_attn.o_proj`, durchläuft der Vektor außerdem das MLP des Layers, bevor er das Residual nach dem Layer erreicht.
- Der Adapter wird nur akzeptiert, wenn das Gate besteht (`maxKL`, `maxPerplexityRise` auf `heldOut`); führen Sie zuerst `steer_search`-Versuche durch und committen Sie einen Skalierungsfaktor mit akzeptabler KL-Divergenz.

## Tokenweiser Trace und bedingte Ablation (#405)

Das Vorgehen für ein Verhalten wie Ablehnung (Refusal): Trace → Kontrast → bedingte Ablation → Vergleich.

1. **`trace_tokens`** auf einigen Prompts, die das Verhalten zeigen. `ForwardProbe.headOutputs` des Forks liefert die Attention-Ausgabe jedes Heads vor der Output-Projektion (vor Qwen3.5s Output-Gate), sodass Fused Attention aktiv bleibt; `routerTopK` liefert die Experten und Gates, die der Layer tatsächlich kombiniert (einschließlich Expert-Bias, Group-Top-k und Skalierung). Pro Token und getractem Layer enthält die Datei:
   - `heads`: L2-Norm der Ausgabe jedes Heads;
   - `experts` (Ganzzahlen) und `gates`; `routerEntropy` (Softmax über die Router-Logits) und `routerMargin` (oberstes minus zweites Gate);
   - `projection`: Residual · Richtung nach dem Layer für jeden Eintrag in `projections`;
   - `attentionEntropy` (Opt-in, explizite Attention, langsamer) und `lens`/`lensProbabilities` (Opt-in Top-k-Logit-Lens).

   Dateiformat: eine JSON-Headerzeile (`layout.floats` und `layout.ints` listen `{layer, field, width}`-Segmente in Spaltenreihenfolge), eine JSON-Zeile pro Token (`position`, `id`, `token`, `generated`, `ints`), gefolgt von `tokens × floatsPerToken` Little-Endian-`float32`-Werten. Vor dem Lauf geprüfte Limits: Prompt plus `maxTokens` ≤ 2048 Tokens, Layer innerhalb des Modells, höchstens 16 Projektionen der Hidden-Größe auf getracten Layern und eine geschätzte Dateigröße ≤ `maxFileBytes` (≤ 64 MiB; die kodierte Größe wird vor dem Schreiben erneut geprüft).
2. **`activation_contrast`** mit Menge `a` (zeigt das Verhalten) und Menge `b` (zeigt es nicht). Die mittlere Aktivierung pro Prompt jeder Komponente (Head-Norm; Experten-Gate falls geroutet, sonst 0) liefert Cohen's d und AUROC; `rateA`/`rateB` zählen Tokens, bei denen die Norm eines Heads über seinem gepoolten Mittelwert liegt bzw. der Experte geroutet wurde; `onsetA`/`onsetB` mitteln die `onsetWindow` Prompt-Tokens vor Beginn der Generierung. `direction` bei `bestLayer` ist die Residual-Richtung zwischen beiden Mengen.
3. **`intervene` mit `when`**: Ablatiert einen eingestuften Head oder Experten (oder steuert mit negativem Skalierungsfaktor, um eine Richtung zu subtrahieren) nur dort, wo ein Auslöser greift – die Projektion auf die Kontrastrichtung über einem Schwellenwert oder ein Experte, der mit einem Gate über `gate` geroutet wurde –, optional für `window` Tokens ab dem ersten Auslöser. Der Layer der Aktion muss nach dem Auslöser-Layer liegen; der Auslöser wird im selben Forward-Pass gelesen (eine GPU-Synchronisation pro bedingtem Overlay und Forward-Pass), sodass die Änderung auf genau dem Token greift, das auslöst – sowohl im Prompt als auch in der Generierung. Ein Auslöser, der niemals feuert, belässt die Logits bit-identisch.
4. **`experiment_compare`** mit dem bedingten Overlay im Vergleich zu einem statischen `ablate_*`-Overlay derselben Komponente und einem `steer_commit`-Adapter: KL-Divergenz, Perplexität, die Texte und `triggerRate` (der Anteil der Tokens, auf die die bedingte Bearbeitung angewendet wurde).

Noch nicht verdrahtet: die System-One-Auslesung als Auslöser oder Labeller, gesampelte (nicht-greedy) Traces, Chat-Nachrichten als Eingabe (Template selbst anwenden) sowie ein dedizierter Evaluations-Runner für Ablehnungsrate, Genauigkeit und Dekodiergeschwindigkeit.

## Aktuelle Erweiterungen (`v0.12`)

- **Automatische BOS-Behandlung in rohen X-Ray-Traces (`#630`):** `trace_run`, `trace_tokens`, `experiment_compare`, `contrast`, `activation_contrast`, `steer_search` und `TokenAttribution` teilen sich `ModelProbe.encoded(_:_:)`, sodass Tokenizer, deren `encode(addSpecialTokens: true)` kein `bosToken` einfügt (wie etwa Gemma 4), dennoch `<bos>` an Position 0 erhalten und BOS-Attention-Sink-Layer (`L43–L56`) ohne doppeltes BOS bewahrt bleiben.
- **Tokenweise Logprobs, Attribution & Halluzinationsrisiko (`#613`, `#614`):** `POST /v1/xray/tokens` und Generierungsrouten stellen tokenweise Logprobs, Top-k-Entropie, Layer-/Head-/MoE-Attribution und Halluzinationsrisiko-Signale bereit.
- **Mechanistische Chirurgie & Einfluss-Attribution (`#600`, `#622`):** GQA-KV-Gruppen-Patching, kognitive Nullraum-Projektion, Anti-Steering-Guard, `MAO-SNMF`-Feature-Zerlegung, `Score d'influence`-Head-Attribution (`HeadInfluenceCore`) und temperatur-skaliertes Routing (`CalibratedRouterGate`).
- **HITL-Governance (`#626`):** `HITLGovernanceCore` / `HITLApprovalActor` sichern wirkungsmächtige MCP-Aktionen über `/v1/mcp/approvals`, gekoppelt mit dem LAAS-CNRS `SymbolicKnowledgeGraph`.
