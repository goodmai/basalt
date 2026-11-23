# Funktionen und Benchmarks

> Languages: [English](../../benchmarks.md) · [Русский](../ru/benchmarks.md) · [中文](../zh/benchmarks.md) · [Español](../es/benchmarks.md) · [Français](../fr/benchmarks.md) · [日本語](../ja/benchmarks.md) · **Deutsch** · [한국어](../ko/benchmarks.md) · [Bahasa Indonesia](../id/benchmarks.md)

## Funktionen im Überblick

Jede Beschleunigung ist opt-in und lässt den Greedy-Output unverändert, sofern in der Zeile nichts anderes steht. Zahlen: M4 Pro 24 GB am Netzteil, Release-Build.

| Funktion | Flag | Was sie tut | Messwert |
|---|---|---|---|
| Prompt-Lookup (spekulatives Decoding) | `serve --speculative-lookup` | entwirft bis zu 16 Tokens aus Prompt oder Output und verifiziert sie in einem Forward-Pass | 1.84–2.66× bei Code-Edits (Bonsai 2 27B), 3.45–3.62× (Gemma 4 12B); ~1.0× bei Prosa |
| MTP-Head-Drafting | `serve --mtp [--mtp-block N]` | entwirft mit dem Multi-Token-Prediction-Head des Modells (`mtp.safetensors`, z. B. MTPLX-Pakete); exakt bei jeder Temperatur. Runden starten bei einem entworfenen Token und wachsen nach vier vollständig akzeptierten Runden auf bis zu N−1 (Standard 4); eine Ablehnung fällt auf eins zurück | 1.30× bei Code, 1.04× bei Prosa, 1.26× bei Reasoning (Bonsai 2 27B, T=0.7); ein Folgeturn nutzt den Prompt-Cache wie normales Decoding: TTFT 12.5 s → 0.75 s bei einem 1,149-Token-Turn (Bonsai 2 27B) |
| Ternary Bonsai 2 | (automatisch) | lädt Hadamard-rotierte ternäre Checkpoints (`prism_hadamard_qwen35`) | 27B-Modell in 7.4 GB |
| Embeddings und Rerank | `serve --embedding-model`, `--rerank-model` | OpenAI `/v1/embeddings`, Cohere/Jina `/v1/rerank` parallel zum Chat-Modell | bge-small 229–299 Texte/s, Qwen3-Embedding-0.6B 20 Texte/s (Batch 32) |
| Prompt-Cache auf Datenträger | `serve --prompt-cache-disk-mb N` | eine fortgesetzte Sitzung oder ein neu gestarteter Server stellt das längste passende Präfix wieder her, statt es neu zu prefillen | — |
| KV-Cache-Quantisierung und Zonen | `serve --kv-bits 4\|8\|auto\|k8v4`, `--kv-zones` | bringt längere Kontexte im gleichen Speicher unter | 16k-Prompt Peak 1.03 GB statt 3.11 GB (Qwen3-4B, 4-bit-Zonen) |
| Gleichzeitige Anfragen | `serve --max-concurrent N` | jede laufende Anfrage reserviert ihren KV-Platz; passt eine nicht mehr, erhält sie 503 `Retry-After` statt Out-of-Memory | — |
| DRY- und XTC-Sampler (experimentell) | `serve --experimental-samplers` | `dry_*` / `xtc_*`-Request-Felder | kein Anti-Loop-Werkzeug: bei einem 2-bit-Modell versteckt DRY Schleifen hinter verfälschten Fakten (#177) |

## Woher die Zahlen im README stammen

Jede Zahl auf dem ersten Bildschirm des README stammt aus einem protokollierten Lauf, und der PR, der die Funktion gelandet hat, enthält die Tabelle, den Commit und den Log-Pfad. M4 Pro 24 GB, Release-Build, sofern der PR nichts anderes angibt.

| Aussage | Quelle |
|---|---|
| Qwen3-4B 78–89 tok/s; Ternary Bonsai 2 27B in 7.4 GB mit 20.9–21.5 tok/s, 31–38 mit Prompt-Lookup bei Code | [#239](https://github.com/goodmai/basalt/pull/239), aus [#218](https://github.com/goodmai/basalt/pull/218), [#231](https://github.com/goodmai/basalt/pull/231), [#232](https://github.com/goodmai/basalt/pull/232) |
| Bonsai 2 27B mit seinem MTP-Head, 27.1 tok/s bei Code | [#263](https://github.com/goodmai/basalt/pull/263) |
| Prompt-Lookup 3.45–3.62× auf Gemma 4 12B, Greedy-Output unverändert | [#265](https://github.com/goodmai/basalt/pull/265) |
| Zweiter Agent-Turn ca. 1 s statt 36 s auf Ornith-1.5-9B (12.6k-Token-Prompt) | [#82](https://github.com/goodmai/basalt/pull/82) |
| 16k-Prompt Peak 1.03 GB statt 3.11 GB mit KV-Zonen auf Qwen3-4B | [#159](https://github.com/goodmai/basalt/pull/159) |
| Mixture of Models: jedes Label korrekt, $0.035 pro 1000 Anfragen, p95 4.8 s gegenüber 10.0 s | [#418](https://github.com/goodmai/basalt/pull/418) |
| Mac-App: warmes Fenster 257 ms, 0 Idle-Wakeups, 7.4 ms p95 pro gestreamtem Token, 40 ms Schnellanfrage | [#463](https://github.com/goodmai/basalt/pull/463) |
| Ein 9B-Judge neben Gemma 4 12B auf 24 GB: 4-bit passt (15.3 GB), 98.9% bei 77% weniger LLM-Tokens | [#465](https://github.com/goodmai/basalt/pull/465) |
| Qwen3.6-35B-A3B-8bit (37.7 GB) auf 24 GB von SSD gestreamt mit 8.77 tok/s bei 0 Swap | [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478) |
| Post-Quantisierung `--experimental-pv-tune`: 8.7–29% Erholung der 2–3-bit-Perplexity in unter 3 Minuten | [#486](https://github.com/goodmai/basalt/pull/486) |
| QTIP-Bitshift-Trellis `--experimental-qtip`: `Qwen3.8-27B` 2.5 bpw in 10.00 GB schlägt 3-bit Affine & PV-Tuning (16k PPL 4.621 vs 4.702 / 4.652); 2.0 bpw in 8.48 GB senkt 2-bit-PPL um 72.1% (5.345 vs 19.146) | [#494](https://github.com/goodmai/basalt/pull/494) |
| Native `.gguf` GSQ-RCO- / `i-quant`-Ausführung auf Metal (`IQ4_XS` / `IQ3_S`) | [#433](https://github.com/goodmai/basalt/pull/433) |
| Rekursives `Flint-40B-A2.5B` (`oQ3e`) mit 67.02 tok/s Decode (273.5 tok/s Prefill) in 17.61 GB auf M4 Pro 24 GB | [#522](https://github.com/goodmai/basalt/pull/522) |
| `Zyphra/ZONOS3-Mini-hf` 44.1 kHz neuronales TTS in 2.24 GB bei RTF 0.228–0.263 (3.8–4.4× Echtzeit) über EN/DE/RU | [#527](https://github.com/goodmai/basalt/pull/527) |
| 84 hardware-gestaffelte Compose-Profile (`16–256 GB × 12`) mit automatischer RAM-Stufen-Empfehlung in `Basalt.app` | [#528](https://github.com/goodmai/basalt/pull/528) |
| Single-Pass `MACaQSwitchLinear` WHT + `gatherQuantizedMM`: `3.41×` MoE-Kernel (`2.06 → 0.60 ms`), `+82%` End-to-End auf Gemma 4 26B-A4B (`49.2–54.8 tok/s`), `64–75 tok/s` auf Ornith 35B-A3B 3-bit | [#628](https://github.com/goodmai/basalt/pull/628) |
| `Qwen3.5-9B-PV-MACaQ-4bit`: `4/5` exakte Ladder-Parität mit 8-bit, `+52%` Decode (`43.2 tok/s` p50), `−33%` RAM (`6.68 vs 9.97 GB`), max. Layer-Drift `3.37%` | [#545](https://github.com/goodmai/basalt/pull/545), [#630](https://github.com/goodmai/basalt/pull/630) |
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`: `15.1 GB` RAM (`18.09 GB` Peak) auf 24-GB-Mac, `4/5` Ladder + `4/6` Lunar bei `14.1 tok/s` p50 (4-bit Ref läuft OOM) | [#628](https://github.com/goodmai/basalt/pull/628), [#630](https://github.com/goodmai/basalt/pull/630) |

Die golden-getesteten Modelle im README werden in der CI durch `scripts/check_readme_models.swift` geprüft: Jedes dort aufgeführte Modell muss auf ein existierendes Fixture in `Tests/BasaltTests/Fixtures/golden` verlinken.

Eine eigenständige statische Benchmark-Seite ([`docs/benchmark-page/index.html`](../../benchmark-page/index.html)) und ein prüfbereites Hugging-Face-Collection-Manifest ([`docs/benchmark-page/hf-collection.json`](../../benchmark-page/hf-collection.json)) werden aus diesen Tabellen, `benchmarks/` und den Golden-Fixtures durch `swift scripts/build_benchmark_page.swift` generiert (und in der CI mit `--check` verifiziert; lokal `--logs logs` übergeben, um `summary`-Tabellen der Runner anzuhängen). Die Veröffentlichung nach außen bleibt eine menschliche Entscheidung.

## Benchmarks

Zwei Test-Harnesses, beide gegen einen laufenden Server gerichtet, beide mit Zahlen, die sich auf deiner eigenen Maschine reproduzieren lassen.

**Die Leiter zum Mond** — `benchmarks/ladder/run.py`. Fünf Stufen, jede strikt schwerer als die darunter, gipfelnd in einer vollständigen Erde-zum-Mond-Mission. Nichts wird danach bewertet, wie die Antwort klingt: Der eigene Code des Modells wird gegen Eingaben ausgeführt, die der Prompt nie gezeigt hat, sodass eine plausibel klingende falsche Antwort trotzdem durchfällt.

| Stufe | Aufgabe | Was tatsächlich getestet wird |
|---|---|---|
| 1 | Biquadratische Gleichung, reelle Nullstellen | geschlossene Form plus Randfälle: keine, doppelte, Null |
| 2 | Biquadratische Gleichung, komplexe Nullstellen | Zweigbehandlung — alle vier Nullstellen, mit Vielfachheit |
| 3 | Fourier-Koeffizienten | numerische Integration, geprüft an Sägezahn, Rechteckschwingung, Kosinus |
| 4 | Pumpenförderhöhe | Einheiten (mm vs. m, L/s vs. m³/s) + Colebrook, für die es keine geschlossene Form gibt |
| 5 | **Erde → Mond-Mission** | Ausströmgeschwindigkeit aus `sqrt(2ηQ)`, Ziolkowski je Stufe, Vierkörper-Propagation, Mondbremsung, weiche Landung — jede Konstante steht im Prompt, nichts aus einem Lehrbuch abrufbar |

Die oberste Stufe wird anhand von sechs Invarianten statt einer einzelnen Antwort bewertet, sodass ein Modell, das die Energetik richtig trifft, die Trajektorie aber nie propagiert, dennoch eine Kennzahl liefert. Ihr Prompt liegt in `benchmarks/ladder/lunar_prompt.md`; die Referenzwerte in `lunar_task.md` werden dem Modell nie gezeigt.

`--warmup` stellt drei günstige Smoke-Tests (Algebra, Fibonacci, Übersetzung) voran, die jedes Instruct-Modell bestehen sollte. Sie fließen nicht in den Ladder-Score ein.

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`, offizielle pass@2-Regeln. Der Korpus ist nicht mitgeliefert; lade ihn zuerst nach `benchmarks/arc-agi/dataset/` herunter (siehe `benchmarks/arc-agi/ARC_AGI.md`).

### Ergebnisse

Gemessen auf einem Mac der M-Serie mit 24 GB Unified Memory. `thinking` ist der `<think>`-Block des Chat-Templates: bei Reasoning-Modellen standardmäßig aktiv, deaktiviert mit `--reasoning-effort none`.

| Modell | Warm-up | Ladder | Oberste Stufe | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` (Thinking aus, 24 GB) | 3/3 | **4/5** | 4/6 Invarianten | — | 11.6–13.5 (p50 14.1) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` (Thinking aus, 24 GB) | 3/3 | **4/5** | 0/6 (length) | — | 9.4–11.4 (p50 11.6) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (Thinking aus) | 3/3 | **4/5** | 0/6 (Prosa) | — | 35.6–41.6 (p50 43.2) |
| `mlx-community/Qwen3.5-9B-8bit` (Ref, Thinking aus) | 3/3 | **4/5** | 0/6 (Prosa) | — | 24.7–27.3 |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` (Thinking aus) | 3/3 | **4/5** | 5/6 Invarianten | — | 56.0–63.4 |
| `goodmai/gemma-4-12B-it-qat-4bit` (Thinking aus) | 3/3 | **4/5** | **6/6 PASS** | — | 18.8–22.2 |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` (Thinking aus, `#628`) | 3/3 | 3/5 | 0/6 | — | 49.2–54.8 |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` (Thinking aus, `#628`) | 3/3 | 1/5 | 0/6 | — | 64.0–74.7 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (Thinking aus) | 3/3 | 1/4 ² | nicht ausgeführt | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (Thinking an) | 2/3 | 0/4 ² | nicht ausgeführt | nicht ausgeführt | ~43 |

Ornith 9B Stufe für Stufe, Thinking aus:

| Stufe | Ergebnis | Detail |
|---|---|---|
| 1 · biquadratisch reell | ✅ | Nullstellen korrekt, einschließlich der doppelten und leeren Fälle |
| 2 · biquadratisch komplex | ❌ | `x⁴ + 1` liefert 2 statt 4 Nullstellen |
| 3 · Fourier | ❌ | `a0 = 10.86` für `f(x) = x`, erwartet 0 — die Integration ist falsch, nicht die Formel |
| 4 · Pumpenförderhöhe | ❌ | `v = 0.0298` m/s, erwartet 1.79 — mm wurden nie in m umgerechnet |
| 5 · Mondmission | — | nicht ausgeführt |

Warm-up, Thinking aus: Algebra ✅ (163 tok, 4 s), Fibonacci ✅ (757 tok, 16 s), Übersetzung ✅ (303 tok, 7 s — Deutsch idiomatisch, Französisch eher paraphrasiert als übersetzt).

² Stufen 1 und 2 wurden gemessen, bevor die Leiter in getrennte Prompts aufgeteilt wurde: Ein Prompt verlangte beide Funktionen, die reelle Hälfte war korrekt und die komplexe nicht. Stufe 5 wurde mit diesem Modell noch nicht ausgeführt.

¹ Die ARC-AGI-Spalte ist ein Teillauf: drei Trainingsaufgaben (`007bbfb7`, `00d62c1b`, `017c7c7b`), keine bei pass@2 gelöst, 330 s gesamt. Der Lauf wurde vor den verbleibenden beiden gestoppt; werte ihn daher als Smoke-Test und nicht als Score.

Die Validatoren werden selbst ohne Modell oder GPU geprüft:

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 assertions, both directions
```

Mit Thinking **an** endeten alle drei schweren Aufgaben und die Übersetzung mit `finish_reason: length` — das Budget floss in den `<think>`-Block und es wurde nie eine Antwort ausgegeben. Das ist ein Budget-Versagen, keine Fähigkeitsgrenze, weshalb die obige Tabelle beide Modi getrennt aufführt.

### Reproduzieren

```bash
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

Ein Modell zur Tabelle hinzuzufügen erfordert jeweils einen Lauf — PRs mit neuen Zeilen sind willkommen.
