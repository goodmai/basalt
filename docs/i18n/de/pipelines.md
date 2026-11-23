# Pipeline-Voreinstellungen und Speichermodi

> Languages: [English](../../pipelines.md) · [Русский](../ru/pipelines.md) · [中文](../zh/pipelines.md) · [Español](../es/pipelines.md) · [Français](../fr/pipelines.md) · [日本語](../ja/pipelines.md) · **Deutsch** · [한국어](../ko/pipelines.md) · [Bahasa Indonesia](../id/pipelines.md)

`/v1/pipeline` (MCP `basalt_pipeline`) führt Wahrnehmung (OCR, Sprache) → Entscheidung (System One) → Antwort (das Chat-Modell) in einem einzigen Prozess aus. Eine Voreinstellung (Preset) füllt die Serve-Flags für ein mitgeliefertes Profil aus; jedes explizit angegebene Flag hat Vorrang:

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One by readout of the chat model, the chat model
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe and the chat model
basalt serve --pipeline-preset support-triage  # System One with the ticket questions, and the chat model
```

Das Chat-Modell ist Gemma 4 12B oder Qwen3.8-27B 3-bit, sofern es neben den anderen Modellen noch mit 8 GB Puffer in den Speicher passt. `--pipeline-memory` legt fest, wie sich die Modelle den Speicher teilen:

- `resident` hält jedes Modell geladen, sodass die Summe aller Modelle in das Working Set passen muss.
- `sequential` lädt jeden Perceiver für seine Phase und entlädt ihn danach wieder. Das Chat-Modell bleibt geladen, wenn es neben dem größten Perceiver Platz findet, und wird andernfalls für diesen Zeitraum entladen. Es muss lediglich die größte Phase plus 2 GB passen, sodass auch ein kleinerer Mac eine umfangreiche Zusammenstellung – wenn auch langsamer – ausführen kann. Sequenzielle Pipeline-Anfragen laufen einzeln nacheinander, und reguläre Chat-Anfragen warten, während Modelle ausgetauscht werden.
- `auto` (Standard) wählt `resident`, wenn die Summe passt, andernfalls `sequential`, und lehnt nur dann ab, wenn eine einzelne Phase bereits für sich genommen zu groß ist.

Gemessen auf einem M4 Pro, 24 GB (19.1 GB Working Set), Release-Build, Commit 1921c3c. Die zweite Anfrage profitiert von einem warmen Page Cache:

| Preset | Modelle | Plan | Anfrage 1 | Anfrage 2 | Davon Modellwechsel | Spitzenauslastung |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 GB + Nemotron Transcribe 30B-A4B 14.9 GB | sequential (resident benötigt 25.9 GB), Chat während Spracherkennung entladen | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 GB |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`, Chat bleibt geladen | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 GB |
