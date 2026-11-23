# Cloud-Entscheidungen mit Gemini Decision 1

> Languages: [English](../../cloud-decisions.md) · [Русский](../ru/cloud-decisions.md) · [中文](../zh/cloud-decisions.md) · [Español](../es/cloud-decisions.md) · [Français](../fr/cloud-decisions.md) · [日本語](../ja/cloud-decisions.md) · **Deutsch** · [한국어](../ko/cloud-decisions.md) · [Bahasa Indonesia](../id/cloud-decisions.md)

Googles `gemini-decision-1` beantwortet System-One-Fragen (`noul`, `choice`, `score`) in einem einzigen Durchgang über Text, Bilder, Audio, Video und PDFs hinweg. Sein Übertragungsformat (Wire Format) entspricht basalts eigenem `/v1/systemone`, sodass es als Cloud-Stufe vor einem lokalen Entscheidungsmodell arbeitet:

```bash
export GEMINI_API_KEY=...        # read from the environment, never a flag or a file
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- Unsichere Fragen (Konfidenz unter `--decision-escalate-below`), fehlgeschlagene Aufrufe und Rate-Limits gehen an das lokale Modell.
- `/v1/systemone` akzeptiert `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]` neben `state`. Medien werden abgelehnt, wenn keine Stufe sie lesen kann.
- In einer Compose-Datei lautet das Modell `source: cloud:gemini/gemini-decision-1` mit `api_key_env: GEMINI_API_KEY`.
  - Gemini-Modelle bedienen ausschließlich Entscheidungskaskaden.
  - Pipelines belassen wahrgenommenen Text auf diesem Mac, sofern dieses Modell nicht `data: media` setzt. In diesem Fall gehen die angehängten Dateien selbst sowie deren wahrgenommener Text zur Entscheidung an Google.
- Die Kosten werden anhand der von der API gemeldeten Token-Nutzung mit $0.10 pro Million Input-Tokens berechnet und auf Compose-Budgets angerechnet.

## Ein Kalibrierungslehrer

Gemini Decision 1 liefert kalibrierte Wahrscheinlichkeiten statt bloßer Labels. Google gibt an, dass es bei der Wahrscheinlichkeitskalibrierung führend ist. Das macht es zu einem idealen Kalibrierungslehrer für basalts lokale Entscheidungsmodelle: Lassen Sie dieselben Zustände durch dieses Modell und durch ein lokales Readout laufen und passen Sie die lokale Temperatur an dessen Wahrscheinlichkeiten an – oder trainieren Sie darauf einen OpenJev-Head. Dann bedeutet die Konfidenz des lokalen Entscheiders dasselbe wie die der Cloud, und Eskalationsschwellenwerte lassen sich direkt übertragen. Die Einbindung in den Kalibrierungsschritt ist zusammen mit der lokalen JEV-Arbeit aus #423 geplant (J6: aus dem Bundle gelesene Kalibrierung).
