# Préréglages de pipeline et modes mémoire

> Languages: [English](../../pipelines.md) · [Русский](../ru/pipelines.md) · [中文](../zh/pipelines.md) · [Español](../es/pipelines.md) · **Français** · [日本語](../ja/pipelines.md) · [Deutsch](../de/pipelines.md) · [한국어](../ko/pipelines.md) · [Bahasa Indonesia](../id/pipelines.md)

`/v1/pipeline` (MCP `basalt_pipeline`) exécute percevoir (OCR, parole) → décider (System One) → répondre (le modèle de chat) dans un seul processus. Un préréglage renseigne les options de `serve` pour un profil fourni ; toute option que vous passez explicitement prévaut :

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One by readout of the chat model, the chat model
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe and the chat model
basalt serve --pipeline-preset support-triage  # System One with the ticket questions, and the chat model
```

Le modèle de chat est Gemma 4 12B, ou Qwen3.8-27B 3-bit lorsqu'il tient à côté des autres modèles avec une marge de 8 Go. `--pipeline-memory` définit la manière dont les modèles se partagent la mémoire :

- `resident` conserve tous les modèles chargés en mémoire ; leur somme doit donc tenir dans le working set.
- `sequential` charge chaque percepteur pour son étape et le décharge ensuite. Le modèle de chat reste chargé s'il tient à côté du plus grand percepteur, et il est déchargé le temps de son exécution dans le cas contraire. Seule la plus grande étape plus 2 Go doit tenir en mémoire, ce qui permet à un Mac plus modeste d'exécuter un ensemble lourd plus lentement. Les requêtes de pipeline séquentiel s'exécutent une par une, et les requêtes de chat simples attendent pendant le basculement des modèles.
- `auto` (par défaut) choisit `resident` lorsque la somme tient en mémoire, sinon `sequential`, et ne refuse qu'une étape trop volumineuse à elle seule.

Mesuré sur un M4 Pro, 24 Go (working set de 19.1 Go), build release, commit 1921c3c. La deuxième requête bénéficie d'un cache de pages chaud :

| Préréglage | Modèles | Plan | Requête 1 | Requête 2 | Dont basculement | Empreinte pic |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 Go + Nemotron Transcribe 30B-A4B 14.9 Go | séquentiel (`resident` nécessite 25.9 Go), chat déchargé pendant la parole | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 Go |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`, le chat reste chargé | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 Go |
