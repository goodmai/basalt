# Preajustes de pipeline y modos de memoria

> Languages: [English](../../pipelines.md) · [Русский](../ru/pipelines.md) · [中文](../zh/pipelines.md) · **Español** · [Français](../fr/pipelines.md) · [日本語](../ja/pipelines.md) · [Deutsch](../de/pipelines.md)

`/v1/pipeline` (MCP `basalt_pipeline`) ejecuta percibir (OCR, voz) → decidir (System One) → responder (el modelo de chat) en un único proceso. Un preajuste rellena las opciones de `serve` para un perfil predefinido; cualquier opción que especifiques explícitamente tiene prioridad:

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One by readout of the chat model, the chat model
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe and the chat model
basalt serve --pipeline-preset support-triage  # System One with the ticket questions, and the chat model
```

El modelo de chat es Gemma 4 12B, o Qwen3.8-27B 3-bit cuando cabe junto a los demás modelos dejando 8 GB libres. `--pipeline-memory` establece cómo comparten la memoria los modelos:

- `resident` mantiene todos los modelos cargados, por lo que su suma debe caber en el conjunto de trabajo.
- `sequential` carga cada perceptor para su etapa y lo descarga después. El modelo de chat permanece cargado cuando cabe junto al perceptor más grande y, en caso contrario, se descarga durante su ejecución. Solo es necesario que quepa la etapa más grande más 2 GB, de modo que un Mac más pequeño ejecuta un conjunto pesado con menor velocidad. Las peticiones de pipeline secuencial se ejecutan de una en una, y las peticiones de chat normales esperan mientras se intercambian los modelos.
- `auto` (predeterminado) elige `resident` cuando la suma cabe; en caso contrario elige `sequential`, y solo rechaza una etapa que sea demasiado grande por sí sola.

Medido en un M4 Pro, 24 GB (19.1 GB de conjunto de trabajo), compilación release, commit 1921c3c. La segunda petición cuenta con una caché de páginas caliente:

| Preajuste | Modelos | Plan | Petición 1 | Petición 2 | De ello intercambio | Huella máxima |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 GB + Nemotron Transcribe 30B-A4B 14.9 GB | sequential (resident necesita 25.9 GB), chat descargado durante el reconocimiento de voz | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 GB |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`, el chat permanece cargado | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 GB |
