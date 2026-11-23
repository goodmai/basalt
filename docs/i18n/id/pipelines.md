# Preset pipeline dan mode memori

> Languages: [English](../../pipelines.md) · [Русский](../ru/pipelines.md) · [中文](../zh/pipelines.md) · [Español](../es/pipelines.md) · [Français](../fr/pipelines.md) · [日本語](../ja/pipelines.md) · [Deutsch](../de/pipelines.md) · [한국어](../ko/pipelines.md) · **Bahasa Indonesia**

`/v1/pipeline` (MCP `basalt_pipeline`) menjalankan tahap persepsi (OCR, suara) → keputusan (System One) → jawaban (model chat) dalam satu proses. Preset mengisi flag `serve` bawaan; flag eksplisit dari Anda selalu menang:

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One, dan model chat
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe dan model chat
basalt serve --pipeline-preset support-triage  # System One untuk triase tiket dan model chat
```

Model chat default adalah Gemma 4 12B, atau Qwen3.8-27B 3-bit bila muat berdampingan dengan sisa 8 GB. `--pipeline-memory` mengatur cara model berbagi memori:

- `resident`: menjaga semua model tetap dimuat; total ukuran harus muat di working set.
- `sequential`: memuat setiap model persepsi saat tahapnya berjalan dan membongkarnya setelah selesai. Model chat tetap dimuat bila muat bersama model persepsi terbesar.
- `auto` (default): memilih `resident` bila totalnya muat, jika tidak memilih `sequential`, dan hanya menolak jika satu tahap saja terlalu besar.

Diukur di M4 Pro 24 GB (working set 19.1 GB), build release (komit `1921c3c`):

| Preset | Model | Rencana | Permintaan 1 | Permintaan 2 | Waktu swap | Puncak memori |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 GB + Nemotron Transcribe 30B-A4B 14.9 GB | sequential (resident butuh 25.9 GB), chat dibongkar saat suara | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 GB |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`, chat tetap dimuat | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 GB |
