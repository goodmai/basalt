# Keputusan cloud dengan Gemini Decision 1

> Languages: [English](../../cloud-decisions.md) · [Русский](../ru/cloud-decisions.md) · [中文](../zh/cloud-decisions.md) · [Español](../es/cloud-decisions.md) · [Français](../fr/cloud-decisions.md) · [日本語](../ja/cloud-decisions.md) · [Deutsch](../de/cloud-decisions.md) · [한국어](../ko/cloud-decisions.md) · **Bahasa Indonesia**

`gemini-decision-1` dari Google menjawab pertanyaan System One (`noul`, `choice`, `score`) dalam satu lintasan atas teks, gambar, audio, video, dan PDF. Format wire-nya cocok dengan `/v1/systemone` milik basalt sehingga dapat dipasang sebagai tahap cloud di depan model keputusan lokal:

```bash
export GEMINI_API_KEY=...        # dibaca dari lingkungan, tidak pernah dari flag atau file
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- Pertanyaan yang belum pasti (kepercayaan di bawah `--decision-escalate-below`), panggilan gagal, dan batas laju akan dialihkan ke model lokal.
- `/v1/systemone` menerima `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]` di samping `state`.
- Di file compose, model didefinisikan dengan `source: cloud:gemini/gemini-decision-1` dan `api_key_env: GEMINI_API_KEY`.
  - Model Gemini hanya melayani kaskade `decide`.
  - Pipeline menjaga teks hasil persepsi tetap di Mac Anda kecuali model menetapkan `data: media`.
- Biaya dihitung dari penggunaan token yang dilaporkan API ($0.10 per juta token input) dan masuk ke anggaran harian compose.

## Guru kalibrasi (Calibration teacher)

Gemini Decision 1 mengembalikan probabilitas terkalibrasi, bukan sekadar label. Dengan menjalankan status yang sama melalui Gemini Decision 1 dan readout lokal lalu menyesuaikan temperature lokal atau kepala OpenJev terhadap probabilitasnya, skor kepercayaan lokal akan setara dengan cloud sehingga ambang eskalasi tetap konsisten.
