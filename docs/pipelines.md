# Pipeline presets and memory modes

`/v1/pipeline` (MCP `basalt_pipeline`) runs perceive (OCR, speech) → decide (System One) → answer (the chat model) in one process. A preset fills the serve flags for a shipped profile; any flag you give explicitly wins:

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One by readout of the chat model, the chat model
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe and the chat model
basalt serve --pipeline-preset support-triage  # System One with the ticket questions, and the chat model
```

The chat model is Gemma 4 12B, or Qwen3.8-27B 3-bit when it fits beside the other models with 8 GB to spare. `--pipeline-memory` sets how the models share memory:

- `resident` keeps every model loaded, so their sum must fit the working set.
- `sequential` loads each perceiver for its stage and unloads it after. The chat model stays loaded when it fits beside the largest perceiver, and is unloaded around it otherwise. Only the largest stage plus 2 GB must fit, so a smaller Mac runs a heavy set more slowly. Sequential pipeline requests run one at a time, and plain chat requests wait while models swap.
- `auto` (default) picks resident when the sum fits, otherwise sequential, and refuses only a stage too big on its own.

Measured on an M4 Pro, 24 GB (19.1 GB working set), release build, commit 1921c3c. The second request has a warm page cache:

| Preset | Models | Plan | Request 1 | Request 2 | Of which swapping | Peak footprint |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 GB + Nemotron Transcribe 30B-A4B 14.9 GB | sequential (resident needs 25.9 GB), chat unloaded around speech | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 GB |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`, chat stays loaded | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 GB |
