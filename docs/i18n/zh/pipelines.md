# 流水线预设与内存模式

> Languages: [English](../../pipelines.md) · [Русский](../ru/pipelines.md) · **中文** · [Español](../es/pipelines.md) · [Français](../fr/pipelines.md) · [日本語](../ja/pipelines.md) · [Deutsch](../de/pipelines.md)

`/v1/pipeline`（MCP 工具为 `basalt_pipeline`）会在同一个进程中依次执行 感知（perceive：OCR、语音）→ 决策（decide：System One）→ 回答（answer：对话模型）。预设（preset）会为内置场景填充对应的 serve 参数；你显式传入的任何参数都会覆盖预设值：

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One by readout of the chat model, the chat model
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe and the chat model
basalt serve --pipeline-preset support-triage  # System One with the ticket questions, and the chat model
```

对话模型默认使用 Gemma 4 12B；如果设备内存允许在加载其他模型的同时还能余出 8 GB 空间，则使用 Qwen3.8-27B 3-bit。`--pipeline-memory` 用于控制各模型如何共享内存：

- `resident` 会让所有模型常驻内存，因此所有模型的内存总和必须能放入可用工作集（working set）中。
- `sequential` 会在执行某个阶段时加载对应的感知模型，并在阶段结束后卸载。如果对话模型能够与体积最大的感知模型同时装入内存，则会保持加载状态；否则在运行该感知模型期间会先卸载对话模型。此时只需满足“最大单阶段模型 + 2 GB”能装入内存即可，因此内存较小的 Mac 也能运行较重的模型组合（只是速度会慢一些）。顺序模式下流水线请求一次只能处理一个，普通对话请求在模型切换期间也会等待。
- `auto`（默认）会在模型总和能够装入内存时选择 `resident`，否则自动回退到 `sequential`，仅当某个单阶段模型自身过大无法装入时才会拒绝运行。

以下数据实测于 M4 Pro 24 GB（19.1 GB 工作集）、Release 构建、commit 1921c3c。第二次请求时系统页缓存（page cache）已预热：

| 预设 | 模型组合 | 调度方案 | 第 1 次请求 | 第 2 次请求 | 其中模型切换耗时 | 峰值内存占用 |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 GB + Nemotron Transcribe 30B-A4B 14.9 GB | sequential（resident 需要 25.9 GB），语音阶段期间卸载对话模型 | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 GB |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`，对话模型保持常驻 | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 GB |
