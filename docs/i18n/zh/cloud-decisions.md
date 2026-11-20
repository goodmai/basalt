# 使用 Gemini Decision 1 进行云端决策

> Languages: [English](../../cloud-decisions.md) · [Русский](../ru/cloud-decisions.md) · **中文** · [Español](../es/cloud-decisions.md) · [Français](../fr/cloud-decisions.md) · [日本語](../ja/cloud-decisions.md) · [Deutsch](../de/cloud-decisions.md)

Google 的 `gemini-decision-1` 可以在单次前向推理中针对文本、图像、音频、视频和 PDF 回答 System One 问题（`noul`、`choice`、`score`）。它的通信协议格式与 basalt 自身的 `/v1/systemone` 一致，因此可以作为本地决策模型前置的云端决策阶段使用：

```bash
export GEMINI_API_KEY=...        # read from the environment, never a flag or a file
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- 不确定的问题（置信度低于 `--decision-escalate-below`）、调用失败或触发速率限制的请求都会转交给本地模型处理。
- `/v1/systemone` 支持在 `state` 旁传入 `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]`。如果没有任何阶段能够读取该媒体类型，请求会被拒绝。
- 在 compose 配置文件中，该模型配置为 `source: cloud:gemini/gemini-decision-1`，并搭配 `api_key_env: GEMINI_API_KEY`。
  - Gemini 模型仅用于决策级联（decide cascades）。
  - 除非该模型设置了 `data: media`，否则流水线会将感知阶段提取的文本保留在本地 Mac 上；当设置了 `data: media` 时，附带的原始文件本身及其感知文本都会发送给 Google 用于决策。
- 费用根据 API 返回的 token 用量按每百万输入 token $0.10 计算，并计入 compose 预算上限。

## 校准教师模型

Gemini Decision 1 不仅返回分类标签，还会返回经过校准的概率值。据 Google 报告，它在概率校准方面处于领先水平。这使其成为 basalt 本地决策模型的理想校准教师（calibration teacher）：将相同的状态分别输入给它和本地 readout，然后依据其输出的概率拟合本地温度参数（temperature），或训练一个 OpenJev 决策头。这样一来，本地决策器输出的置信度就具备了与云端一致的语义，升级（escalation）阈值也可以直接迁移复用。将该能力接入校准步骤的工作计划与 #423 的本地 JEV 工作（J6：从 bundle 中读取校准参数）一并推进。
