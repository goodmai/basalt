# 파이프라인 프리셋 및 메모리 모드

> Languages: [English](../../pipelines.md) · [Русский](../ru/pipelines.md) · [中文](../zh/pipelines.md) · [Español](../es/pipelines.md) · [Français](../fr/pipelines.md) · [日本語](../ja/pipelines.md) · [Deutsch](../de/pipelines.md) · **한국어** · [Bahasa Indonesia](../id/pipelines.md)

`/v1/pipeline`(MCP `basalt_pipeline`)은 단일 프로세스에서 인지(OCR, 음성) → 판단(System One) → 응답(채팅 모델)을 실행합니다. 프리셋은 기본 `serve` 플래그를 채워주며, 사용자가 명시한 플래그가 항상 우선합니다:

```bash
basalt serve --pipeline-preset document-qa     # PaddleOCR-VL, System One, 채팅 모델
basalt serve --pipeline-preset voice-agent     # Nemotron Transcribe 및 채팅 모델
basalt serve --pipeline-preset support-triage  # 티켓 질문용 System One 및 채팅 모델
```

채팅 모델은 Gemma 4 12B이며, 다른 모델과 함께 8 GB 여유를 두고 적재 가능하면 Qwen3.8-27B 3-bit를 사용합니다. `--pipeline-memory`는 모델 간 메모리 공유 방식을 지정합니다:

- `resident`: 모든 모델을 메모리에 상주시킵니다. 총합이 워킹 셋에 맞아야 합니다.
- `sequential`: 각 단계마다 인지 모델을 로드하고 종료 후 언로드합니다. 채팅 모델은 가장 큰 인지 모델과 함께 들어갈 수 있으면 상주하고, 그렇지 않으면 필요 시 언로드됩니다.
- `auto`(기본값): 합계가 맞으면 `resident`, 그렇지 않으면 `sequential`을 선택하며 단독 단계가 메모리를 초과할 때만 거부합니다.

M4 Pro 24 GB(워킹 셋 19.1 GB), 릴리스 빌드(커밋 `1921c3c`) 측정 결과:

| 프리셋 | 모델 | 계획 | 요청 1 | 요청 2 | 스와핑 시간 | 피크 메모리 |
|---|---|---|---|---|---|---|
| voice-agent | Gemma 4 12B 11.0 GB + Nemotron Transcribe 30B-A4B 14.9 GB | sequential (`resident`는 25.9 GB 필요), 음성 처리 중 채팅 언로드 | 21.4 s | 8.6 s | 6.0 s / 5.4 s | 15 GB |
| document-qa | Qwen3.8-27B 3-bit + PaddleOCR-VL 1.5 | `--pipeline-memory sequential`, 채팅 상주 | 23.2 s | 11.2 s | 0.4 s / 0.2 s | 14 GB |
