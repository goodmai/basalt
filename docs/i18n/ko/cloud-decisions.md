# Gemini Decision 1을 활용한 클라우드 의사결정

> Languages: [English](../../cloud-decisions.md) · [Русский](../ru/cloud-decisions.md) · [中文](../zh/cloud-decisions.md) · [Español](../es/cloud-decisions.md) · [Français](../fr/cloud-decisions.md) · [日本語](../ja/cloud-decisions.md) · [Deutsch](../de/cloud-decisions.md) · **한국어** · [Bahasa Indonesia](../id/cloud-decisions.md)

Google의 `gemini-decision-1`은 텍스트, 이미지, 오디오, 비디오, PDF에 대해 한 번의 패스로 System One 질문(`noul`, `choice`, `score`)에 답합니다. 와이어 포맷이 basalt의 `/v1/systemone`과 일치하므로 로컬 결정 모델 앞단의 클라우드 스테이지로 바로 사용할 수 있습니다:

```bash
export GEMINI_API_KEY=...        # 환경 변수에서만 읽으며 플래그나 파일에 넣지 않습니다
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- 불확실한 질문(신뢰도가 `--decision-escalate-below` 미만), 호출 실패, 레이트 리밋은 로컬 모델로 폴백됩니다.
- `/v1/systemone`은 `state` 옆에 `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]`를 받습니다.
- compose 파일에서는 `source: cloud:gemini/gemini-decision-1`과 `api_key_env: GEMINI_API_KEY`로 지정합니다.
  - Gemini 모델은 `decide` 캐스케이드 전용으로 동작합니다.
  - 모델에 `data: media`를 명시하지 않는 한 인지된 텍스트는 로컬 Mac을 벗어나지 않습니다.
- 비용은 API가 보고한 토큰 사용량 기준(입력 100만 토큰당 $0.10)으로 집계되어 compose 일일 예산에 반영됩니다.

## 보정 교사 (Calibration teacher)

Gemini Decision 1은 단순 레이블이 아닌 보정된 확률을 반환합니다. 동일한 상태에 대해 클라우드와 로컬 리드아웃을 함께 실행하고 로컬 temperature 또는 OpenJev 헤드를 이 확률에 맞춰 학습시키면, 로컬 결정기의 신뢰도가 클라우드와 같은 의미를 갖게 되어 에스컬레이션 임계값을 그대로 이식할 수 있습니다.
