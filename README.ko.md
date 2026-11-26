<p align="center"><img src="images/logo.svg" alt="basalt logo" width="128"></p>

# basalt — 이제 당신의 Mac이 AI 연구실입니다.

> Apple Silicon에서 오픈 웨이트 LLM, 뉴럴 음성, 비디오, OCR 및 이미지 모델을 서빙, 학습, 양자화, 구성(compose)하고 내부까지 분석하세요. 단일 Swift 6.4 바이너리가 **OpenAI API, Anthropic API, MCP**를 모두 지원하며, 그 위에 네이티브 macOS 27 Liquid Glass 앱이 올라갑니다. 런타임 Python 불필요. 요청하지 않는 한 클라우드 통신 없음.

[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · **한국어** · [Bahasa Indonesia](README.id.md) · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

📖 **단행본:** [신경망 아키텍처의 진화, 1958–2027](docs/monograph/README.md) (저자 Aleksei Boklag, 유엔 6개 공용어) · **문제집:** basalt가 이미 실행하는 내용 체크리스트 [#816](https://github.com/goodmai/basalt/issues/816)

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.13.0](https://img.shields.io/badge/release-v0.13.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.13.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2865%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/i18n/ko/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/i18n/ko/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/i18n/ko/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/i18n/ko/benchmarks.md#benchmarks)

[![Quality Metrics](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml)
[![Concurrency Hatches](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/concurrency-hatches.json)](scripts/concurrency-hatches.allow)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/swift-testing.json)](Tests/BasaltTests)
[![Files ≤600 LOC](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/file-size.json)](scripts/check_size_limits.swift)
[![Crash Traps](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/crash-traps.json)](scripts/check_quality_metrics.swift)
[![L10n EN/RU](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/l10n-parity.json)](Localization)
[![Supply Chain](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/supply-chain.json)](.github/workflows/quality-metrics.yml)
[![Golden Logits](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/golden-logits.json)](Tests/BasaltTests/Fixtures/ReferenceLogits)

<p align="center"><img src="images/models-hive.svg" alt="Open models basalt runs: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Apple Silicon에서 오픈 웨이트 모델을 실행합니다. 여기서 로짓 검증 완료: Gemma 4, Qwen3.8, LFM2.5(아래 표 참조); <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a>에서 측정 및 패키징: Ornith, Bonsai, Qwen3.6-35B-A3B, DeepSeek-OCR-2, OpenJev, Laya(<a href="docs/i18n/ko/models.md#verified-models">모델 표</a> 참조); gpt-oss 및 Llama는 MLX 호환. 로고는 각 소유자의 상표이며 호환성 표시 목적으로만 사용됩니다. Anthropic 모델은 클로즈드 웨이트이므로 포함되지 않으며, Claude Code가 로컬 모델과 통신할 수 있도록 Anthropic <i>API</i>를 지원합니다.</sub></p>

**하나의 저장소에 담긴 다섯 가지 도구:** 기존 도구들이 바로 연결되는 로컬 추론 서버, LoRA 증류 및 양자화 후 튜닝, 멀티 모델 시스템을 위한 docker-compose, 모델 내부를 실시간으로 들여다보는 X-ray 현미경, 그리고 단축키 한 번으로 이 모든 것을 여는 네이티브 Mac 앱.

**추론 경로의 모든 변경마다 참조 로짓과 대조 검증:** [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift)가 기록된 프롬프트를 재생하며 greedy 토큰이나 로짓이 달라지면 실패합니다. qwen3, qwen3_5, gemma4, lfm2, nemotron_h(dense 및 MoE) 패밀리별 참조 로짓은 [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits)에 있습니다.

| 골든 검증 모델 | 아키텍처 계열 | 픽스처 |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | sliding-window + full attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | hybrid linear/full attention, dense | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | hybrid conv + attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) 원클릭 실행 (검증된 모델 65종 이상)

[`goodmai`](https://huggingface.co/goodmai)의 모든 모델 카드에는 **Run with Basalt** 버튼([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en))이 포함되어 있으며, `basalt://run?model=...` 링크로 `Basalt.app`을 열거나 정확한 `basalt serve` / `basalt compose` CLI 명령을 출력합니다:

| 모델 / 스택 | 역할 및 24 GB Mac 메모리 점유 | 실행 |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT 일상 채팅 & 비전 (~7.6 GB, Ladder 4/5 + Lunar 6/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE (`A4B`), 골든 검증 완료 (14.9 GB, 56–63 tok/s, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 24 GB Mac용 31B dense PV-MACaQ 3.2 bpw (15.1 GB, 14.1 tok/s, Ladder 4/5 + Lunar 4/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B dense PV-MACaQ 4-bit (15.3 GB, 11.6 tok/s p50, PPL 1.23, MSE −19.5%, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE (`A3B`) 3-bit MACaQ (14.6 GB, `#628` switch WHT로 64–75 tok/s) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw 트렐리스 (10.0 GB, 16k PPL 4.621) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO 네이티브 i-quant (11.8 GB, PPL 4.17, System One 300/300) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 37.7 GB 35B MoE를 SSD에서 8.77 tok/s, 0 swap으로 스트리밍 | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B Hadamard ternary를 7.4 GB에 적재, `--mtp`로 27.1 tok/s | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B 하이브리드 conv + attention MoE, 골든 검증 완료 (~12.6 GB) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | 43개 언어 지원 44.1 kHz 뉴럴 TTS (2.24 GB, RTF 0.23–0.26) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | 공식 768 타일링 비전 인코더 OCR (`/v1/pipeline`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | 9B 로컬 System One 판단기 (정확도 98.9%, LLM 토큰 −77%) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/i18n/ko/compose.md#bundled-hardware-tiered-profiles-526) | 추천 24 GB Compose 스택 (Chat + ZONOS3 + Kokoro ANE) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

### v0.13.0 새 기능 ([변경 기록](CHANGELOG.md))

- **Inkling 2 Mini.** 루프형 MoE가 도구 호출과 에이전트 턴 간 프롬프트 캐시 재사용까지 처음부터 끝까지 동작합니다. 체크포인트의 DSpark 헤드를 쓰는 `--mtp`로 코드 디코딩이 28.0에서 49.8 tok/s로 빨라지고(M4 Pro, 24 GB), 37.5K 토큰 프롬프트도 최대 14.5 GB, 스왑 0으로 처리합니다([#650](https://github.com/goodmai/basalt/issues/650)).
- **어떤 소스에서든 NVFP4.** `models convert --to nvfp4`는 모든 체크포인트에서 변환합니다. NVIDIA ModelOpt NVFP4는 `models download`와 `serve` 시점에 자동으로 무손실 가져오기됩니다. KV 공간이 부족한 MoE는 Metal OOM 대신 전문가를 스트리밍합니다([#677](https://github.com/goodmai/basalt/issues/677)).
- **새 아키텍처와 KV.** 전문가 스트리밍을 지원하는 Qwen3-Coder-Next([#642](https://github.com/goodmai/basalt/issues/642)), BailingHybrid / Ling-3.0([#672](https://github.com/goodmai/basalt/issues/672)), 32K–128K에서 KV 메모리를 1.40–1.53배 줄이는 실험적 Gemma 4 K=V 공유 행 KV 캐시([#645](https://github.com/goodmai/basalt/issues/645)).
- **어댑터와 컨텍스트.** PEFT LoRA 어댑터를 폴더나 Hub에서 불러와 내보내는 체크포인트에 융합할 수 있습니다([#675](https://github.com/goodmai/basalt/issues/675)). `--rope-yarn`은 Qwen 3.5–3.8의 컨텍스트를 늘립니다([#729](https://github.com/goodmai/basalt/issues/729)).
- **번역과 자기 재작성.** `basalt:translation` 벤치마크(chrF++, COMET, MetricX-24)와 TranslateGemma-4 프로필, 쓰기 시 복사 가중치 위에서 동작하는 실험적 자기 재작성(`basalt selfedit`, [#676](https://github.com/goodmai/basalt/issues/676)).
- **안정화.** 두 에이전트의 결함 탐색으로 REST/MCP 경계, 클라우드 예산, 채점, 변환, 캐시, Mac 앱 전반의 64건을 수정했습니다([#809](https://github.com/goodmai/basalt/pull/809), [#810](https://github.com/goodmai/basalt/pull/810), [#811](https://github.com/goodmai/basalt/pull/811)).

---

## 추론: 40B MoE 67 tok/s, 27B를 7.4 GB에, 37.7 GB MoE를 스왑 없이 — 익숙한 API 그대로

- **한 줄로 내 GPU에서 Claude Code 실행.** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — 실제 API 키는 건드리지 않습니다. OpenAI SDK, OpenCode, Continue, Cline, aider는 OpenAI API로, Claude Desktop과 Cursor는 MCP로 연결됩니다(도구/그룹별 `--mcp-enable` / `--mcp-disable`, [#488](https://github.com/goodmai/basalt/pull/488)).
- **노트북에서 빠른 속도.** M4 Pro 24 GB 기준: Recursive `Flint-40B-A2.5B`(`oQ3e`) 17.61 GB에서 67.02 tok/s ([#522](https://github.com/goodmai/basalt/pull/522)); `Ornith-1.5-35B-A3B-MACaQ-3bit`는 단일 패스 `MACaQSwitchLinear` WHT + `gatherQuantizedMM` 커널로 14.6 GB에서 64–75 tok/s(MoE 커널 `3.41×` 가속, Gemma 4 26B-A4B 종단간 `+82%`, [#628](https://github.com/goodmai/basalt/pull/628)); `Qwen3.5-9B-PV-MACaQ-4bit`는 `43.2 tok/s` p50(8-bit 대비 속도 `+52%`, RAM `−33%`)과 정확한 `4/5` Ladder 동등성; `Gemma-4-31B-it-PV-MACaQ-3.2bpw`는 4-bit가 OOM 나는 24 GB Mac에서 31B dense를 `15.1 GB`에 실행(`14.1 tok/s`, Ladder `4/5` + Lunar `4/6`); Qwen3-4B 78–89 tok/s; Ternary Bonsai 2 27B는 7.4 GB에서 20.9–21.5 tok/s, MTP 헤드로 27.1 tok/s, 코드 편집 시 prompt lookup으로 31–38 tok/s ([#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)); `.gguf` GSQ-RCO / `i-quant` 체크포인트는 커스텀 Metal 커널에서 직접 실행됩니다 ([#433](https://github.com/goodmai/basalt/pull/433), [#515](https://github.com/goodmai/basalt/pull/515)).
- **RAM보다 큰 모델도 스왑 0으로.** `Qwen3.6-35B-A3B-8bit`(37.7 GB)를 배치 POSIX AIO와 `LIO_NOWAIT` 다음 레이어 SSD 익스퍼트 프리페치(`--expert-streaming`, [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478), [#516](https://github.com/goodmai/basalt/pull/516))로 24 GB Mac에서 스왑 0, 8.77 tok/s로 디코딩하며, UMA 인식 L0/L1/L2 계층형 KV 캐싱(`TieredKVHierarchyCore`)과 적응형 이기종 플래닝을 지원합니다 ([#627](https://github.com/goodmai/basalt/pull/627)).
- **정확도를 해치지 않는 투기적 디코딩.** Prompt lookup은 greedy 출력을 바꾸지 않고 코드 편집에서 1.84–2.66×(Bonsai 2 27B, [#239](https://github.com/goodmai/basalt/pull/239)), 3.45–3.62×(Gemma 4 12B, [#265](https://github.com/goodmai/basalt/pull/265)) 가속을 제공합니다; `SpecTr` OT + `Residual Race`, `EAGLE-3`, `SmallM`(`M=2..16`), `Fast-dLLM` + `DFlash`, `DiffuCD`가 무손실 멀티 드래프트 및 블록 디퓨전 디코딩을 확장합니다 ([#593](https://github.com/goodmai/basalt/pull/593), [#603](https://github.com/goodmai/basalt/pull/603)).
- **다음 턴은 새 토큰 비용만 지불.** 하이브리드 모델까지 지원하는 프리픽스 트리 프롬프트 캐시: 12.6k 토큰 에이전트 프롬프트가 있는 Ornith-1.5-9B에서 두 번째 턴은 약 36초 대신 약 1초가 걸립니다 ([#82](https://github.com/goodmai/basalt/pull/82)).
- **더 적은 메모리로 긴 컨텍스트.** 정확한 구간 기반 KV 양자화(`4|8|k8v4`, `KVarN` 2D 밸런싱, 비편향 1-bit `QJL`, `ChanMix`/`ThinKV`, 크로스 링구얼 `HeadKV` RBO 전이, 센트로이드 `RQ-KV`, `turbo3|turbo4|turbo8v4`, `polar4` RoPE 페어 키, [#472](https://github.com/goodmai/basalt/pull/472), [#590](https://github.com/goodmai/basalt/pull/590), [#602](https://github.com/goodmai/basalt/pull/602)): Qwen3-4B에서 16k 프롬프트 피크 메모리가 3.11 GB에서 1.03 GB로 줄어듭니다 ([#159](https://github.com/goodmai/basalt/pull/159)).
- **비디오, 뉴럴 TTS, 전이중 음성, 음악 및 2K MLX 이미지 합성.** 24 GB UMA Video Studio는 `LTX-2.5 AV-DiT`, `Kandinsky 6.0 CrossDiT`, `LingBot-Video MoE`, `Wan 2.2`를 causal 3D-VAE, brick-tiled STA Metal 커널, `AVAssetWriter` `.mp4` 먹싱으로 실행합니다 ([#608](https://github.com/goodmai/basalt/pull/608)–[#611](https://github.com/goodmai/basalt/pull/611)); `LFM2.5-Audio`, `SmartTurn` + `Sortformer`, `MossFormer2`, `Soprano-80M`, `Qwen3ForcedAligner`, `Zyphra/ZONOS3-Mini-hf`(44.1 kHz, 43개 언어, 2.24 GB, RTF 0.228–0.263, [#527](https://github.com/goodmai/basalt/pull/527)), `Kokoro-82M`이 `/v1/audio/*`를 구동합니다 ([#604](https://github.com/goodmai/basalt/pull/604)); `FLUX.3`, `SeedVR2`, `ACE-Step 1.5`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`가 `/v1/images/*` 및 `/v1/music/*`를 구동합니다 ([#542](https://github.com/goodmai/basalt/pull/542), [#605](https://github.com/goodmai/basalt/pull/605)).
- **완전한 로컬 스택 + Swift SDK.** 도구 호출, JSON 스키마·정규식·문법 제약 출력(XGrammar), Meta `Muse-Glimmer-30B` ([#534](https://github.com/goodmai/basalt/pull/534)), `Ripple-65B`, `Nemotron-3.5-Lightning` ([#522](https://github.com/goodmai/basalt/pull/522)), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)), `/v1/embeddings`, `/v1/rerank`, `BasaltFoundationModels` Swift SDK ([#606](https://github.com/goodmai/basalt/pull/606)), `swift-huggingface` Xet 다운로드 ([#587](https://github.com/goodmai/basalt/pull/587)), 요청별 LoRA/PHM/OFT 어댑터 선택, LAN 우선 클라우드 폴백 계층을 지원합니다 ([#480](https://github.com/goodmai/basalt/pull/480)).
- **PV-MACaQ 3-bit로 27B가 24 GB에서 빠르게 동작.** `Qwen3.8-27B-PV-MACaQ-3bit`는 M4 Pro 11.5–12.4 GB에서 15.6–16.7 tok/s로 공식 Ladder 5/5(`lunar` 6/6 포함)를 통과합니다(기본 4-bit는 4/5, 15.3 GB, 11.6 tok/s). `--speculative-lookup`은 비전 타워가 포함된 Qwen3.5/3.8 체크포인트에서도 동작하며, `--kv-bits kvarn|kvarn-qjl`(2-bit KV)과 `--attention-backend moba`가 `basalt serve`에 연결되어 4k passkey를 통과합니다 ([#632](https://github.com/goodmai/basalt/issues/632), [#634](https://github.com/goodmai/basalt/issues/634)). 체크포인트: [`goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX), [`goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX).

자세히 보기: [REST API](docs/i18n/ko/rest-api.md) · [연동 가이드](docs/i18n/ko/integrations.md) · [명령어 및 플래그](docs/i18n/ko/commands.md) · [모델](docs/i18n/ko/models.md) · [벤치마크](docs/i18n/ko/benchmarks.md)

## 학습 및 양자화: 범용 any-to-any 변환기, PV-MACaQ, 분산 학습 및 QAT

- **범용 any-to-any 양자화 변환기 (`basalt models convert`).** `MACaQ`(`affine` / `mxfp4` / `moe` 2.8 bpw 라우터 가중 Hessian, [#537](https://github.com/goodmai/basalt/pull/537), [#599](https://github.com/goodmai/basalt/pull/599)), `FlatQuant` & `LIRQuant` ([#588](https://github.com/goodmai/basalt/pull/588), [#594](https://github.com/goodmai/basalt/pull/594)), `SINQ` / `A-SINQ` / `EAQuant` ([#561](https://github.com/goodmai/basalt/pull/561)), `QuIP# E8P` & `TurboBoA` Babai–LLL ([#589](https://github.com/goodmai/basalt/pull/589)), `QTIP` (`2.0–3.0 bpw`, [#494](https://github.com/goodmai/basalt/pull/494)), `SVD-LLM V2` / `OBR` ([#592](https://github.com/goodmai/basalt/pull/592)), `.gguf` i-quants & `MACaQ-R`, 도메인 인식 `QAT STE` 정제 및 양자화 후 품질·공정성·누출 감사(`QuantizationAuditCore`, [#623](https://github.com/goodmai/basalt/pull/623))를 CLI, REST `/v1/models/convert*`, macOS `ConvertModelSheet`에서 동일하게 지원합니다 ([#546](https://github.com/goodmai/basalt/issues/546)).
- **양자화 후 PV-Tuning 및 PV-MACaQ.** `basalt models convert --to macaq:affine:4:64 --pv-tune`은 Walsh–Hadamard 블록 위에서 연속 스케일/바이어스 업데이트와 이산 Hessian 플립 정제를 번갈아 수행합니다: `Qwen3.5-9B`에서 `8-bit → 4-bit PV-MACaQ`는 RAM을 33% 절감(`9.97 → 6.68 GB`)하고 디코딩을 `+52%` 가속(`27.3 → 41.6 tok/s`, p50 `43.2`)하면서도 `4/5` Ladder 동등성(`valKL = 0.0044`)을 유지합니다 ([#486](https://github.com/goodmai/basalt/pull/486), [#545](https://github.com/goodmai/basalt/pull/545)).
- **증류, 사후 학습 및 멀티 Mac 분산 학습.** `basalt models distill`은 OpenAI/Anthropic 교사 모델을 `lora`, `phm`, `oft` 또는 양자화 `AdaLoRA (LR/40)` 어댑터로 증류합니다 ([#501](https://github.com/goodmai/basalt/pull/501), [#508](https://github.com/goodmai/basalt/pull/508), [#624](https://github.com/goodmai/basalt/pull/624)); 2단계 `SFT → DPO`, 제로카피 `ADPO`/`cADPO`, `EMA-QLoRA`, `Self-Appraisal`, `RLVR` 감사가 선호 정렬 품질을 유지하며 ([#601](https://github.com/goodmai/basalt/pull/601)), 멀티 Mac 클러스터 학습은 인증 전송, `DDP + ZeRO-2`, `DiLoCo` + `PowerSGD`, `Zero-Bubble ZB-H1`, `Expert-Parallel MoE`를 지원합니다 ([#595](https://github.com/goodmai/basalt/pull/595)–[#598](https://github.com/goodmai/basalt/pull/598)).
- **품질 게이트와 함께 서빙 중 학습.** 타임슬라이스 MCP `distill`, 검증 전용 `train_experts`, `train-jepa`, 그리고 홀드아웃 KL 또는 perplexity 회귀 시 비트 단위 롤백을 제공합니다.

자세히 보기: [학습 명령어](docs/i18n/ko/commands.md) · [자가 학습, 증류 및 품질 게이트](docs/i18n/ko/research-tools.md#self-training-train_experts)

## Compose: 선언형 `basalt.yaml v2.2`, 거버넌스 및 117개 하드웨어 프로필

- **모델을 위한 docker-compose (`basalt.yaml v2.2`).** 로컬 및 클라우드 모델, RAM 중복 제거 플래너, 전문가 프로필, `resident` / `on-demand` / `sequential` 메모리 모드, `BusinessOrchestrator` 5요소 라우팅, `LANTERN` 시맨틱 캐싱, `HITL` 승인 게이트(`/v1/mcp/approvals`), `LAAS-CNRS` 뉴로-심볼릭 지식 코어(`SymbolicKnowledgeGraph`)를 하나의 선언형 파일로 정의합니다 ([#607](https://github.com/goodmai/basalt/pull/607), [#625](https://github.com/goodmai/basalt/pull/625), [#626](https://github.com/goodmai/basalt/pull/626)).
- **모든 Mac(`16 GB`~`256 GB`)을 위한 117개 내장 프로필.** 9개 통합 메모리 티어 × 13개 워크로드 아키타입을 한 줄로 실행합니다 (`basalt compose up 24gb-voice-zonos`, [#528](https://github.com/goodmai/basalt/pull/528)).
- **적재 불가능한 구성은 사전 거부, 변경은 라이브 반영.** `basalt compose plan`은 다운로드 전 Hub에서 크기를 계산해 오버플로를 차단하며, `compose diff` / `compose up`이 무중단 리로드를 수행하고, `compose bench`가 선언형 벤치마크 스위트를 실행합니다 ([#505](https://github.com/goodmai/basalt/pull/505)).
- **로컬 판별기, 보정된 라우팅 및 클라우드 예산 제한.** `OpenJev-Flash-9B-4bit`는 다운스트림 LLM 토큰을 77% 절감하면서 98.9% 분류 정확도를 달성하고 ([#465](https://github.com/goodmai/basalt/pull/465)), `CalibratedRouterGate` + `Gemini Decision 1`이 일일 USD 예산 내에서 어려운 질의만 라우팅합니다 ([#418](https://github.com/goodmai/basalt/pull/418), [#622](https://github.com/goodmai/basalt/pull/622)).

자세히 보기: [compose 레퍼런스 및 측정](docs/i18n/ko/compose.md) · [파이프라인 프리셋 및 메모리 모드](docs/i18n/ko/pipelines.md) · [Gemini 클라우드 의사결정](docs/i18n/ko/cloud-decisions.md)

## 연구 도구: 실행 중인 모델을 열어 내부까지 분석

- **Model X-ray, 토큰별 기여도 및 Belnap 인식론적 게이팅.** `model_describe` ([#630](https://github.com/goodmai/basalt/pull/630)), `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run`, `trace_tokens`, `/v1/xray/tokens` 토큰별 logprobs 및 환각 위험 ([#613](https://github.com/goodmai/basalt/pull/613)), `Score d'influence` 헤드 기여도 ([#622](https://github.com/goodmai/basalt/pull/622)), Marchenko–Pastur 스펙트럼, 4치 Belnap `SIXTEEN_3` 증거 상태 ([#500](https://github.com/goodmai/basalt/pull/500)), 복소 MVN 페이저 프로브 ([#509](https://github.com/goodmai/basalt/pull/509)), Fisher–Rao 드리프트 ([#503](https://github.com/goodmai/basalt/pull/503)), `/api/v1/metrics` OpenMetrics ([#458](https://github.com/goodmai/basalt/pull/458)).
- **가역적 모델 수술 및 메커니즘 회로 분석.** `serve --research-tools`로 `intervene` ([#600](https://github.com/goodmai/basalt/pull/600)), `MAO-SNMF`, `weights_delta`, 라이브 `tensor_pack`/`tensor_unpack`, `activation_contrast`, `steer_search`/`steer_commit`, `experiment_compare`, `introspect`를 실행합니다. 모든 실험은 TTL과 저널이 있는 되돌리기 가능한 오버레이이며 디스크의 가중치는 절대 수정되지 않습니다.
- **모델이 작성한 코드를 직접 실행하는 벤치마크.** 지구-달 임무까지 이어지는 5단계 Ladder가 보이지 않는 입력으로 모델 코드를 실행하며, ARC-AGI는 공식 pass@2 규칙으로 평가합니다.

자세히 보기: [연구 도구](docs/i18n/ko/research-tools.md) · [연구 기반 (논문 95편+)](docs/i18n/ko/research-foundations.md) · [벤치마크](docs/i18n/ko/benchmarks.md) · [serve 플래그](docs/i18n/ko/commands.md)

## Basalt for Mac: macOS 27 Liquid Glass, 단축키 또는 클릭 한 번으로

- **관리형 엔진 위의 네이티브 SwiftUI 앱.** macOS 27 Liquid Glass ([#479](https://github.com/goodmai/basalt/pull/479))용으로 설계되었으며 영어·러시아어를 지원합니다 ([#456](https://github.com/goodmai/basalt/pull/456)): 257 ms 웜 윈도우, 유휴 웨이크업 0, 2,000개 메시지 대화에서 스트리밍 토큰당 p95 7.4 ms, `⌥Space` 퀵 질문 오버헤드 40 ms ([#463](https://github.com/goodmai/basalt/pull/463)).
- **Compose Video Studio, 2K Image Studio 및 범용 변환기 UI.** 24 GB Mac에서 A/V `.mp4` 비디오 생성·재생 ([#611](https://github.com/goodmai/basalt/pull/611)), Before/After 슬라이더로 2K RGBA 이미지 편집 ([#519](https://github.com/goodmai/basalt/pull/519)), `ConvertModelSheet`에서 모델 변환 및 QAT 감사 ([#623](https://github.com/goodmai/basalt/pull/623)), 10 Hz SSE 텔레메트리 및 Xet 다운로드로 Hugging Face 탐색·배포를 지원합니다 ([#587](https://github.com/goodmai/basalt/pull/587)).
- **Compose 프로필 (`⌥⌘C`), 전이중 음성 및 `basalt://` 원클릭 실행.** Mac RAM 티어(`16–256 GB`, [#528](https://github.com/goodmai/basalt/pull/528))에 맞는 최적 프로필을 자동 추천하고, `SpeechAnalyzer` / `LFM2.5-Audio` / `ZONOS3-Mini` ([#604](https://github.com/goodmai/basalt/pull/604)) 핸즈프리 음성, `basalt://run?model=...` 딥링크, 단축어, Spotlight, 공증 `.dmg` 패키징을 지원합니다.

**Basalt.app: 엔진에서 실행 중인 Compose와 인스펙터가 열린 채팅 화면** (macOS 27 Liquid Glass, 다크 & 라이트):

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-compose-dark.png"><img src="docs/assets/screenshots/app-compose-light.png" alt="Basalt.app — Compose profile active on the engine" width="49%"></picture></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-chat-dark.png"><img src="docs/assets/screenshots/app-chat-light.png" alt="Basalt.app — chat with the inspector open" width="49%"></picture></p>

자세히 보기: [Basalt for Mac](docs/i18n/ko/mac-app.md)

---

## 빠른 시작

요구 사항: macOS 27+, Xcode 27.0+ / Swift 6.4 및 Metal 툴체인(`xcodebuild -downloadComponent MetalToolchain`), Apple Silicon M1–M4, 모델에 따라 디스크 공간 2–38 GB. 전체 단계와 소스 빌드, `./basalt-claude` 런처는 [docs/i18n/ko/install.md](docs/i18n/ko/install.md)를 참조하세요.

**Xcode·빌드 없이 두 줄로 사전 빌드 Apple Silicon 바이너리 실행** (macOS 27+, Apple Silicon; `mlx.metallib` 포함; 첫 실행 시 12 GB 모델 다운로드, 가볍게 시작하려면 `goodmai/gemma-4-12B-it-qat-4bit` 사용):

```bash
curl -fsSL https://github.com/goodmai/basalt/releases/latest/download/basalt-macos-arm64.tar.gz | tar -xz
./basalt-macos-arm64/basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --rest
```

```bash
brew tap goodmai/basalt https://github.com/goodmai/basalt
brew trust goodmai/basalt
brew install goodmai/basalt/basalt          # builds CLI & server from source

basalt onboard                              # picks a model that fits your Mac and downloads it
basalt serve --model goodmai/gemma-4-12B-it-qat-4bit --rest

# in another terminal: Claude Code, backed by your own GPU
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude
```

24 GB Mac에서 37.7 GB MoE 모델을 스트리밍하거나 전체 멀티 모델 compose 시스템을 실행하세요:

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 117 profiles across 16–256 GB Macs
```

소스 체크아웃에서 Mac 앱 빌드(최초 실행 시 `basalt://` 원클릭 링크 자동 등록):

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

로컬 4B–12B 모델은 프런티어 모델의 완전한 대체재가 아니라, 프라이빗·오프라인·비용 0 에이전트 작업과 하이브리드 로컬/클라우드 시스템을 위한 것입니다. `basalt fit`으로 현재 Mac에서 실제로 실행 가능한 모델을 확인할 수 있습니다.

Swift 기반 MLX 네이티브 도구를 응원하신다면 **Star**를 눌러주시고, 사용 중인 Mac과 tok/s 측정 결과를 이슈나 [벤치마크 표 PR](docs/i18n/ko/benchmarks.md#benchmarks)로 공유해 주세요.

---

## 문서

> 번역된 문서는 `docs/i18n/`에 있습니다: [Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/) · [한국어](docs/i18n/ko/) · [Bahasa Indonesia](docs/i18n/id/).

| 주제 | 문서 |
|---|---|
| 설치, 요구 사항, 소스 빌드, `./basalt-claude` | [docs/i18n/ko/install.md](docs/i18n/ko/install.md) |
| CLI 명령어, `serve` 플래그, 프롬프트 캐시, CacheBlend, PV-Tuning | [docs/i18n/ko/commands.md](docs/i18n/ko/commands.md) |
| REST API: 보안, 모델 관리, OpenAI, Anthropic, 이미지, SSE, OpenMetrics, Swagger | [docs/i18n/ko/rest-api.md](docs/i18n/ko/rest-api.md) |
| Claude Code, OpenCode, OpenAI SDK, Claude Desktop·Cursor용 MCP | [docs/i18n/ko/integrations.md](docs/i18n/ko/integrations.md) |
| 검증된 모델, `goodmai` Hub 스택 및 다운로드 | [docs/i18n/ko/models.md](docs/i18n/ko/models.md) |
| 기능 요약, 벤치마크 래더, ARC-AGI, 프로비넌스 | [docs/i18n/ko/benchmarks.md](docs/i18n/ko/benchmarks.md) |
| 파이프라인 프리셋 및 메모리 모드 | [docs/i18n/ko/pipelines.md](docs/i18n/ko/pipelines.md) |
| 클라우드 의사결정: Gemini Decision 1, 멀티모달 미디어, 보정 교사 | [docs/i18n/ko/cloud-decisions.md](docs/i18n/ko/cloud-decisions.md) |
| `basalt compose` 레퍼런스 및 측정 | [docs/i18n/ko/compose.md](docs/i18n/ko/compose.md) |
| Basalt for Mac 및 `basalt://` 딥링크 | [docs/i18n/ko/mac-app.md](docs/i18n/ko/mac-app.md) |
| Model X-ray, 텔레메트리, 실험, 자가 학습, 증류 | [docs/i18n/ko/research-tools.md](docs/i18n/ko/research-tools.md) |
| 수학적·과학적 연구 기반 (논문 85편+ 및 감사의 글) | [docs/i18n/ko/research-foundations.md](docs/i18n/ko/research-foundations.md) |
| 프로젝트 구조, 아키텍처 노트, 스크립트, 의존성 | [docs/i18n/ko/architecture.md](docs/i18n/ko/architecture.md) |
| 아키텍처 패턴 및 모듈 (러시아어) | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| 클라우드: OpenRouter 설정, 비용 최적화, 모델 비교 | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| AI 에이전트와 함께 basalt 작업하기 | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| 로드맵, 개인정보 처리방침 | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| 에이전트 CLI 연구 노트, 체크리스트, 스펙 (러시아어) | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| 기여 가이드, 변경 이력 | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## 라이선스

[MIT](LICENSE) © 2026 goodmai

모델 가중치는 이 라이선스에 포함되지 **않으며**, HuggingFace의 각 모델에는 자체 라이선스 조건이 적용됩니다.
