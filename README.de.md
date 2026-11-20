<p align="center"><img src="images/logo.svg" alt="basalt logo" width="128"></p>

# basalt — dein Mac ist jetzt das KI-Labor.

> Offene LLMs, neuronale Sprachsynthese, Video-, OCR- und Bildmodelle auf Apple Silicon bereitstellen, trainieren, quantisieren, komponieren und analysieren. Ein einziges Swift-6.4-Binary spricht die **OpenAI API, die Anthropic API und MCP**, mit einer nativen macOS-27-Liquid-Glass-App obendrauf. Kein Python zur Laufzeit. Keine Cloud, außer du forderst sie an.

[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · **Deutsch** · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.11.0](https://img.shields.io/badge/release-v0.11.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.11.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2855%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/i18n/de/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/i18n/de/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/i18n/de/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/i18n/de/benchmarks.md#benchmarks)

<p align="center"><img src="images/models-hive.svg" alt="Offene Modelle, die basalt ausführt: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Führt Open-Weight-Modelle auf Apple Silicon aus. Logit-verifiziert: Gemma 4, Qwen3.8 und LFM2.5 (siehe Tabelle unten); vermessen und paketiert auf <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a>: Ornith, Bonsai, Qwen3.6-35B-A3B, DeepSeek-OCR-2, OpenJev und Laya (siehe <a href="docs/i18n/de/models.md#verified-models">die Modelltabelle</a>); gpt-oss und Llama sind MLX-kompatibel. Logos sind Marken ihrer jeweiligen Eigentümer und dienen ausschließlich der Anzeige von Kompatibilität. Die Modelle von Anthropic sind Closed-Weight und daher nicht im Hive enthalten; die Anthropic-<i>API</i> wird unterstützt, damit Claude Code mit einem lokalen Modell sprechen kann.</sub></p>

**Fünf Dinge in einem Repo:** ein lokaler Inferenzserver, den deine Tools bereits ansprechen können, LoRA-Distillation und Post-Quantization-Tuning, ein docker-compose für Multi-Modell-Systeme, ein Live-Röntgenmikroskop für das Innenleben des Modells und eine native Mac-App, die alles auf einen Tastendruck bereitstellt.

**Bei jeder Änderung am Inferenzpfad gegen Referenz-Logits geprüft:** [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) spielt aufgezeichnete Prompts erneut ab und schlägt fehl, sobald sich ein Greedy-Token oder ein Logit verschiebt. Familienspezifische Referenz-Logits für qwen3, qwen3_5, gemma4, lfm2 und nemotron_h — sowohl Dense als auch MoE — liegen in [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits).

| Golden-getestetes Modell | Klasse | Fixture |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | Sliding-Window- + Full-Attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | Hybride Linear-/Full-Attention, Dense | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | Hybride Conv- + Attention-Architektur, MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### Ein-Klick-Start über [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) (55+ verifizierte Modelle)

Jede Model Card unter [`goodmai`](https://huggingface.co/goodmai) enthält eine Schaltfläche **Run with Basalt** ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en)), die `Basalt.app` via `basalt://run?model=...` öffnet oder den exakten `basalt serve`- bzw. `basalt compose`-CLI-Befehl ausgibt:

| Modell / Stack | Rolle & Speicherbedarf auf 24-GB-Mac | Start |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT Alltagstauglicher Chat & Vision (~7.6 GB, Ladder 4/5 + Lunar 6/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE (`A4B`), golden-verifiziert (14.9 GB, 56–63 tok/s, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 31B Dense PV-MACaQ 3.2 bpw für 24-GB-Mac (15.1 GB, 14.1 tok/s, Ladder 4/5 + Lunar 4/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B Dense PV-MACaQ 4-bit (15.3 GB, 11.6 tok/s p50, PPL 1.23, MSE −19.5%, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE (`A3B`) 3-bit MACaQ (14.6 GB, 64–75 tok/s mit `#628` Switch-WHT) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw Trellis (10.0 GB, 16k PPL 4.621) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO natives i-Quant (11.8 GB, PPL 4.17, 300/300 System One) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 35B MoE in 37.7 GB direkt von SSD gestreamt mit 8.77 tok/s, 0 Swap | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B Hadamard-ternär in 7.4 GB, 27.1 tok/s mit `--mtp` | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B Hybrid-Conv- + Attention-MoE, golden-verifiziert (~12.6 GB) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | 44.1 kHz neuronales TTS in 43 Sprachen (2.24 GB, RTF 0.23–0.26) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | Vision-Encoder-OCR mit offiziellem 768-Tiling (`/v1/pipeline`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | 9B lokaler System-One-Entscheider (98.9% Genauigkeit, −77% LLM-Tokens) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/i18n/de/compose.md#bundled-hardware-tiered-profiles-526) | Empfohlener 24-GB-Compose-Stack (Chat + ZONOS3 + Kokoro ANE) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

---

## Inferenz: 40B-MoE mit 67 tok/s, 27B in 7.4 GB oder 37.7-GB-MoE ohne Swap — hinter den APIs, die du bereits nutzt

- **Claude Code auf deiner eigenen GPU in einer Zeile.** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — dein echter API-Key wird nicht angerührt. OpenAI-SDKs, OpenCode, Continue, Cline und aider verbinden sich über die OpenAI API, Claude Desktop und Cursor über MCP (mit `--mcp-enable` / `--mcp-disable` pro Tool und Gruppe, [#488](https://github.com/goodmai/basalt/pull/488)).
- **Schnell auf einem Laptop.** Auf einem M4 Pro mit 24 GB: Rekursives `Flint-40B-A2.5B` (`oQ3e`) dekodiert mit 67.02 tok/s in 17.61 GB ([#522](https://github.com/goodmai/basalt/pull/522)); `Ornith-1.5-35B-A3B-MACaQ-3bit` dekodiert mit 64–75 tok/s in 14.6 GB dank unseres Single-Pass-`MACaQSwitchLinear`-WHT- + `gatherQuantizedMM`-Kernels (`3.41×` schnellerer MoE-Kernel, `+82%` End-to-End auf Gemma 4 26B-A4B, [#628](https://github.com/goodmai/basalt/pull/628)); `Qwen3.5-9B-PV-MACaQ-4bit` erreicht `43.2 tok/s` p50 (`+52%` Geschwindigkeit, `−33%` RAM ggü. 8-bit) bei exakter `4/5`-Ladder-Parität; `Gemma-4-31B-it-PV-MACaQ-3.2bpw` passt als 31B Dense in `15.1 GB` (`14.1 tok/s`, Ladder `4/5` + Lunar `4/6`) auf einen 24-GB-Mac, auf dem 4-bit OOM läuft; Qwen3-4B dekodiert mit 78–89 tok/s; Ternary Bonsai 2 27B passt in 7.4 GB und läuft mit 20.9–21.5 tok/s, 27.1 mit seinem MTP-Head und 31–38 mit Prompt-Lookup bei Code ([#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)); `.gguf` GSQ-RCO- / `i-quant`-Checkpoints laufen direkt auf eigenen Metal-Kernels ([#433](https://github.com/goodmai/basalt/pull/433), [#515](https://github.com/goodmai/basalt/pull/515)).
- **Modelle größer als RAM, null Swap.** `Qwen3.6-35B-A3B-8bit` (37.7 GB) dekodiert auf einem 24-GB-Mac mit 8.77 tok/s bei 0 Swap über gebatchtes POSIX-AIO und `LIO_NOWAIT`-SSD-Expert-Prefetch für den nächsten Layer (`--expert-streaming`, [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478), [#516](https://github.com/goodmai/basalt/pull/516)), ergänzt um UMA-bewusstes L0/L1/L2-Tiered-KV-Caching (`TieredKVHierarchyCore`) und adaptive heterogene Planung ([#627](https://github.com/goodmai/basalt/pull/627)).
- **Spekulatives Decoding, das exakt bleibt.** Prompt-Lookup liefert 1.84–2.66× bei Code-Edits (Bonsai 2 27B, [#239](https://github.com/goodmai/basalt/pull/239)) und 3.45–3.62× (Gemma 4 12B, [#265](https://github.com/goodmai/basalt/pull/265)) bei unverändertem Greedy-Output; `SpecTr` OT + `Residual Race`, `EAGLE-3`, `SmallM` (`M=2..16`), `Fast-dLLM` + `DFlash` und `DiffuCD` erweitern verlustfreies Multi-Draft- und Block-Diffusions-Decoding ([#593](https://github.com/goodmai/basalt/pull/593), [#603](https://github.com/goodmai/basalt/pull/603)).
- **Der nächste Turn kostet nur seine neuen Tokens.** Ein Präfixbaum-Prompt-Cache, auch für Hybrid-Modelle: Bei Ornith-1.5-9B mit einem 12.6k-Token-Agent-Prompt dauert der zweite Turn etwa 1 s statt rund 36 s ([#82](https://github.com/goodmai/basalt/pull/82)).
- **Lange Kontexte mit weniger Speicher.** KV-Quantisierung mit exakten Zonen (`4|8|k8v4`, plus `KVarN`-2D-Balancing, unverzerrtem 1-bit-`QJL`, `ChanMix`/`ThinKV`, sprachübergreifendem `HeadKV`-RBO-Transfer, Centroid-`RQ-KV`, `turbo3|turbo4|turbo8v4` und `polar4`-RoPE-Paar-Keys, [#472](https://github.com/goodmai/basalt/pull/472), [#590](https://github.com/goodmai/basalt/pull/590), [#602](https://github.com/goodmai/basalt/pull/602)): Ein 16k-Prompt erreicht bei Qwen3-4B einen Peak von 1.03 GB statt 3.11 GB ([#159](https://github.com/goodmai/basalt/pull/159)).
- **Video, neuronales TTS, Vollduplex-Sprache, Musik & 2K-MLX-Bildsynthese.** Ein phasenweises 24-GB-UMA-Video-Studio führt `LTX-2.5 AV-DiT`, `Kandinsky 6.0 CrossDiT`, `LingBot-Video MoE` und `Wan 2.2` mit kausalem 3D-VAE, Brick-Tiled-STA-Metal-Kernels und `AVAssetWriter`-`.mp4`-Muxing aus ([#608](https://github.com/goodmai/basalt/pull/608)–[#611](https://github.com/goodmai/basalt/pull/611)); `LFM2.5-Audio`-Mimi/SNAC-Duplex-Codecs, `SmartTurn` + `Sortformer`, `MossFormer2`, `Soprano-80M`, `Qwen3ForcedAligner`, `Zyphra/ZONOS3-Mini-hf` (`44.1 kHz`, 43 Sprachen, 2.24 GB, RTF 0.228–0.263, [#527](https://github.com/goodmai/basalt/pull/527)) und `Kokoro-82M` betreiben `/v1/audio/*` ([#604](https://github.com/goodmai/basalt/pull/604)); 4-stufiges `FLUX.3`-Compose, `SeedVR2`-Upscaling, `ACE-Step 1.5`-Musik, `Z-Image-Turbo`, `Iris-3B` und `SANA-1.5` treiben `/v1/images/*` und `/v1/music/*` an ([#542](https://github.com/goodmai/basalt/pull/542), [#605](https://github.com/goodmai/basalt/pull/605)).
- **Ein kompletter lokaler Stack + Swift SDK.** Tool-Aufrufe, JSON-Schema-, Regex- und Grammatik-Ausgabe (XGrammar), Meta `Muse-Glimmer-30B` ([#534](https://github.com/goodmai/basalt/pull/534)), `Ripple-65B`, `Nemotron-3.5-Lightning` ([#522](https://github.com/goodmai/basalt/pull/522)), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)), `/v1/embeddings`, `/v1/rerank`, `BasaltFoundationModels` Swift SDK ([#606](https://github.com/goodmai/basalt/pull/606)), `swift-huggingface`-Xet-Downloads ([#587](https://github.com/goodmai/basalt/pull/587)), pro Request wählbare LoRA/PHM/OFT-Adapter sowie geordnete LAN-vor-Cloud-Fallback-Stufen ([#480](https://github.com/goodmai/basalt/pull/480)).
- **PV-MACaQ 3-Bit: 27B passt und fliegt auf 24 GB.** `Qwen3.8-27B-PV-MACaQ-3bit` besteht die Official Ladder 5/5 (inkl. `lunar` 6/6) in 11.5–12.4 GB mit 15.6–16.7 tok/s auf einem M4 Pro, gegenüber 4/5 Sprossen, 15.3 GB und 11.6 tok/s beim Standard-4-Bit. `--speculative-lookup` läuft jetzt auch auf Qwen3.5/3.8-Checkpoints mit Vision-Tower (43–51 % Draft-Akzeptanz, +17 % bei Code-Edits), und `--kv-bits kvarn|kvarn-qjl` (2-Bit-KV) sowie `--attention-backend moba` sind in `basalt serve` verdrahtet und bestehen einen 4k-Passkey ([#632](https://github.com/goodmai/basalt/issues/632), [#634](https://github.com/goodmai/basalt/issues/634)). Checkpoints: [`goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX), [`goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX).

Details: [REST-API](docs/i18n/de/rest-api.md) · [Integrationen](docs/i18n/de/integrations.md) · [Befehle und Flags](docs/i18n/de/commands.md) · [Modelle](docs/i18n/de/models.md) · [Benchmarks](docs/i18n/de/benchmarks.md)

## Training & Quantisierung: universeller Any-to-Any-Konverter, PV-MACaQ, verteiltes Training & QAT

- **Universeller Any-to-Any-Quantisierungskonverter (`basalt models convert`).** Konvertiere zwischen `MACaQ` (`affine` / `mxfp4` / `moe` 2.8 bpw Router-gewichtete Hesse-Matrix, [#537](https://github.com/goodmai/basalt/pull/537), [#599](https://github.com/goodmai/basalt/pull/599)), `FlatQuant` & `LIRQuant` ([#588](https://github.com/goodmai/basalt/pull/588), [#594](https://github.com/goodmai/basalt/pull/594)), `SINQ` / `A-SINQ` / `EAQuant` Dual-Axis-Normalisierung ([#561](https://github.com/goodmai/basalt/pull/561)), `QuIP# E8P` 2-bit-Gitter & `TurboBoA` Babai–LLL ([#589](https://github.com/goodmai/basalt/pull/589)), `QTIP` Bitshift-Trellis (`2.0–3.0 bpw`, [#494](https://github.com/goodmai/basalt/pull/494)), `SVD-LLM V2` / `OBR` ([#592](https://github.com/goodmai/basalt/pull/592)), `.gguf` i-Quants & `MACaQ-R`, plus domänenbewusstes `QAT STE`-Refinement und Post-Quantisierungs-Qualitäts-/Fairness-/Leakage-Audit (`QuantizationAuditCore`, [#623](https://github.com/goodmai/basalt/pull/623)) — identisch verdrahtet über CLI, REST `/v1/models/convert*` und macOS `ConvertModelSheet` ([#546](https://github.com/goodmai/basalt/issues/546)).
- **Post-Quantisierungs-PV-Tuning & PV-MACaQ.** `basalt models convert --to macaq:affine:4:64 --pv-tune` wechselt kontinuierliche Scale/Bias-Updates mit diskretem Hesse-Flip-Refinement über Walsh–Hadamard-Blöcke ab: Bei `Qwen3.5-9B` reduziert `8-bit → 4-bit PV-MACaQ` den RAM um 33% (`9.97 → 6.68 GB`) und beschleunigt das Decoding um `+52%` (`27.3 → 41.6 tok/s`, p50 `43.2`) bei exakter `4/5`-Ladder-Parität (`valKL = 0.0044`); bei 2–3-bit-Checkpoints senkt PV-Tuning den Perplexity-Verlust um 8.7–29% in unter 3 Minuten ([#486](https://github.com/goodmai/basalt/pull/486), [#545](https://github.com/goodmai/basalt/pull/545)).
- **Distillieren, Nachtrainieren & über Macs verteilen.** `basalt models distill` destilliert OpenAI/Anthropic-Teacher in `lora`-, `phm`-, `oft`- oder quantisierte `AdaLoRA (LR/40)`-Adapter ([#501](https://github.com/goodmai/basalt/pull/501), [#508](https://github.com/goodmai/basalt/pull/508), [#624](https://github.com/goodmai/basalt/pull/624)); zweistufiges `SFT → DPO`, Zero-Copy-`ADPO`/`cADPO`, `EMA-QLoRA`, `Self-Appraisal` und `RLVR`-Anti-Shortcut-Audit halten das Präferenz-Alignment scharf ([#601](https://github.com/goodmai/basalt/pull/601), [#624](https://github.com/goodmai/basalt/pull/624)); Multi-Mac-Cluster-Training unterstützt authentifizierten Transport, `DDP + ZeRO-2`, `DiLoCo` + `PowerSGD`, `Zero-Bubble ZB-H1` und `Expert-Parallel MoE` ([#595](https://github.com/goodmai/basalt/pull/595)–[#598](https://github.com/goodmai/basalt/pull/598)).
- **Trainieren im laufenden Betrieb, mit Quality Gate.** Zeitscheibenbasiertes MCP-`distill`, nur verifiziertes `train_experts`, `train-jepa` sowie bitgenauer Rollback, falls Held-out-KL oder Perplexity regressieren.

Details: [Trainingsbefehle](docs/i18n/de/commands.md) · [Selbsttraining, Distillation und Quality Gate](docs/i18n/de/research-tools.md#self-training-train_experts)

## Compose: deklaratives `basalt.yaml v2.2`, Business-Governance & 84 Hardware-Profile

- **docker-compose, für Modelle (`basalt.yaml v2.2`).** Lokale und Cloud-Modelle, RAM-Deduplizierungsplaner, Spezialistenprofile, `resident` / `on-demand` / `sequential`-Speichermodi, 5-Komponenten-Routing mit `BusinessOrchestrator`, semantisches `LANTERN`-Caching, `HITL`-Freigabe-Gate (`/v1/mcp/approvals`) und neuro-symbolischer Wissenskern nach `LAAS-CNRS` (`SymbolicKnowledgeGraph`) — alles in einer deklarativen Datei ([#607](https://github.com/goodmai/basalt/pull/607), [#625](https://github.com/goodmai/basalt/pull/625), [#626](https://github.com/goodmai/basalt/pull/626)).
- **84 integrierte Profile für jeden Mac (`16 GB` bis `256 GB`).** Sieben Unified-Memory-Stufen (`16gb`, `24gb`, `32-36gb`, `48gb`, `64gb`, `128gb`, `256gb`) × 12 Workload-Archetypen starten mit einem Befehl (`basalt compose up 24gb-voice-zonos`, [#528](https://github.com/goodmai/basalt/pull/528)).
- **Lehnt ab, was nicht passt, und ändert im laufenden Betrieb.** `basalt compose plan` bemisst Modelle vom Hub vor dem Download und verweigert Überläufe; `compose diff` / `compose up` laden Änderungen sauber im laufenden Betrieb neu; `compose bench` führt deklarative Benchmark-Suites aus und exportiert Leaderboard-Bundles ([#505](https://github.com/goodmai/basalt/pull/505)).
- **Lokale Judges, kalibriertes Routing & Cloud-Budgets.** `OpenJev-Flash-9B-4bit` erreicht 98.9% Triage-Genauigkeit bei 77% weniger nachgelagerten LLM-Tokens ([#465](https://github.com/goodmai/basalt/pull/465)), temperatur-skaliertes `CalibratedRouterGate` + `Gemini Decision 1` leiten schwierige Anfragen sauber innerhalb täglicher USD-Obergrenzen ([#418](https://github.com/goodmai/basalt/pull/418), [#622](https://github.com/goodmai/basalt/pull/622)), und wahrgenommener Dokument- oder Audio-Text verlässt deinen Mac standardmäßig nie.

Details: [Compose-Referenz und Messwerte](docs/i18n/de/compose.md) · [Pipeline-Presets und Speichermodi](docs/i18n/de/pipelines.md) · [Cloud-Entscheidungen mit Gemini](docs/i18n/de/cloud-decisions.md)

## Forschung: das Modell im laufenden Betrieb öffnen

- **Model X-Ray, Pro-Token-Attribution & epistemisches Belnap-Gating.** `model_describe` (mit `MACaQ`- / `FlatQuant`- / `LIRQuant`-Modulmanifesten, [#630](https://github.com/goodmai/basalt/pull/630)), `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run`, `trace_tokens` (mit automatischem Gemma-BOS-Handling, [#630](https://github.com/goodmai/basalt/pull/630)), `/v1/xray/tokens` Pro-Token-Logprobs & Halluzinationsrisiko ([#613](https://github.com/goodmai/basalt/pull/613)), `Score d'influence` Head-Attribution ([#622](https://github.com/goodmai/basalt/pull/622)), Marchenko–Pastur-Spektren, vierwertige Belnap-`SIXTEEN_3`-Evidenzzustände ([#500](https://github.com/goodmai/basalt/pull/500)), komplexe MVN-Phasor-Probes ([#509](https://github.com/goodmai/basalt/pull/509)), Fisher–Rao-Drift ([#503](https://github.com/goodmai/basalt/pull/503)) und OpenMetrics unter `/api/v1/metrics` ([#458](https://github.com/goodmai/basalt/pull/458)).
- **Reversible Chirurgie & mechanistische Schaltkreise.** Mit `serve --research-tools`: `intervene` (inklusive bedingter Ablation, GQA-Group-Patching, kognitivem Nullspace & Anti-Steering-Guard, [#600](https://github.com/goodmai/basalt/pull/600)), `MAO-SNMF`-Feature-Zerlegung, `weights_delta`, Live-`tensor_pack`/`tensor_unpack`, `activation_contrast`, `steer_search`/`steer_commit`, `experiment_compare` und `introspect`. Jedes Experiment ist ein rückgängig machbares Overlay mit TTL und Journal; Gewichte auf der Festplatte werden nie verändert.
- **Benchmarks, die den Code des Modells ausführen.** Eine fünfstufige Leiter, die in einer Erde-zum-Mond-Mission gipfelt, führt den eigenen Code des Modells gegen ungesehene Eingaben aus, und ARC-AGI läuft nach offiziellen pass@2-Regeln.

Details: [Forschungstools](docs/i18n/de/research-tools.md) · [Wissenschaftliche Grundlagen (95+ Paper)](docs/i18n/de/research-foundations.md) · [Benchmarks](docs/i18n/de/benchmarks.md) · [Serve-Flags](docs/i18n/de/commands.md)

## Basalt for Mac: macOS 27 Liquid Glass, nur einen Tastendruck oder Klick entfernt

- **Eine native SwiftUI-App auf einer überwachten Engine.** Gebaut für macOS 27 Liquid Glass ([#479](https://github.com/goodmai/basalt/pull/479)) auf Englisch und Russisch ([#456](https://github.com/goodmai/basalt/pull/456)): 257 ms warmes Fenster, 0 Idle-Wakeups, 7.4 ms p95 pro gestreamtem Token in einem 2,000-Nachrichten-Transkript und 40 ms `⌥Space`-Schnellanfrage-Overhead ([#463](https://github.com/goodmai/basalt/pull/463)).
- **Compose Video Studio, 2K Image Studio & Universal Converter UI.** Erzeuge und spiele A/V-`.mp4`-Videos auf 24-GB-Macs ([#611](https://github.com/goodmai/basalt/pull/611)), bearbeite 2K-RGBA-Bilder mit Vorher/Nachher-Schieberegler ([#519](https://github.com/goodmai/basalt/pull/519)), konvertiere und QAT-auditiere Modelle direkt in `ConvertModelSheet` ([#623](https://github.com/goodmai/basalt/pull/623)) und durchstöbere/veröffentliche auf Hugging Face mit 10-Hz-SSE-Telemetrie & Xet-Downloads ([#587](https://github.com/goodmai/basalt/pull/587)).
- **Compose Profiles (`⌥⌘C`), Vollduplex-Sprache & `basalt://`-1-Klick-Start.** Empfiehlt automatisch passende Top-Profile für die RAM-Stufe deines Macs (`16–256 GB`, [#528](https://github.com/goodmai/basalt/pull/528)), streamt Freisprech-Sprache über `SpeechAnalyzer` / `LFM2.5-Audio` / `ZONOS3-Mini` ([#604](https://github.com/goodmai/basalt/pull/604)) und unterstützt `basalt://run?model=...`-Links, `⌥Space`-Schnellanfrage, Kurzbefehle, Spotlight und notarisierte `.dmg`-Paketierung.


**Basalt.app: laufendes Compose auf der Engine und der Chat mit Inspektor** (macOS 27 Liquid Glass, dunkel & hell):

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-compose-dark.png"><img src="docs/assets/screenshots/app-compose-light.png" alt="Basalt.app — Compose-Profil aktiv auf der Engine" width="49%"></picture></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-chat-dark.png"><img src="docs/assets/screenshots/app-chat-light.png" alt="Basalt.app — Chat mit geöffnetem Inspektor" width="49%"></picture></p>

Details: [Basalt for Mac](docs/i18n/de/mac-app.md)

---

## Schnellstart

Voraussetzungen: macOS 27+, Xcode 27.0+ / Swift 6.4 mit der Metal-Toolchain (`xcodebuild -downloadComponent MetalToolchain`), Apple Silicon M1–M4, 2–38 GB Speicherplatz je nach Modell. Alle Schritte, Bauen aus dem Quelltext und der `./basalt-claude`-Launcher stehen in [docs/i18n/de/install.md](docs/i18n/de/install.md).

**Ohne Xcode, ohne Build — das vorkompilierte Apple-Silicon-Binary in zwei Zeilen starten** (macOS 27+, Apple Silicon; `mlx.metallib` liegt im Archiv; der erste Start lädt das 12-GB-Modell, für einen leichteren Einstieg `goodmai/gemma-4-12B-it-qat-4bit` nehmen):

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

Streame ein 37.7-GB-MoE-Modell auf einem 24-GB-Mac oder starte ein vollständiges Multi-Modell-Compose-System:

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 84 profiles across 16–256 GB Macs
```

Die Mac-App aus einem Checkout (registriert `basalt://`-1-Klick-Links beim ersten Öffnen automatisch):

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

Lokale 4B–12B-Modelle sind kein direkter Ersatz für ein Frontier-Modell — sie sind für private, offline laufende, kostenfreie Agentenarbeit und für hybride Lokal/Cloud-Systeme gedacht. `basalt fit` sagt dir, was deine Maschine tatsächlich ausführen kann. Die Modelltabelle führt auch auf, was *nicht funktioniert hat*.

**Vergib einen Stern**, wenn du dir mehr MLX-native Werkzeuge in Swift wünschst, und eröffne ein Issue mit deinem Mac und deinen tok/s — [PRs mit einer neuen Zeile für die Benchmark-Tabelle](docs/i18n/de/benchmarks.md#benchmarks) sind willkommen.

---

## Dokumentation

> Übersetzte Dokumentation liegt unter `docs/i18n/`: [Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/).

| Thema | Dokument |
|---|---|
| Installation, Voraussetzungen, Bauen aus dem Quelltext, `./basalt-claude` | [docs/i18n/de/install.md](docs/i18n/de/install.md) |
| CLI-Befehle, `serve`-Flags, Prompt-Cache, CacheBlend, PV-Tuning | [docs/i18n/de/commands.md](docs/i18n/de/commands.md) |
| REST-API: Sicherheit, Modellverwaltung, OpenAI, Anthropic, Bilder, SSE, OpenMetrics, Swagger | [docs/i18n/de/rest-api.md](docs/i18n/de/rest-api.md) |
| Claude Code, OpenCode, OpenAI-SDKs, MCP für Claude Desktop und Cursor | [docs/i18n/de/integrations.md](docs/i18n/de/integrations.md) |
| Verifizierte Modelle, `goodmai`-Hub-Stack und Download | [docs/i18n/de/models.md](docs/i18n/de/models.md) |
| Funktionen im Überblick, Benchmark-Leiter, ARC-AGI, Herkunftsnachweis | [docs/i18n/de/benchmarks.md](docs/i18n/de/benchmarks.md) |
| Pipeline-Presets und Speichermodi | [docs/i18n/de/pipelines.md](docs/i18n/de/pipelines.md) |
| Cloud-Entscheidungen: Gemini Decision 1, multimodale Medien, Kalibrierungs-Teacher | [docs/i18n/de/cloud-decisions.md](docs/i18n/de/cloud-decisions.md) |
| `basalt compose`-Referenz und Messwerte | [docs/i18n/de/compose.md](docs/i18n/de/compose.md) |
| Basalt for Mac & `basalt://`-Deep-Links | [docs/i18n/de/mac-app.md](docs/i18n/de/mac-app.md) |
| Model X-Ray, Telemetrie, Experimente, Selbsttraining, Distillation | [docs/i18n/de/research-tools.md](docs/i18n/de/research-tools.md) |
| Mathematische & wissenschaftliche Grundlagen (85+ Paper & Danksagungen) | [docs/i18n/de/research-foundations.md](docs/i18n/de/research-foundations.md) |
| Projektstruktur, Architekturhinweise, Skripte, Abhängigkeiten | [docs/i18n/de/architecture.md](docs/i18n/de/architecture.md) |
| Architekturmuster und Module (Russisch) | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| Cloud: OpenRouter-Einrichtung, Kostenoptimierung, Modellvergleich | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| Mit KI-Agenten an basalt arbeiten | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| Roadmap, Datenschutzrichtlinie | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| Forschungsnotizen zu Agent-CLIs, Checklisten, Spezifikationen (Russisch) | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| Mitwirken, Changelog | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## Lizenz

[MIT](LICENSE) © 2026 goodmai

Modellgewichte sind **nicht** von dieser Lizenz abgedeckt — jedes Modell auf HuggingFace unterliegt seinen eigenen Nutzungsbedingungen.
