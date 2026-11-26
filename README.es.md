<p align="center"><img src="images/logo.svg" alt="logo de basalt" width="128"></p>

# basalt — tu Mac es ahora el laboratorio de IA.

> Sirve, entrena, cuantiza, compone y disecciona LLMs abiertos, TTS neuronal, ASR, OCR y modelos de visión (más andamiajes de imagen/vídeo/TTS) en Apple Silicon. Un único binario de Swift 6.4 habla la **API de OpenAI, la API de Anthropic y MCP**, con una app nativa de macOS 27 Liquid Glass encima. Sin Python en tiempo de ejecución. Sin nube a menos que tú la pidas.

[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md) · **Español** · [Français](README.fr.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Bahasa Indonesia](README.id.md) · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

📖 **Monografía:** [Evolución de las arquitecturas de redes neuronales, 1958–2027](docs/monograph/evo.es.md), de Aleksei Boklag, en los seis idiomas de la ONU ([todos](docs/monograph/README.md)) · **libro de ejercicios:** lo que basalt ya ejecuta de ella, lista [#816](https://github.com/goodmai/basalt/issues/816)

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.13.0](https://img.shields.io/badge/release-v0.13.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.13.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2865%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![mlx-community LLM/VLM loadable](https://img.shields.io/badge/mlx--community%20LLM%2FVLM-82%25%20loadable-0A84FF.svg)](scripts/hf_model_coverage.swift)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/i18n/es/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/i18n/es/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/i18n/es/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/i18n/es/benchmarks.md#benchmarks)

[![Quality Metrics](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml)
[![Concurrency Hatches](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/concurrency-hatches.json)](scripts/concurrency-hatches.allow)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/swift-testing.json)](Tests/BasaltTests)
[![Files ≤600 LOC](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/file-size.json)](scripts/check_size_limits.swift)
[![Crash Traps](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/crash-traps.json)](scripts/check_quality_metrics.swift)
[![L10n EN/RU](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/l10n-parity.json)](Localization)
[![Supply Chain](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/supply-chain.json)](.github/workflows/quality-metrics.yml)
[![Golden Logits](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/golden-logits.json)](Tests/BasaltTests/Fixtures/ReferenceLogits)

<p align="center"><img src="images/models-hive.svg" alt="Modelos abiertos que ejecuta basalt: Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Ejecuta modelos de pesos abiertos en Apple Silicon. Verificados por logits aquí: Gemma 4, Qwen3.8 y LFM2.5 (la tabla siguiente); medidos y empaquetados en <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a>: Ornith, Bonsai, Qwen3.6-35B-A3B, DeepSeek-OCR-2, OpenJev y Laya (consulta <a href="docs/i18n/es/models.md#verified-models">la tabla de modelos</a>); gpt-oss y Llama son compatibles con MLX. Los logotipos son marcas comerciales de sus propietarios y se muestran únicamente para indicar compatibilidad. Los modelos de Anthropic son de pesos cerrados, por lo que no están en la colmena; la <i>API</i> de Anthropic es compatible para que Claude Code pueda comunicarse con un modelo local.</sub></p>

**Cinco cosas en un mismo repositorio:** un servidor de inferencia local con el que tus herramientas ya saben comunicarse, destilación LoRA y ajuste poscuantización, un docker-compose para sistemas multimodelo, un microscopio de rayos X en vivo para ver qué ocurre dentro del modelo y una app nativa para Mac que lo pone todo a una pulsación de tecla.

**Comprobado frente a logits de referencia en cada cambio** de la ruta de inferencia: [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) reproduce prompts grabados y falla cuando cambia un token greedy o un logit. Los logits de referencia por familia para qwen3, qwen3_5, gemma4, lfm2 y nemotron_h, densos y MoE, están en [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits).

| Modelo con pruebas golden | Clase | Fixture |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | atención por ventana deslizante + completa, MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | atención híbrida lineal/completa, denso | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | híbrido convolución + atención, MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### Lanzamiento con un clic desde [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) (más de 65 modelos verificados)

Cada ficha de modelo bajo [`goodmai`](https://huggingface.co/goodmai) incluye un botón **Run with Basalt** ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en)) que abre `Basalt.app` mediante `basalt://run?model=...` o imprime el comando exacto de la CLI `basalt serve` / `basalt compose`:

| Modelo / Stack | Rol y consumo en Mac de 24 GB | Lanzamiento |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT para chat diario y visión (~7.6 GB, Ladder 4/5 + Lunar 6/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE (`A4B`), verificado con golden (14.9 GB, 56–63 tok/s, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 31B denso PV-MACaQ 3.2 bpw para Mac de 24 GB (15.1 GB, 14.1 tok/s, Ladder 4/5 + Lunar 4/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B denso PV-MACaQ 4-bit (15.3 GB, 11.6 tok/s p50, PPL 1.23, MSE −19.5%, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE (`A3B`) 3-bit MACaQ (14.6 GB, 64–75 tok/s con switch WHT de `#628`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw trellis (10.0 GB, PPL 16k 4.621) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO i-quant nativo (11.8 GB, PPL 4.17, 300/300 System One) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 35B MoE en 37.7 GB transmitido por streaming desde SSD a 8.77 tok/s, 0 swap | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B ternario de Hadamard en 7.4 GB, 27.1 tok/s con `--mtp` | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B MoE híbrido conv + atención, verificado con golden (~12.6 GB) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | TTS neuronal a 44.1 kHz en 43 idiomas (2.24 GB, RTF 0.23–0.26) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | OCR con codificador de visión y mosaico oficial 768 (`/v1/pipeline`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | Decisor local System One de 9B (98.9% acc, −77% tokens LLM) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/i18n/es/compose.md#bundled-hardware-tiered-profiles-526) | Stack Compose recomendado para 24 GB (Chat + ZONOS3 TTS; Kokoro andamiaje [#821](https://github.com/goodmai/basalt/issues/821)) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

### Novedades de v0.13.0 ([registro de cambios](CHANGELOG.md))

- **Inkling 2 Mini.** El MoE en bucle funciona de principio a fin, con llamadas a herramientas y reutilización de la caché del prompt entre turnos del agente. `--mtp` con la cabeza DSpark del checkpoint sube la decodificación de 28.0 a 49.8 tok/s en código (M4 Pro, 24 GB), y un prompt de 37.5K tokens alcanza un pico de 14.5 GB sin swap ([#650](https://github.com/goodmai/basalt/issues/650)).
- **NVFP4 desde cualquier origen.** `models convert --to nvfp4` funciona con cualquier checkpoint. NVIDIA ModelOpt NVFP4 se importa sin pérdidas, solo, en `models download` y `serve`. Un MoE con poco espacio para KV transmite sus expertos en vez de sufrir un Metal OOM ([#677](https://github.com/goodmai/basalt/issues/677)).
- **Nuevas arquitecturas y KV.** Qwen3-Coder-Next con expertos transmitidos ([#642](https://github.com/goodmai/basalt/issues/642)), BailingHybrid / Ling-3.0 ([#672](https://github.com/goodmai/basalt/issues/672)) y una caché KV experimental de fila compartida K=V para Gemma 4 que necesita 1.40–1.53× menos memoria KV en 32K–128K ([#645](https://github.com/goodmai/basalt/issues/645)).
- **Adaptadores y contexto.** Los adaptadores PEFT LoRA se cargan desde una carpeta o el Hub y se fusionan en los checkpoints exportados ([#675](https://github.com/goodmai/basalt/issues/675)). `--rope-yarn` amplía el contexto de Qwen 3.5–3.8 ([#729](https://github.com/goodmai/basalt/issues/729)).
- **Traducción y autorreescritura.** Benchmarks `basalt:translation` (chrF++, COMET, MetricX-24) con perfiles TranslateGemma-4, y autorreescritura experimental sobre pesos con copia en escritura (`basalt selfedit`, [#676](https://github.com/goodmai/basalt/issues/676)).
- **Robustez.** Una búsqueda de defectos entre dos agentes corrigió 64 hallazgos en el perímetro REST/MCP, los presupuestos de nube, la puntuación, la conversión, las cachés y la app para Mac ([#809](https://github.com/goodmai/basalt/pull/809), [#810](https://github.com/goodmai/basalt/pull/810), [#811](https://github.com/goodmai/basalt/pull/811)).

---

## Inferencia: MoE de 40B a 67 tok/s, 27B en 7.4 GB o MoE de 37.7 GB sin swap — detrás de las APIs que ya usas

- **Claude Code en tu propia GPU en una línea.** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — tu clave de API real no se toca nunca. Los SDKs de OpenAI, OpenCode, Continue, Cline y aider se conectan mediante la API de OpenAI; Claude Desktop y Cursor mediante MCP (con `--mcp-enable` / `--mcp-disable` por herramienta y por grupo, [#488](https://github.com/goodmai/basalt/pull/488)).
- **Rápido en un portátil.** En un M4 Pro con 24 GB: Recursive `Flint-40B-A2.5B` (`oQ3e`) decodifica a 67.02 tok/s en 17.61 GB ([#522](https://github.com/goodmai/basalt/pull/522)); `Ornith-1.5-35B-A3B-MACaQ-3bit` decodifica a 64–75 tok/s en 14.6 GB con nuestro kernel `MACaQSwitchLinear` WHT + `gatherQuantizedMM` de una sola pasada (kernel MoE `3.41×` más rápido, `+82%` de extremo a extremo en Gemma 4 26B-A4B, [#628](https://github.com/goodmai/basalt/pull/628)); `Qwen3.5-9B-PV-MACaQ-4bit` alcanza `43.2 tok/s` p50 (`+52%` de velocidad, `−33%` de RAM frente a 8-bit) con paridad exacta `4/5` en Ladder; `Gemma-4-31B-it-PV-MACaQ-3.2bpw` cabe como 31B denso en `15.1 GB` (`14.1 tok/s`, Ladder `4/5` + Lunar `4/6`) en un Mac de 24 GB donde 4-bit da OOM; Qwen3-4B decodifica a 78–89 tok/s; Ternary Bonsai 2 27B cabe en 7.4 GB y corre a 20.9–21.5 tok/s, 27.1 con su cabezal MTP y 31–38 con prompt lookup en código ([#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)); los checkpoints `.gguf` GSQ-RCO / `i-quant` se ejecutan directamente en kernels Metal propios ([#433](https://github.com/goodmai/basalt/pull/433), [#515](https://github.com/goodmai/basalt/pull/515)).
- **Modelos más grandes que la RAM, cero swap.** `Qwen3.6-35B-A3B-8bit` (37.7 GB) decodifica a 8.77 tok/s en un Mac de 24 GB con 0 swap mediante POSIX AIO por lotes y precarga de expertos de la siguiente capa desde SSD con `LIO_NOWAIT` (`--expert-streaming`, [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478), [#516](https://github.com/goodmai/basalt/pull/516)), junto con caché KV jerárquica L0/L1/L2 consciente de UMA (`TieredKVHierarchyCore`) y planificación heterogénea adaptativa ([#627](https://github.com/goodmai/basalt/pull/627)).
- **Decodificación especulativa y experimental.** Prompt lookup logra 1.84–2.66× en ediciones de código (Bonsai 2 27B, [#239](https://github.com/goodmai/basalt/pull/239)) y 3.45–3.62× (Gemma 4 12B, [#265](https://github.com/goodmai/basalt/pull/265)), sin alterar la salida greedy; `--mtp` funciona en Bonsai 2 e Inkling 2, `--experimental-smallm-kernel` acelera ventanas `M=4..8` ([#208](https://github.com/goodmai/basalt/issues/208)), `--experimental-dllm` ejecuta Fast-dLLM en `llada` ([#819](https://github.com/goodmai/basalt/issues/819)) y `--experimental-loop-exit` sale anticipadamente en modelos en bucle ([#820](https://github.com/goodmai/basalt/issues/820)). `SpecTr` OT + `Residual Race`, `EAGLE-3` y `DiffuCD` son kernels internos de investigación (sin conectar a `serve`), y `--experimental-dflash` se rechaza hasta conectarlo ([#574](https://github.com/goodmai/basalt/issues/574)).
- **El siguiente turno cuesta solo sus tokens nuevos.** Una caché de prompts en árbol de prefijos, incluso para modelos híbridos: en Ornith-1.5-9B con un prompt de agente de 12.6k tokens, el segundo turno tarda alrededor de 1 s en lugar de unos 36 s ([#82](https://github.com/goodmai/basalt/pull/82)).
- **Contextos largos en menos memoria.** Cuantización KV con zonas exactas (`4|8|k8v4`, además de balanceo 2D `KVarN`, `QJL` insesgado de 1 bit, `ChanMix`/`ThinKV`, transferencia RBO multilingüe `HeadKV`, `RQ-KV` por centroides, `turbo3|turbo4|turbo8v4` y claves de pares RoPE `polar4`, [#472](https://github.com/goodmai/basalt/pull/472), [#590](https://github.com/goodmai/basalt/pull/590), [#602](https://github.com/goodmai/basalt/pull/602)): un prompt de 16k alcanza un pico de 1.03 GB en vez de 3.11 GB en Qwen3-4B ([#159](https://github.com/goodmai/basalt/pull/159)).
- **TTS neuronal, ASR, OCR y visión — más andamiajes de imagen/vídeo/TTS.** Backends verificados con pesos reales hoy: `Zyphra/ZONOS3-Mini-hf` (`44.1 kHz`, 43 idiomas, 2.24 GB, RTF 0.228–0.263, [#527](https://github.com/goodmai/basalt/pull/527)), `Nemotron Transcribe` / Apple `SpeechAnalyzer` (`--speech-analyzer`), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)) y VLMs de visión (Gemma 4, Qwen3.5). Video Studio (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`), generadores de imagen/música (`FLUX.3`, `Z-Image-Turbo`, `SeedVR2`, `ACE-Step 1.5`, `Iris-3B`, `SANA-1.5`) y otros motores TTS/voz (`Kokoro-82M`, `Qwen3-TTS`, `CosyVoice 3`, `Chatterbox`, `Pocket TTS`, `LFM2.5-Audio`) se distribuyen hoy como **andamiajes (scaffolds)** con semilla fija y carga parcial — el trabajo con pesos reales se sigue en [#824](https://github.com/goodmai/basalt/issues/824), [#821](https://github.com/goodmai/basalt/issues/821), [#818](https://github.com/goodmai/basalt/issues/818) y [#826](https://github.com/goodmai/basalt/issues/826).
- **Todo un stack local + SDK en Swift.** Llamadas a herramientas, salida con esquema JSON, regex y gramática (XGrammar), Meta `Muse-Glimmer-30B` ([#534](https://github.com/goodmai/basalt/pull/534)), `Ripple-65B`, `Nemotron-3.5-Lightning` ([#522](https://github.com/goodmai/basalt/pull/522)), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)), `/v1/embeddings`, `/v1/rerank`, el SDK Swift `BasaltFoundationModels` ([#606](https://github.com/goodmai/basalt/pull/606)), descargas Xet con `swift-huggingface` ([#587](https://github.com/goodmai/basalt/pull/587)), adaptadores LoRA/PHM/OFT seleccionados por petición y niveles ordenados de respaldo LAN antes que nube ([#480](https://github.com/goodmai/basalt/pull/480)).
- **PV-MACaQ de 3 bits: 27B cabe y vuela en 24 GB.** `Qwen3.8-27B-PV-MACaQ-3bit` supera el Official Ladder 5/5 (incluido `lunar` 6/6) con 11.5–12.4 GB a 15.6–16.7 tok/s en un M4 Pro, frente a 4/5 peldaños, 15.3 GB y 11.6 tok/s del 4-bit estándar. `--speculative-lookup` ya funciona también en checkpoints Qwen3.5/3.8 con torre de visión (43–51 % de aceptación de borradores, +17 % en ediciones de código), y `--kv-bits kvarn|kvarn-qjl` (KV de 2 bits) y `--attention-backend moba` están conectados a `basalt serve` y superan un passkey de 4k ([#632](https://github.com/goodmai/basalt/issues/632), [#634](https://github.com/goodmai/basalt/issues/634)). Checkpoints: [`goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX), [`goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX).

Detalles: [REST API](docs/i18n/es/rest-api.md) · [integraciones](docs/i18n/es/integrations.md) · [comandos y flags](docs/i18n/es/commands.md) · [modelos](docs/i18n/es/models.md) · [benchmarks](docs/i18n/es/benchmarks.md)

## Entrenamiento y Cuantización: conversor universal any-to-any, PV-MACaQ y LoRA/PHM/OFT

- **Conversor universal de cuantización any-to-any (`basalt models convert`).** Convierte entre `affine`, `mixed`, `mxfp4`, `nvfp4`, `nf4`, `MACaQ` (`affine` / `mxfp4` / `moe` 2.8 bpw con Hessiano ponderado por router, [#537](https://github.com/goodmai/basalt/pull/537), [#599](https://github.com/goodmai/basalt/pull/599)), `FlatQuant` y `LIRQuant` ([#588](https://github.com/goodmai/basalt/pull/588), [#594](https://github.com/goodmai/basalt/pull/594)), normalización `SINQ` / `A-SINQ` / `EAQuant` ([#561](https://github.com/goodmai/basalt/pull/561)), `QTIP` (`2.0–3.0 bpw`, [#494](https://github.com/goodmai/basalt/pull/494)), `GSQ-RCO`, i-quants `.gguf` y `MACaQ-R` — conectado de forma idéntica en CLI, REST `/v1/models/convert*` y `ConvertModelSheet` ([#546](https://github.com/goodmai/basalt/issues/546)). `QuIP# E8P` (`E8PLinear`), `TurboBoA` Babai–LLL y `SVD-LLM V2` / `OptimalBrainRestoration` son módulos internos aún no expuestos en `--to`; `--qat-refine` (`QAT STE`) y `--audit` (`QuantizationAuditCore`) siguen planificados sin conectar en el conversor ([#616](https://github.com/goodmai/basalt/issues/616)).
- **PV-Tuning y PV-MACaQ poscuantización.** `basalt models convert --to macaq:affine:4:64 --pv-tune` alterna actualizaciones continuas de escala/sesgo con refinamiento discreto por inversión de bits guiado por Hessiano sobre bloques Walsh–Hadamard: en `Qwen3.5-9B`, `8-bit → 4-bit PV-MACaQ` reduce la RAM en un 33% (`9.97 → 6.68 GB`) y acelera la decodificación un `+52%` (`27.3 → 41.6 tok/s`, p50 `43.2`) con paridad exacta `4/5` en Ladder (`valKL = 0.0044`); en checkpoints de 2–3 bits, PV-Tuning reduce la pérdida de perplejidad entre un 8.7% y un 29% en menos de 3 minutos ([#486](https://github.com/goodmai/basalt/pull/486), [#545](https://github.com/goodmai/basalt/pull/545)).
- **Destilación, postentrenamiento y calibración en clúster.** `basalt models distill` y `basalt models train` (SFT) entrenan adaptadores `lora`, `phm` y `oft` con `--quantized-base-bits` (`LR/40`, [#501](https://github.com/goodmai/basalt/pull/501), [#508](https://github.com/goodmai/basalt/pull/508)), `ADPO` online corre dentro del motor, y `basalt cluster` + `--cluster-peers` emparejan Macs para calibración distribuida ([#565](https://github.com/goodmai/basalt/issues/565)). `--dpo-pairs` (`SFT → DPO`) y `--adalora-target-rank` son rechazados por `models train` (DPO/GRPO offline en [#822](https://github.com/goodmai/basalt/issues/822)); `DDP + ZeRO-2`, `DiLoCo` + `PowerSGD`, `ZB-H1` y `Expert-Parallel MoE` son núcleos internos aún no conectados al entrenamiento multi-Mac.
- **Entrena mientras sirves, con puerta de calidad.** `distill` por MCP dividido en rodajas de tiempo, `train_experts` solo con datos verificados, `train-jepa` y reversión bit a bit si empeoran la KL o la perplejidad en el conjunto de validación.

Detalles: [comandos de entrenamiento](docs/i18n/es/commands.md) · [autoentrenamiento, destilación y puerta de calidad](docs/i18n/es/research-tools.md#self-training-train_experts)

## Compose: `basalt.yaml v2.2` declarativo, gobernanza empresarial y 117 perfiles de hardware

- **docker-compose, para modelos (`basalt.yaml v2.2`).** Modelos locales y en la nube, planificador de deduplicación de RAM, perfiles de especialistas, modos de memoria `resident` / `on-demand` / `sequential`, enrutamiento de 5 componentes `BusinessOrchestrator`, caché semántica `LANTERN`, puerta de aprobación `HITL` (`/v1/mcp/approvals`) y núcleo de conocimiento neurosimbólico `LAAS-CNRS` (`SymbolicKnowledgeGraph`) — todo en un único archivo declarativo ([#607](https://github.com/goodmai/basalt/pull/607), [#625](https://github.com/goodmai/basalt/pull/625), [#626](https://github.com/goodmai/basalt/pull/626)).
- **117 perfiles integrados para cada Mac (`16 GB` a `256 GB`).** Nueve niveles de memoria unificada (`16gb`, `24gb`, `32-36gb`, `48gb`, `64gb`, `96gb`, `128gb`, `192gb`, `256gb`) × 13 arquetipos de carga de trabajo que se lanzan con un comando (`basalt compose up 24gb-voice-zonos`, [#528](https://github.com/goodmai/basalt/pull/528)).
- **Rechaza lo que no cabe, aplica cambios en caliente.** `basalt compose plan` calcula el tamaño de los modelos desde el Hub antes de descargarlos y rechaza desbordamientos; `compose diff` / `compose up` recargan cambios en caliente de forma limpia; `compose bench` ejecuta suites de benchmarks declarativas y exporta paquetes de clasificación ([#505](https://github.com/goodmai/basalt/pull/505)).
- **Jueces locales, enrutamiento calibrado y presupuestos en la nube.** `OpenJev-Flash-9B-4bit` alcanza un 98.9% de precisión en triaje mientras recorta en un 77% los tokens de LLM posteriores ([#465](https://github.com/goodmai/basalt/pull/465)), `CalibratedRouterGate` con escalado de temperatura + `Gemini Decision 1` enrutan consultas difíciles de forma limpia dentro de límites diarios en USD ([#418](https://github.com/goodmai/basalt/pull/418), [#622](https://github.com/goodmai/basalt/pull/622)), y el texto percibido de documentos o audio nunca sale de tu Mac por defecto.

Detalles: [referencia de compose y mediciones](docs/i18n/es/compose.md) · [preajustes de pipeline y modos de memoria](docs/i18n/es/pipelines.md) · [decisiones en la nube con Gemini](docs/i18n/es/cloud-decisions.md)

## Investigación: abre el modelo mientras se ejecuta

- **Rayos X del modelo, atribución por token y compuertas epistémicas de Belnap.** `model_describe` (que informa manifiestos de módulos `MACaQ` / `FlatQuant` / `LIRQuant`, [#630](https://github.com/goodmai/basalt/pull/630)), `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run`, `trace_tokens` (con manejo automático de BOS en Gemma, [#630](https://github.com/goodmai/basalt/pull/630)), logprobs por token y riesgo de alucinación en `/v1/xray/tokens` ([#613](https://github.com/goodmai/basalt/pull/613)), atribución de cabezales `Score d'influence` ([#622](https://github.com/goodmai/basalt/pull/622)), espectros de Marchenko–Pastur, estados de evidencia de cuatro valores Belnap `SIXTEEN_3` ([#500](https://github.com/goodmai/basalt/pull/500)), sondas fasoriales complejas MVN ([#509](https://github.com/goodmai/basalt/pull/509)), deriva de Fisher–Rao ([#503](https://github.com/goodmai/basalt/pull/503)) y OpenMetrics en `/api/v1/metrics` ([#458](https://github.com/goodmai/basalt/pull/458)).
- **Cirugía reversible y circuitos mecanísticos.** Con `serve --research-tools`: `intervene` (incluyendo ablación condicional, parcheo de grupos GQA, espacio nulo cognitivo y guardia anti-steering, [#600](https://github.com/goodmai/basalt/pull/600)), descomposición de características `MAO-SNMF`, `weights_delta`, `tensor_pack`/`tensor_unpack` en vivo, `activation_contrast`, `steer_search`/`steer_commit`, `experiment_compare` e `introspect`. Cada experimento es una superposición reversible con TTL y diario; los pesos en disco nunca se modifican.
- **Benchmarks que ejecutan el código del modelo.** Una escalera de cinco peldaños que culmina en una misión Tierra-Luna ejecuta el propio código del modelo frente a entradas nunca vistas, y ARC-AGI se ejecuta bajo las reglas oficiales de pass@2.

Detalles: [herramientas de investigación](docs/i18n/es/research-tools.md) · [fundamentos de investigación (más de 95 artículos)](docs/i18n/es/research-foundations.md) · [benchmarks](docs/i18n/es/benchmarks.md) · [flags de serve](docs/i18n/es/commands.md)

## Basalt para Mac: macOS 27 Liquid Glass, a una tecla o un clic de distancia

- **Una app nativa en SwiftUI sobre un motor supervisado.** Diseñada para macOS 27 Liquid Glass ([#479](https://github.com/goodmai/basalt/pull/479)) en inglés y ruso ([#456](https://github.com/goodmai/basalt/pull/456)): ventana en caliente en 257 ms, 0 despertares en reposo, 7.4 ms p95 por token transmitido en una conversación de 2,000 mensajes y 40 ms de sobrecarga en consulta rápida con `⌥Space` ([#463](https://github.com/goodmai/basalt/pull/463)).
- **Interfaz del Conversor Universal, Hugging Face Hub y paneles Studio experimentales.** Convierte modelos directamente en `ConvertModelSheet` (opciones QAT/auditoría planificadas [#616](https://github.com/goodmai/basalt/issues/616)), explora/publica en Hugging Face con telemetría SSE a 10 Hz y descargas Xet ([#587](https://github.com/goodmai/basalt/pull/587)) y previsualiza Video/Image Studio (respaldado hoy por andamiajes; pesos reales en [#824](https://github.com/goodmai/basalt/issues/824)).
- **Perfiles de Compose (`⌥⌘C`), voz local y lanzamiento en 1 clic `basalt://`.** Recomienda automáticamente los mejores perfiles para la RAM de tu Mac (`16–256 GB`, [#528](https://github.com/goodmai/basalt/pull/528)), ejecuta voz con pesos reales mediante Apple `SpeechAnalyzer` + `ZONOS3-Mini` (otros motores TTS/dúplex son andamiajes: [#821](https://github.com/goodmai/basalt/issues/821), [#826](https://github.com/goodmai/basalt/issues/826)) y gestiona enlaces `basalt://run?model=...`, `⌥Space`, Atajos, Spotlight y `.dmg` notarizado.


**Basalt.app: Compose activo en el motor y el chat con su inspector** (macOS 27 Liquid Glass, oscuro y claro):

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-compose-dark.png"><img src="docs/assets/screenshots/app-compose-light.png" alt="Basalt.app — perfil de Compose activo en el motor" width="49%"></picture></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-chat-dark.png"><img src="docs/assets/screenshots/app-chat-light.png" alt="Basalt.app — chat con el inspector abierto" width="49%"></picture></p>

Detalles: [Basalt para Mac](docs/i18n/es/mac-app.md)

---

## Inicio rápido

Requisitos: macOS 27+, Xcode 27.0+ / Swift 6.4 con la cadena de herramientas de Metal (`xcodebuild -downloadComponent MetalToolchain`), Apple Silicon M1–M4 y de 2 a 38 GB de disco según el modelo. Los pasos completos, la compilación desde el código fuente y el lanzador `./basalt-claude` están en [docs/i18n/es/install.md](docs/i18n/es/install.md).

**Sin Xcode ni compilar: ejecuta el binario precompilado para Apple Silicon en dos líneas** (macOS 27+, Apple Silicon; `mlx.metallib` viene dentro del archivo; la primera ejecución descarga el modelo de 12 GB, usa `goodmai/gemma-4-12B-it-qat-4bit` para empezar más ligero):

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

Transmite por streaming un modelo MoE de 37.7 GB en un Mac de 24 GB o ejecuta un sistema compose completo multimodelo:

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 117 profiles across 16–256 GB Macs
```

La app de Mac, desde una copia del repositorio (registra automáticamente los enlaces de 1 clic `basalt://` al abrirla por primera vez):

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

Los modelos locales de 4B–12B no son un reemplazo directo de un modelo frontera: están pensados para trabajo agéntico privado, sin conexión y a coste cero, y para sistemas híbridos local/nube. `basalt fit` te indica qué puede ejecutar realmente tu máquina. La tabla de modelos también enumera qué *falló*.

**Dale una estrella** si quieres más herramientas nativas de MLX en Swift, y abre un issue con tu Mac y tus tok/s — los [PRs que añaden una fila a la tabla de benchmarks](docs/i18n/es/benchmarks.md#benchmarks) son bienvenidos.

---

## Documentación

> Hay documentación traducida disponible en `docs/i18n/`: [Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/) · [한국어](docs/i18n/ko/) · [Bahasa Indonesia](docs/i18n/id/).

| Tema | Documento |
|---|---|
| Instalación, requisitos, compilación desde fuentes, `./basalt-claude` | [docs/i18n/es/install.md](docs/i18n/es/install.md) |
| Comandos de CLI, flags de `serve`, caché de prompts, CacheBlend, PV-Tuning | [docs/i18n/es/commands.md](docs/i18n/es/commands.md) |
| API REST: seguridad, gestión de modelos, OpenAI, Anthropic, imágenes, SSE, OpenMetrics, Swagger | [docs/i18n/es/rest-api.md](docs/i18n/es/rest-api.md) |
| Claude Code, OpenCode, SDKs de OpenAI, MCP para Claude Desktop y Cursor | [docs/i18n/es/integrations.md](docs/i18n/es/integrations.md) |
| Modelos verificados, stack en Hub de `goodmai` y descarga | [docs/i18n/es/models.md](docs/i18n/es/models.md) |
| Resumen de funciones, escalera de benchmarks, ARC-AGI, procedencia | [docs/i18n/es/benchmarks.md](docs/i18n/es/benchmarks.md) |
| Preajustes de pipeline y modos de memoria | [docs/i18n/es/pipelines.md](docs/i18n/es/pipelines.md) |
| Decisiones en la nube: Gemini Decision 1, medios multimodales, profesor de calibración | [docs/i18n/es/cloud-decisions.md](docs/i18n/es/cloud-decisions.md) |
| Referencia de `basalt compose` y mediciones | [docs/i18n/es/compose.md](docs/i18n/es/compose.md) |
| Basalt para Mac y enlaces profundos `basalt://` | [docs/i18n/es/mac-app.md](docs/i18n/es/mac-app.md) |
| Rayos X del modelo, telemetría, experimentos, autoentrenamiento, destilación | [docs/i18n/es/research-tools.md](docs/i18n/es/research-tools.md) |
| Fundamentos matemáticos y científicos (más de 85 artículos y agradecimientos) | [docs/i18n/es/research-foundations.md](docs/i18n/es/research-foundations.md) |
| Estructura del proyecto, notas de arquitectura, scripts, dependencias | [docs/i18n/es/architecture.md](docs/i18n/es/architecture.md) |
| Patrones de arquitectura y módulos (ruso) | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| Nube: configuración de OpenRouter, optimización de costes, comparación de modelos | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| Trabajo en basalt con agentes de IA | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| Hoja de ruta, política de privacidad | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| Notas de investigación sobre CLIs de agentes, listas de verificación, especificaciones (ruso) | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| Contribuir, registro de cambios | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## Licencia

[MIT](LICENSE) © 2026 goodmai

Los pesos de los modelos **no** están cubiertos por esta licencia — cada modelo en HuggingFace tiene sus propios términos.
