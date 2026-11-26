<p align="center"><img src="images/logo.svg" alt="logo basalt" width="128"></p>

# basalt — votre Mac est désormais un laboratoire d'IA.

> Servez, entraînez, quantifiez, composez et disséquez des LLM ouverts, du TTS neuronal, de l'ASR, de l'OCR et des modèles de vision (plus des échafaudages image/vidéo/TTS) sur Apple Silicon. Un seul binaire Swift 6.4 parle **l'API OpenAI, l'API Anthropic et MCP**, surmonté d'une application native macOS 27 Liquid Glass. Pas de Python à l'exécution. Pas de cloud, sauf si vous le demandez.

[English](README.md) · [Русский](README.ru.md) · [中文](README.zh.md) · [Español](README.es.md) · **Français** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [한국어](README.ko.md) · [Bahasa Indonesia](README.id.md) · **[Website — bas-alt.vercel.app](https://bas-alt.vercel.app)**

📖 **Monographie :** [Évolution des architectures de réseaux de neurones, 1958–2027](docs/monograph/evo.fr.md), par Aleksei Boklag, dans les six langues de l'ONU ([toutes](docs/monograph/README.md)) · **recueil d'exercices :** ce que basalt en exécute déjà, liste [#816](https://github.com/goodmai/basalt/issues/816)

[![Website](https://img.shields.io/badge/website-bas--alt.vercel.app-FF5E36.svg)](https://bas-alt.vercel.app)
[![Release v0.13.0](https://img.shields.io/badge/release-v0.13.0-0A84FF.svg)](https://github.com/goodmai/basalt/releases/tag/v0.13.0)
[![CI Release](https://github.com/goodmai/basalt/actions/workflows/release.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/release.yml)
[![Security Audit](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/security-audit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-goodmai%20%2865%2B%29-black.svg)](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc)
[![mlx-community LLM/VLM loadable](https://img.shields.io/badge/mlx--community%20LLM%2FVLM-82%25%20loadable-0A84FF.svg)](scripts/hf_model_coverage.swift)
[![Run with Basalt](https://img.shields.io/badge/%E2%96%B6%20Run%20with%20Basalt-1--click-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?lang=en)

[![Platform](https://img.shields.io/badge/platform-macOS%2027%2B%20%C2%B7%20Apple%20Silicon-black.svg)](docs/i18n/fr/install.md#requirements)
[![Swift 6.4](https://img.shields.io/badge/Swift-6.4-black.svg)](Package.swift)
[![Homebrew](https://img.shields.io/badge/brew-goodmai%2Fbasalt%2Fbasalt-black.svg)](docs/i18n/fr/install.md#install-with-homebrew)
[![Ladder](https://img.shields.io/badge/ladder-6%20tasks-black.svg)](docs/i18n/fr/benchmarks.md#benchmarks)
[![ARC-AGI](https://img.shields.io/badge/ARC--AGI-pass%402-black.svg)](docs/i18n/fr/benchmarks.md#benchmarks)

[![Quality Metrics](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml/badge.svg)](https://github.com/goodmai/basalt/actions/workflows/quality-metrics.yml)
[![Concurrency Hatches](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/concurrency-hatches.json)](scripts/concurrency-hatches.allow)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/swift-testing.json)](Tests/BasaltTests)
[![Files ≤600 LOC](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/file-size.json)](scripts/check_size_limits.swift)
[![Crash Traps](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/crash-traps.json)](scripts/check_quality_metrics.swift)
[![L10n EN/RU](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/l10n-parity.json)](Localization)
[![Supply Chain](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/supply-chain.json)](.github/workflows/quality-metrics.yml)
[![Golden Logits](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/goodmai/basalt/badges/golden-logits.json)](Tests/BasaltTests/Fixtures/ReferenceLogits)

<p align="center"><img src="images/models-hive.svg" alt="Modèles ouverts exécutés par basalt : Qwen, Gemma, Ornith, gpt-oss, DeepSeek, Llama" width="240"></p>
<p align="center"><sub>Exécute des modèles à poids ouverts sur Apple Silicon. Vérifiés par logits ici : Gemma 4, Qwen3.8 et LFM2.5 (tableau ci-dessous) ; mesurés et empaquetés sur <a href="https://huggingface.co/goodmai">huggingface.co/goodmai</a> : Ornith, Bonsai, Qwen3.6-35B-A3B, DeepSeek-OCR-2, OpenJev et Laya (voir <a href="docs/i18n/fr/models.md#verified-models">le tableau des modèles</a>) ; gpt-oss et Llama sont compatibles MLX. Les logos sont des marques déposées de leurs propriétaires respectifs, affichés uniquement pour indiquer la compatibilité. Les modèles d'Anthropic étant à poids fermés, ils ne figurent pas dans la ruche ; l'<i>API</i> Anthropic est prise en charge afin que Claude Code puisse communiquer avec un modèle local.</sub></p>

**Cinq outils en un seul dépôt :** un serveur d'inférence local avec lequel vos outils savent déjà dialoguer, la distillation LoRA et le réglage post-quantification, un équivalent de docker-compose pour les systèmes multimodèles, un microscope à rayons X en direct pour observer ce qui se passe à l'intérieur du modèle, et une application Mac native qui met tout cela à portée d'un raccourci clavier.

**Vérifié par rapport aux logits de référence à chaque modification** du chemin d'inférence : [`GoldenLogitsTests`](Tests/BasaltTests/GoldenLogitsTests.swift) rejoue des prompts enregistrés et échoue dès qu'un token glouton (greedy) ou un logit dérive. Les logits de référence par famille pour qwen3, qwen3_5, gemma4, lfm2 et nemotron_h, denses et MoE, se trouvent dans [`Fixtures/ReferenceLogits`](Tests/BasaltTests/Fixtures/ReferenceLogits).

| Modèle vérifié sur golden | Classe | Fixture |
|---|---|---|
| [`mlx-community/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) | sliding-window + full attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/mlx-community--gemma-4-26B-A4B-it-qat-4bit.json) |
| [`lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | attention hybride linéaire/complète, dense | [golden](Tests/BasaltTests/Fixtures/golden/lukaskremla--Qwen3.8-27B-3bit-MLX-TextOnly.json) |
| [`LiquidAI/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) | hybride conv + attention, MoE | [golden](Tests/BasaltTests/Fixtures/golden/LiquidAI--LFM2.5-24B-A2B-MLX-4bit.json) |

### Lancement en un clic depuis [`huggingface.co/goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc) (65+ modèles vérifiés)

Chaque fiche de modèle sous [`goodmai`](https://huggingface.co/goodmai) comporte un bouton **Run with Basalt** ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en)) qui ouvre `Basalt.app` via `basalt://run?model=...` ou affiche la commande CLI exacte `basalt serve` / `basalt compose` :

| Modèle / Pile | Rôle et empreinte sur Mac 24 Go | Lancement |
|---|---|---|
| [`goodmai/gemma-4-12B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-12B-it-qat-4bit) | 12B QAT polyvalent chat et vision (~7.6 GB, Ladder 4/5 + Lunar 6/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-12B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-26B-A4B-it-qat-4bit`](https://huggingface.co/goodmai/gemma-4-26B-A4B-it-qat-4bit) | 26B MoE (`A4B`), vérifié sur golden (14.9 GB, 56–63 tok/s, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-26B-A4B-it-qat-4bit&role=chat&lang=en) |
| [`goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX`](https://huggingface.co/goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX) | 31B dense PV-MACaQ 3.2 bpw pour Mac 24 Go (15.1 GB, 14.1 tok/s, Ladder 4/5 + Lunar 4/6) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2Fgemma-4-31B-it-PV-MACaQ-3.2bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX) | 27B dense PV-MACaQ 4-bit (15.3 GB, 11.6 tok/s p50, PPL 1.23, MSE −19.5%, Ladder 4/5) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-PV-MACaQ-4bit-Affine-MLX&role=chat&lang=en) |
| [`goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX`](https://huggingface.co/goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX) | 35B MoE (`A3B`) 3-bit MACaQ (14.6 GB, 64–75 tok/s avec WHT switch `#628`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOrnith-1.5-35B-A3B-MACaQ-3bit-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-QTIP-2.5bpw-MLX) | 27B QTIP 2.5 bpw en treillis (10.0 GB, 16k PPL 4.621) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-QTIP-2.5bpw-MLX&role=chat&lang=en) |
| [`goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native`](https://huggingface.co/goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native) | 27B GSQ-RCO i-quant natif (11.8 GB, PPL 4.17, 300/300 System One) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native&role=chat&lang=en) |
| [`goodmai/Qwen3.6-35B-A3B-8bit`](https://huggingface.co/goodmai/Qwen3.6-35B-A3B-8bit) | 35B MoE de 37.7 GB diffusé depuis le SSD à 8.77 tok/s, 0 swap | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FQwen3.6-35B-A3B-8bit&role=moe-stream&lang=en) |
| [`goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed`](https://huggingface.co/goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed) | 27B ternaire Hadamard en 7.4 GB, 27.1 tok/s avec `--mtp` | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FTernary-Bonsai-2-27B-MTPLX-Optimized-Speed&role=mtp&lang=en) |
| [`goodmai/LFM2.5-24B-A2B-MLX-4bit`](https://huggingface.co/goodmai/LFM2.5-24B-A2B-MLX-4bit) | 24B MoE hybride conv + attention, vérifié sur golden (~12.6 GB) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FLFM2.5-24B-A2B-MLX-4bit&role=chat&lang=en) |
| [`Zyphra/ZONOS3-Mini-hf`](https://huggingface.co/Zyphra/ZONOS3-Mini-hf) | TTS neuronal 44.1 kHz dans 43 langues (2.24 GB, RTF 0.23–0.26) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=Zyphra%2FZONOS3-Mini-hf&role=tts&lang=en) |
| [`goodmai/DeepSeek-OCR-2-8bit`](https://huggingface.co/goodmai/DeepSeek-OCR-2-8bit) | OCR par encodeur de vision avec découpage officiel en tuiles 768 (`/v1/pipeline`) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FDeepSeek-OCR-2-8bit&role=ocr&lang=en) |
| [`goodmai/OpenJev-Flash-9B-MLX-4bit`](https://huggingface.co/goodmai/OpenJev-Flash-9B-MLX-4bit) | Décideur local System One 9B (98.9% de précision, −77% de tokens LLM) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?model=goodmai%2FOpenJev-Flash-9B-MLX-4bit&role=decide&lang=en) |
| [`24gb-voice-zonos`](docs/i18n/fr/compose.md#bundled-hardware-tiered-profiles-526) | Pile Compose 24 Go recommandée (Chat + ZONOS3 TTS ; Kokoro échafaudage [#821](https://github.com/goodmai/basalt/issues/821)) | [![Run with Basalt](https://img.shields.io/badge/%E2%96%B6_Run_with_Basalt-0A84FF.svg)](https://goodmai-run-with-basalt.static.hf.space/?profile=24gb-voice-zonos&lang=en) |

### Nouveautés de la v0.13.0 ([journal des modifications](CHANGELOG.md))

- **Inkling 2 Mini.** Le MoE en boucle fonctionne de bout en bout, avec appels d'outils et réutilisation du cache de prompt entre les tours d'agent. `--mtp` avec la tête DSpark du checkpoint fait passer le décodage de 28.0 à 49.8 tok/s sur du code (M4 Pro, 24 GB), et un prompt de 37.5K tokens culmine à 14.5 GB sans swap ([#650](https://github.com/goodmai/basalt/issues/650)).
- **NVFP4 depuis n'importe quelle source.** `models convert --to nvfp4` accepte tout checkpoint. NVIDIA ModelOpt NVFP4 s'importe sans perte, automatiquement, lors de `models download` et `serve`. Un MoE qui manque de place pour le KV diffuse ses experts au lieu de provoquer un Metal OOM ([#677](https://github.com/goodmai/basalt/issues/677)).
- **Nouvelles architectures et KV.** Qwen3-Coder-Next avec experts diffusés ([#642](https://github.com/goodmai/basalt/issues/642)), BailingHybrid / Ling-3.0 ([#672](https://github.com/goodmai/basalt/issues/672)), et un cache KV expérimental à ligne partagée K=V pour Gemma 4 qui demande 1.40–1.53× moins de mémoire KV entre 32K et 128K ([#645](https://github.com/goodmai/basalt/issues/645)).
- **Adaptateurs et contexte.** Les adaptateurs PEFT LoRA se chargent depuis un dossier ou le Hub et se fusionnent dans les checkpoints exportés ([#675](https://github.com/goodmai/basalt/issues/675)). `--rope-yarn` étend le contexte de Qwen 3.5–3.8 ([#729](https://github.com/goodmai/basalt/issues/729)).
- **Traduction et auto-réécriture.** Benchmarks `basalt:translation` (chrF++, COMET, MetricX-24) avec profils TranslateGemma-4, et auto-réécriture expérimentale sur des poids en copie sur écriture (`basalt selfedit`, [#676](https://github.com/goodmai/basalt/issues/676)).
- **Robustesse.** Une chasse aux défauts à deux agents a corrigé 64 problèmes : périmètre REST/MCP, budgets cloud, évaluation, conversion, caches et app Mac ([#809](https://github.com/goodmai/basalt/pull/809), [#810](https://github.com/goodmai/basalt/pull/810), [#811](https://github.com/goodmai/basalt/pull/811)).

---

## Inférence : MoE 40B à 67 tok/s, 27B dans 7.4 GB, ou MoE de 37.7 GB sans swap — derrière les API que vous utilisez déjà

- **Claude Code sur votre propre GPU en une ligne.** `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_AUTH_TOKEN=local claude` — votre vraie clé API n'est jamais touchée. Les SDK OpenAI, OpenCode, Continue, Cline et aider se connectent via l'API OpenAI ; Claude Desktop et Cursor via MCP (avec activation/désactivation par outil et par groupe via `--mcp-enable` / `--mcp-disable`, [#488](https://github.com/goodmai/basalt/pull/488)).
- **Rapide sur un ordinateur portable.** Sur un M4 Pro 24 Go : le modèle récursif `Flint-40B-A2.5B` (`oQ3e`) décode à 67.02 tok/s dans 17.61 GB ([#522](https://github.com/goodmai/basalt/pull/522)) ; `Ornith-1.5-35B-A3B-MACaQ-3bit` décode à 64–75 tok/s dans 14.6 GB grâce à notre noyau passe unique `MACaQSwitchLinear` WHT + `gatherQuantizedMM` (noyau MoE `3.41×` plus rapide, `+82%` de bout en bout sur Gemma 4 26B-A4B, [#628](https://github.com/goodmai/basalt/pull/628)) ; `Qwen3.5-9B-PV-MACaQ-4bit` atteint `43.2 tok/s` p50 (`+52%` de vitesse, `−33%` de RAM par rapport au 8-bit) avec parité exacte `4/5` sur le Ladder ; `Gemma-4-31B-it-PV-MACaQ-3.2bpw` fait tenir un 31B dense dans `15.1 GB` (`14.1 tok/s`, Ladder `4/5` + Lunar `4/6`) sur Mac 24 Go là où le 4-bit provoque un OOM ; Qwen3-4B décode à 78–89 tok/s ; Ternary Bonsai 2 27B tient dans 7.4 GB et tourne à 20.9–21.5 tok/s, 27.1 avec sa tête MTP et 31–38 avec la recherche dans le prompt sur du code ([#239](https://github.com/goodmai/basalt/pull/239), [#263](https://github.com/goodmai/basalt/pull/263)) ; les checkpoints `.gguf` GSQ-RCO / `i-quant` s'exécutent directement sur des noyaux Metal sur mesure ([#433](https://github.com/goodmai/basalt/pull/433), [#515](https://github.com/goodmai/basalt/pull/515)).
- **Des modèles plus grands que la RAM, zéro swap.** `Qwen3.6-35B-A3B-8bit` (37.7 GB) décode à 8.77 tok/s sur un Mac 24 Go avec 0 swap via POSIX AIO par lots et préchargement SSD `LIO_NOWAIT` des experts de la couche suivante (`--expert-streaming`, [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478), [#516](https://github.com/goodmai/basalt/pull/516)), aux côtés d'un cache KV hiérarchisé L0/L1/L2 conscient de l'UMA (`TieredKVHierarchyCore`) et d'une planification hétérogène adaptative ([#627](https://github.com/goodmai/basalt/pull/627)).
- **Décodage spéculatif et expérimental.** Le prompt lookup donne 1.84–2.66× sur les éditions de code (Bonsai 2 27B, [#239](https://github.com/goodmai/basalt/pull/239)) et 3.45–3.62× (Gemma 4 12B, [#265](https://github.com/goodmai/basalt/pull/265)), sans changer la sortie gloutonne ; `--mtp` tourne sur Bonsai 2 et Inkling 2, `--experimental-smallm-kernel` accélère les fenêtres `M=4..8` ([#208](https://github.com/goodmai/basalt/issues/208)), `--experimental-dllm` exécute Fast-dLLM sur `llada` ([#819](https://github.com/goodmai/basalt/issues/819)) et `--experimental-loop-exit` sort tôt sur les modèles bouclés ([#820](https://github.com/goodmai/basalt/issues/820)). `SpecTr` OT + `Residual Race`, `EAGLE-3` et `DiffuCD` sont des noyaux de recherche internes (non câblés sur `serve`), et `--experimental-dflash` est refusé tant qu'il n'est pas câblé ([#574](https://github.com/goodmai/basalt/issues/574)).
- **Le tour suivant ne coûte que ses nouveaux tokens.** Un cache de prompts en arbre de préfixes, même pour les modèles hybrides : sur Ornith-1.5-9B avec un prompt d'agent de 12.6k tokens, le deuxième tour prend environ 1 s au lieu d'environ 36 s ([#82](https://github.com/goodmai/basalt/pull/82)).
- **De longs contextes dans moins de mémoire.** Quantification KV avec zones exactes (`4|8|k8v4`, plus équilibrage 2D `KVarN`, `QJL` 1-bit sans biais, `ChanMix`/`ThinKV`, transfert RBO interlingue `HeadKV`, `RQ-KV` à centroïdes, `turbo3|turbo4|turbo8v4` et clés par paires RoPE `polar4`, [#472](https://github.com/goodmai/basalt/pull/472), [#590](https://github.com/goodmai/basalt/pull/590), [#602](https://github.com/goodmai/basalt/pull/602)) : un prompt de 16k culmine à 1.03 GB au lieu de 3.11 GB sur Qwen3-4B ([#159](https://github.com/goodmai/basalt/pull/159)).
- **TTS neuronal, ASR, OCR et vision — plus échafaudages image/vidéo/TTS.** Backends vérifiés sur vrais poids aujourd'hui : `Zyphra/ZONOS3-Mini-hf` (`44.1 kHz`, 43 langues, 2.24 GB, RTF 0.228–0.263, [#527](https://github.com/goodmai/basalt/pull/527)), `Nemotron Transcribe` / Apple `SpeechAnalyzer` (`--speech-analyzer`), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)) et VLM vision (Gemma 4, Qwen3.5). Video Studio (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`), les générateurs image/musique (`FLUX.3`, `Z-Image-Turbo`, `SeedVR2`, `ACE-Step 1.5`, `Iris-3B`, `SANA-1.5`) et les autres moteurs TTS/audio (`Kokoro-82M`, `Qwen3-TTS`, `CosyVoice 3`, `Chatterbox`, `Pocket TTS`, `LFM2.5-Audio`) sont actuellement des **échafaudages (scaffolds)** à graine fixe et chargement partiel — le travail sur vrais poids est suivi dans [#824](https://github.com/goodmai/basalt/issues/824), [#821](https://github.com/goodmai/basalt/issues/821), [#818](https://github.com/goodmai/basalt/issues/818) et [#826](https://github.com/goodmai/basalt/issues/826).
- **Une pile locale complète + SDK Swift.** Appels d'outils, sorties contraintes par schéma JSON, regex et grammaire (XGrammar), Meta `Muse-Glimmer-30B` ([#534](https://github.com/goodmai/basalt/pull/534)), `Ripple-65B`, `Nemotron-3.5-Lightning` ([#522](https://github.com/goodmai/basalt/pull/522)), `DeepSeek-OCR-2` ([#476](https://github.com/goodmai/basalt/pull/476)), `/v1/embeddings`, `/v1/rerank`, le SDK Swift `BasaltFoundationModels` ([#606](https://github.com/goodmai/basalt/pull/606)), les téléchargements Xet `swift-huggingface` ([#587](https://github.com/goodmai/basalt/pull/587)), les adaptateurs LoRA/PHM/OFT sélectionnés par requête, et des paliers de repli ordonnés LAN-avant-cloud ([#480](https://github.com/goodmai/basalt/pull/480)).
- **PV-MACaQ 3 bits : le 27B tient et vole sur 24 Go.** `Qwen3.8-27B-PV-MACaQ-3bit` réussit l'Official Ladder 5/5 (dont `lunar` 6/6) en 11.5–12.4 Go à 15.6–16.7 tok/s sur un M4 Pro, contre 4/5 barreaux, 15.3 Go et 11.6 tok/s pour le 4 bits standard. `--speculative-lookup` fonctionne désormais aussi sur les checkpoints Qwen3.5/3.8 avec tour de vision (43–51 % d'acceptation des brouillons, +17 % sur les éditions de code), et `--kv-bits kvarn|kvarn-qjl` (KV 2 bits) et `--attention-backend moba` sont branchés dans `basalt serve` et réussissent un passkey de 4k ([#632](https://github.com/goodmai/basalt/issues/632), [#634](https://github.com/goodmai/basalt/issues/634)). Checkpoints : [`goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX), [`goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX`](https://huggingface.co/goodmai/Qwen3.8-27B-Heretic-PV-MACaQ-3bit-Affine-MLX).

Détails : [API REST](docs/i18n/fr/rest-api.md) · [intégrations](docs/i18n/fr/integrations.md) · [commandes et options](docs/i18n/fr/commands.md) · [modèles](docs/i18n/fr/models.md) · [benchmarks](docs/i18n/fr/benchmarks.md)

## Entraînement et quantification : convertisseur universel tout-en-tout, PV-MACaQ et LoRA/PHM/OFT

- **Convertisseur universel de quantification tout-en-tout (`basalt models convert`).** Convertissez entre `affine`, `mixed`, `mxfp4`, `nvfp4`, `nf4`, `MACaQ` (`affine` / `mxfp4` / `moe` 2.8 bpw avec Hessien pondéré par le routeur, [#537](https://github.com/goodmai/basalt/pull/537), [#599](https://github.com/goodmai/basalt/pull/599)), `FlatQuant` & `LIRQuant` ([#588](https://github.com/goodmai/basalt/pull/588), [#594](https://github.com/goodmai/basalt/pull/594)), `SINQ` / `A-SINQ` / `EAQuant` ([#561](https://github.com/goodmai/basalt/pull/561)), `QTIP` (`2.0–3.0 bpw`, [#494](https://github.com/goodmai/basalt/pull/494)), `GSQ-RCO`, i-quants `.gguf` & `MACaQ-R` — câblés de manière identique entre CLI, REST `/v1/models/convert*` et `ConvertModelSheet` ([#546](https://github.com/goodmai/basalt/issues/546)). `QuIP# E8P` (`E8PLinear`), `TurboBoA` Babai–LLL et `SVD-LLM V2` / `OptimalBrainRestoration` sont des modules internes pas encore exposés via `--to` ; `--qat-refine` (`QAT STE`) et `--audit` (`QuantizationAuditCore`) restent planifiés/non câblés dans le convertisseur ([#616](https://github.com/goodmai/basalt/issues/616)).
- **PV-Tuning et PV-MACaQ post-quantification.** `basalt models convert --to macaq:affine:4:64 --pv-tune` alterne des mises à jour continues d'échelle/biais et un affinement discret par basculement hessien sur des blocs de Walsh–Hadamard : sur `Qwen3.5-9B`, `8-bit → 4-bit PV-MACaQ` réduit la RAM de 33% (`9.97 → 6.68 GB`) et accélère le décodage de `+52%` (`27.3 → 41.6 tok/s`, p50 `43.2`) avec une parité exacte `4/5` sur le Ladder (`valKL = 0.0044`) ; sur les checkpoints 2–3-bit, PV-Tuning réduit la perte de perplexité de 8.7–29% en moins de 3 minutes ([#486](https://github.com/goodmai/basalt/pull/486), [#545](https://github.com/goodmai/basalt/pull/545)).
- **Distillation, post-entraînement et calibration en cluster.** `basalt models distill` et `basalt models train` (SFT) entraînent des adaptateurs `lora`, `phm` et `oft` avec `--quantized-base-bits` (`LR/40`, [#501](https://github.com/goodmai/basalt/pull/501), [#508](https://github.com/goodmai/basalt/pull/508)), `ADPO` en ligne tourne dans le moteur, et `basalt cluster` + `--cluster-peers` associent des Macs pour la calibration distribuée ([#565](https://github.com/goodmai/basalt/issues/565)). `--dpo-pairs` (`SFT → DPO`) et `--adalora-target-rank` sont refusés par `models train` (DPO/GRPO hors-ligne dans [#822](https://github.com/goodmai/basalt/issues/822)) ; `DDP + ZeRO-2`, `DiLoCo` + `PowerSGD`, `ZB-H1` et `Expert-Parallel MoE` sont des modules de recherche internes non câblés à l'entraînement multi-Mac.
- **Entraîner tout en servant, avec contrôle qualité.** `distill` MCP à découpage temporel, `train_experts` uniquement sur réponses vérifiées, `train-jepa`, et retour arrière bit-à-bit si la divergence KL ou la perplexité sur jeu réservé régresse.

Détails : [commandes d'entraînement](docs/i18n/fr/commands.md) · [auto-entraînement, distillation et contrôle qualité](docs/i18n/fr/research-tools.md#self-training-train_experts)

## Compose : `basalt.yaml v2.2` déclaratif, gouvernance d'entreprise et 117 profils matériels

- **docker-compose, pour les modèles (`basalt.yaml v2.2`).** Modèles locaux et cloud, planificateur de déduplication RAM, profils spécialistes, modes mémoire `resident` / `on-demand` / `sequential`, routage à 5 composantes `BusinessOrchestrator`, cache sémantique `LANTERN`, porte d'approbation `HITL` (`/v1/mcp/approvals`), et noyau de connaissances neuro-symbolique `LAAS-CNRS` (`SymbolicKnowledgeGraph`) — dans un seul fichier déclaratif ([#607](https://github.com/goodmai/basalt/pull/607), [#625](https://github.com/goodmai/basalt/pull/625), [#626](https://github.com/goodmai/basalt/pull/626)).
- **117 profils intégrés pour chaque Mac (`16 GB` à `256 GB`).** Neuf paliers de mémoire unifiée (`16gb`, `24gb`, `32-36gb`, `48gb`, `64gb`, `96gb`, `128gb`, `192gb`, `256gb`) × 13 archétypes de charge se lancent en une commande (`basalt compose up 24gb-voice-zonos`, [#528](https://github.com/goodmai/basalt/pull/528)).
- **Refuse ce qui ne tient pas, modifie à chaud.** `basalt compose plan` dimensionne les modèles depuis le Hub avant téléchargement et refuse les débordements ; `compose diff` / `compose up` appliquent les différences à chaud proprement ; `compose bench` exécute des suites de benchmarks déclaratives et exporte des paquets de classement ([#505](https://github.com/goodmai/basalt/pull/505)).
- **Juges locaux, routage calibré et budgets cloud.** `OpenJev-Flash-9B-4bit` atteint 98.9% de précision de triage tout en réduisant les tokens LLM en aval de 77% ([#465](https://github.com/goodmai/basalt/pull/465)), `CalibratedRouterGate` mis à l'échelle par température + `Gemini Decision 1` routent proprement les requêtes difficiles dans la limite de plafonds quotidiens en USD ([#418](https://github.com/goodmai/basalt/pull/418), [#622](https://github.com/goodmai/basalt/pull/622)), et le texte issu de documents ou d'audio perçus ne quitte jamais votre Mac par défaut.

Détails : [référence et mesures compose](docs/i18n/fr/compose.md) · [préréglages de pipeline et modes mémoire](docs/i18n/fr/pipelines.md) · [décisions cloud avec Gemini](docs/i18n/fr/cloud-decisions.md)

## Recherche : ouvrir le modèle pendant qu'il s'exécute

- **Rayons X du modèle, attribution par token et filtrage épistémique de Belnap.** `model_describe` (qui signale les manifestes de modules `MACaQ` / `FlatQuant` / `LIRQuant`, [#630](https://github.com/goodmai/basalt/pull/630)), `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run`, `trace_tokens` (avec gestion automatique du BOS Gemma, [#630](https://github.com/goodmai/basalt/pull/630)), logprobs par token et risque d'hallucination `/v1/xray/tokens` ([#613](https://github.com/goodmai/basalt/pull/613)), attribution de têtes `Score d'influence` ([#622](https://github.com/goodmai/basalt/pull/622)), spectres de Marchenko–Pastur, états de preuve tétravalents de Belnap `SIXTEEN_3` ([#500](https://github.com/goodmai/basalt/pull/500)), sondes de phaseur MVN complexes ([#509](https://github.com/goodmai/basalt/pull/509)), dérive de Fisher–Rao ([#503](https://github.com/goodmai/basalt/pull/503)), et OpenMetrics sur `/api/v1/metrics` ([#458](https://github.com/goodmai/basalt/pull/458)).
- **Chirurgie réversible et circuits mécanistes.** Avec `serve --research-tools` : `intervene` (incluant ablation conditionnelle, patching de groupes GQA, espace nul cognitif et garde anti-steering, [#600](https://github.com/goodmai/basalt/pull/600)), décomposition de caractéristiques `MAO-SNMF`, `weights_delta`, `tensor_pack`/`tensor_unpack` en direct, `activation_contrast`, `steer_search`/`steer_commit`, `experiment_compare` et `introspect`. Chaque expérience est une surcouche réversible dotée d'un TTL et d'un journal ; les poids sur disque ne sont jamais modifiés.
- **Des benchmarks qui exécutent le code du modèle.** Une échelle à cinq échelons culminant avec une mission Terre-Lune exécute le propre code du modèle sur des entrées inédites, et ARC-AGI tourne selon les règles officielles pass@2.

Détails : [outils de recherche](docs/i18n/fr/research-tools.md) · [fondements de recherche (95+ articles)](docs/i18n/fr/research-foundations.md) · [benchmarks](docs/i18n/fr/benchmarks.md) · [options de serve](docs/i18n/fr/commands.md)

## Basalt pour Mac : macOS 27 Liquid Glass, à une touche ou un clic

- **Une application SwiftUI native sur un moteur supervisé.** Conçue pour macOS 27 Liquid Glass ([#479](https://github.com/goodmai/basalt/pull/479)) en anglais et en russe ([#456](https://github.com/goodmai/basalt/pull/456)) : fenêtre chaude en 257 ms, 0 réveil au repos, 7.4 ms p95 par token diffusé dans une transcription de 2 000 messages, et 40 ms de surcoût pour la requête rapide `⌥Space` ([#463](https://github.com/goodmai/basalt/pull/463)).
- **Interface du convertisseur universel, Hugging Face Hub et panneaux Studio expérimentaux.** Convertissez les modèles dans `ConvertModelSheet` (options QAT/audit planifiées [#616](https://github.com/goodmai/basalt/issues/616)), parcourez/publiez sur Hugging Face avec télémétrie SSE à 10 Hz et Xet ([#587](https://github.com/goodmai/basalt/pull/587)), et prévisualisez Video/Image Studio (échafaudages aujourd'hui ; vrais poids suivis dans [#824](https://github.com/goodmai/basalt/issues/824)).
- **Profils Compose (`⌥⌘C`), voix locale et lancement en 1 clic `basalt://`.** Recommande automatiquement les meilleurs profils pour la RAM de votre Mac (`16–256 GB`, [#528](https://github.com/goodmai/basalt/pull/528)), exécute la voix sur vrais poids via Apple `SpeechAnalyzer` + `ZONOS3-Mini` (autres moteurs TTS/duplex en échafaudage : [#821](https://github.com/goodmai/basalt/issues/821), [#826](https://github.com/goodmai/basalt/issues/826)), et gère `basalt://run?model=...`, `⌥Space`, Raccourcis, Spotlight et `.dmg` notarié.


**Basalt.app : Compose actif sur le moteur et le chat avec son inspecteur** (macOS 27 Liquid Glass, sombre et clair) :

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-compose-dark.png"><img src="docs/assets/screenshots/app-compose-light.png" alt="Basalt.app — profil Compose actif sur le moteur" width="49%"></picture></p>

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/screenshots/app-chat-dark.png"><img src="docs/assets/screenshots/app-chat-light.png" alt="Basalt.app — chat avec l'inspecteur ouvert" width="49%"></picture></p>

Détails : [Basalt pour Mac](docs/i18n/fr/mac-app.md)

---

## Démarrage rapide

Prérequis : macOS 27+, Xcode 27.0+ / Swift 6.4 avec la chaîne d'outils Metal (`xcodebuild -downloadComponent MetalToolchain`), Apple Silicon M1–M4, 2–38 GB d'espace disque selon le modèle. Les étapes complètes, la compilation depuis les sources et le lanceur `./basalt-claude` sont dans [docs/i18n/fr/install.md](docs/i18n/fr/install.md).

**Sans Xcode ni compilation : lancez le binaire précompilé pour Apple Silicon en deux lignes** (macOS 27+, Apple Silicon ; `mlx.metallib` est dans l'archive ; le premier lancement télécharge le modèle de 12 Go, utilisez `goodmai/gemma-4-12B-it-qat-4bit` pour démarrer plus léger) :

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

Diffusez un modèle MoE de 37.7 GB sur un Mac 24 Go, ou exécutez un système compose multimodèle complet :

```bash
basalt serve --model goodmai/Qwen3.6-35B-A3B-8bit --expert-streaming --rest

basalt compose plan 24gb-voice-zonos
basalt compose up 24gb-voice-zonos          # 117 profiles across 16–256 GB Macs
```

L'application Mac, depuis un dépôt cloné (enregistre automatiquement les liens en 1 clic `basalt://` à la première ouverture) :

```bash
./scripts/bundle_app.swift                  # release build, Metal kernels, then .build/Basalt.app
open .build/Basalt.app
```

Les modèles locaux 4B–12B ne remplacent pas directement un modèle frontière — ils sont conçus pour le travail agentique privé, hors ligne, sans coût, et pour les systèmes hybrides local/cloud. `basalt fit` vous indique ce que votre machine peut réellement exécuter. Le tableau des modèles indique aussi ce qui a *cassé*.

**Mettez une étoile** si vous souhaitez davantage d'outillage natif MLX en Swift, et ouvrez une issue avec votre Mac et vos tok/s — [les PR ajoutant une ligne au tableau des benchmarks](docs/i18n/fr/benchmarks.md#benchmarks) sont les bienvenues.

---

## Documentation

> La documentation traduite est disponible sous `docs/i18n/` : [Русский](docs/i18n/ru/) · [中文](docs/i18n/zh/) · [Español](docs/i18n/es/) · [Français](docs/i18n/fr/) · [日本語](docs/i18n/ja/) · [Deutsch](docs/i18n/de/) · [한국어](docs/i18n/ko/) · [Bahasa Indonesia](docs/i18n/id/).

| Sujet | Document |
|---|---|
| Installation, prérequis, compilation depuis les sources, `./basalt-claude` | [docs/i18n/fr/install.md](docs/i18n/fr/install.md) |
| Commandes CLI, options de `serve`, cache de prompts, CacheBlend, PV-Tuning | [docs/i18n/fr/commands.md](docs/i18n/fr/commands.md) |
| API REST : sécurité, gestion des modèles, OpenAI, Anthropic, images, SSE, OpenMetrics, Swagger | [docs/i18n/fr/rest-api.md](docs/i18n/fr/rest-api.md) |
| Claude Code, OpenCode, SDK OpenAI, MCP pour Claude Desktop et Cursor | [docs/i18n/fr/integrations.md](docs/i18n/fr/integrations.md) |
| Modèles vérifiés, pile Hub `goodmai` et téléchargement | [docs/i18n/fr/models.md](docs/i18n/fr/models.md) |
| Aperçu des fonctionnalités, échelle de benchmarks, ARC-AGI, provenance | [docs/i18n/fr/benchmarks.md](docs/i18n/fr/benchmarks.md) |
| Préréglages de pipeline et modes mémoire | [docs/i18n/fr/pipelines.md](docs/i18n/fr/pipelines.md) |
| Décisions cloud : Gemini Decision 1, médias multimodaux, enseignant de calibration | [docs/i18n/fr/cloud-decisions.md](docs/i18n/fr/cloud-decisions.md) |
| Référence et mesures de `basalt compose` | [docs/i18n/fr/compose.md](docs/i18n/fr/compose.md) |
| Basalt pour Mac et liens profonds `basalt://` | [docs/i18n/fr/mac-app.md](docs/i18n/fr/mac-app.md) |
| Rayons X du modèle, télémétrie, expériences, auto-entraînement, distillation | [docs/i18n/fr/research-tools.md](docs/i18n/fr/research-tools.md) |
| Fondements mathématiques et scientifiques (85+ articles et remerciements) | [docs/i18n/fr/research-foundations.md](docs/i18n/fr/research-foundations.md) |
| Structure du projet, notes d'architecture, scripts, dépendances | [docs/i18n/fr/architecture.md](docs/i18n/fr/architecture.md) |
| Patrons d'architecture et modules (russe) | [docs/ARCHITECTURE_PATTERNS_AND_MODULES.md](docs/ARCHITECTURE_PATTERNS_AND_MODULES.md) |
| Cloud : configuration OpenRouter, optimisation des coûts, comparaison des modèles | [docs/cloud/OPENROUTER_SETUP.md](docs/cloud/OPENROUTER_SETUP.md) · [docs/cloud/COST_OPTIMIZATION.md](docs/cloud/COST_OPTIMIZATION.md) · [docs/cloud/MODEL_COMPARISON.md](docs/cloud/MODEL_COMPARISON.md) |
| Travailler sur basalt avec des agents IA | [docs/AI_SETUP.md](docs/AI_SETUP.md) · [docs/AGENT_OPERATIONS.md](docs/AGENT_OPERATIONS.md) |
| Feuille de route, politique de confidentialité | [docs/ROADMAP.md](docs/ROADMAP.md) · [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) |
| Notes de recherche sur les CLI d'agents, listes de contrôle, spécifications (russe) | [docs/features.md](docs/features.md) · [docs/specs/FEATURES_CHECKLIST.md](docs/specs/FEATURES_CHECKLIST.md) · [docs/specs/AUDIT_CHECKLIST.md](docs/specs/AUDIT_CHECKLIST.md) · [docs/specs/EMOJI_AS_CODE.md](docs/specs/EMOJI_AS_CODE.md) |
| Contribuer, journal des modifications | [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) |

## Licence

[MIT](LICENSE) © 2026 goodmai

Les poids des modèles ne sont **pas** couverts par cette licence — chaque modèle sur HuggingFace possède ses propres conditions.
