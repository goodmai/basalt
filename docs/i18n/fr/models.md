# Modèles

> Languages: [English](../../models.md) · [Русский](../ru/models.md) · [中文](../zh/models.md) · [Español](../es/models.md) · **Français** · [日本語](../ja/models.md) · [Deutsch](../de/models.md)

## Modèles vérifiés

Les modèles sont téléchargés depuis HuggingFace et mis en cache dans `~/.cache/huggingface/hub/`. Plus de 42 checkpoints vérifiés de chat, MoE, QTIP, PV-Tuned, GSQ-RCO, OCR, TTS neuronal, embeddings, rerank et décision System One sont publiés avec des boutons **Run with Basalt** en un clic ([`goodmai-run-with-basalt.static.hf.space`](https://goodmai-run-with-basalt.static.hf.space/?lang=en) → `basalt://run?model=...`) dans la [collection Hugging Face `goodmai`](https://huggingface.co/collections/goodmai/basalt-inference-and-quantization-stack-6af94bed95c64cce97e188fc).

```bash
basalt models download goodmai/gemma-4-12B-it-qat-4bit
basalt models download goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native
```

### Mesuré pour 0.12

M4 Pro, 24 Go, compilation release (`620641c` / `82a99d4`), novembre 2026 (`--reasoning-effort none --kv-bits 4`).

| Modèle | Rôle / Architecture | Mémoire active | Décodage tok/s | Ladder / Notes |
|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` | 31B hybride dense (attn/embed 4-bit + MLP PV-MACaQ affine 3-bit) | 15.1 GB (pic 18.09 GB) | 11.6–13.5 (p50 `14.07`, p95 `14.42`) | **3/3 préchauffage, 4/5 échelons + Lunar 4/6**, `probe_ppl = 2.39`, 50.6% d'acceptation lookup ; tient sur Mac 24 Go là où le 4-bit provoque un OOM (`--prompt-cache-mb 0 --wired-limit`) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` | 27B hybride dense (48 attn linéaires + 16 complètes, 496 modules MACaQ 4-bit g64) | 15.3 GB (pic 17.45 GB) | 9.4–11.4 (p50 `11.59`, p95 `11.86`) | **3/3 préchauffage, 4/5 échelons**, `probe_ppl = 1.23`, `SQNR 20.61 → 21.59 dB` (`MSE −19.5%`), dérive max sur 64 couches `5.53%` (`#630`) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (depuis `Qwen3.5-9B-8bit`) | 9B hybride (24 attn linéaires + 8 complètes, 200 modules MACaQ 4-bit g64) | 6.68 GB (−33% vs 8-bit) | 35.6–41.6 (p50 `43.22`, +52% vs 8-bit) | **3/3 préchauffage, 4/5 échelons** (parité 100% avec la réf. 8-bit `24.7–27.3 tok/s`), `valKL = 0.0044`, `probe_ppl = 1.28`, dérive max par couche `3.37%` |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` | 26B MoE (`A4B`, 128 experts, top-8, MACaQ affine g32 + MLP partagé 8-bit) | 16.75 GB | 49.2–54.8 (+82% via switch WHT `#628`) | **3/3 préchauffage, 3/5 échelons** (`biquad-complex`, `fourier`, `pump`) ; accélération noyau `3.41×` (`2.06 → 0.60 ms`, `#628`) |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` | 35B MoE (`A3B`, 120 couches `MACaQSwitchLinear`) | 14.64 GB | 64.0–74.7 (68.7 à 8k) | **3/3 préchauffage, 1/5 échelons**, `probe_ppl = 1.46`, switch WHT en passe unique (`#628`) |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` | 26B MoE (`A4B`, QAT 4-bit) | 14.92 GB | 56.0–63.4 | **3/3 préchauffage, 4/5 échelons** (`pump` ✅, `lunar 5/6`) |
| `goodmai/gemma-4-12B-it-qat-4bit` | 12B dense QAT 4-bit | 10.51 GB | 18.8–22.2 | **3/3 préchauffage, 4/5 échelons** (`lunar` ✅ 6/6) |
| Vidéo / Parole / Musique (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`, `LFM2.5-Audio`, `FLUX.3`, `SeedVR2`, `ACE-Step 1.5`) | Video Studio phasé pour UMA 24 Go, audio duplex Mimi/SNAC et image/musique en 4 étapes | ≤ 18 GB phasé | 3D-VAE + noyau Metal STA | Vérifié dans `#542`, `#604`, `#605`, `#608`–`#611` |

### Mesuré pour 0.11

M4 Pro, 24 Go, compilation release, novembre 2026.

| Modèle | Rôle / Architecture | Mémoire active | Débit / Latence | Notes |
|---|---|---|---|---|
| `Recursive/Flint-40B-A2.5B` (`flint-oQ3e`) | 40B-A2.5B MoE hybride sliding/full (`512` routés, top-10 + 1 partagé) | 17.61 GB | 67.02 tok/s décodage · 273.5 tok/s préfill | chargeur natif `oQ3e` (`groupSize = 32`) (#518, #522), 0 swap sur 24 Go |
| `Zyphra/ZONOS3-Mini-hf` | 1.6B Llama + TTS DAC multi-têtes FSQ à 32 codebooks (`44.1 kHz`) | 2.24 GB | RTF 0.228–0.263 (3.8–4.4× TR) | 43 langues, 27 voix, `embedding` interlingue + CFG `1.5` (#525, #527) |
| `hexgrad/Kokoro-82M` | 82M TTS neuronal (ANE + MLX) | ~0.35 GB | TTFA inférieur à 80 ms | étage de repli en streaming par phrase dans `pipeline.speak` (#493, #513) |
| `Tongyi-MAI/Z-Image-Turbo` | Synthèse d'images DiT | ~6.4 GB | 4–8 étapes (`1024×1024`) | `pipeline.imagine` avec TeaCache et encodeur de texte partagé (#517, #519) |
| `Ricursive/Ripple-65B-4bit` | 65B MoE clairsemé à attention à portes (`128` experts, top-8) | ~35 GB (résident sur 64 Go+, ou streaming SSD) | architecture vérifiée | NoPE / RoPE partiel (#520, #522) |
| `nvidia/Nemotron-3.5-Lightning-30B-A3B` | 30B-A3B Mamba-2 + MoE à routage latent (`lr_moe`) | ~17 GB (4-bit) | architecture vérifiée | GQA entrelacé + Mamba-2 (#521, #522) |
| `meta-models/Muse-Glimmer-30B` (`mlx-community/Muse-Glimmer-30B-4bit`) | 30B hybride SWA (`2048`) + attention complète NoPE + VLM ViT-G/14 | ~18.1 GB (4-bit) · ~19.8 GB (`oQ4e`) | architecture et shards HF vérifiés | raisonnement Onyx `to=self` + appels d'outils ATEM + import de normes décentrées GGUF (#533) |

### Mesuré pour 0.8

M4 Pro, 24 Go, sur adaptateur secteur 68 W, compilation release, décodage glouton (greedy), novembre 2026.

| Modèle | Paramètres | Mémoire active | Décodage, tok/s | Notes |
|---|---|---|---|---|
| `mlx-community/Qwen3-4B-4bit` | 4B | ~2.2 GB | 78–89 | dense ; la recherche dans le prompt et tous les modes KV s'appliquent |
| `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` | 27B ternaire | 7.4 GB | 20.9–21.5 brut · 27.1 avec `--mtp` sur code · 31–38 avec `--speculative-lookup` sur code | hybride (Gated DeltaNet) ; en miroir sur `goodmai/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` |
| `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly` | 27B | ~11.8 GB | 15–19 | hybride ; vérifié sur golden ; en miroir sur `goodmai/Qwen3.8-27B-3bit-MLX-TextOnly` |
| `goodmai/Qwen3.8-27B-GSQ-RCO-IQ3_S-Basalt-Native` | 27B i-quant | ~12.4 GB | 5.2 | import natif GSQ-RCO `IQ3_S` (#433), WikiText-2 PPL 4.169, 300/300 System One |
| `goodmai/Qwen3.6-35B-A3B-8bit` | 35B MoE (37.7 GB) | ~13.9 GB | 8.77 | `--experimental-expert-streaming` (#478), 0 swap sur 24 Go via AIO par lots + `LIO_NOWAIT` |
| `lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly` | 27B | non mesuré | 18–22 | boucle sur certains prompts (3 sur 8) ; `RepetitionGuard` les interrompt |

| Modèle d'embedding / rerank | Tâche | Mesuré |
|---|---|---|
| `mlx-community/bge-small-en-v1.5-bf16` | embeddings (384-d) | 229–299 textes/s, lot de 32 × ~240 tokens |
| `mlx-community/Qwen3-Embedding-0.6B-8bit` | embeddings (1024-d) | 20 textes/s, même lot |
| `mlx-community/Qwen3-Reranker-0.6B-4bit` | rerank | paires connues classées correctement |
| `jinaai/jina-reranker-v3-mlx` | rerank | paires connues classées correctement |

### Rapports précédents (non remesurés pour 0.8)

| Modèle | Paramètres | RAM | Statut sur Mac 24 Go |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 2B | 2.7 GB | ✅ ~110 TPS |
| `mlx-community/gemma-4-e4b-it-4bit` | 4B | 4.3 GB | ✅ ~85 TPS |
| `mlx-community/Qwen3.5-4B-4bit` | 4B | 2.3 GB | ✅ ~92 TPS |
| `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` | 7B | 4.1 GB | ✅ ~60 TPS |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | 9B | 5.8 GB | ✅ ~37 TPS |
| `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit` | 27B | 15.2 GB | ✅ ~12 TPS (Abliterated) |
| `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX` | 26B MoE | 14.5 GB | ✅ ~14 TPS (Quantification dynamique) |
| `Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4` | 35B MoE | 20.5 GB | ✅ ~8 TPS (NVFP4) |
| `Ex0bit/Elbaz-Olmo-3-7B-Instruct-abliterated` | 7B | 4.5 GB | ✅ ~55 TPS |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` | 9B | 4.7 GB | ✅ ~48 TPS (raisonnement — voir note) |
| `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit` | 35B MoE | 21 GB | ✅ non testé au-dessus de 24 Go |
| `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` | 8B MoE | 4.5 GB | ✅ ~76 TPS (raisonne toujours) |
| `LiquidAI/LFM2.5-24B-A2B-MLX-4bit` | 24B MoE | 12.6 GB | ✅ ~40 TPS (raisonne toujours) |
| `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (Base BF16) | 27B | ~54 GB | ❌ Nécessite 64 Go+ (utiliser la version MLX 4-bit sur 24 Go) |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B MoE | 14.5 GB | ❌ Sortie incohérente |
| `mlx-community/Qwen3.6-27B-4bit` | 27B | 14.5 GB | ❌ Sortie incohérente |
| `mlx-community/gemma-4-31b-it-4bit` | 31B | 17 GB | ❌ OOM (Nécessite 32 Go+) |

> **Note sur les Mac à 24 Go de RAM :**
> - Les modèles 27B/35B non quantifiés (~54 GB de poids) dépassent la mémoire physique. Utilisez des poids quantifiés MLX 4-bit / Dynamic Quant (DQ) (par ex. `AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit`, `Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX`) qui tiennent confortablement dans ~15 GB de RAM.
> - Les modèles > 10 Go contraignent la taille du cache KV sous des contextes multi-tours chargés. Le cache KV occupe `2 · layers · kv_heads · head_dim · 2 B` par token (fp16), lus depuis le `config.json` du modèle ; les modèles hybrides ne comptent que leurs couches à attention complète plus un état récurrent fixe, les couches sliding-window s'arrêtent à la fenêtre, et `--kv-bits 4|8` stocke `bits + 0.5` bits par élément (`--kv-bits` ne s'applique pas aux couches sliding-window). Qwen3-4B : 144 KiB par token, 4.5 GiB à 32k tokens ; Ornith-1.5-9B (8 couches sur 32 en attention complète) : 32 KiB par token, 1.05 GiB à 32k. `serve --dry-run`, `models check`, `fit` (sa grille de quantification et son estimation de vitesse de décodage — la colonne RAM et le niveau d'adéquation ne comptent encore que les poids) et le budget de tokens dérivé de la mémoire utilisent ce chiffre (plafond = min(max_tokens demandé, --max-tokens, espace KV − tokens du prompt), où l'espace KV correspond au working set recommandé par Metal moins les poids chargés et le budget du cache de prompts, et le prompt est compté avec le tokenizer du modèle ; un prompt qui ne tient pas du tout est refusé avec un code 413 avant la génération) ; sans configuration téléchargée, ils se rabattent sur une estimation basée sur le nombre de paramètres qui est proche pour les modèles denses 7B–32B mais plusieurs fois fausse pour les petits modèles, les hybrides et les MoE. Estimations, et non mesures : l'état récurrent des couches Mamba (Jamba, NemotronH) n'est pas compté, seules leurs couches d'attention le sont ; les modèles MLA (DeepSeek V2/V3) sont comptés comme les K et V décompressés par tête que MLX met en cache, et non comme le latent compressé.

## Téléchargement de modèles (mlx-community, huihui-ai, Ex0bit)

Basalt peut parcourir, inspecter et télécharger des modèles depuis n'importe quel créateur ou organisation sur Hugging Face :

```bash
# List models by author
basalt models list --author Ex0bit
basalt models list --author huihui-ai
basalt models list --author mlx-community

# Search with filters
basalt models list --author Ex0bit --search PRISM
basalt models list --search Huihui-Qwen3.8

# Download specific models
basalt models download AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit
basalt models download Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX
basalt models download Ex0bit/Qwen3.6-35B-A3B-PRISM-MLX-NVFP4

# Interactive picker with custom author
basalt models download --author Ex0bit
```
