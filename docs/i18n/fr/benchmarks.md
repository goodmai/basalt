# Fonctionnalités et benchmarks

> Languages: [English](../../benchmarks.md) · [Русский](../ru/benchmarks.md) · [中文](../zh/benchmarks.md) · [Español](../es/benchmarks.md) · **Français** · [日本語](../ja/benchmarks.md) · [Deutsch](../de/benchmarks.md) · [한국어](../ko/benchmarks.md) · [Bahasa Indonesia](../id/benchmarks.md)

## Aperçu des fonctionnalités

Chaque accélération est activable à la demande et garde une sortie gloutonne (greedy) identique, sauf indication contraire dans la ligne. Chiffres : M4 Pro 24 Go sur secteur, compilation release.

| Fonctionnalité | Option | Ce qu'elle fait | Mesuré |
|---|---|---|---|
| Recherche dans le prompt (décodage spéculatif) | `serve --speculative-lookup` | ébauche jusqu'à 16 tokens copiés depuis le prompt ou la sortie, vérifiés en une seule passe avant | 1.84–2.66× sur les éditions de code (Bonsai 2 27B), 3.45–3.62× (Gemma 4 12B) ; ~1.0× sur la prose |
| Ébauchage par tête MTP | `serve --mtp [--mtp-block N]` | ébauche avec la tête de prédiction multi-tokens du modèle (`mtp.safetensors`, par ex. packs MTPLX) ; exact à toute température. Les tours démarrent à un token ébauché et montent jusqu'à N−1 (4 par défaut) après quatre tours entièrement acceptés ; un rejet ramène à un | 1.30× sur le code, 1.04× sur la prose, 1.26× sur le raisonnement (Bonsai 2 27B, T=0.7) ; un tour suivant réutilise le cache de prompts comme le décodage simple : TTFT 12.5 s → 0.75 s sur un tour de 1 149 tokens (Bonsai 2 27B) |
| Ternary Bonsai 2 | (automatique) | charge des checkpoints ternaires à rotation de Hadamard (`prism_hadamard_qwen35`) | modèle 27B dans 7.4 GB |
| Embeddings et rerank | `serve --embedding-model`, `--rerank-model` | OpenAI `/v1/embeddings`, Cohere/Jina `/v1/rerank` à côté du modèle de chat | bge-small 229–299 textes/s, Qwen3-Embedding-0.6B 20 textes/s (lot de 32) |
| Cache de prompts sur disque | `serve --prompt-cache-disk-mb N` | une session reprise ou un serveur redémarré restaure le plus long préfixe correspondant au lieu de le préfill | — |
| Quantification et zones du cache KV | `serve --kv-bits 4\|8\|auto\|k8v4`, `--kv-zones` | fait tenir des contextes plus longs dans la même mémoire | pic prompt 16k à 1.03 GB contre 3.11 GB (Qwen3-4B, zones 4-bit) |
| Requêtes concurrentes | `serve --max-concurrent N` | chaque requête en cours réserve son espace KV ; celle qui ne tient pas reçoit 503 `Retry-After` au lieu d'un dépassement mémoire | — |
| Échantillonneurs DRY et XTC (expérimental) | `serve --experimental-samplers` | champs de requête `dry_*` / `xtc_*` | pas un outil anti-boucle : sur un modèle 2-bit, DRY masque les boucles derrière des faits corrompus (#177) |

## D'où proviennent les chiffres du README

Chaque chiffre affiché sur le premier écran du README provient d'une exécution journalisée, et la PR qui a livré la fonctionnalité contient le tableau, le commit et le chemin du journal. M4 Pro 24 Go, compilation release, sauf mention contraire dans la PR.

| Affirmation | Source |
|---|---|
| Qwen3-4B 78–89 tok/s ; Ternary Bonsai 2 27B dans 7.4 GB à 20.9–21.5 tok/s, 31–38 avec recherche dans le prompt sur le code | [#239](https://github.com/goodmai/basalt/pull/239), issu de [#218](https://github.com/goodmai/basalt/pull/218), [#231](https://github.com/goodmai/basalt/pull/231), [#232](https://github.com/goodmai/basalt/pull/232) |
| Bonsai 2 27B avec sa tête MTP, 27.1 tok/s sur le code | [#263](https://github.com/goodmai/basalt/pull/263) |
| Recherche dans le prompt 3.45–3.62× sur Gemma 4 12B, sortie gloutonne inchangée | [#265](https://github.com/goodmai/basalt/pull/265) |
| Deuxième tour d'agent environ 1 s au lieu de 36 s sur Ornith-1.5-9B (prompt de 12.6k tokens) | [#82](https://github.com/goodmai/basalt/pull/82) |
| Pic prompt 16k à 1.03 GB au lieu de 3.11 GB avec les zones KV sur Qwen3-4B | [#159](https://github.com/goodmai/basalt/pull/159) |
| Mélange de modèles : chaque étiquette juste, $0.035 pour 1000 requêtes, p95 4.8 s contre 10.0 s | [#418](https://github.com/goodmai/basalt/pull/418) |
| Application Mac : fenêtre chaude 257 ms, 0 réveil au repos, 7.4 ms p95 par token diffusé, 40 ms requête rapide | [#463](https://github.com/goodmai/basalt/pull/463) |
| Un juge 9B à côté de Gemma 4 12B sur 24 Go : 4-bit tient (15.3 GB), 98.9% avec 77% de tokens LLM en moins | [#465](https://github.com/goodmai/basalt/pull/465) |
| Qwen3.6-35B-A3B-8bit (37.7 GB) diffusé depuis le SSD sur 24 Go à 8.77 tok/s avec 0 swap | [#477](https://github.com/goodmai/basalt/pull/477), [#478](https://github.com/goodmai/basalt/pull/478) |
| `--experimental-pv-tune` post-quantification : récupération de 8.7–29% de perplexité en 2–3-bit en moins de 3 minutes | [#486](https://github.com/goodmai/basalt/pull/486) |
| Treillis à décalage de bits QTIP `--experimental-qtip` : `Qwen3.8-27B` 2.5 bpw en 10.00 GB bat l'affine 3-bit et PV-Tuning (16k PPL 4.621 vs 4.702 / 4.652) ; 2.0 bpw en 8.48 GB réduit la PPL 2-bit de 72.1% (5.345 vs 19.146) | [#494](https://github.com/goodmai/basalt/pull/494) |
| Exécution native `.gguf` GSQ-RCO / `i-quant` sur Metal (`IQ4_XS` / `IQ3_S`) | [#433](https://github.com/goodmai/basalt/pull/433) |
| `Flint-40B-A2.5B` récursif (`oQ3e`) à 67.02 tok/s en décodage (273.5 tok/s en préfill) dans 17.61 GB sur M4 Pro 24 Go | [#522](https://github.com/goodmai/basalt/pull/522) |
| `Zyphra/ZONOS3-Mini-hf` TTS neuronal 44.1 kHz en 2.24 GB à RTF 0.228–0.263 (3.8–4.4× temps réel) sur EN/DE/RU | [#527](https://github.com/goodmai/basalt/pull/527) |
| 84 profils Compose par paliers matériels (`16–256 GB × 12`) avec recommandation automatique du palier RAM dans `Basalt.app` | [#528](https://github.com/goodmai/basalt/pull/528) |
| `MACaQSwitchLinear` WHT en passe unique + `gatherQuantizedMM` : noyau MoE `3.41×` (`2.06 → 0.60 ms`), `+82%` de bout en bout sur Gemma 4 26B-A4B (`49.2–54.8 tok/s`), `64–75 tok/s` sur Ornith 35B-A3B 3-bit | [#628](https://github.com/goodmai/basalt/pull/628) |
| `Qwen3.5-9B-PV-MACaQ-4bit` : parité exacte `4/5` sur le Ladder avec le 8-bit, `+52%` en décodage (`43.2 tok/s` p50), `−33%` de RAM (`6.68 vs 9.97 GB`), dérive max par couche `3.37%` | [#545](https://github.com/goodmai/basalt/pull/545), [#630](https://github.com/goodmai/basalt/pull/630) |
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` : `15.1 GB` de RAM (pic `18.09 GB`) sur Mac 24 Go, `4/5` Ladder + `4/6` Lunar à `14.1 tok/s` p50 (la réf. 4-bit provoque un OOM) | [#628](https://github.com/goodmai/basalt/pull/628), [#630](https://github.com/goodmai/basalt/pull/630) |

Les modèles vérifiés sur golden dans le README sont contrôlés par `scripts/check_readme_models.swift` en CI : un modèle qui y figure doit lier une fixture existante dans `Tests/BasaltTests/Fixtures/golden`.

Une page de benchmarks statique autonome ([`docs/benchmark-page/index.html`](../../benchmark-page/index.html)) et un manifeste de collection Hugging Face prêt pour relecture ([`docs/benchmark-page/hf-collection.json`](../../benchmark-page/hf-collection.json)) sont générés à partir de ces tableaux, de `benchmarks/` et des fixtures golden par `swift scripts/build_benchmark_page.swift` (et vérifiés en CI avec `--check` ; passez `--logs logs` en local pour ajouter les tableaux `summary` des exécutions). La publication externe reste une décision humaine.

## Benchmarks

Deux bancs de test, tous deux pointés vers un serveur en cours d'exécution, et tous deux rapportant des chiffres reproductibles sur votre propre machine.

**L'échelle vers la Lune** — `benchmarks/ladder/run.py`. Cinq échelons, chacun strictement plus difficile que le précédent, culminant par une mission Terre-Lune complète. Rien n'est noté sur la forme de la réponse : le propre code du modèle est exécuté sur des entrées que le prompt n'a jamais montrées, de sorte qu'une réponse fausse mais plausible échoue tout de même.

| Échelon | Tâche | Ce qu'il teste réellement |
|---|---|---|
| 1 | Bicarrée, racines réelles | forme analytique et cas limites : aucune, double, zéro |
| 2 | Bicarrée, racines complexes | gestion des branches — les quatre racines, avec multiplicité |
| 3 | Coefficients de Fourier | intégration numérique, vérifiée sur dent de scie, onde carrée, cosinus |
| 4 | Hauteur manométrique de pompe | unités (mm vs m, L/s vs m³/s) + Colebrook, qui n'a pas de forme analytique |
| 5 | **Mission Terre → Lune** | vitesse d'éjection depuis `sqrt(2ηQ)`, Tsiolkovski par étage, propagation à quatre corps, freinage lunaire, alunissage en douceur — chaque constante figure dans le prompt, rien à réciter d'un manuel |

L'échelon supérieur est noté sur six invariants plutôt que sur une seule réponse, afin qu'un modèle qui réussit l'énergétique sans propager la trajectoire obtienne tout de même un score. Son prompt se trouve dans `benchmarks/ladder/lunar_prompt.md` ; les valeurs de référence de `lunar_task.md` ne sont jamais montrées au modèle.

`--warmup` ajoute en tête trois tests rapides peu coûteux (algèbre, Fibonacci, traduction) que tout modèle instruct devrait réussir. Ils ne comptent pas dans le score de l'échelle.

**ARC-AGI** — `benchmarks/arc-agi/arc_agi_benchmark.py`, règles officielles pass@2. Le corpus n'est pas inclus dans le dépôt ; récupérez-le d'abord dans `benchmarks/arc-agi/dataset/` (voir `benchmarks/arc-agi/ARC_AGI.md`).

### Résultats

Mesuré sur un Mac de série M avec 24 Go de mémoire unifiée. `thinking` correspond au bloc `<think>` du gabarit de chat : activé par défaut pour les modèles de raisonnement, désactivé avec `--reasoning-effort none`.

| Modèle | Préchauffage | Échelle | Échelon supérieur | ARC-AGI mini | tok/s |
|---|---|---|---|---|---|
| `goodmai/gemma-4-31B-it-PV-MACaQ-3.2bpw-MLX` (thinking off, 24 Go) | 3/3 | **4/5** | 4/6 invariants | — | 11.6–13.5 (p50 14.1) |
| `goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX` (thinking off, 24 Go) | 3/3 | **4/5** | 0/6 (longueur) | — | 9.4–11.4 (p50 11.6) |
| `Qwen3.5-9B-PV-MACaQ-4bit` (thinking off) | 3/3 | **4/5** | 0/6 (prose) | — | 35.6–41.6 (p50 43.2) |
| `mlx-community/Qwen3.5-9B-8bit` (réf., thinking off) | 3/3 | **4/5** | 0/6 (prose) | — | 24.7–27.3 |
| `goodmai/gemma-4-26B-A4B-it-qat-4bit` (thinking off) | 3/3 | **4/5** | 5/6 invariants | — | 56.0–63.4 |
| `goodmai/gemma-4-12B-it-qat-4bit` (thinking off) | 3/3 | **4/5** | **6/6 PASS** | — | 18.8–22.2 |
| `goodmai/gemma-4-26B-A4B-it-PV-MACaQ-4bit-Affine-MLX` (thinking off, `#628`) | 3/3 | 3/5 | 0/6 | — | 49.2–54.8 |
| `goodmai/Ornith-1.5-35B-A3B-MACaQ-3bit-MLX` (thinking off, `#628`) | 3/3 | 1/5 | 0/6 | — | 64.0–74.7 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking off) | 3/3 | 1/4 ² | non exécuté | 0/3 ¹ | ~45 |
| `ornith-ai/Ornith-1.5-9B-MLX-4bit` (thinking on) | 2/3 | 0/4 ² | non exécuté | non exécuté | ~43 |

Ornith 9B échelon par échelon, thinking off :

| Échelon | Résultat | Détail |
|---|---|---|
| 1 · bicarrée réelle | ✅ | racines correctes, y compris les cas double et vide |
| 2 · bicarrée complexe | ❌ | `x⁴ + 1` renvoie 2 racines au lieu de 4 |
| 3 · Fourier | ❌ | `a0 = 10.86` pour `f(x) = x`, attendu 0 — c'est l'intégration qui est fausse, pas la formule |
| 4 · hauteur de pompe | ❌ | `v = 0.0298` m/s, attendu 1.79 — n'a jamais converti les mm en m |
| 5 · mission lunaire | — | non exécuté |

Préchauffage, thinking off : algèbre ✅ (163 tok, 4 s), Fibonacci ✅ (757 tok, 16 s), traduction ✅ (303 tok, 7 s — allemand idiomatique, français paraphrasé plutôt que traduit).

² Les échelons 1 et 2 ont été mesurés avant la séparation de l'échelle en prompts distincts : un seul prompt demandait les deux fonctions, la moitié réelle était correcte et la moitié complexe ne l'était pas. L'échelon 5 n'a pas encore été exécuté sur ce modèle.

¹ La colonne ARC-AGI est une exécution partielle : trois tâches d'entraînement (`007bbfb7`, `00d62c1b`, `017c7c7b`), aucune résolue en pass@2, 330 s au total. L'exécution a été arrêtée avant les deux suivantes ; il convient donc de la considérer comme un test rapide plutôt que comme un score.

Les validateurs sont eux-mêmes vérifiés, sans modèle ni GPU :

```bash
python3 benchmarks/ladder/selfcheck.py    # 9 assertions, both directions
```

Avec thinking **activé**, les trois tâches difficiles et la traduction se sont terminées avec `finish_reason: length` — tout le budget est passé dans le bloc `<think>` et aucune réponse n'a jamais été émise. Il s'agit d'un échec de budget, et non d'un plafond de capacité, ce qui explique pourquoi le tableau ci-dessus rapporte les deux modes séparément.

### Reproduire

```bash
basalt serve --model ornith-ai/Ornith-1.5-9B-MLX-4bit --rest --reasoning-effort none &

python3 benchmarks/ladder/run.py --port 8080 --warmup --json ladder.json
python3 benchmarks/arc-agi/arc_agi_benchmark.py --engine mlx --split training --limit 5
```

Ajouter un modèle au tableau correspond à une exécution de chaque banc — les PR avec de nouvelles lignes sont les bienvenues.
