# basalt compose

> Languages: [English](../../compose.md) · [Русский](../ru/compose.md) · [中文](../zh/compose.md) · [Español](../es/compose.md) · **Français** · [日本語](../ja/compose.md) · [Deutsch](../de/compose.md)

`basalt compose` sert un système de modèles décrit dans un seul fichier YAML. Le fichier répertorie les modèles locaux et cloud ainsi que la manière dont chacun est conservé en mémoire. Il décrit également les pipelines qui acheminent une requête entre eux. Chaque pipeline exécute la perception (OCR, parole), puis la décision (System One), puis la réponse (un modèle de chat). Il s'agit d'un mélange de modèles, dans le style d'un fichier docker-compose pour modèles.

```bash
basalt compose plan -f basalt.yaml        # validate and show the compute plan on this Mac
basalt compose up -f basalt.yaml          # serve it; run it again with a changed file to apply the diff
basalt compose ps                         # what is loaded, when on-demand models unload, cloud spend
basalt compose down <name>                # stop it
basalt compose up document-qa             # legacy bundled files: document-qa, voice-agent, zonos-voice
basalt compose up 24gb-voice-zonos        # 84 hardware-tiered profiles: <tier>-<archetype>
```

`compose up` démarre `basalt serve` avec les options produites par l'expansion du fichier ; il les affiche au démarrage. Il expose les mêmes surfaces REST et MCP. Les options placées après `--` sont transmises à `serve`, par exemple `basalt compose up -f basalt.yaml -- --port 8090`.

Le serveur ouvre d'abord son port et charge les modèles ensuite. Pendant le chargement, `compose ps` les affiche comme `loading`, `GET /v1/health/ready` répond 503, et l'enregistrement d'exécution existe déjà, de sorte qu'un second `compose up` retrouve le système au lieu d'en démarrer un autre. Une fois les modèles chargés, chacun passe une sonde de démarrage :
- le modèle de chat et chaque modèle de réponse local génèrent un token ;
- chaque modèle de décision local répond à une question ;
- les fichiers d'un modèle à la demande (`on-demand`) sont vérifiés sans le charger.

Un modèle qui échoue à sa sonde apparaît comme `failed`, avec la raison, dans `compose ps`. `/v1/health/ready` reste à 503 jusqu'à ce qu'un `compose up` remplace le modèle. Un modèle que `compose up` charge ou recharge subit la même sonde. Avec `--mcp`, les modèles sont chargés avant le démarrage du serveur, comme auparavant.

## Le fichier

```yaml
version: 1
name: support-desk            # the system's name, used by ps, down and up-as-diff
memory: auto                  # resident | sequential | auto
budget: { usd_per_day: 2 }    # optional cap on all cloud spend

models:
  chat:     { source: mlx-community/gemma-4-12B-it-qat-4bit }
  big:      { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, idle_unload: 10m }
  ocr:      { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 2m }
  asr:      { source: divergentlabs/Nemotron-3.5-Transcribe-30B-A4B-MLX, keep: sequential }
  jev:      { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
  frontier: { source: "cloud:openrouter/anthropic/claude-haiku-5.5", api_key_env: OPENROUTER_API_KEY, budget: { usd_per_day: 1 } }

pipeline:                     # the default pipeline: /v1/pipeline and MCP basalt_pipeline
  perceive:
    documents: [ocr, apple-vision]     # the first available reads pdf and image attachments
    audio: [asr, speech-analyzer]
  decide:
    cascade: [jev, chat]               # up to four stages, cloud and local in any order, the last one local
    escalate_below: 0.6
    abstain_below: 0.75                # questions below this confidence go to the answer model
  answer:
    routes:                            # the first matching rule wins
      - when: { decide: { route: code }, cost_left: true }
        model: frontier
        fallback: chat
      - when: { prompt_tokens: { over: 8000 } }
        model: big
      - model: chat                    # the last rule has no when and names a local model

pipelines:                    # more pipelines, served at /v1/pipelines/<name>
  triage:
    decide: { cascade: [chat] }
    answer: chat
```

### Modèles
- **`source` :** un dépôt Hugging Face, un chemin local (`./`, `/` ou `~`), `cloud:openrouter/<model>`, ou `cloud:gemini/<model>` pour les modèles de décision Google tels que `gemini-decision-1` (cascades de décision uniquement, clé généralement dans `GEMINI_API_KEY`).
- **`keep`** s'applique aux modèles locaux :
  - `resident` est la valeur par défaut et reste chargé ;
  - `on-demand` se charge lors de la première requête nécessitant le modèle et se décharge après `idle_unload`. Les durées s'écrivent sous la forme `30s`, `5m` ou `1h`, et la valeur par défaut est de 120 s ;
  - `sequential` ne charge un module de perception que pour son étape. Dans le pipeline par défaut, il peut également décharger le modèle de chat autour du module de perception si les deux ne tiennent pas ensemble en mémoire.
- **Modèles cloud :**
  - Indiquez le nom de la variable d'environnement contenant la clé avec `api_key_env`. Une clé écrite en clair dans le fichier est refusée, tout comme l'interpolation `${…}`.
  - `budget: { usd_per_day }` plafonne les dépenses d'un modèle.
  - `data: no-perceived` est la valeur par défaut : le texte lu depuis des documents ou de l'audio n'est jamais envoyé à un modèle cloud.
  - `data: media`, uniquement pour les modèles de décision `cloud:gemini`, envoie les fichiers joints du pipeline eux-mêmes (PDF, images, audio au format WAV) ainsi que leur texte perçu à ce modèle pour la décision. Ne l'utilisez que pour des documents autorisés à quitter ce Mac.

### Mémoire et plan
`compose plan` additionne les octets résidents de chaque modèle local. Il compte un modèle `on-demand` comme résident, car il peut être chargé à tout moment.

- `memory: resident` exige que la somme tienne dans l'ensemble de travail (*working set*).
- `memory: sequential` ne nécessite que la plus grande étape plus 2 GB.
- `memory: auto` choisit `resident` lorsque cela tient en mémoire.

Un système qui ne tient pas en mémoire est refusé avec les chiffres à l'appui avant que quoi que ce soit ne soit chargé. Un modèle qui n'est pas téléchargé est signalé comme étant de taille inconnue.

### Décision (`decide`)
- **Ordre :** `cascade` répertorie les modèles de décision dans l'ordre. Chaque étape répond aux questions pour lesquelles son niveau de confiance est suffisant (au moins `escalate_below`) et transmet le reste à l'étape suivante.
- **Étapes dans n'importe quel ordre :** jusqu'à quatre étapes, cloud et locales dans n'importe quel ordre. Par exemple, un petit juge local d'abord, un modèle cloud ensuite et un modèle local plus grand en dernier : `[laya, jev, chat]`.
  - La dernière étape est locale.
  - Chaque étape locale est maintenue en tant que moteur de décision : le modèle de chat, lu en tant que `self`, ou un modèle de décision tel qu'un bundle Laya Core AI ou un bundle OpenJev. Un modèle de décision local dédié peut définir `keep: on-demand` (et `idle_unload`) afin de se charger lors de la première décision qui l'atteint et de se décharger en cas d'inactivité.
  - Chaque pipeline nomme ses propres étapes. Un modèle de décision nommé par plusieurs pipelines n'est chargé qu'une seule fois.
  - Un modèle de décision ne fait que décider ; seul le modèle de chat répond et décide à la fois.
- **Calibration :** un modèle de décision par lecture (*readout*) récupère ses températures depuis un fichier `basalt_readout.json` dans son dossier (`{"temperature": …, "noul_temperature": …, "noul_bias": …}`), s'il existe. Sinon, il utilise les valeurs connues pour son nom, et à défaut aucune. #448 prévoit d'ajuster ces fichiers par rapport à Gemini Decision 1.
- **Tailles :** `compose plan` dimensionne un bundle à tête OpenJev comme son modèle de base plus son adaptateur, car le LoRA est appliqué sans fusion sur la base telle qu'elle est stockée, 4 bits inclus. Le plan propre à `serve` fait de même.
  - Mesuré sur OpenJev-0.6B : 100 états étiquetés, 300 questions, M4 Pro, commit 066da0a. La référence est le chemin float32 fusionné, qui correspond à PyTorch à 1e-4 près (#406).

    | Base, LoRA non fusionné | Précision | Même meilleure réponse que float32 | Mémoire active | Médiane par état |
    |---|---|---|---|---|
    | float32 fusionné (référence) | 176/300 | — | 2,273 MB | 185 ms |
    | bf16, tel que nommé par le bundle (par défaut) | 175/300 | 288/300 | 1,141 MB | 181 ms |
    | 8-bit | 176/300 | 279/300 | 608 MB | 184 ms |
    | 4-bit | 149/300 | 215/300 | 324 MB | 185 ms |

  - Une base 4 bits coûte 9 points à ce modèle 0.6B, tandis que le 8 bits conserve sa précision pour un quart de la mémoire. `serve` et `compose` chargent la base nommée par le bundle, de sorte qu'une base quantifiée n'est accessible que directement via `JevEvaluator`.
  - Relancez ce test avec `BASALT_E2E=1 BASALT_E2E_JEV_MODEL=<bundle> BASALT_E2E_JEV_BASE=<bf16 base> BASALT_E2E_JEV_BASE_8BIT=… BASALT_E2E_JEV_BASE_4BIT=… swift test --filter JevRealModelTests/unmergedArms`.
- **Un juge 9B aux côtés de Gemma 4 12B sur 24 GB :** mesuré avec `openjev/OpenJev-Flash-9B-MLX` (8-bit, 9.5 GB) et `openjev/OpenJev-Flash-9B-MLX-4bit` (5.9 GB). Gemma 4 12B (`gemma-4-12B-it-qat-4bit`) répond aux escalades. Le jeu comprend 60 états étiquetés, 180 décisions, sur un M4 Pro 24 GB au commit 8294bf7.
  - Utilisez le juge 4-bit. Aux côtés de Gemma, il occupe 15.3 GB actifs (pic à 15.9 GB) sur 24 GB.
  - Le juge 8-bit occupe environ 21 GB avec le modèle de chat et une tête 0.6B. Il décide très bien seul (92.8% sans abstention, 1.85 s), mais dès que Gemma décode les escalades, le cache KV pousse le Mac dans le swap. Une branche de deux minutes a alors tourné pendant plus de 24 minutes avant d'être arrêtée.

    | Décideur aux côtés de Gemma 4 12B (juge 4-bit) | Précision | Envoyé au LLM | Tokens LLM économisés | Décision p50 |
    |---|---|---|---|---|
    | LLM uniquement | 97.8% | 180 | — | — |
    | OpenJev-Flash-9B 4-bit, sans abstention | 96.7% | 0 | 100% | 2.0 s |
    | OpenJev-Flash-9B 4-bit, abstention sous 0.6 | 98.9% | 20 | 88.7% | 2.2 s |
    | OpenJev-Flash-9B 4-bit, abstention sous 0.75 | 98.9% | 41 | 76.7% | 2.4 s |
    | OpenJev-Flash-9B 4-bit, abstention sous 0.9 | 97.8% | 74 | 58.5% | 2.5 s |
    | lecture de Gemma lui-même (`self`), abstention sous 0.75 | 99.4% | 1 | 99.4% | 1.9 s |
    | tête OpenJev-0.6B (exécution juge 8-bit), abstention sous 0.75 | 82.8% | 86 | 48% | 0.26 s |

  - Décider pendant que Gemma décode une longue réponse fait chuter le décodage de 21 à 3 tok/s (−86%). Le temps réel est de 341 s contre 202 s en exécution séquentielle. Les décideurs GPU restent donc sérialisés entre les générations.
  - Relancez ce test avec `BASALT_E2E=1 BASALT_E2E_BUDGET_CHAT_MODEL=<chat> BASALT_E2E_BUDGET_FLASH=<judge dir> swift test --filter SystemOneBudgetBenchmarkTests`.
- **Ce que reçoit le cloud :** les requêtes contenant des documents ou de l'audio ignorent les étapes cloud et ne passent que par les étapes locales, dans leur ordre.
- **Seuils :** les questions dont la confiance est inférieure à `abstain_below`, ou dont l'écart entre les deux meilleures options est inférieur à `min_margin`, vont au modèle de réponse.
- **Porte épistémique (version 2) :** `epistemic` classifie `(support, refutation, needs_tool)` avant de décoder un seul token de réponse : `enabled`, `support_threshold`, `refutation_threshold`, `tool_threshold`, `on_no_evidence` (`abstain` ou `escalate`), `on_conflict` (`report` ou `escalate`), et le champ optionnel `abstain_message`.

### Routes de réponse
- **Correspondance :** le `when` d'une règle correspond lorsque toutes ses conditions sont vérifiées. Conditions :
  - `modality` : documents, audio ou texte ;
  - `decide` : identifiant de question vers étiquette, issu de System One ;
  - `prompt_tokens: { over: N }` : estimé à 4 octets par token ;
  - `escalated` : l'étape de décision a escaladé, ou une question a été transmise au modèle de réponse ;
  - `cost_left` : le modèle de la règle dispose encore de budget.
- **Ordre des essais :** d'abord le `model` de la règle, puis son `fallback`, puis le modèle local par défaut du pipeline.
- **Omissions :** un modèle cloud est ignoré pour le texte perçu, en l'absence de budget, après un refus de clé, ou en cas de limitation de débit (*rate limit*). Chaque omission est nommée dans le reçu.

### Speak (`#493`, `#525`)
Un pipeline peut déclarer une étape optionnelle `speak` pour synthétiser de l'audio (`Kokoro-82M`, `Qwen3-TTS`, `Fun-CosyVoice3`, `Chatterbox`, `Zyphra/ZONOS3-Mini-hf`) avec repli déterministe GPU → Neural Engine → voix système :

```yaml
pipeline:
  answer: chat
  speak:
    chain: [tts_zonos, tts_ane, apple-speech]
    voice: Savannah
    language: en_us
    reference_audio: ./speaker.wav
    on_gpu_busy: fallback_ane
    sentence_streaming: true
    format: wav
    exaggeration: 0.5
    max_ttfa_ms: 80
```

- **`speak` :** accepte un nom unique (`speak: apple-speech`), une liste de repli (`speak: [tts_zonos, tts_ane, apple-speech]`), ou le mappage complet.
- **`chain` :** modèles TTS locaux ordonnés se terminant optionnellement par le synthétiseur système sans poids `apple-speech`.
- **`voice` :** identifiant de voix par défaut (`Savannah`, `alloy`, `af_heart`, ou un identifiant `AVSpeechSynthesisVoice`).
- **`language` :** code de langue ou de locale par défaut (`en_us`, `ru`, `de`, `cmn`, etc.).
- **`reference_audio` :** extrait audio local de référence optionnel (`.wav` / `.pcm` / `.caf` / `.aiff` / `.flac`) pour le clonage vocal zero-shot.
- **`on_gpu_busy` :** `keep`, `fallback_ane` ou `fallback_apple_speech` lorsque Metal est occupé à décoder.
- **`sentence_streaming` :** diffuse en full-duplex des segments de phrases vers le TTS à mesure que les tokens arrivent.
- **`format` :** conteneur de sortie ou encodage d'échantillons (`wav`, `pcm`, `pcm_24k`, `mp3`, `opus`, `aac`, `flac`).
- **`exaggeration` :** contrôle d'expressivité prosodique dans `0.0...2.0` lorsqu'il est pris en charge.
- **`max_ttfa_ms` :** budget maximal de latence jusqu'au premier audio avant de basculer sur la chaîne de repli.

### Imagine (`#517`, `#539`)
Un pipeline peut déclarer une étape optionnelle `imagine` pour synthétiser ou éditer des images sur Apple Silicon à travers jusqu'à quatre étapes explicites (réécrivain `upsample_via` -> encodeur texte/vision `share_encoder` -> peintre DiT `chain` -> décodeur VAE latent `decoder_via`), ainsi que le saut d'étapes TeaCache et le jugement VLM best-of-N :

```yaml
pipeline:
  answer: chat
  imagine:
    chain: [z_turbo, sana_fast]
    upsample_via: chat
    share_encoder: encoder
    decoder_via: vae
    steps: 8
    size: 1024x1024
    format: png
    teacache_threshold: 0.15
    preview_every_steps: 2
    candidates: 3
    judge_via: vlm_judge
    on_gpu_busy: fallback
```

- **`imagine` :** accepte un nom de générateur unique (`imagine: z_turbo`), une liste de repli (`imagine: [z_turbo, sana_fast]`), ou le mappage complet.
- **`chain` :** synthétiseurs d'images locaux ordonnés (`FLUX.2/3-klein`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`, `Ming-Image`, `Bonsai-4B`, `Step1X-Edit`, `SonicDiffusion`).
- **`upsample_via` :** emplacement LLM/VLM local réécrivain utilisé pour développer des prompts concis en descriptions de scène détaillées avant la synthèse (utilise la convention de prompt complète de description de scène de BFL lors du ciblage de `FLUX.3`).
- **`share_encoder` :** emplacement LLM/VLM local (tel que `Qwen3-VL-4B-Instruct` en 4-bit, 8-bit ou BF16) dont les embeddings de conditionnement pilotent le synthétiseur d'images.
- **`decoder_via` :** emplacement optionnel de décodeur VAE latent local (tel que le décodeur distillé `black-forest-labs/FLUX.2-small-decoder` ou le décodeur complet `unsloth/FLUX.2-VAE` / `Felldude/FLUX.2-HDR-VAE`).
- **`steps` :** nombre d'étapes de débruitage par défaut dans `1...100`.
- **`size` :** résolution de sortie par défaut `<width>x<height>` (multiples de 16, par ex. `1024x1024` ou `2048x2048`).
- **`format` :** format d'image de sortie (`png`, `webp` ou `jpeg`).
- **`teacache_threshold` :** seuil résiduel L1 (`>= 0`) pour le saut d'étapes internes TeaCache.
- **`preview_every_steps` :** émet des images d'aperçu TAEF intermédiaires toutes les N étapes pendant le streaming SSE (`>= 0`).
- **`candidates` :** nombre d'images candidates avec seed (`1...64`) synthétisées pour la sélection best-of-N.
- **`judge_via` :** emplacement VLM/LLM local qui évalue `candidates` par rapport au prompt et renvoie l'artefact gagnant.
- **`on_gpu_busy` :** `queue` ou `fallback` lorsque Metal est occupé à décoder.

Préréglages `FLUX.3` en 4 étapes fournis (`basalt compose up <preset>`) : `flux3-16gb-fast`, `flux3-24gb-balanced`, `flux3-36gb-moe-vision`, `flux3-64gb-studio-base` et `flux3-128gb-reference`.

### Version 2 : variables et profils
`version: 2` ajoute deux blocs afin qu'un seul fichier serve plusieurs Mac. Un fichier version 1 continue de fonctionner ; l'utilisation de l'un de ces blocs dans un tel fichier est refusée avec l'erreur `"needs version: 2"`.

```yaml
version: 2
name: desk
variables:
  chat_repo: mlx-community/gemma-4-12B-it-qat-4bit
profiles:
  small:
    when: { memory_gb: { under: 20 } }
    variables: { chat_repo: mlx-community/gemma-4-E4B-it-4bit }
  big:
    when: { memory_gb: { over: 48 }, chip: Max }
models:
  chat:  { source: "${chat_repo}" }
  large: { source: lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly, keep: on-demand, profiles: [big] }
```

- **`variables` :** noms en minuscules avec des valeurs simples sur une ligne. `${name}` est remplacé partout dans le fichier. Un `${…}` non déclaré est refusé, de sorte que les variables d'environnement n'atteignent toujours jamais le fichier ; les secrets proviennent uniquement de `api_key_env`.
- **`profiles` :** un profil est activé lorsque son `when` est vérifié sur ce Mac : `memory_gb` avec `under` et `over` (mémoire unifiée en GB), et `chip` (une sous-chaîne insensible à la casse du nom de la puce, comme `M4 Pro` ou `Max`). Un profil sans `when` n'est activé que lorsqu'il est nommé avec `--profile`.
  - Les `variables` d'un profil activé remplacent celles du fichier.
  - Un modèle ou un pipeline nommé avec `profiles: [...]` n'existe que tant que l'un d'eux est activé.
- **Surcharges :** `--set name=value` remplace une variable déclarée et `--profile name` active exactement les profils nommés, pour `config`, `plan` et `up`. `compose up` les transmet au serveur, et la mise à jour différentielle (*up-as-diff*) les envoie avec le fichier.
- **Chemins :** un chemin de modèle `./` ou `../` est relatif au dossier du fichier compose.
- **`basalt compose config -f <file>`** affiche le fichier tel qu'il est résolu sur ce Mac, avec les profils activés. **`basalt compose schema`** affiche le schéma JSON du format pour les éditeurs.

### Section expérimentale et MCP (`#485`)
Un fichier compose peut déclarer des sections de niveau supérieur `experimental` et `mcp` (toutes deux prises en charge en version 1 et 2) au lieu de passer de longues listes d'options après `--` :

```yaml
experimental:
  zero_copy_body: true
  smallm_kernel: true
  speech_analyzer: true
  image_fallback: ocr
  ane_vision_tower: ./tower-bundle
  telemetry_confidence_every: 4
  adaptive_thinking_budget: true
  samplers: true
  mtp_draft_head: true
  expert_streaming: true
  expert_cache_gb: 2.5
  kv_head_split: 0.5            # or quest_top_k: 64 (mutually exclusive)

mcp:
  enabled: true
  rest: true
  research_tools: false
  enable: [surgery, intervene]
  disable: [switch_model]
  image_dirs: [./images]
```

- **`experimental` :** se traduit vers les options `serve` correspondantes (`zero_copy_body`, `smallm_kernel`, `speech_analyzer`, `image_fallback`, `ane_vision_tower`, `telemetry_confidence_every`, `adaptive_thinking_budget`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k`), validées avec les numéros de ligne YAML exacts (`telemetry_confidence_every` >= 0, `image_fallback` `ocr`, `ane_vision_tower` contenant `metadata.json` et `tower.aimodel`, `expert_cache_gb` > 0 avec `expert_streaming: true`, `kv_head_split` dans `0.05...1`, `quest_top_k` dans `1...4096`). Les chemins relatifs (`./`, `../`, `~`) sont résolus par rapport au dossier du fichier compose.
- **`mcp` :** configure le transport stdio MCP (`enabled`, `rest`, `research_tools`, `enable`, `disable`, `image_dirs`) à l'aide des mêmes sélecteurs de groupes et d'outils que `--mcp-enable` / `--mcp-disable` (`serving`, `gemma`, `media`, `harness`, `xray`, `surgery`, `research`, `all`, ou des noms d'outils individuels ; `disable` l'emporte toujours).
- **Rechargement à chaud vs redémarrage :** `speech_analyzer`, `image_fallback`, `telemetry_confidence_every` et `adaptive_thinking_budget` se mettent à jour à chaud lors de `compose up` (`POST /v1/compose/apply`, affiché sous `experimental:`). Les options liées au moteur et au transport (`zero_copy_body`, `smallm_kernel`, `ane_vision_tower`, `samplers`, `mtp_draft_head`, `expert_streaming`, `expert_cache_gb`, `kv_head_split`, `quest_top_k` et `mcp`) signalent `needs a restart` dans `compose diff` et sont refusées lors d'un `apply` à chaud.

### Benchmarks et soumissions aux classements (version 2, `#492`)
Un fichier compose `version: 2` peut déclarer un bloc de niveau supérieur `benchmarks` contenant des valeurs partagées `defaults` et des suites nommées `suites` (soit sous `suites:`, soit comme clés de suite abrégées directement sous `benchmarks:`) :

```yaml
benchmarks:
  defaults:
    seed: 42
    temperature: 0.0
    gpu_lock: true
    output_dir: ./bench-out
  suites:
    smoke:
      - adapter: basalt:throughput
        target: model:chat
        params: { iterations: 2, max_tokens: 64 }
        gates: { min_decode_tps: 15, max_peak_gb: 20, max_p50_ms: 5000 }
    frontier:
      profiles: [big]
      budget: { usd_max: 5.0, max_minutes: 30 }
      tasks:
        - adapter: arc-agi-2
          dataset: ./arc-eval.json
          target: pipeline
          gates: { min_accuracy: 0.25, max_ppl: 12.0, max_ece: 0.08 }
          submit:
            target: arc-prize
            api_key_env: ARC_PRIZE_API_KEY
            endpoint: https://arcprize.org/api/v1/submissions
            subset: public-eval
            split: test
```

- **`defaults` :** champs optionnels `seed`, `temperature`, `gpu_lock` et `output_dir`.
- **`suites` :** chaque suite est soit une liste abrégée de tâches (`smoke: [...]`), soit `{ profiles, budget: { usd_max, max_minutes }, tasks: [...] }`.
- **`tasks` :** chaque tâche nomme un `adapter` (`basalt:throughput`, `basalt:perplexity`, `basalt:system-one`, `basalt:passkey`, `basalt:mom`, `basalt:harness`, `arc-agi`, `arc-agi-2`, `terminal-bench`, `swe-bench`, `lm-eval`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `evalplus`), un `dataset` optionnel, un `target` (`pipeline`, `pipeline:<name>` ou `model:<slot>`), des paramètres scalaires `params`, des seuils de régression `gates` (`min_decode_tps`, `max_peak_gb`, `min_accuracy`, `max_p50_ms`, `max_ppl`, `max_ece`) et un bloc `submit` optionnel (`target`, `api_key_env`, `endpoint`, `subset`, `split`).
- **Commandes :** `basalt compose plan` affiche les suites résolues ; `basalt compose bench -f basalt.yaml --suite smoke --submit-dry-run` exécute la suite et exporte des bundles de classement vérifiés ; `basalt bench list|compare|submit` inspecte les adaptateurs, compare les artefacts et exporte des bundles à blanc sur l'ensemble des dix cibles (`arc-prize`, `terminal-bench`, `swe-bench`, `hf-leaderboard`, `livecodebench`, `bfcl`, `inspect-ai`, `tau-bench`, `gaia`, `weave-otel`).

### Échecs cloud et dépenses
- Une clé refusée (HTTP 401/403) désactive ce modèle cloud jusqu'au redémarrage du serveur, avec une ligne de journal l'indiquant.
- Une limitation de débit (429) le met en pause pendant `Retry-After`, ou 60 s si l'en-tête est absent.
- Pendant qu'un modèle cloud est désactivé ou en pause, l'étape suivante ou le modèle de repli répond.
- Les dépenses du jour par modèle et au total sont conservées dans `~/.basalt/cloud-spend.json`, afin qu'un redémarrage ne réinitialise pas un plafond. Les journées sont en UTC.

## Exécuter des systèmes
- **`compose pull` :** télécharge chaque modèle local du fichier dans le cache Hugging Face et vérifie chaque fichier, en affichant la progression par dixièmes. Après un `pull`, `compose plan` dispose des tailles exactes et `compose up` charge sans attendre de téléchargements.
- **Tailles dans `compose plan` :** un modèle présent dans le cache est mesuré sur disque. Un modèle non téléchargé est dimensionné à partir de la liste des fichiers de l'API Hugging Face. Si ni l'un ni l'autre ne connaît la taille, `plan` refuse et nomme le modèle : exécutez `compose pull`, ou passez `--assume-size <GB>` pour le comptabiliser à cette taille. `plan` ne compte plus un modèle inconnu comme 0 octet.
- **`compose diff` :** montre ce que `compose up` modifierait sur le système en cours d'exécution portant le même nom, sans le modifier (`POST /v1/compose/diff`). Il répertorie les chargements, déchargements, clés et budgets, un redémarrage nécessaire, ou un refus : une clé manquante, ou un système qui ne tient pas en mémoire. Il quitte avec un code non nul en cas de redémarrage requis ou de refus.
- **Événements :** `GET /v1/compose/events` est un flux Server-Sent Events décrivant l'activité du système. Les types d'événements sont `loading`, `ready`, `registered` (à la demande), `failed` (sonde de démarrage), `applied`, `refused` et `up`. Un nouveau client reçoit d'abord les 100 derniers événements. `compose up --wait` les affiche pendant son attente.
- **`compose up` sur un système en cours d'exécution :** si un système portant le même `name` est déjà actif, `compose up` lui envoie le fichier (`POST /v1/compose/apply`) et applique la différence.
  - Tant que le système est encore en cours de chargement ou applique un autre changement, il répond 503. `compose up` attend et renvoie le fichier, jusqu'à `--wait-timeout` secondes (1800 par défaut).
  - Avec `--wait`, `compose up` attend ensuite que `GET /v1/health/ready` réponde 200, affiche les modèles attendus, et quitte avec un code non nul si le système n'est pas prêt dans le délai imparti.
  - Il charge les modèles ajoutés, recharge ceux modifiés et décharge ceux supprimés.
  - Il permute les pipelines entre les requêtes.
  - Il vérifie d'abord le nouveau système par rapport à la mémoire de ce Mac, comme le fait `compose plan`, en comptant les modèles de chaque pipeline. Un système qui ne tient pas est refusé, et rien ne change.
  - Lorsque le nouveau système tient en mémoire mais que les anciens et nouveaux modèles ne tiennent pas côte à côte, il décharge les modèles qu'il remplace avant de charger leurs successeurs. Une requête qui a besoin de l'un d'eux dans cet intervalle attend le chargement ou échoue. Si un chargement échoue ensuite, il recharge les anciens modèles.
  - Il affiche `load`, `reload`, `unload` et `keep` par modèle, `budget` pour chaque budget cloud modifié (`total` pour celui du fichier) et `key` pour chaque modèle cloud dont `api_key_env` a changé.
  - Le budget ou la variable de clé d'un modèle cloud change sans rechargement. Les nouvelles limites s'appliquent immédiatement, et un modèle désactivé par une clé refusée est réactivé lorsque sa variable de clé change.
  - Un modèle de chat différent est chargé à la place du modèle actif et affiché comme `chat`. Les requêtes attendent pendant le chargement. Si une étape ultérieure de l'application échoue, l'ancien modèle de chat est rechargé.
  - Un nouveau modèle de décision local d'un pipeline est chargé comme tout modèle ajouté. Celui du pipeline par défaut est également affiché comme `decide`. Les pipelines et `/v1/systemone` l'utilisent dès la requête suivante. Un modèle qui commence ou cesse de décider est rechargé dans son nouveau rôle.
- **Changements nécessitant un redémarrage :** un mode mémoire différent, un module de perception séquentiel, les options de démarrage moteur/HTTP sous `experimental`, ou des changements sous `mcp`. Exécutez `compose down` puis `compose up`.
- **`compose ps` :** répertorie chaque système en cours d'exécution (`GET /v1/compose`) avec ses pipelines et, par modèle :
  - son état, parmi `loading`, `ready`, `unloading`, `unloaded`, `idle-unloaded` (à la demande, non chargé actuellement), `decision` (détenu par le moteur de décision), `cloud`, `no-key`, `over-budget`, `rate-limited`, `refused` et `failed` ;
  - pour un modèle à la demande, quand il a été utilisé pour la dernière fois et quand il se déchargera ;
  - pour un modèle cloud, ses dépenses du jour par rapport à son plafond, et un refus le cas échéant.
- **Santé :** `GET /v1/health/live` répond 200 tant que le serveur répond en HTTP. `GET /v1/health/ready` répond 200 dès que chaque modèle peut servir. Avant cela, il répond 503 et répertorie les modèles encore en attente, comme `chat (loading)`. Un modèle à la demande inactif, un modèle cloud hors budget et un modèle cloud limité en débit comptent comme aptes à servir ; la cascade ou le routeur les contourne.
- **Clés cloud :** `compose up` refuse de démarrer lorsque la variable `api_key_env` d'un modèle cloud n'est pas définie, et nomme la variable. La mise à jour différentielle vérifie plutôt l'environnement du serveur en cours d'exécution.
- **`compose down` :** envoie `SIGINT` au serveur. Celui-ci termine les requêtes en cours et décharge les modèles. Si le processus tourne encore après 30 s, `down` l'indique et quitte avec un code non nul.
- **Emplacement de l'état :** le PID, le port et l'heure de démarrage du processus du système actif se trouvent dans `~/.basalt/compose/<name>.json`. Un enregistrement dont le PID appartient désormais à un autre processus, parce que le PID a été réutilisé, est considéré comme non actif.
- **Clé API :** lorsque le serveur possède une clé API, définissez `BASALT_API_KEY` pour `ps` et la mise à jour différentielle.

## Adaptateurs de domaine (`version: 2`)
Un fichier `version: 2` peut déclarer `adapters:` entraînés sur les préréglages Hugging Face NVIDIA Nemotron (`nemotron:post-v2`, `nemotron:post-v1`, `nemotron:code`, `nemotron:math`, `nemotron:crossthink`, `nemotron:personas`), des jeux de données `hf:<org>/<repo>`, ou des fichiers `.jsonl` locaux, et router les requêtes entre eux sur un seul modèle `base` résident sans recharger les poids :

```yaml
adapters:
  code_oft:
    base: chat
    method: oft              # lora | phm | oft
    rank: 16
    scale: 10.0
    hypercomplex_dim: 8
    factor_rank: 1
    shared_basis: true
    block_size: 16
    parameterization: cayley # cayley | matrixExp
    block_shared: false
    layers: [0, 1, 2, 3]
    output: ./adapters/code_oft
    iterations: 200
    batch_size: 2
    learning_rate: 0.0001
    max_seq_len: 2048
    checkpoint_every: 50
    corpus:
      source: nemotron:code
      subset: default
      split: split_0
      max_rows: 512
      min_pass_rate: 0.9
      include_reasoning: true
      seed: 42
    gate:
      max_kl: 0.2
      max_ppl_rise: 0.05
      held_out: ["Explain what a hash table is in two sentences."]
```

Exécutez `basalt compose train -f basalt.yaml` (avec en option `--adapter`, `--force`, `--resume`) pour entraîner les adaptateurs déclarés sur disque, puis sélectionnez un `adapter:` dans `pipeline.answer` ou dans n'importe quelle entrée de `pipeline.answer.routes`.

## Reçus
Chaque réponse de `/v1/pipeline` et chaque enregistrement de journal de session comportent un reçu. Par étape, il indique le modèle, les millisecondes et les tokens. Les étapes de décision indiquent aussi leur coût et les éventuelles étapes cloud ignorées. Les étapes de réponse indiquent la règle déclenchée (`route`), le coût et les modèles ignorés. `swap_ms` est le temps passé à charger les modèles à la demande et séquentiels.

## Profils matériels intégrés par palier (`#526`)

`basalt compose` et `Basalt.app` (`Compose Profiles`, `⌥⌘C`, ainsi que **Settings → Engine → Bundled Profile**) embarquent **84 profils calibrés selon le matériel** répartis sur 7 paliers de mémoire unifiée Apple Silicon (`16gb`, `24gb`, `32-36gb` [alias `32gb` et `36gb`], `48gb`, `64gb`, `128gb`, `256gb` [alias `192gb` et `236gb`]) × 12 archétypes de charge de travail :

| Archétype | Principal | Objectif |
|---|---|---|
| `daily-chat` | ★ | Chat quotidien, perception de documents Apple Vision, entrée vocale locale |
| `coding-agent` | ★ | Noyau spéculatif small-M, routage spécialisé pour le code, outils MCP |
| `voice-zonos` | ★ | TTS neuronal Zyphra `ZONOS3-Mini` 44.1 kHz + repli ANE `Kokoro-82M` |
| `multimodal-desk` | ★ | OCR, perception audio, triage System One, chat et voix ZONOS3 |
| `reasoning-lab` | | Budgets de réflexion adaptatifs + cascade System One pour les maths et la logique |
| `document-ocr` | | OCR neuronal (`PaddleOCR-VL-1.5`) + porte épistémique de Belnap à quatre valeurs |
| `voice-duplex` | | Transcription full-duplex et synthèse en streaming sous les 80 ms de TTFA |
| `creative-studio` | | Synthèse d'images MLX native (`Z-Image-Turbo` / `FLUX`) avec TeaCache |
| `rag-knowledge` | | Recherche ancrée en contexte long avec filtrage des conflits et d'abstention |
| `hybrid-cloud` | | Chat local respectueux de la confidentialité avec escalade cloud plafonnée à `$1/day` |
| `moe-frontier` | | MoE épars (`Flint 40B-A2.5B`, `Nemotron-3.5`, `Qwen3.8`) + streaming d'experts |
| `research-xray` | | Model X-Ray, chirurgie des poids et benchmarks rapides `basalt:throughput` |

Au lancement, `Basalt.app` détecte la mémoire unifiée physique (`ComposeMemoryTier.detect`) et épingle les 4 profils principaux étoilés pour ce Mac sous **Recommended for this Mac**, tout en gardant les 84 profils interrogeables et sélectionnables. Chaque profil s'exécute à l'identique depuis la CLI (`basalt compose plan 24gb-voice-zonos`, `basalt compose up 36gb-moe-frontier`, `basalt compose up 236gb-multimodal-desk`).

## Exemples
- `docs/examples/compose/any-mac.yaml` : un fichier version 2 pour les Mac de 16, 24 et 64 GB. Le modèle de chat provient d'un profil choisi selon la mémoire, et le profil 64 GB ajoute un modèle 27B à la demande ainsi qu'un pipeline `long`.
- `docs/examples/compose/mac-16gb.yaml` : Ornith 9B local, OCR à la demande, prompts longs vers Claude Haiku dans la limite de $0.50 par jour.
- `docs/examples/compose/mac-24gb.yaml` : Gemma 4 12B, OCR à la demande, code vers Haiku, un second pipeline `support`, $1 par jour au total.
- `docs/examples/compose/mac-64gb.yaml` : Gemma 4 26B-A4B avec Qwen3.8-27B et Nemotron Transcribe à la demande, aucune réponse cloud.
- `docs/examples/compose/document-qa-cloud.yaml`, `mom.yaml` et `too-big.yaml` : les fichiers à l'origine des mesures ci-dessous.

D'autres mesures figurent dans #412.

## Mesures

```yaml
version: 1
name: document-qa-cloud
models:
  chat:  { source: mlx-community/gemma-4-12B-it-qat-4bit }
  ocr:   { source: mlx-community/PaddleOCR-VL-1.5-8bit, keep: on-demand, idle_unload: 30s }
  jev:   { source: "cloud:openrouter/~typesafe/jev-latest", api_key_env: OPENROUTER_API_KEY }
pipeline:
  perceive: { documents: [ocr, apple-vision] }
  decide:   { cascade: [jev, chat], escalate_below: 0.6 }
  answer: chat
```

```bash
basalt compose plan -f docs/examples/compose/document-qa-cloud.yaml
basalt compose up -f docs/examples/compose/document-qa-cloud.yaml -- --port 8080
basalt compose up voice-agent            # bundled: document-qa, voice-agent
```

```yaml
pipeline:
  decide: { cascade: [jev, chat], escalate_below: 0.6 }
  answer:
    routes:
      - when: { decide: { route: code }, cost_left: true }
        model: haiku          # cloud:openrouter/anthropic/claude-haiku-5.5, budget: { usd_per_day: 0.5 }
        fallback: chat
      - model: chat
```

Charge mixte de 55 requêtes via `/v1/pipeline` (30 décisions, 10 code, 10 chat, 5 documents), M4 Pro, build release (`scripts/compose_mom_bench.swift`, `docs/examples/compose/mom.yaml`) :

| Configuration | Étiquettes exactes | Cloud pour 1000 requêtes | p50 | p95 | Temps d'étape locale |
|---|---|---|---|---|---|
| local uniquement (Gemma 4 12B décide et répond) | 55/55 | $0 | 0.92 s | 10.0 s | 176 s |
| cloud d'abord (Jev décide, Haiku répond à tout texte) | 55/55 | $0.046 | 1.25 s | 4.8 s | 24 s |
| MoM (Jev décide, code vers Haiku tant que le budget dure, le reste vers Gemma) | 55/55 | $0.035 | 1.21 s | 4.8 s | 53 s |

Mesuré avec `compose up` sur un M4 Pro de 24 GB, build release :

| Fichier | Requête | Reçu |
|---|---|---|
| document-qa-cloud | document (première) | OCR chargé à la demande (489 ms), décidé localement, répondu : 6.1 s |
| | document (à nouveau) | OCR déjà chargé (0 ms) : 5.0 s |
| | après 40 s d'inactivité | OCR déchargé, empreinte 12 → 11 GB |
| | question saisie | décidé par Jev dans le cloud : 0.6 s |
| | document | OCR rechargé (267 ms) : 5.3 s |
| voice-agent (intégré) | audio | chat déchargé autour de Nemotron Transcribe (swap 6.9 s) : 17.3 s |
| too-big (exemple) | `compose plan` | refusé : une étape nécessite 20.7 GB plus 2 GB face à un working set de 19.1 GB |


## Extensions du schéma `version: "2.2"`

Compose `version: "2.2"` (compatible avec `version: 2`) ajoute la configuration déclarative pour les LLM à diffusion par blocs, les modèles brouillons spéculatifs `DFlash`, la parole-à-parole en duplex, les encodeurs de texte VLM/DiT partagés, la mise à l'échelle `SeedVR2` en 1 étape, le DiT musical `ACE-Step`, les DiT vidéo VAE 3D causaux, l'entraînement par préférences par segments et les clusters multi-Mac Thunderbolt/LAN :

- Par modèle : `speculative` (`mode`, `drafter`, `block_size`, `verify`) et `diffusion_text` (`dual_kv_cache`, `confidence_threshold`, `contrastive_alpha`).
- Extensions `imagine` : `share_text_encoder_with`, `kv_cache_references`, `rgba`, et `upscale` (`model`, `steps`, `target_resolution`).
- Racine `voice` (`mode`, `sts_model`, `denoise`, `turn_detection` avec `model` et `diarization`, `tts_fallback`, `aligner`).
- Racine `music` (`model`, `steps`, `duration_seconds`) et `video` (`model`, `vae_tiling`, `sta_sparse`, `fps`, `frames`).
- Racine `train` (`method`, `segments`, `ema_decay`) et `cluster` (`topology`, `peers`, `compression`).
- Routage, mise en cache et gouvernance d'entreprise (`v0.12`) : routage à 5 composants `BusinessOrchestratorCore`, cache sémantique `LANTERN` (`SemanticCacheCore` / `SemanticCacheActor`, `#625`), porte d'approbation `HITL` (`/v1/mcp/approvals`) + `SymbolicKnowledgeGraph` du LAAS-CNRS (`#626`), et `TieredKVHierarchyCore` L0/L1/L2 sensible à l'UMA + `AdaptiveHeterogeneousPlanner` (`#627`).
