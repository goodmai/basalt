# Radiographie du modèle (Model X-ray) et outils de recherche (#328)

> Languages: [English](../../research-tools.md) · [Русский](../ru/research-tools.md) · [中文](../zh/research-tools.md) · [Español](../es/research-tools.md) · **Français** · [日本語](../ja/research-tools.md) · [Deutsch](../de/research-tools.md)

Le serveur MCP de Basalt peut montrer ce que fait le modèle chargé et exécuter de petites expériences réversibles sur celui-ci. Les outils d'inspection en lecture seule sont toujours listés. Les outils d'expérimentation ne sont listés et acceptés que lorsque le serveur démarre avec `basalt serve --mcp --research-tools`.

Chaque appel passe par la porte de génération (generation gate) en priorité d'arrière-plan, afin que les requêtes utilisateur passent en premier. Les réponses sont bornées : des statistiques et de petites tranches (slices), jamais un tenseur entier ni une matrice d'attention complète.

## Outils

Les exemples correspondent à l'objet `arguments` d'une requête MCP `tools/call`.

### Inspection en lecture seule

| Outil | Ce qu'il renvoie | Exemples d'arguments |
|---|---|---|
| `model_describe` | Type de modèle, architecture, nature par couche (attention complète, fenêtre glissante, attention linéaire, mamba, conv), têtes, têtes KV, taille de tête, fenêtre, experts, classe de cache KV, quantification (y compris les manifestes de modules `MACaQ` / `FlatQuant` / `LIRQuant` en plus de la précision de base, `#630`), nombre de paramètres et octets | `{}` |
| `model_tensors` | Noms des paramètres avec forme (shape), dtype et indication s'ils sont quantifiés ; `filter` est une sous-chaîne du nom | `{"filter": "layers.3.mlp"}` |
| `model_tensor` | Moyenne, écart-type, min, max, L2 et les centiles p1/p5/p50/p95/p99 de \|x\| pour un tenseur (déquantifié), plus jusqu'à `maxValues` valeurs d'une tranche. Avec `"spectrum": true`, renvoie également le spectre de la matrice, ou d'un expert d'une pile 3D sélectionné par la tranche : variance du bruit de Marchenko–Pastur, bords du bulk et masse des valeurs propres au-dessus du bulk, rang de Gavish–Donoho, exposant de queue en loi de puissance `alpha` et durée de calcul en secondes (matrice de Gram WᵀW du plus petit côté sur le GPU, `eigvalsh` sur le CPU ; plus petit côté ≤ 8192 ; uniquement avec `serve --research-tools`, car cela immobilise le modèle pendant plusieurs secondes) | `{"name": "model.layers.3.input_layernorm.weight", "slice": [[0, 64]], "maxValues": 64}`, `{"name": "model.layers.3.mlp.down_proj.weight", "maxValues": 0, "spectrum": true}` |
| `kv_inspect` | L'arbre du cache de prompt par espace de noms : nœuds, feuilles, jetons, octets, instantanés récurrents, épingles (pins), dernière utilisation ; budget et évictions | `{"namespace": "claude-code"}` |
| `trace_run` | Une exécution de capture sur une invite : norme de l'état caché par couche, top-k du logit lens, positions les plus suivies et entropie par tête à une position donnée, choix du routeur MoE | `{"prompt": "The capital of France is", "layers": [0, 12, 23], "topK": 5}` |
| `telemetry_query` | Agrégats en direct du processus serveur (#450) : totaux ; fenêtres glissantes avec TTFT et tok/s p50/p95, taux de succès du cache de prompt et taux d'acceptation du brouillon (draft) ; compteurs, coût et déciles de confiance par modèle de décision ; compteurs du cache d'experts en streaming et routage par couche ; déciles de confiance par jeton ; santé et mémoire. Uniquement des compteurs et des mesures temporelles | `{"sections": ["windows", "decisions"]}` |

### Expériences (`--research-tools`)

| Outil | Ce qu'il fait | Exemples d'arguments |
|---|---|---|
| `intervene` | Ajoute un vecteur de pilotage (steering vector) au flux résiduel après une couche, ou met à zéro une tête d'attention ou un expert. Avec `when`, s'applique uniquement sur les jetons sélectionnés par un déclencheur (ablation conditionnelle). S'applique uniquement aux appels de recherche (`experiment_compare`, `trace_run`). | `{"kind": "ablate_head", "layer": 10, "head": 3}`, `{"kind": "ablate_expert", "layer": 20, "expert": 7, "when": {"kind": "projection", "layer": 14, "vector": [0.01, …], "threshold": 2, "window": 16}}` |
| `weights_delta` | Ajoute `values × scale` à une tranche d'un tenseur flottant en mémoire (une valeur unique est diffusée par broadcast). Modifie le modèle en direct, donc les requêtes utilisateur le voient également (voir ci-dessous). | `{"tensor": "model.layers.10.post_attention_layernorm.weight", "slice": [[0, 128]], "values": [0.05], "scale": 1}` |
| `overlays_list` | Surcouches (overlays) actives : identifiant, description, heure de création et d'expiration | `{}` |
| `overlay_undo` | Annule une surcouche par identifiant ; un delta de poids est restauré bit à bit | `{"id": 4}` |
| `experiment_compare` | Comparaison gloutonne (greedy) avant/après sur un ensemble d'invites avec les surcouches données : les deux textes, divergence KL des logits, les deux perplexités | `{"prompts": ["2 + 2 =", "Name a prime above 100."], "overlays": [3, 4], "maxTokens": 32}` |
| `adapter_apply` | Applique un adaptateur LoRA au modèle en direct : un répertoire d'adaptateur mlx-lm (`adapter_config.json` + `adapters.safetensors`), ou une mise à jour de rang 1 `scale × (x·input) output` d'une couche linéaire, couches quantifiées incluses. Marque le modèle comme modifié à la manière d'un delta de poids ; `overlay_undo` restaure les couches bit à bit. Avec `gate`, la porte de qualité s'exécute après l'application : en cas de succès, l'adaptateur reste sans expiration ; en cas d'échec, il est annulé et le rapport revient avec `overlay: null`. Sans `gate`, il expire après 30 minutes. | `{"module": "model.layers.10.mlp.down_proj", "input": [0.01, …], "output": [0.02, …], "scale": 1, "gate": {"heldOut": ["The capital of France is"], "maxKL": 0.5}}` |
| `gate_check` | Exécute la porte de qualité sur le modèle tel qu'il est actuellement par rapport au modèle d'origine (chaque adaptateur actif et chaque delta de poids étant levés pour la référence) : chaque contrôle avec sa valeur, sa limite et son verdict, ainsi que les lignes de comparaison sur l'ensemble retenu (held-out) | `{"heldOut": ["The capital of France is", "2 + 2 ="], "maxKL": 0.5, "maxPerplexityRise": 0.1, "target": [{"prompt": "17 + 25 =", "expected": "42"}]}` |
| `tensor_pack` | Recompresse le poids d'une couche linéaire en mémoire : `affine` (`bits` 2, 3, 4, 5, 6 ou 8, `groupSize` 32, 64 ou 128), `mxfp4`, `nf4:32`, `nf4:64` ou `nf4:128` (le dictionnaire NormalFloat à 16 niveaux de QLoRA avec une échelle absmax fp16 par groupe, conservé comme `QuantizedLinear` nf4 natif servi par les noyaux nf4 du fork), ou `ternary` (échelle absmean·{-1, 0, 1} par groupe de 64). Une couche quantifiée est d'abord déquantifiée. La couche est permutée en direct, comme un delta de poids (voir ci-dessous) | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "affine", "bits": 2, "groupSize": 64}` |
| `tensor_unpack` | Remplace en direct une couche linéaire quantifiée par une couche dense `float16` ou `bfloat16` (par défaut `bfloat16`) contenant son poids déquantifié | `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "bfloat16"}` |
| `train_experts` | Auto-entraînement (#332 mode 3, longue durée) : collecte des réponses à des tâches vérifiables, ne conserve que celles qui sont vérifiées, entraîne un LoRA, l'enregistre, rapporte la perte et la précision sur l'ensemble retenu avant/après, puis l'applique via la porte de qualité. Voir [Auto-entraînement](#auto-entraînement-train_experts) | `{"count": 200, "seed": 1, "layers": [20, 21, 22, 23], "iterations": 200, "heldOut": ["Summarise the plot of Hamlet in two sentences."]}` |
| `distill` | Distillation enseignant → élève (longue durée) : un enseignant derrière un point de terminaison compatible OpenAI ou Anthropic répond aux invites (avec cache et budget), un LoRA est entraîné uniquement sur ses complétions, la NLL de l'enseignant sur l'ensemble retenu est rapportée avant et après, et l'adaptateur est appliqué via la porte de qualité. Découpé en tranches temporelles : le modèle est verrouillé un pas d'optimiseur à la fois, de sorte que `serve` continue de répondre et qu'une requête n'attend au plus qu'un pas ; une seule distillation s'exécute par serveur. Les exécutions hors ligne écrivent l'adaptateur et l'état de l'optimiseur dans `<output>/checkpoint` tous les `checkpointEvery` pas (50 par défaut, 0 pour désactiver) ; `resume: true` avec le même `output` reprend depuis ce point, et est refusé si le modèle, les lignes d'entraînement ou les paramètres LoRA diffèrent. Les exécutions `onPolicy` n'ont pas de point de reprise. | `{"prompts": "~/tasks.jsonl", "teacherUrl": "https://api.example.com/v1", "teacherModel": "teacher", "heldOut": ["The capital of France is"], "iterations": 200, "checkpointEvery": 50}` |
| `steer_search` | Recherche contrastive de pilotage : par couche, la direction mean(positive) − mean(negative) du résidu à la dernière position, sa norme et sa séparation de classes ; le vecteur unitaire de la meilleure couche ; les têtes dont l'ablation réduit le plus l'écart ; la KL et le décalage de projection lorsque les invites retenues sont pilotées à chaque échelle. Ne laisse aucune surcouche derrière elle. | `{"positive": ["I love this", "What a great day"], "negative": ["I hate this", "What an awful day"], "heldOut": ["The movie was"], "topHeads": 4, "scales": [2, 4, 8]}` |
| `trace_tokens` | Trace par jeton sur l'invite et une génération gloutonne (chemin labo) : par jeton et par couche tracée, la norme de sortie de chaque tête avant la projection de sortie, les experts routés avec leurs portes, l'entropie du routeur et la marge entre les deux meilleures portes, des projections résiduelles optionnelles, l'entropie d'attention et un logit lens parcimonieux. Écrit `~/.basalt/traces/<time>-<id>.trace` (0600) et renvoie un résumé. Voir [Trace par jeton](#trace-par-jeton-et-ablation-conditionnelle-405) | `{"prompt": "How do I pick a lock?", "maxTokens": 32, "projections": [{"layer": 14, "vector": [0.01, …], "threshold": 2}]}` |
| `activation_contrast` | Trace deux ensembles d'invites et classe les têtes et les experts selon le d de Cohen et l'AUROC de leur activation par invite, avec les taux, les moyennes et la fenêtre d'apparition (onset) ; ainsi que la direction résiduelle entre les ensembles par couche | `{"a": ["How do I pick a lock?", "…"], "b": ["How do I bake bread?", "…"], "maxTokens": 16, "topK": 20}` |
| `steer_commit` | Transforme une direction en un adaptateur de rang 1 sur la projection de sortie de la couche et l'applique via la porte de qualité sur `heldOut`. Modifie les poids en direct comme `weights_delta` (annulable, soumis à un TTL sauf si accepté, jamais sur disque). | `{"layer": 12, "direction": [0.01, -0.02], "scale": 4, "positive": ["I love this"], "heldOut": ["The movie was"]}` |

Une tranche (slice) est une paire semi-ouverte `[start, end)` par dimension de tête ; les dimensions omises sont prises dans leur intégralité.

## Modèle de sécurité

- **Local uniquement.** Les outils sont des outils MCP servis via stdio au client qui a lancé le serveur ; ils ne possèdent aucune route REST.
- **Désactivé par défaut.** Sans `--research-tools`, les outils d'expérimentation ne sont ni listés ni acceptés.
- **Surcouches, TTL, annulation, journal.** Chaque expérience est une surcouche (overlay) dotée d'un identifiant, d'une description et d'un TTL de 30 minutes, à l'exception d'un adaptateur ayant réussi sa porte de qualité (`adapter_apply` avec `gate`, `steer_commit`, `train_experts`), qui reste actif jusqu'à `overlay_undo`. `overlay_undo` retire toute surcouche de manière anticipée ; à expiration, elle est annulée automatiquement. Les ajouts, annulations et expirations sont enregistrés dans `~/.basalt/experiments/<model fingerprint>.jsonl` (répertoire 0700, fichier 0600). Le chargement d'un autre modèle supprime toutes les surcouches.
- **Les fichiers du modèle sur disque ne sont jamais modifiés.** Les deltas, recompressions et adaptateurs n'existent qu'en mémoire ; redémarrer le serveur ou recharger le modèle les supprime. Les seuls fichiers écrits sont le journal et l'adaptateur que `train_experts` enregistre dans un nouveau répertoire.
- **Les interventions restent cantonnées aux appels de recherche.** Le pilotage et les ablations s'appliquent à `experiment_compare` et `trace_run`, jamais aux requêtes utilisateur.
- **Les deltas de poids sont visibles et isolés des caches.** Un delta modifie les poids utilisés par toutes les requêtes, donc tant qu'un delta est actif :
  - chaque réponse porte `modelModified: true` (y compris une réponse dont la génération a chevauché une application ou une annulation) ;
  - l'instantané de télémétrie (MCP `status`, `GET /api/v1/stats`) liste les deltas actifs sous `activeDeltas` ;
  - la réutilisation du cache de prompt est désactivée, y compris les restaurations depuis le cache disque, car le KV mis en cache a été calculé avec les anciens poids ;
  - les caches de prompt en mémoire sont purgés lors de l'application et de l'annulation, et toute requête qui s'exécutait pendant l'un ou l'autre de ces changements ne stocke pas son cache.
- **Validé à la frontière.** Le tenseur doit exister, la tranche doit être dans les bornes et non vide (au plus 16M éléments), il doit y avoir une valeur par élément ou une valeur unique à diffuser par broadcast, chaque valeur doit être finie et `|scale| ≤ 10`. Un seul delta actif par tenseur.
- **Tenseurs flottants uniquement.** Les deltas modifient les normes, les biais et les plongements (embeddings) ou projections non quantifiés (float16, bfloat16, float32). Tout tenseur d'une couche quantifiée (son poids compressé ainsi que ses échelles et biais) est refusé : un chemin d'écriture quantifié est hors périmètre.

## Porte de qualité et adaptateurs (#332)

`gate_check` et le paramètre `gate` de `adapter_apply` comparent le modèle actuel au modèle d'origine, à l'intérieur d'un appel modèle exclusif qui lève temporairement chaque adaptateur actif et chaque delta de poids pour établir la référence, puis les remet en place. Les contrôles :

- **`probe perplexity`** : la perplexité de la sonde de chargement sur son texte fixe doit rester sous le plafond de la famille ou, si le modèle d'origine est déjà au-dessus, ne pas augmenter de plus de `maxPerplexityRise` par rapport à la valeur d'origine.
- **`probe known answer …`** : chaque réponse connue de la sonde de chargement que le modèle d'origine réussit doit toujours être réussie (valeur 1 ou 0, limite égale au résultat d'origine).
- **`held-out mean KL`** et **`held-out perplexity rise`** : sur chaque invite `heldOut`, la mécanique de `experiment_compare` s'exécute avec le modèle d'origine comme référence ; la divergence KL moyenne des distributions du prochain jeton le long du texte de référence doit être d'au plus `maxKL` (0.5 par défaut) et la hausse relative moyenne de perplexité de ce texte d'au plus `maxPerplexityRise` (0.1 par défaut).
- **`target …`** : la réponse gloutonne de chaque invite cible (modèle de chat d'abord, puis invite brute, 16 jetons) doit contenir `expected`, sans tenir compte de la casse.

Les valeurs non finies sont signalées comme le plus grand flottant fini et entraînent un échec. Adaptateurs :

- Un adaptateur sur répertoire est chargé avec la prise en charge LoRA de mlx-swift-lm ; un seul adaptateur sur répertoire à la fois, et ses tenseurs doivent correspondre aux formes des couches et au rang configuré.
- Un adaptateur de rang 1 nécessite une valeur `input` par dimension d'entrée et une valeur `output` par dimension de sortie de la couche ; la couche doit être une simple couche `Linear` ou `QuantizedLinear` sans adaptateur.
- Les adaptateurs n'enveloppent que des couches `Linear` et `QuantizedLinear` simples, de sorte que l'annulation remet en place les mêmes poids et que les logits correspondent bit à bit au modèle d'origine. Rien n'est écrit sur disque.

## Comment cela mène aux boucles d'apprentissage (#328 couche 3)

Ces outils constituent l'instrument de mesure et le bac à sable pour une couche ultérieure : `distill.collect` rassemble des invites et des réponses de référence, `adapter.train` / `apply` / `evaluate` entraînent et testent un petit adaptateur, et `weights.propose` reçoit une correction proposée de l'extérieur. Chaque candidat s'exécute d'abord ici comme surcouche ou delta, est comparé via `experiment_compare`, et n'est accepté que si la sonde au chargement, les logits de référence, le lanceur de parité et la matrice de fonctionnalités passent toujours. Rien dans cette couche ne rend un changement persistant ; c'est la boucle d'apprentissage qui décide ce qui devient permanent, le cas échéant.

## Compression et décompression de tenseurs (#332)

`tensor_pack` et `tensor_unpack` modifient tous deux une couche linéaire sur place : ils lisent son poids (en déquantifiant une couche quantifiée), le convertissent et permutent le module entier (`Linear` ↔ `QuantizedLinear`), de sorte que les octets changent réellement. La réponse indique le coût de la conversion (valeurs illustratives) :

```json
{"overlay": {"id": 7, "description": "tensor_pack model.layers.10.mlp.down_proj.weight: 4-bit affine g64 → 2-bit affine g64", "createdAt": "…", "expiresAt": "…"},
 "bytesBefore": 9437184, "bytesAfter": 5505024, "rmse": 0.0041, "maxAbsError": 0.031, "relativeError": 0.29}
```

`rmse`, `maxAbsError` et `relativeError` (‖W − Ŵ‖ / ‖W‖) comparent le poids déquantifié de la nouvelle couche aux valeurs que détenait l'ancienne couche. La décompression vers le type flottant propre aux échelles est exacte (les trois valeurs valent 0).

La recompression est une surcouche au même titre qu'un delta de poids : le modèle est marqué comme modifié, la réutilisation du cache de prompt est désactivée, `overlay_undo` (ou le TTL de 30 minutes) remet en place la couche d'origine bit à bit, et rien n'est écrit sur disque. Seules les couches `Linear` et `QuantizedLinear` simples sont compressées ; les piles d'experts MoE, les plongements, les couches enveloppées par un LoRA et toute couche portant un delta ou une autre compression sont refusés.

`experiment_compare` s'exécute sur les poids en direct, de sorte que l'effet d'une recompression correspond à la différence entre une exécution avant et une exécution après :

1. `experiment_compare` `{"prompts": ["2 + 2 =", "Name a prime above 100."], "maxTokens": 32}` : notez `baselineText` et `baselinePerplexity`.
2. `tensor_pack` `{"tensor": "model.layers.10.mlp.down_proj.weight", "format": "ternary"}` : notez l'identifiant de la surcouche et l'erreur.
3. `experiment_compare` avec les mêmes arguments : la variation de `baselineText` et `baselinePerplexity` par rapport à l'étape 1 mesure l'effet de la recompression sur les logits.
4. `overlay_undo` `{"id": 7}`, puis relancer l'étape 1 redonne la réponse de l'étape 1 bit à bit.

## Introspection (`introspect`, #332 mode 1)

Avec `--research-tools`, `introspect` permet au modèle servi de s'étudier lui-même. Étant donné un objectif, le serveur exécute le modèle via le chemin de génération normal en lui proposant les outils de radiographie comme définitions d'outils, exécute chaque appel qu'il effectue via le même chemin de recherche contrôlé que les outils MCP, lui renvoie le résultat et l'interroge de nouveau.

| Argument | Signification |
|---|---|
| `goal` (requis) | Ce qu'il faut découvrir, par ex. `"find which heads attend to the last tool call"` |
| `maxSteps` | Tours de modèle avant que la boucle ne s'arrête, 1–32 (8 par défaut) |
| `allowInterventions` | Proposer également `intervene`, `experiment_compare` et `overlay_undo` (false par défaut ; les outils en lecture seule sont toujours proposés) |
| `resultBudget` | Nombre de caractères de chaque résultat d'outil renvoyé au modèle, 200–20000 (2000 par défaut) ; les résultats plus longs sont tronqués et signalés |

- **Un appel par tour.** La boucle s'arrête sur une réponse sans appel d'outil, après `maxSteps`, ou lorsque le modèle répète un appel déjà effectué (même outil et mêmes arguments, l'ordre des clés étant ignoré) ; `stop` dans le rapport indique la raison.
- **Hypothèses.** Il est demandé au modèle d'écrire chaque hypothèse sous forme de lignes `Hypothesis:` / `Test:` / `Outcome:`. Elles sont analysées avec souplesse (markdown, numérotation, `Result:` à la place de `Outcome:`), et une hypothèse reformulée remplace la précédente tout en conservant son `Test:` ou son `Outcome:` si la reformulation omet cette ligne.
- **Les erreurs sont renvoyées au modèle.** Un appel d'outil mal formé ou en échec devient un résultat `error: …` visible par le modèle ; une génération échouée ou une annulation met fin à la boucle.
- **Les interventions ne survivent jamais à la boucle.** Chaque surcouche créée pendant la boucle est annulée à sa terminaison, y compris en cas d'erreur ou d'annulation ; `undone` en dresse la liste. Si le serveur refuse le nettoyage (occupé), `cleanupFailure` le signale et ces surcouches expirent à leur TTL. `overlay_undo` refuse les surcouches qui existaient avant la boucle.
- **Un tour échoué conserve le rapport.** Si une génération échoue en cours de boucle, les étapes obtenues jusque-là sont retournées avec `stop: "failed"` et l'erreur dans `failure`.
- **Journal.** Chaque étape ajoute une ligne `introspect` (modèle, étape, outil, arguments et début du résultat) dans `~/.basalt/experiments/introspect.jsonl`.

Le rapport contient `goal`, `steps` (le texte du modèle et, pour un appel, son `tool`, ses `arguments` et son `result` tronqué), `hypotheses` (`statement`, `test`, `outcome`), `answer`, `stop`, `failure`, `undone` et `cleanupFailure`.

```json
{"goal": "find which heads attend to the last tool call", "maxSteps": 6, "allowInterventions": true}
```

## Auto-entraînement (`train_experts`)

Un seul appel exécute toute la boucle en priorité d'arrière-plan à travers la porte de génération, en verrouillant le modèle pour l'entraînement lui-même :

1. **Tâches.** `count` + `seed` génèrent des problèmes entiers `+`, `-`, `*` avec des réponses connues (la même graine donne les mêmes problèmes) ; `file` lit à la place des lignes JSONL `{"prompt": "...", "answer": "..."}`. Un cinquième des tâches (au moins une) est mis de côté et n'est jamais utilisé pour l'entraînement.
2. **Collecte.** Le modèle répond à chaque tâche de façon gloutonne via le modèle de chat (réflexion désactivée, 64 jetons). Avec `teacher` (URL de base d'un serveur compatible OpenAI ; `teacherModel` désigne son modèle, `BASALT_TEACHER_API_KEY` est envoyé en jeton bearer uniquement lorsque le schéma, l'hôte et le port de l'enseignant correspondent à `BASALT_TEACHER_URL`, afin qu'un client ne puisse pas envoyer la clé ailleurs), l'enseignant répond également aux tâches d'entraînement.
3. **Vérification.** Une réponse n'est conservée que si son entier final (virgules ignorées) égale une réponse numérique connue, ou si sa dernière ligne, casse et ponctuation environnante ignorées, égale une réponse textuelle (ou se termine par `: <answer>`). Les réponses fausses, vides ou mal formées sont rejetées. **Rien de non vérifié n'est jamais appris :** si aucune réponse n'est vérifiée, le rapport indique `verified: 0`, aucun adaptateur n'est entraîné, rien n'est écrit et le modèle reste intact.
4. **Entraînement.** Un LoRA (`rank`, Adam à `learningRate`, `iterations`) est entraîné avec `LoRATrain` du fork sur les paires invite + réponse vérifiées, rendues telles que le modèle les a vues, sur les projections d'attention et les projections denses ou d'experts partagés de `layers` (toutes par défaut). Le LoRA par expert n'est pas disponible — les couches d'experts `SwitchLinear` du fork n'acceptent pas d'adaptateur — donc `experts` est refusé avec une erreur. Un modèle qui porte déjà un adaptateur est également refusé. L'annulation de l'appel arrête l'entraînement à l'itération suivante et laisse le modèle inchangé.
5. **Sauvegarde.** `adapter_config.json` + `adapters.safetensors` (le format d'adaptateur mlx-lm) sont enregistrés dans `output` ou `~/.basalt/adapters/<model fingerprint>/<timestamp>` (0700). `output` ne doit pas encore exister, et un chemin situé à l'intérieur du répertoire du modèle est refusé : rien n'est écrasé et rien n'est écrit à côté des poids source.
6. **Mesure et application.** La précision sur l'ensemble retenu est mesurée avec l'adaptateur en place, les couches d'entraînement sont retirées de nouveau, et l'adaptateur est appliqué via `applyAdapter` avec la porte de qualité construite à partir de `heldOut`, `maxKL` et `maxPerplexityRise` : il devient une surcouche marquée sans expiration que `overlay_undo` retire, ou est refusé par la porte.

Le rapport liste `collected` (réponses recueillies pour l'entraînement), `verified`, `losses` (perte moyenne d'entraînement par pas de suivi, au plus 20), `accuracyBefore` / `accuracyAfter` sur les tâches retenues, `adapterDirectory` et le `verdict` de la porte.

## Pilotage contrastif (« méditation », #332 mode 2)

Il s'agit d'une recherche dirigée de directions d'activation, et non d'une conscience de soi. Rien ici ne signifie que le modèle réfléchit sur lui-même : basalt passe deux ensembles d'invites dans le modèle, soustrait les moyennes des états cachés et mesure l'effet de cette différence.

`steer_search` exécute chaque invite positive et négative dans une passe avant sondée et conserve le résidu après chaque couche à la dernière position.

- **Direction par couche :** mean(positive) − mean(negative). `norm` est sa longueur ; `separation` est l'écart entre les moyennes de classe des projections sur la direction unitaire (égal à `norm`) divisé par l'écart-type groupé de ces projections. La meilleure couche possède la séparation la plus élevée, et `direction` est son vecteur unitaire (de la taille cachée).
- **Têtes :** chaque tête d'attention de la meilleure couche et de la couche précédente est soumise à une ablation à tour de rôle (l'intervention `ablate_head`), et `effect` mesure de combien l'écart de projection positif-moins-négatif à la meilleure couche diminue. Les têtes des couches ultérieures ne peuvent pas modifier ce résidu et ne sont pas testées. Au plus `topHeads` sont rapportées, par ordre décroissant d'effet. Cela représente une passe avant par tête et par invite : gardez donc des ensembles d'invites réduits sur les grands modèles, ou passez `topHeads: 0`.
- **Essais :** pour chaque échelle, les invites retenues sont pilotées par `direction × scale` après la meilleure couche (l'intervention `steer`, réservée à la recherche). `klDivergence` est la KL moyenne de la distribution du prochain jeton par rapport à l'exécution non pilotée ; `projectionShift` est la variation moyenne de la projection sur `direction` lue après la dernière couche. À la couche pilotée, ce décalage vaut l'échelle par construction ; la dernière couche montre donc quelle part de la poussée subsiste.
- Aucune surcouche n'est enregistrée : les ablations et le pilotage n'existent qu'à l'intérieur de l'appel.

`steer_commit` rend permanente pour la session une direction qui s'est montrée utile, en passant par la porte. Il sélectionne la projection dense de la couche qui écrit dans le résidu (`mlp.down_proj`, sinon `feed_forward.w2`, sinon `self_attn.o_proj` ; une couche MoE sans projection dense est refusée), mesure l'entrée moyenne `m` de ce module à la dernière position sur les invites `positive`, et appelle le chemin adaptateur avec `.rankOne(module: <full module path>, input: m / (m·m), output: direction, scale: scale)`, de sorte que la mise à jour vaut ΔW = scale · direction ⊗ m / (m·m).

L'approximation, énoncée clairement :

- La mise à jour ajoute `scale · (x·m / m·m) · direction` à la sortie de la projection pour chaque jeton. Cela n'égale exactement le vecteur de pilotage que lorsque l'entrée `x` du module a la même projection sur `m` que la moyenne positive ; les autres jetons reçoivent plus, moins ou l'opposé. `steer` ajoute le même vecteur à chaque position.
- Elle n'est exacte dans le résidu que lorsque la sortie de la projection est ajoutée directement au flux résiduel (blocs pre-norm tels que Qwen et LFM). Lorsqu'une normalisation suit la projection (normes post-attention et post-feedforward de Gemma), le vecteur est remis à l'échelle et seule sa direction est approximativement conservée. Lorsque la solution de repli `self_attn.o_proj` est utilisée, le vecteur traverse également le MLP de la couche avant d'atteindre le résidu après la couche.
- L'adaptateur n'est accepté que si la porte passe (`maxKL`, `maxPerplexityRise` sur `heldOut`) ; exécutez d'abord les essais `steer_search` et validez une échelle dont la KL était acceptable.


## Trace par jeton et ablation conditionnelle (#405)

La procédure pour un comportement tel que le refus : tracer → contraster → ablation conditionnelle → comparer.

1. **`trace_tokens`** sur quelques invites qui manifestent le comportement. `ForwardProbe.headOutputs` du fork fournit la sortie d'attention de chaque tête avant la projection de sortie (avant la porte de sortie de Qwen3.5), de sorte que l'attention fusionnée reste active ; `routerTopK` fournit les experts et les portes que la couche combine réellement (biais d'expert, top-k de groupe et mise à l'échelle inclus). Par jeton et par couche tracée, le fichier contient :
   - `heads` : norme L2 de la sortie de chaque tête ;
   - `experts` (entiers) et `gates` ; `routerEntropy` (softmax sur les logits du routeur) et `routerMargin` (meilleure porte moins deuxième porte) ;
   - `projection` : residual · direction après la couche, pour chaque entrée de `projections` ;
   - `attentionEntropy` (optionnel, attention explicite, plus lent) et `lens`/`lensProbabilities` (logit lens top-k optionnel).

   Format de fichier : une ligne d'en-tête JSON (`layout.floats` et `layout.ints` listent les segments `{layer, field, width}` dans l'ordre des colonnes), une ligne JSON par jeton (`position`, `id`, `token`, `generated`, `ints`), puis `tokens × floatsPerToken` valeurs float32 little-endian. Plafonds vérifiés avant l'exécution : invite plus `maxTokens` ≤ 2048 jetons, couches dans les bornes du modèle, au plus 16 projections de la taille cachée sur les couches tracées, et une taille estimée ≤ `maxFileBytes` (≤ 64 MiB ; la taille encodée est vérifiée de nouveau avant l'écriture).
2. **`activation_contrast`** avec l'ensemble `a` (manifeste le comportement) et l'ensemble `b` (ne le manifeste pas). L'activation moyenne par invite de chaque composant (norme de tête ; porte de l'expert lorsqu'il est routé, sinon 0) donne le d de Cohen et l'AUROC ; `rateA`/`rateB` comptent les jetons où la norme d'une tête dépasse sa moyenne groupée ou où l'expert a été routé ; `onsetA`/`onsetB` moyennent les `onsetWindow` jetons d'invite avant le début de la génération. `direction` à `bestLayer` est la direction résiduelle entre les ensembles.
3. **`intervene` avec `when`** : soumet à ablation une tête ou un expert classé (ou pilote avec une échelle négative pour soustraire une direction) uniquement lorsqu'un déclencheur s'active — la projection sur la direction de contraste au-dessus d'un seuil, ou un expert routé avec une porte supérieure à `gate` — optionnellement pendant `window` jetons à partir du premier déclenchement. La couche de l'action doit être postérieure à la couche du déclencheur ; le déclencheur est lu dans la même passe avant (une synchronisation GPU par surcouche conditionnelle et par passe avant), de sorte que la modification s'applique au jeton qui déclenche, aussi bien dans l'invite que dans la génération. Un déclencheur qui ne s'active jamais laisse les logits identiques bit à bit.
4. **`experiment_compare`** avec la surcouche conditionnelle, face à une surcouche statique `ablate_*` du même composant et un adaptateur `steer_commit` : KL, perplexité, textes et `triggerRate` (la part des jetons auxquels la modification conditionnelle s'est appliquée).

Pas encore câblé : la lecture System One comme déclencheur ou étiqueteur, les traces échantillonnées (non gloutonnes), les messages de chat en entrée (appliquez vous-même le gabarit), et un lanceur d'évaluation dédié pour le taux de refus, la précision et la vitesse de décodage.

## Ajouts récents (`v0.12`)

- **Gestion automatique du BOS dans les traces X-Ray brutes (`#630`) :** `trace_run`, `trace_tokens`, `experiment_compare`, `contrast`, `activation_contrast`, `steer_search` et `TokenAttribution` partagent `ModelProbe.encoded(_:_:)` afin que les tokenizers dont `encode(addSpecialTokens: true)` omet `bosToken` (comme Gemma 4) reçoivent tout de même `<bos>` à la position 0, préservant ainsi les couches puits d'attention BOS (`L43–L56`) sans double BOS.
- **Logprobs par jeton, attribution et risque d'hallucination (`#613`, `#614`) :** `POST /v1/xray/tokens` et les routes de génération exposent les logprobs par jeton, l'entropie top-k, l'attribution par couche/tête/MoE et les signaux de risque d'hallucination.
- **Chirurgie mécaniste et attribution d'influence (`#600`, `#622`) :** patch par groupe KV GQA, projection sur l'espace nul cognitif, garde-fou anti-pilotage, décomposition en caractéristiques `MAO-SNMF`, attribution par tête `Score d'influence` (`HeadInfluenceCore`) et routage calibré en température (`CalibratedRouterGate`).
- **Gouvernance HITL (`#626`) :** `HITLGovernanceCore` / `HITLApprovalActor` contrôlent les actions MCP à fort impact via `/v1/mcp/approvals`, couplés au `SymbolicKnowledgeGraph` du LAAS-CNRS.
