# API REST

> Languages: [English](../../rest-api.md) · [Русский](../ru/rest-api.md) · [中文](../zh/rest-api.md) · [Español](../es/rest-api.md) · **Français** · [日本語](../ja/rest-api.md) · [Deutsch](../de/rest-api.md)

URL de base : `http://127.0.0.1:8080` — **aucune authentification requise**.

## Sécurité

Sur loopback, tout processus local peut appeler l'API sans clé, mais une page web dans votre navigateur ne le peut pas :

- **Host.** Lié à loopback, le serveur ne répond qu'aux requêtes dont le `Host` est `localhost`, `127.0.0.1` ou `[::1]` (n'importe quel port), tout autre reçoit 403 ; cela empêche le DNS rebinding, où le domaine d'un attaquant se résout à nouveau vers 127.0.0.1 et sa page lit vos réponses. Lié à une adresse spécifique, cette adresse est également acceptée. Lié à `0.0.0.0`/`::`, tout `Host` est accepté : la clé API y est obligatoire.
- **Origin.** Une requête comportant un en-tête `Origin` est refusée (403) à moins qu'il ne s'agisse de la propre origine loopback du serveur (`http(s)://localhost|127.0.0.1|[::1]` sur le port d'écoute du serveur ; l'absence de port signifie 80 pour http, 443 pour https) ou qu'elle soit listée avec `--allow-origin` / `allowed_origins` (`scheme://host[:port]`, sans chemin, sans barre oblique finale, sans `*` ; les ports par défaut correspondent, donc `https://app.example:443` équivaut à `https://app.example`). Les autres ports locaux ne sont pas approuvés : une page sur un serveur de développement (`localhost:3000`) ou Jupyter pourrait sinon charger des modèles et lancer des générations, une telle interface nécessite donc `--allow-origin http://localhost:3000`. Les clients qui n'envoient pas d'`Origin` (curl, SDK, Claude Code, Codex) ne sont pas affectés.
- **Content-Type.** Chaque `POST` doit envoyer `Content-Type: application/json` (un paramètre `charset` est accepté), sinon 415. `curl -d` sans `-H "Content-Type: application/json"` envoie un type de formulaire et est refusé.
- **Identifiants de modèles.** Les valeurs `model` REST sont des identifiants de dépôts Hugging Face : `/v1/models/load` n'accepte que `org/name` ; dans le champ `model` d'une requête de génération, une forme de chemin (commençant par `/`, `.` ou `~`, ou contenant `..`, y compris dissimulé dans `claude-local/..--x` ; les espaces entourants sont ignorés) provoque une erreur 400, tandis que toute autre valeur qui n'est pas un identifiant de dépôt (`basalt`, `qwen/qwen3-8b-instruct:free`) est un alias : le modèle chargé répond et rien n'est basculé. Les chemins locaux restent disponibles pour `basalt serve --model-path` et MCP. Un chargement en échec laisse le modèle actuel chargé, et les corps d'erreur ne contiennent jamais de chemins du système de fichiers (ils figurent dans le journal du serveur).

## Gestion des modèles

```bash
# List all locally cached models (OpenAI-compatible format)
curl http://localhost:8080/v1/models

# Currently loaded model + readiness status
curl http://localhost:8080/v1/models/current

# Hot-swap to a different model at runtime (blocks until loaded)
curl -s http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/gemma-4-31b-it-4bit"}'
```

`GET /v1/models` renvoie des identifiants préfixés par `claude-local/` afin que le sélecteur de modèles automatique de Claude Code les ajoute. Le champ `display_name` porte l'identifiant de dépôt HuggingFace d'origine.

## Génération brute

```bash
curl -s http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum entanglement.", "maxTokens": 256}'
```

| Champ | Type | Par défaut | Description |
|---|---|---|---|
| `prompt` | string | requis | Texte d'entrée |
| `maxTokens` | int | 8192 | Nombre maximal de jetons à générer |
| `temperature` | float | modèle | Température d'échantillonnage (0–2) ; si omise, prend `generation_config.json` du modèle, sinon 0.7 |
| `topP` | float | modèle | Échantillonnage par noyau (*nucleus sampling*, 0–1) ; si omis, prend `generation_config.json`, sinon 0.9 |
| `topK` | int | modèle | Conserve les k jetons les plus probables ; si omis, prend `--top-k`, puis `generation_config.json`, sinon désactivé |
| `presencePenalty` | float | désactivé | −2…2, soustrait à chaque jeton figurant parmi les 1024 derniers jetons générés (l'invite n'est pas pénalisée) |
| `repetitionPenalty` | float | `--repetition-penalty` | (0, 2], pénalité multiplicative sur les jetons récents |
| `thinkingBudget` | int | désactivé | Jetons de raisonnement avant que le serveur force la fin de la réflexion (Qwen : "Considering the limited time by the user…" + `</think>`) pour que la réponse tienne dans `maxTokens` ; abaissé pour tenir dans `maxTokens` moins la transition et une réserve de réponse de min(512, maxTokens/4) ; un `maxTokens` laissant moins de 64 jetons après la réserve renvoie 400 ; ignoré lorsque la réflexion est désactivée, que le modèle n'a pas de protocole de raisonnement ou qu'il s'agit d'un modèle non-Qwen dont le protocole ne déclare aucune transition |
| `minP` | float | serveur | Min-P 0–1 : élimine les jetons sous `minP × max p`. Remplace `--min-p` et le profil du modèle ; `0` le désactive |
| `dynamicTemperature` | bool | false | Température dynamique basée sur l'entropie (EDT) : par jeton `T = T0·0.8^(1.5/H) + 0.5·R`, bornée à 0.05–2 (une requête plus froide garde sa propre température comme plancher ; `top_p`/`min_p` coupent la distribution d'origine), où `H` est l'entropie de la distribution du jeton suivant et `R ∈ [0,1]` un indice de répétition. Quasi glouton sur les jetons sûrs, plus chaud lorsqu'une boucle démarre. Sans effet à `temperature: 0` |
| `speculativeLookup` | bool | `--speculative-lookup` | Décodage spéculatif par recherche dans l'invite (*prompt-lookup*) pour cette requête : jusqu'à 16 jetons copiés depuis une correspondance de n-grammes dans l'invite ou la sortie sont vérifiés en une seule passe avant ; après un brouillon rejeté, il suspend l'ébauche pendant 1, 2, 4… jusqu'à 64 jetons. Avec `serve --speculative-recycling`, lorsqu'aucun n-gramme ne correspond, il ébauche jusqu'à 6 jetons à partir des propres successeurs argmax du modèle enregistrés deux fois de suite (Token Recycling, #38), avec sa propre pause. Modèles à attention seule et hybrides récurrents denses (Qwen3.5/Ornith, Nemotron-H : un brouillon rejeté restaure l'état récurrent et rejoue les jetons acceptés), modèles à fenêtre glissante (Gemma 4 : chaque tour de vérification est préparé à part, un brouillon rejeté n'écrase donc jamais l'anneau de la fenêtre), sans `--kv-bits` et en entrée texte uniquement ; les hybrides récurrents avec routage d'experts (LFM2.5 MoE) sont ignorés. Sur Gemma 4 26B-A4B (MoE), la recherche apporte un gain de 1.97× sur les modifications de code et coûte 3–6% sur la prose et le raisonnement |

`/v1/chat/completions`, `/v1/responses` et `/v1/messages` acceptent les mêmes deux champs sous les noms `min_p` et `dynamic_temperature` ; `gemma_generate` côté MCP sous `minP` et `dynamicTemperature`. Les points de terminaison OpenAI acceptent `top_k`, `presence_penalty`, `repetition_penalty` et `thinking_budget` ; `/v1/messages` accepte `top_k` et `thinking: {"type": "enabled", "budget_tokens": N}` comme budget de réflexion (1 ≤ N < `max_tokens`, sinon 400). Tous trois acceptent `speculative_lookup` (une valeur non booléenne renvoie 400) ; `gemma_generate` côté MCP accepte `speculativeLookup` (une valeur non booléenne renvoie -32602). Les statistiques d'utilisation rapportent `completion_tokens_details.reasoning_tokens` (chat ; diffusé en streaming avec `stream_options.include_usage`) et `output_tokens_details.reasoning_tokens` (Responses) : `0` lorsque le modèle n'a pas raisonné, et la transition forcée compte comme du raisonnement.

## Compatible OpenAI

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

Prend en charge le streaming SSE, les invites système et les conversations multi-tours. L'envoi d'un identifiant de modèle HuggingFace dans le champ `"model"` déclenche automatiquement un échange à chaud (*hot-swap*).

La réflexion d'un modèle de raisonnement revient dans `message.reasoning` (flux : `delta.reasoning`), le nom OpenRouter / Ollama, et ce même texte également dans `reasoning_content`, le nom lu par les clients vLLM, DeepSeek, llama.cpp et LM Studio. Les tours d'assistant renvoyés avec l'un ou l'autre champ sont acceptés ; la réflexion elle-même n'est pas réinjectée dans l'invite.

## Compatible Anthropic

```bash
curl -s http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "basalt",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Implémente la séquence complète d'événements SSE Anthropic (`message_start`, `content_block_start`, `content_block_delta`, `message_stop`). Accepte les formats de contenu en chaîne simple et en tableau de blocs.

Les blocs `thinking` portent un champ `signature` vide : basalt ne dispose d'aucune clé pour les signer. Ils font l'aller-retour sans problème avec basalt, mais la véritable API Anthropic rejette une conversation qui les rejoue ; supprimez-les donc avant d'y envoyer le même historique.

**Sortie contrainte.** `response_format` contraint la réponse, après tout bloc de raisonnement, à l'aide de XGrammar :
- `{"type": "json_object"}` ou `{"type": "json_schema", "json_schema": {"schema": …}}` ;
- `{"type": "regex", "regex": "[A-Z]{3}-\\d{4}"}` ;
- `{"type": "grammar", "grammar": "root ::= …"}`, dialecte EBNF/GBNF de XGrammar avec règle racine `root` ;
- `{"type": "structural_tag", "format": …}`, JSON structural-tag de XGrammar (`sequence`, `const_string`, `regex`, `json_schema`, `tag`, …).

Les champs de requête `guided_regex` et `guided_grammar` de vLLM fonctionnent également, un seul à la fois. L'outil MCP `generate` accepte `regex` ou `grammar`. Une source dépassant 64 KB, ou qu'XGrammar ne parvient pas à compiler, provoque une erreur 400 qui nomme le format.

**Pré-remplissage (*prefill*).** Une conversation qui se termine par un message d'assistant prolonge ce message au lieu d'y répondre dans un nouveau tour, sur `/v1/messages`, `/v1/chat/completions` et `/v1/responses`, comme sur l'API Anthropic, `continue_final_message` de vLLM et llama.cpp. Les tours précédents sont rendus jusqu'à l'invite de génération avec la réflexion désactivée, le texte de l'assistant suit sans être fermé, et la réponse ne contient que ce que le modèle écrit à la suite : pré-remplissez `{"name":` et la réponse est ` "Whiskers", "age": 3}`. Avec `response_format` ou un outil forcé, le pré-remplissage constitue le début du document JSON (un pré-remplissage avec lequel le schéma ne peut pas démarrer renvoie 400), et les arguments d'un outil forcé l'incluent. Un pré-remplissage qui ouvre un bloc de réflexion (`<think>…`) conserve la réflexion demandée et se poursuit comme raisonnement. Plusieurs messages d'assistant consécutifs en fin de liste sont concaténés en un seul pré-remplissage ; un message vide est ignoré ; un message d'assistant contenant des appels d'outils reste un tour terminé. Une invite contenant des images ne peut pas être prolongée (400). Les jetons de contrôle dans le pré-remplissage (`<|im_end|>`, `<|im_start|>`, …) sont neutralisés comme le reste de la conversation, afin qu'un pré-remplissage ne puisse pas forger un tour ; seuls les marqueurs de réflexion qui ouvrent et ferment son raisonnement restent réels, et sur un modèle de discussion qui ouvre déjà `<think>` lui-même (Ornith, Qwen3.5+), le `<think>` initial du pré-remplissage est supprimé au lieu d'être doublé. Avec `--cloud-fallback`, une requête pré-remplie reste toujours locale, car tous les fournisseurs ne prolongent pas un message d'assistant final. Sur Nemotron Nano v2, qui désactive la réflexion avec une ligne `/no_think` dans le tour système, une requête pré-remplie génère un tour système différent d'une requête avec réflexion activée sur le même historique, les deux ne partagent donc que le cache d'invite situé avant ce tour.

## Entrée image

Un modèle de vision (Gemma 4, Qwen3.5-VL, … — tout modèle chargé par MLXVLM) accepte des images sur `/v1/chat/completions`, `/v1/responses`, `/v1/messages` et l'outil MCP `generate` :

- **OpenAI :** une partie de contenu `{"type": "image_url", "image_url": {"url": "data:image/png;base64,…"}}` (`/v1/responses` : `{"type": "input_image", "image_url": "data:…"}`).
- **Anthropic :** un bloc de contenu `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "…"}}`.
- **MCP :** `images: [{"data": "<base64>", "mime": "image/png"}]`, ou `[{"path": "/abs/file.png"}]` pour un fichier situé dans un répertoire indiqué via `serve --image-dir <dir>` (répétable). Le chemin est vérifié après résolution des liens symboliques ; sans `--image-dir`, tout chemin est refusé.

PNG, JPEG, WebP et HEIC, au plus 8 images par requête, 20 MB d'octets encodés par image et 32 MB pour toutes les images d'une requête ; chaque dépassement renvoie une erreur 400 qui indique la limite. Les URL distantes (`http(s)://` dans `image_url`, `source.type: "url"`) sont refusées avec 400 — basalt ne les télécharge pas ; envoyez l'image sous forme d'URI `data:`. Une image envoyée à un modèle qui n'est pas un modèle de vision renvoie 400 (`<model> does not take images`), jamais une suppression silencieuse, et une requête contenant des images ne part jamais vers `--cloud-fallback`. Le cache d'invite est indexé par le SHA-256 de chaque image : les mêmes images dans le même ordre réutilisent le préfixe en cache, des images différentes ne réutilisent rien. Les journaux de session enregistrent les condensats et les tailles en octets, jamais les octets d'image. Les images dans les résultats d'outils sont encore présentées au modèle sous la forme `[image omitted]`.

## Streaming SSE (API native)

```bash
curl -N http://localhost:8080/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "maxTokens": 512}'
```

Chaque événement est une ligne `data: {...}` ; le flux se termine par `data: [DONE]`. Une défaillance en cours de flux arrive sous la forme `event: error` avec `data: {"error": ...}`.

## Télémétrie et Prometheus

- **`GET /api/v1/stats`** renvoie l'instantané de télémétrie en JSON.
  - Il contient également `windows` : les fenêtres glissantes sur 1 minute, 5 minutes et 1 heure avec requêtes, jetons, temps jusqu'au premier jeton p50/p95 et tok/s, taux de succès du cache d'invite et taux d'acceptation du brouillon.
  - Il contient également `decisions` : par modèle de décision, requêtes, réponses, abstentions, escalades, coût cloud, latence et réponses par décile de confiance.
- Avec `--experimental-expert-streaming`, il contient également `experts` : les succès, défauts, évictions et octets lus du cache d'experts en streaming, et par couche MoE les experts routés depuis le chargement (nombre, distincts, les huit plus utilisés et leur part).
- Avec `--telemetry-confidence-every N`, chaque Nième génération enregistre pour chaque jeton la probabilité du meilleur jeton, l'entropie et la marge entre les deux meilleurs. Les requêtes qui le demandent sont toujours enregistrées. `confidence` compte ensuite les générations par décile de la plus faible probabilité du meilleur jeton, de la probabilité moyenne et de la plus petite marge, ainsi que par classe d'un demi-nat de l'entropie moyenne. Chaque jeton mesuré coûte une synchronisation GPU, la valeur par défaut 0 ne mesure donc que les requêtes qui le demandent.
- **`GET /api/v1/metrics`** renvoie les mêmes données au format texte OpenMetrics pour Prometheus ou Grafana.
  - Les noms de métriques commencent par `basalt_`. La fenêtre et le quantile sont des étiquettes (*labels*), tout comme le modèle de décision.
  - Ce point de terminaison passe par la clé API du serveur comme tous les autres : fournissez le jeton bearer au collecteur (*scraper*).

```yaml
scrape_configs:
  - job_name: basalt
    metrics_path: /api/v1/metrics
    authorization: { credentials_file: /etc/prometheus/basalt.token }
    static_configs: [{ targets: ["127.0.0.1:8080"] }]
```

La télémétrie ne contient que des compteurs, des temps et des histogrammes : aucun texte d'invite ni de réponse (#450).

## Points de terminaison multimodaux, Convertisseur, X-Ray et Gouvernance (`v0.12`)

- **Studio Vidéo (`/v1/videos/generations`, `/v1/videos/{id}`)** — Génération phasée sur UMA 24 GB (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video MoE`, `Wan 2.2`) renvoyant un `.mp4` multiplexé via `AVAssetWriter` (`#608`–`#611`).
- **Audio et Musique (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/music/generations`)** — `ZONOS3-Mini`, `Kokoro-82M`, codecs duplex Mimi/SNAC `LFM2.5-Audio`, `SmartTurn`/`Sortformer`, `MossFormer2`, `Soprano-80M`, `Qwen3ForcedAligner`, `ACE-Step 1.5` (`#604`, `#605`).
- **Images (`/v1/images/generations`, `/v1/images/edits`)** — `FLUX.3`, `Z-Image-Turbo`, `Iris-3B`, `SANA-1.5`, mise à l'échelle `SeedVR2`, prise en charge RGBA (`#542`, `#605`).
- **Convertisseur Universel (`/v1/models/convert`, `/v1/models/convert/jobs/{id}`)** — Quantification asynchrone tout-en-tout (`MACaQ`, `FlatQuant`, `LIRQuant`, `SINQ`, `QTIP`, `E8P`, `PV-Tuning`, `QAT STE` + `QuantizationAuditCore`, `#546`, `#623`).
- **Attribution des jetons et risque d'hallucination (`POST /v1/xray/tokens`)** — Logprobs par jeton, entropie top-k, attribution couche/tête/MoE et signaux de risque d'hallucination (`#613`, `#614`).
- **Gouvernance HITL (`/v1/mcp/approvals`)** — File d'approbation humaine (*human-in-the-loop*) pour les appels d'outils MCP à fort impact, associée à `HITLGovernanceCore` et `SymbolicKnowledgeGraph` (`#626`).

## Interface Swagger

Rendez-vous sur `http://localhost:8080/swagger` pour la documentation interactive de l'API.
