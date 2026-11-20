# Intégrations

> Languages: [English](../../integrations.md) · [Русский](../ru/integrations.md) · [中文](../zh/integrations.md) · [Español](../es/integrations.md) · **Français** · [日本語](../ja/integrations.md) · [Deutsch](../de/integrations.md)

## Utiliser depuis vos outils

basalt parle trois protocoles clients en même temps, choisissez donc celui que votre outil utilise déjà.

| Client | Point de terminaison | Configuration |
|---|---|---|
| Tout client OpenAI (SDK Python/JS, Continue, Cline, aider) | `http://localhost:8080/v1` | URL de base + n'importe quelle clé API sur loopback |
| OpenCode | `/v1/chat/completions` | voir [Intégration OpenCode](#intégration-opencode) |
| Claude Code dans le terminal | `/v1/messages` (Anthropic) | voir [Intégration Claude Code](#intégration-claude-code) |
| Claude Desktop, Cursor | MCP stdio | voir [Intégration MCP](#intégration-mcp-claude-desktop--cursor) |

Démarrez un seul serveur pour tous ces clients, avec les options supplémentaires souhaitées :

```bash
basalt serve --rest --model Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed --mtp \
  --embedding-model mlx-community/bge-small-en-v1.5-bf16 \
  --rerank-model mlx-community/Qwen3-Reranker-0.6B-4bit
```

Le SDK Python officiel d'OpenAI fonctionne sans modification (vérifié avec `openai` 3.19.2) :

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="local")
reply = client.chat.completions.create(model="basalt", messages=[{"role": "user", "content": "Hello"}])
vectors = client.embeddings.create(model="basalt", input=["first text", "second text"])
```

Rerank utilise le format Cohere/Jina :

```bash
curl -s http://localhost:8080/v1/rerank -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "documents": ["Paris is the capital of France.", "Berlin is in Germany."], "top_n": 1}'
```

## Intégration Claude Code

### Option A — variables d'environnement (par session de terminal)

```bash
# Start Basalt
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen3.5-4B-4bit --rest

# In another terminal — scope vars only to this claude process
ANTHROPIC_BASE_URL=http://localhost:8080 \
ANTHROPIC_AUTH_TOKEN=local \
claude
```

`ANTHROPIC_AUTH_TOKEN` s'envoie en tant que `Authorization: Bearer local` (et non `x-api-key`), votre véritable `ANTHROPIC_API_KEY` n'est donc jamais modifiée.

### Option B — fonction shell (à ajouter dans `~/.zshrc`)

```bash
function claude-local() {
  ANTHROPIC_BASE_URL=http://localhost:8080          \
  ANTHROPIC_AUTH_TOKEN=local                        \
  ANTHROPIC_DEFAULT_HAIKU_MODEL=mlx-community/gemma-4-e4b-it-4bit    \
  ANTHROPIC_DEFAULT_SONNET_MODEL=mlx-community/Qwen3.5-4B-4bit       \
  ANTHROPIC_DEFAULT_OPUS_MODEL=mlx-community/gemma-4-31b-it-4bit     \
  claude "$@"
}

claude-local                   # sonnet alias → Qwen 4B
claude-local --model haiku     # haiku alias → Gemma 4B (fastest)
claude-local --model opus      # opus alias → Gemma 31B (strongest)
```

### Option C — lanceur `./basalt-claude`

Démarre le serveur et Claude Code en une seule commande (voir [le lanceur dans Installation](install.md#lanceur-en-une-commande--basalt-claude)).

### Appel d'outils

Les outils de Claude Code fonctionnent sur `/v1/messages` : les `tools` qu'il envoie atteignent le modèle de discussion (*chat template*) du modèle, les appels du modèle reviennent sous forme de blocs `tool_use` (diffusés en streaming via `content_block_start` / `input_json_delta` / `content_block_stop`, `stop_reason: "tool_use"`), et les blocs `tool_use` / `tool_result` de l'historique sont rendus dans l'invite, afin que le modèle voie ce que ses outils ont retourné. `/v1/chat/completions` procède de même au format OpenAI (`tools`, `message.tool_calls` / `delta.tool_calls`, `finish_reason: "tool_calls"`, messages de rôle `tool`), comme le faisait déjà `/v1/responses`. Les résultats d'outils dans l'historique sont associés à leurs appels par `tool_use_id` / `tool_call_id`, les appels parallèles conservent donc leurs noms ; les images dans un résultat d'outil apparaissent comme `[image omitted]`. `tool_choice: none` garde les outils dans l'invite et ne laisse passer aucun appel ; un outil nommé garde tous les outils dans l'invite et ne laisse passer que celui-ci (nommer un outil absent de `tools` renvoie 400) ; `any` / `required` ne sont pas imposés et fonctionnent comme `auto` (voir #42) ; `parallel_tool_calls: false` / `disable_parallel_tool_use: true` terminent la génération après le premier appel. Une entrée d'outil qui n'est pas un objet JSON sort sous la forme `{}` avec les arguments d'origine dans une note visible. Un appel irrécupérable produit par le modèle, ou un appel vers un outil qui n'était pas proposé, arrive sous la forme d'une note visible `[tool call … rejected: …]` dans le texte. Les anciens champs `functions` / `function_call` de Chat Completions sont ignorés. Vérifié avec Claude Code 2.1 sur `mlx-community/Qwen3-4B-4bit` : lister un répertoire avec `Bash`, puis lire un fichier et le modifier avec `Read` / `Edit`.

### Découverte de modèles

Claude Code (v2.1.126+) interroge `GET /v1/models` au démarrage et ajoute les modèles retournés au sélecteur `/model` — mais uniquement si l'identifiant commence par `claude` ou `anthropic`. Basalt renvoie les identifiants sous la forme `claude-local/<hf-id>` afin que la découverte fonctionne automatiquement.

---

## Intégration OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) est un agent de codage IA natif du terminal. Basalt fournit une API native compatible OpenAI (`/v1/chat/completions`, `/v1/models`) ainsi qu'une API compatible Anthropic qui se connecte directement à OpenCode sans dépendances cloud ni clés API.

### 1. Démarrer le serveur Basalt

Démarrez Basalt avec le modèle local souhaité (par ex. Huihui Qwen3.8 Abliterated, Ex0bit MYTHOS 26B ou Qwen3.5 4B) :

```bash
# High capability (Abliterated / MoE)
"$(swift build -c release --show-bin-path)/Basalt" serve --model AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit --rest

# Or fast coding model
"$(swift build -c release --show-bin-path)/Basalt" serve --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --rest
```

### 2. Connecter OpenCode

#### Option A — Variables d'environnement du terminal

```bash
# In your project folder:
OPENAI_BASE_URL=http://localhost:8080/v1 \
OPENAI_API_KEY=local \
OPENAI_MODEL=basalt \
opencode
```

#### Option B — Fichier de configuration OpenCode (`~/.config/opencode/config.json` ou `opencode.json`)

```json
{
  "provider": "openai",
  "base_url": "http://localhost:8080/v1",
  "api_key": "local",
  "model": "basalt",
  "temperature": 0.7,
  "max_tokens": 16384
}
```

#### Option C — Alias shell (à ajouter dans `~/.zshrc`)

```bash
function basalt-opencode() {
  OPENAI_BASE_URL=http://localhost:8080/v1 \
  OPENAI_API_KEY=local \
  OPENAI_MODEL=basalt \
  opencode "$@"
}
```

## Intégration MCP (Claude Desktop / Cursor)

Ajoutez à votre configuration MCP (`~/.config/claude/claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "basalt": {
      "command": "/path/to/basalt",
      "args": ["serve", "--model", "mlx-community/Qwen3.5-4B-4bit", "--mcp"]
    }
  }
}
```

Outils MCP disponibles :

| Outil | Description |
|---|---|
| `gemma_generate` | Générer du texte — `prompt`, `maxTokens`, `temperature`, `topP`, `minP`, `dynamicTemperature`, `images` (voir [Entrée image](rest-api.md#entrée-image)) |
| `gemma_status` | Renvoie l'état de disponibilité du serveur, sa version et le modèle actuel |
| `status` | Santé du moteur, modèle chargé, instantané mémoire et télémétrie — `modelId` optionnel à vérifier |
| `check_hardware` | Ressources Apple Silicon : puce, mémoire unifiée, GPU, disque |
| `inspect_model` | `modelId` (identifiant de dépôt ou chemin local) — vérifie les fragments de poids et détecte la largeur de bits réelle des poids MLX/Safetensors |
| `switch_model` | Bascule vers un autre modèle depuis le cache local — `modelId` |
| `gemma_add_knowledge` | Enregistre une leçon dans la bibliothèque d'expérience du harnais (`~/.basalt/harness-library`) |
| `harness_validate` | Valide un harnais Claude Code / opencode / AGENTS.md sous le répertoire de travail ou `BASALT_HARNESS_ROOTS` |
| `harness_optimize` | Exécute une passe Harness² propose–probe–compose sur le modèle chargé (également `basalt harness optimize`) |
| `harness_library_query` | Interroge la bibliothèque d'expérience |
| `model_describe`, `model_tensors`, `model_tensor`, `kv_inspect`, `trace_run` | Model X-ray (lecture seule, priorité d'arrière-plan) : anatomie des couches, liste des paramètres et statistiques bornées de tenseurs, arbre du cache d'invite, une exécution avec capture d'une invite (#328) |
| `intervene`, `weights_delta`, `overlays_list`, `overlay_undo`, `experiment_compare` | Expériences en bac à sable, uniquement avec `serve --research-tools` : pilotage (*steering*) et ablation de têtes/experts sous forme de surcouches annulables appliquées uniquement aux appels de recherche ; les deltas de poids modifient le modèle actif, de sorte que tant que l'un d'eux est actif, chaque réponse est marquée comme modifiée et la réutilisation du cache d'invite est désactivée ; voir [research-tools.md](research-tools.md) (#328) |
