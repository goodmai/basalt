# Architecture

> Languages: [English](../../architecture.md) · [Русский](../ru/architecture.md) · [中文](../zh/architecture.md) · [Español](../es/architecture.md) · **Français** · [日本語](../ja/architecture.md) · [Deutsch](../de/architecture.md) · [한국어](../ko/architecture.md) · [Bahasa Indonesia](../id/architecture.md)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

Deux transports partagent une seule instance d'acteur — **MCP stdio** pour l'intégration dans les IDE (Claude Desktop, Cursor) et **REST HTTP** pour les workflows agent-à-agent et les backends Claude Code.

## Structure du projet

```
Sources/                    — one SwiftPM target per directory; dependencies point down only
  BasaltCore/               — DTOs, errors, config, logger, profilers, calculators, HF cache, orchestrator
    Chat/                   — ModelProfile: everything that differs between model families
    Config/                 — ServerConfig value type
  BasaltInference/          — MLX inference engine, ContextDegradationProfiler, RepetitionGuard  (→ Core)
  BasaltCloud/              — OpenRouter / cloud model fallback (CostTracker, ModelRouter)    (→ Core)
  BasaltServer/             — REST (Hummingbird 2.x, Host/Origin/Content-Type guard, API key and rate-limit middleware) and MCP JSON-RPC stdio server (→ Core)
  BasaltUI/                 — Terminal UI (Markdown, spinner, progress bar, diff, table)       (→ Core)
  BasaltApp/                — SwiftUI REST chat client, not linked into the CLI (launch: #55)  (→ Core)
  BasaltCLI/                — Subcommands and the CLI root command (App/)                     (→ all of the above)
  BasaltBin/                — Thin executable wrapper (product `Basalt`, installed as `basalt`)
  PerformanceBenchmark/     — Standalone benchmark CLI target

Tests/
  BasaltTests/              — Unit + integration tests
    CLI/                    — Command parsing, Cloud, Fit, PromptContextBuilder
    Cloud/                  — CostTracker, ModelRouter, OpenRouterClient
    UI/                     — Clipboard, Diff, Markdown, ProgressBar, Spinner, Table, TerminalUI
    MockInferenceEngine.swift — Controllable mock for orchestrator tests

basalt-claude                       — Executable Swift launcher (build + serve + claude)
scripts/                    — Build and maintenance Swift scripts
docs/                       — Extended documentation
```

Seule `BasaltApp` est autorisée à importer SwiftUI/AppKit ; chaque cible liée par la CLI reste sans interface graphique (vérifié par `ModuleBoundaryTests`).
Les symboles nouvellement partagés entre cibles utilisent le niveau d'accès `package` (l'API publique existante reste `public`).

---

## Notes d'architecture

**Quantification dynamique, MACaQ et MoE** — `MLXInferenceEngine` analyse `config.json` (y compris `macaq_quantization`, `flatquant_quantization`, `lirquant_quantization`) pour répartir les checkpoints Dense et Sparse MoE sur des noyaux Metal optimisés (`SwitchGLU`, `MACaQSwitchLinear` WHT en une passe + `MLX.gatherQuantizedMM` `#628`, treillis 2-bit `E8P`, `QTIP`, attention vidéo 3D `STA` `#609`).

**SDK, gouvernance et mémoire hiérarchisée (`v0.12`)** — `BasaltFoundationModels` fournit un pont SDK Swift (`#606`) ; `VideoRunnerCore` cadence la génération vidéo sur UMA 24 Go (`#611`) ; `TieredKVHierarchyCore` gère les niveaux KV L0/L1/L2 en mémoire unifiée (`#627`) ; `SemanticCacheActor`, `HITLApprovalActor` et `SymbolicKnowledgeGraph` appliquent le routage d'entreprise et les validations Human-in-the-Loop (`#625`, `#626`).

**Isolation par acteur** — `ModelOrchestratorActor` est un acteur Swift 6. Tous les appels d'inférence sont sérialisés (FIFO) sans verrouillage explicite. MCP et REST partagent une seule instance de cet acteur.

**Remplacement à chaud de modèle** — `switchModel(to:)` résout l'identifiant de dépôt HuggingFace depuis le cache local, décharge le modèle courant (`container = nil` + `MLX.GPU.clearCache()`), puis charge les nouveaux poids. L'acteur garantit que les requêtes en cours s'achèvent avant le début du basculement.

**Protection par délai d'expiration** — chaque appel à `generate` et `generateStream` est encadré par un timeout de 5 minutes avec annulation coopérative des `Task`, empêchant qu'un blocage du modèle ne paralyse le serveur indéfiniment.

**Streaming** — le corps de réponse `AsyncStream<ByteBuffer>` de Hummingbird 2 est utilisé pour les deux formats SSE : OpenAI (`data: {...}`) et Anthropic (`event: content_block_delta`).

**Suppression des blocs de réflexion** — `generateStream` exécute une machine à états pour filtrer les blocs de raisonnement `<think>...</think>` avant de transmettre les tokens aux clients.

**Traduction d'identifiants** — `ModelsController` convertit les identifiants de dépôt HuggingFace (`mlx-community/Qwen3.5-4B-4bit`) vers la forme `claude-local/mlx-community--Qwen3.5-4B-4bit` exigée par le filtre de découverte de modèles de Claude Code. Les deux formes sont acceptées dans les requêtes API.

---

## Scripts de maintenance

| Script | Rôle |
|---|---|
| `scripts/build_metal.swift` | Compile les noyaux GPU Metal (ignoré s'ils sont à jour) |
| `scripts/cleanup.swift` | `--mode=daily` effectue la rotation des journaux/images/rapports, `--mode=release` supprime les artefacts de build et d'exécution avant publication, `--mode=all` combine les deux ; simulation à blanc sauf avec `--apply` |
| `scripts/archive.sh` | Archive les journaux, rapports et captures d'écran |

## Dépendances

Tout repose sur des paquets SwiftPM ; il n'y a ni CocoaPods, ni Node, ni Python à
l'exécution. `Package.resolved` fige les révisions exactes.

| Paquet | Rôle |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | la couche de tableaux et réseaux de neurones sur Metal |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | architectures et chargement des modèles — `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | tokenizers et modèles de chat |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | le serveur HTTP |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | la CLI |
| [Rainbow](https://github.com/onevcat/Rainbow) | couleurs et mise en forme dans le terminal |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | rendu Markdown et coloration syntaxique dans le terminal |

`MLXVLM` n'est pas optionnel : Gemma 4 et les checkpoints Ornith déclarent
`*ForConditionalGeneration`, et ces architectures y sont enregistrées plutôt
que dans `MLXLLM`.

Les mises à jour arrivent sous forme de PR Dependabot — chaque semaine pour SwiftPM, chaque mois pour les GitHub
Actions. mlx-swift évolue rapidement et une architecture de modèle n'est souvent qu'à une version
près, ce qui justifie cette cadence hebdomadaire.

---

## Contribuer

Les issues et les pull requests sont les bienvenues. Avant d'ouvrir une PR :

```bash
swift build          # must compile
swift test           # must stay green
```

Ne commitez jamais d'identifiants, de jetons ou de fichiers d'état `*.sqlite3` — `.gitignore` bloque les
fichiers habituels, et le workflow de publication exécute une analyse de secrets sur l'arborescence.
