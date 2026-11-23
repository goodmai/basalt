# Arquitectura

> Languages: [English](../../architecture.md) · [Русский](../ru/architecture.md) · [中文](../zh/architecture.md) · **Español** · [Français](../fr/architecture.md) · [日本語](../ja/architecture.md) · [Deutsch](../de/architecture.md) · [한국어](../ko/architecture.md) · [Bahasa Indonesia](../id/architecture.md)

```
┌──────────────────────────────────────────────────────┐
│                       Basalt                           │
│                                                      │
│   MCP stdio ──┐                                      │
│               ├──► ModelOrchestratorActor ──► MLX   │
│   REST :8080 ─┘        (actor, FIFO)      Metal GPU  │
└──────────────────────────────────────────────────────┘
```

Dos transportes comparten una única instancia de actor: **MCP stdio** para la integración con IDEs (Claude Desktop, Cursor) y **REST HTTP** para flujos de trabajo agente a agente y backends de Claude Code.

## Estructura del proyecto

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

Solo `BasaltApp` puede importar SwiftUI/AppKit; todos los targets que enlaza la CLI permanecen sin interfaz gráfica (aplicado por `ModuleBoundaryTests`).
Los símbolos recién compartidos entre targets utilizan el nivel de acceso `package` (la API pública existente permanece `public`).

---

## Notas de arquitectura

**Cuantización dinámica, MACaQ y MoE** — `MLXInferenceEngine` analiza `config.json` (incluyendo `macaq_quantization`, `flatquant_quantization`, `lirquant_quantization`) para despachar checkpoints MoE densos y dispersos hacia kernels de Metal optimizados (`SwitchGLU`, `MACaQSwitchLinear` WHT en una sola pasada + `MLX.gatherQuantizedMM` `#628`, retículo de 2 bits `E8P`, `QTIP`, atención de vídeo 3D `STA` `#609`).

**SDK, gobernanza y memoria por niveles (`v0.12`)** — `BasaltFoundationModels` proporciona un puente SDK de Swift (`#606`); `VideoRunnerCore` divide en fases la generación de vídeo en 24 GB UMA (`#611`); `TieredKVHierarchyCore` gestiona los niveles KV L0/L1/L2 en memoria unificada (`#627`); `SemanticCacheActor`, `HITLApprovalActor` y `SymbolicKnowledgeGraph` aplican el enrutamiento empresarial y las aprobaciones Human-in-the-Loop (`#625`, `#626`).

**Aislamiento de actores** — `ModelOrchestratorActor` es un actor de Swift 6. Todas las llamadas de inferencia se serializan (FIFO) sin bloqueo explícito. Tanto MCP como REST comparten una única instancia de actor.

**Intercambio en caliente de modelos** — `switchModel(to:)` resuelve el ID de repositorio de HuggingFace desde la caché local, descarga el modelo actual (`container = nil` + `MLX.GPU.clearCache()`) y luego carga los nuevos pesos. El actor garantiza que las peticiones en curso finalicen antes de que comience el intercambio.

**Protección por tiempo de espera** — cada llamada a `generate` y `generateStream` está envuelta en un tiempo de espera de 5 minutos con cancelación cooperativa de `Task`, evitando que los bloqueos del modelo paralicen el servidor indefinidamente.

**Streaming** — se utiliza el cuerpo de respuesta `AsyncStream<ByteBuffer>` de Hummingbird 2 para ambos formatos SSE: OpenAI (`data: {...}`) y Anthropic (`event: content_block_delta`).

**Eliminación de bloques think** — `generateStream` ejecuta una máquina de estados para suprimir los bloques de razonamiento `<think>...</think>` antes de reenviar los tokens a los clientes.

**Traducción de IDs** — `ModelsController` traduce entre los IDs de repositorio de HuggingFace (`mlx-community/Qwen3.5-4B-4bit`) y la forma `claude-local/mlx-community--Qwen3.5-4B-4bit` requerida por el filtro de descubrimiento de modelos de Claude Code. Ambas formas se aceptan en las peticiones a la API.

---

## Scripts de mantenimiento

| Script | Propósito |
|---|---|
| `scripts/build_metal.swift` | Compila los kernels GPU de Metal (se omite si están actualizados) |
| `scripts/cleanup.swift` | `--mode=daily` rota registros/imágenes/informes, `--mode=release` elimina artefactos de compilación/ejecución antes de publicar, `--mode=all` hace ambas cosas; ejecución en seco salvo que se pase `--apply` |
| `scripts/archive.sh` | Archiva registros/informes/capturas de pantalla |

## Dependencias

Todo es un paquete SwiftPM; no hay CocoaPods, ni Node, ni Python en tiempo de ejecución. `Package.resolved` fija las revisiones exactas.

| Paquete | Usado para |
|---|---|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | la capa de arrays/NN sobre Metal |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | arquitecturas de modelos y carga — `MLXLLM`, `MLXVLM`, `MLXLMCommon` |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | tokenizadores y plantillas de chat |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | el servidor HTTP |
| [swift-argument-parser](https://github.com/apple/swift-argument-parser) | la CLI |
| [Rainbow](https://github.com/onevcat/Rainbow) | color y diseño en la terminal |
| [swift-markdown](https://github.com/apple/swift-markdown), [Splash](https://github.com/JohnSundell/Splash) | renderizado de Markdown y resaltado de sintaxis en la terminal |

`MLXVLM` no es opcional: Gemma 4 y los checkpoints de Ornith declaran
`*ForConditionalGeneration`, y esas arquitecturas están registradas allí en lugar
de en `MLXLLM`.

Las actualizaciones llegan como PRs de Dependabot — semanalmente para SwiftPM y mensualmente para las GitHub
Actions. mlx-swift avanza rápido y una arquitectura de modelo suele estar a una sola versión
de distancia, motivo por el cual se mantiene la cadencia semanal.

---

## Contribuir

Los issues y pull requests son bienvenidos. Antes de abrir un PR:

```bash
swift build          # must compile
swift test           # must stay green
```

Nunca hagas commit de credenciales, tokens ni estado `*.sqlite3` — `.gitignore` bloquea los
elementos habituales y el flujo de lanzamiento ejecuta un escaneo de secretos sobre el árbol.
