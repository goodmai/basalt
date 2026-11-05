import BasaltCore
import BasaltHarness

struct MCPInitializeResult: Encodable, Sendable {
    struct ServerInfo: Encodable, Sendable {
        let name: String
        let version: String
    }

    let protocolVersion: String
    let capabilities: [String: [String: AnyCodable]]
    let serverInfo: ServerInfo

    static let current = MCPInitializeResult(
        protocolVersion: "2024-11-05",
        capabilities: ["tools": [:]],
        serverInfo: ServerInfo(name: "Basalt", version: HealthResponse.version)
    )
}

struct MCPToolDescriptor: Encodable, Sendable {
    let name: String
    let description: String
    let inputSchema: AnyCodable
}

struct MCPToolsList: Encodable, Sendable {
    let tools: [MCPToolDescriptor]
}

enum MCPToolCatalog {
    static let list = MCPToolsList(tools: [
        MCPToolDescriptor(
            name: "status",
            description: "Returns the health, loaded model ID, memory snapshot, and readiness status of the inference engine.",
            inputSchema: AnyCodable([
                "type": "object",
                "properties": [
                    "modelId": ["type": "string", "description": "Optional model repo ID to check"]
                ]
            ])
        ),
        MCPToolDescriptor(
            name: "check_hardware",
            description: "Detects Apple Silicon hardware resources (chip, available unified memory, GPU, disk) without downloading models.",
            inputSchema: emptySchema
        ),
        MCPToolDescriptor(
            name: "inspect_model",
            description: "Verifies weight shard integrity and detects the actual bit width of MLX/Safetensors weights.",
            inputSchema: modelIdSchema("Hugging Face repo ID or local path to inspect")
        ),
        MCPToolDescriptor(
            name: "switch_model",
            description: "Switches the currently loaded model to a different model in the local cache.",
            inputSchema: modelIdSchema("Model ID in local cache to switch to")
        ),
        MCPToolDescriptor(name: "basalt_generate", description: "Generate text using the locally-running model via MLX.", inputSchema: generateSchema),
        MCPToolDescriptor(name: "basalt_status", description: statusDescription, inputSchema: emptySchema),
        MCPToolDescriptor(name: "basalt_add_knowledge", description: addKnowledgeDescription, inputSchema: addKnowledgeSchema),
        MCPToolDescriptor(name: "gemma_generate", description: "Generate text using the locally-running Gemma model via MLX.", inputSchema: generateSchema),
        MCPToolDescriptor(name: "gemma_status", description: statusDescription, inputSchema: emptySchema),
        MCPToolDescriptor(name: "gemma_add_knowledge", description: addKnowledgeDescription, inputSchema: addKnowledgeSchema),
        MCPToolDescriptor(
            name: "harness_validate",
            description: "Validate an agent harness: a Claude Code (CLAUDE.md, .claude/agents, .claude/skills, .claude/settings.json hooks), opencode (AGENTS.md, .opencode/agent, .opencode/skill, .opencode/plugin) or AGENTS.md + .agents/skills layout. Reports the always-on budget (instruction files and their @imports), skill and agent frontmatter, hook events, missing script references and untriggered skills. A directory must lie under the server's working directory (unless that is / or $HOME) or a root listed in BASALT_HARNESS_ROOTS (colon-separated); loading, inline files included, is capped at 2000 files, 256 KiB per file, 16 MiB in total, 8 levels deep and 10000 directory entries visited; .claude/worktrees and node_modules are not read.",
            inputSchema: AnyCodable([
                "type": "object",
                "properties": [
                    "directory": ["type": "string", "description": "Repository root that contains the harness (e.g. the folder holding CLAUDE.md and .claude/)"],
                    "files": [
                        "type": "object",
                        "description": "Alternative to directory: the harness as {\"path relative to the repository root\": \"file content\"}",
                        "additionalProperties": ["type": "string"]
                    ],
                    "surface": surfaceProperty,
                    "alwaysOnMaxChars": ["type": "integer", "description": "Budget for the always-on files in characters (default 8000)"]
                ]
            ])
        ),
        MCPToolDescriptor(
            name: "harness_optimize",
            description: "Run one Harness² propose–probe–compose pass on the loaded model: 2k proposer calls (invariant and domain edits of the harness), k base rollouts, a rollout per admitted candidate, k compose calls, then an anonymised judge among the candidates tied on measured stages (delivered, required terms, no loops). Budget R=4k rollouts, P=3k proposer calls. Nothing is written to the harness; the winning edit is returned and stored in the Experience Library. Same directory rules and limits as harness_validate.",
            inputSchema: AnyCodable([
                "type": "object",
                "required": ["task"],
                "properties": [
                    "task": ["type": "string", "description": "The task the harness should get better at"],
                    "slug": ["type": "string", "description": "Short task id used for the library entry and the prompt-cache namespace (default: derived from task)"],
                    "domain": ["type": "string", "description": "Experience Library domain (default general)"],
                    "requiredTerms": ["type": "array", "items": ["type": "string"], "description": "Terms the task demands in the deliverable (conformance stage)"],
                    "expectsJSON": ["type": "boolean", "description": "The deliverable must be a JSON object"],
                    "k": ["type": "integer", "minimum": 1, "maximum": 4, "description": "Budget parameter (default 1)"],
                    "proposerMaxTokens": [
                        "type": "integer", "minimum": HarnessOptimizer.TokenBudgets.proposerRange.lowerBound,
                        "maximum": HarnessOptimizer.TokenBudgets.proposerRange.upperBound,
                        "description": "Token budget of each proposer reply (default: fits the largest editable file, 2048...12288)"
                    ],
                    "directory": ["type": "string", "description": "Repository root that contains the harness"],
                    "files": ["type": "object", "additionalProperties": ["type": "string"], "description": "Alternative to directory: {path: content}"],
                    "surface": surfaceProperty
                ]
            ])
        ),
        MCPToolDescriptor(
            name: "harness_library_query",
            description: "Query stored lessons and component attributions from the Harness Experience Library.",
            inputSchema: AnyCodable([
                "type": "object",
                "properties": [
                    "query": ["type": "string", "description": "Optional substring query for topic, cue, or content"],
                    "domain": ["type": "string", "description": "Optional domain filter"]
                ]
            ])
        ),
    ])

    private static let statusDescription = "Returns the health and readiness status of the inference engine."

    private static let addKnowledgeDescription = "Store a lesson in the persistent Harness Experience Library (~/.basalt/harness-library/<domain>/): EXPERIENCE.md, entries.json and the domain INDEX.md that future harness_optimize proposers read. Limits: topic 200 characters, cue 500, domain 64, content 32 KiB, 1000 entries."

    private static let emptySchema = AnyCodable(["type": "object", "properties": [String: Any]()])

    private static var surfaceProperty: [String: Any] {
        ["type": "string", "enum": HarnessSurfaceSpec.all.map(\.name), "description": "Harness layout (default claude)"]
    }

    private static let generateSchema = AnyCodable([
        "type": "object",
        "required": ["prompt"],
        "properties": [
            "prompt":      ["type": "string",  "description": "Input prompt"],
            "maxTokens":   ["type": "integer", "description": "Max tokens to generate (default 65536)"],
            "temperature": ["type": "number",  "description": "Sampling temperature 0–2 (default: the model's generation_config, else 0.7)"],
            "topP":        ["type": "number",  "description": "Nucleus sampling p (default: the model's generation_config, else 0.9)"],
            "minP":        ["type": "number",  "description": "Min-P 0–1, overrides the server default (0 = off)"],
            "dynamicTemperature": ["type": "boolean", "description": "Entropy-based dynamic temperature (EDT)"],
            "speculativeLookup": ["type": "boolean", "description": "Prompt-lookup speculative decoding for this call; unset takes serve --speculative-lookup"],
            "grammarJumpForward": ["type": "boolean", "description": "Grammar jump-forward constrained decoding for this call; unset takes serve --grammar-jump-forward"]
        ]
    ])

    private static let addKnowledgeSchema = AnyCodable([
        "type": "object",
        "required": ["topic", "content"],
        "properties": [
            "topic": ["type": "string", "description": "Subject of the knowledge"],
            "content": ["type": "string", "description": "Detailed information to store"],
            "domain": ["type": "string", "description": "Optional domain bucket in the Experience Library"],
            "cue": ["type": "string", "description": "Optional trigger cue for retrieving this lesson"],
            "confidence": ["type": "string", "enum": ["low", "medium", "high"], "description": "Confidence level (default medium)"]
        ]
    ])

    private static func modelIdSchema(_ description: String) -> AnyCodable {
        AnyCodable([
            "type": "object",
            "required": ["modelId"],
            "properties": [
                "modelId": ["type": "string", "description": description]
            ]
        ])
    }
}

extension MCPToolCatalog {
    static func catalog(researchTools: Bool) -> MCPToolsList {
        let research = ResearchTool.allCases.filter { researchTools || !$0.isExperiment }
        return MCPToolsList(tools: list.tools + research.map { MCPToolDescriptor(name: $0.rawValue, description: $0.summary, inputSchema: $0.schema) })
    }
}

extension ResearchTool {
    var summary: String {
        switch self {
        case .modelDescribe: "Anatomy of the loaded model: type, quantization, parameter count and bytes, and per layer its kind (full attention, sliding window, linear attention, mamba, conv, mlp, moe), heads, KV heads, head dim, window, experts and KV-cache class."
        case .modelTensors: "List the loaded model's parameters with shape, dtype and whether they are quantized, optionally filtered by a name substring."
        case .modelTensor: "Statistics of one parameter tensor by name (mean, std, min, max, L2, percentiles of |x|), dequantized, with at most maxValues values of an optional slice. Never returns a whole tensor."
        case .kvInspect: "The prompt-cache tree: budget, bytes, evictions and per namespace its nodes, leaves, tokens, bytes, recurrent snapshots, pins and last use."
        case .traceRun: "One capture run of a prompt at background priority: per-layer hidden-state norms, logit lens, per-head attention for one position and MoE router choices, each bounded to topK."
        case .intervene: "Experiment: add a steering vector to a layer's residual stream, or zero one attention head or one expert. Creates an overlay (id, 30 min TTL, undo) that applies to research calls only, never to user requests."
        case .weightsDelta: "Experiment: a temporary in-memory delta on one tensor (optionally a slice), undoable; weights on disk are never written."
        case .overlaysList: "Experiment: the active overlays with id, description, creation and expiry time."
        case .overlayUndo: "Experiment: undo one overlay by id."
        case .experimentCompare: "Experiment: greedy before/after on a prompt set with the given overlays: text, KL divergence of the logits and perplexity."
        case .tensorPack: "Experiment: re-pack the weight of one linear layer in memory (affine 2-8 bit with a group size, mxfp4, or ternary absmean scale·{-1,0,1} per group of 64), dequantizing a quantized layer first. The layer is swapped live (the model is marked modified, prompt-cache reuse is off) as an overlay with TTL and bit-exact undo. Returns bytes before/after, RMSE, max abs error and relative error ||W-Ŵ||/||W||."
        case .tensorUnpack: "Experiment: replace one quantized linear layer by a dense float16 or bfloat16 layer holding its dequantized weight, live, as an overlay with TTL and bit-exact undo. Returns bytes before/after and the error of the cast."
        }
    }

    var schema: AnyCodable {
        switch self {
        case .modelDescribe, .overlaysList: Self.object([:])
        case .modelTensors: Self.object(["filter": ["type": "string", "description": "Substring of the parameter name, e.g. layers.3.mlp (default: all)"]])
        case .kvInspect: Self.object(["namespace": ["type": "string", "description": "Only this prompt-cache namespace (default: all)"]])
        case .modelTensor: Self.object(Self.tensorProperties, required: ["name"])
        case .traceRun: Self.object(Self.traceProperties, required: ["prompt"])
        case .intervene: Self.object(Self.interveneProperties, required: ["kind", "layer"])
        case .weightsDelta: Self.object(Self.deltaProperties, required: ["tensor"])
        case .overlayUndo: Self.object(["id": ["type": "integer", "minimum": 0, "description": "Overlay id from intervene, weights_delta or overlays_list"]], required: ["id"])
        case .experimentCompare: Self.object(Self.compareProperties, required: ["prompts"])
        case .tensorPack: Self.object(Self.packProperties, required: ["tensor"])
        case .tensorUnpack: Self.object(Self.unpackProperties, required: ["tensor"])
        }
    }

    private static func object(_ properties: [String: Any], required: [String] = []) -> AnyCodable {
        var schema: [String: Any] = ["type": "object", "properties": properties]
        if !required.isEmpty { schema["required"] = required }
        return AnyCodable(schema)
    }

    private static var slice: [String: Any] {
        [
            "type": "array", "description": "Half-open [start, end) range per leading dimension",
            "items": ["type": "array", "items": ["type": "integer", "minimum": 0], "minItems": 2, "maxItems": 2],
        ]
    }

    private static var numbers: [String: Any] { ["type": "array", "items": ["type": "number"], "minItems": 1] }

    private static var tensorProperties: [String: Any] {
        [
            "name": ["type": "string", "description": "Parameter name as model_tensors lists it"],
            "slice": slice,
            "maxValues": ["type": "integer", "minimum": ResearchCall.maxValues.lowerBound, "maximum": ResearchCall.maxValues.upperBound, "description": "Values returned from the slice (default 256; 0 = statistics only)"],
        ]
    }

    private static var traceProperties: [String: Any] {
        [
            "prompt": ["type": "string", "description": "Prompt to capture"],
            "position": ["type": "integer", "minimum": 0, "description": "Token position to inspect (default: the last prompt token)"],
            "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layers to report (default: all)"],
            "topK": ["type": "integer", "minimum": ResearchCall.topK.lowerBound, "maximum": ResearchCall.topK.upperBound, "description": "Entries per list (default 10)"],
            "attention": ["type": "boolean", "description": "Per-head attention at position (default true)"],
            "router": ["type": "boolean", "description": "MoE router choices (default true)"],
            "logitLens": ["type": "boolean", "description": "Logit lens per layer (default true)"],
        ]
    }

    private static var interveneProperties: [String: Any] {
        [
            "kind": ["type": "string", "enum": Intervention.kinds],
            "layer": ["type": "integer", "minimum": 0],
            "vector": numbers.merging(["description": "steer: vector of the hidden size added after the layer"]) { $1 },
            "scale": ["type": "number", "description": "steer: factor on the vector (default 1)"],
            "head": ["type": "integer", "minimum": 0, "description": "ablate_head: head to zero"],
            "expert": ["type": "integer", "minimum": 0, "description": "ablate_expert: expert to zero"],
        ]
    }

    private static var deltaProperties: [String: Any] {
        [
            "tensor": ["type": "string", "description": "Parameter name as model_tensors lists it"],
            "slice": slice,
            "values": numbers.merging(["description": "Added to the slice; one value is broadcast"]) { $1 },
            "scale": ["type": "number", "description": "Factor on the values (default 1)"],
        ]
    }

    private static var compareProperties: [String: Any] {
        [
            "prompts": ["type": "array", "items": ["type": "string"], "minItems": ResearchCall.comparePrompts.lowerBound, "maxItems": ResearchCall.comparePrompts.upperBound],
            "overlays": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Overlay ids applied on the after side (default: none)"],
            "maxTokens": ["type": "integer", "minimum": ResearchCall.compareTokens.lowerBound, "maximum": ResearchCall.compareTokens.upperBound, "description": "Greedy tokens per side (default 32)"],
        ]
    }

    private static var packProperties: [String: Any] {
        [
            "tensor": ["type": "string", "description": "Weight of a linear layer as model_tensors lists it, e.g. model.layers.3.mlp.down_proj.weight"],
            "format": ["type": "string", "enum": PackRequest.packFormats, "description": "Target format (default affine)"],
            "bits": ["type": "integer", "minimum": PackRequest.packBits.lowerBound, "maximum": PackRequest.packBits.upperBound, "description": "affine: bits per weight, 2, 3, 4, 5, 6 or 8 (default 4)"],
            "groupSize": ["type": "integer", "enum": [32, 64, 128], "description": "affine: weights per scale and bias (default 64)"],
        ]
    }

    private static var unpackProperties: [String: Any] {
        [
            "tensor": ["type": "string", "description": "Weight of a quantized linear layer as model_tensors lists it"],
            "format": ["type": "string", "enum": PackRequest.unpackFormats, "description": "Float type of the dense layer (default bfloat16)"],
        ]
    }
}
