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
            description: "Verifies weight shard integrity and detects the actual bit width of MLX/Safetensors weights. With verify, re-hashes every file against its sidecar or Hub tree API digest and reports verified \"true\", \"false\" or \"unknown\" per file; a mismatch makes intact false.",
            inputSchema: ResearchTool.object([
                "modelId": ["type": "string", "description": "Hugging Face repo ID or local path to inspect"],
                "verify": ["type": "boolean", "description": "Re-hash every file and report its digest verdict (default: false)"],
            ], required: ["modelId"])
        ),
        MCPToolDescriptor(
            name: "switch_model",
            description: "Switches the currently loaded model to a different model in the local cache.",
            inputSchema: modelIdSchema("Model ID in local cache to switch to")
        ),
        MCPToolDescriptor(name: "basalt_generate", description: "Generate text using the locally-running model via MLX.", inputSchema: generateSchema),
        MCPToolDescriptor(name: "basalt_status", description: statusDescription, inputSchema: emptySchema),
        MCPToolDescriptor(name: "basalt_add_knowledge", description: addKnowledgeDescription, inputSchema: addKnowledgeSchema),
        MCPToolDescriptor(
            name: "basalt_transcribe",
            description: "Transcribe speech with the loaded speech model (or serve --experimental-speech-analyzer). The audio is base64 in audio, at most 25 MB and 30 minutes, decoded and resampled to 16 kHz mono.",
            inputSchema: ResearchTool.object([
                "audio": ["type": "string", "description": "The audio file, base64-encoded"],
                "format": ["type": "string", "description": "File extension of the audio: \(AudioClip.fileExtensions.sorted().joined(separator: ", "))"],
                "language": ["type": "string", "description": "ISO-639-1 code of the spoken language, such as en (default: detected)"],
                "temperature": ["type": "number", "description": "Sampling temperature 0-1 (default 0)"],
            ], required: ["audio", "format"])
        ),
        MCPToolDescriptor(
            name: "basalt_systemone_decide",
            description: "Single-pass typed decisions with the loaded OpenJev decision model (serve --decision-model): noul (P(true)), choice (one of 2–255 named candidates) and score (expected level of 2–10 ordered levels). The state is encoded once and every candidate is scored as a branch over it; no text is generated. Probabilities are calibrated per type unless calibrated is false.",
            inputSchema: ResearchTool.object([
                "state": ["description": "The evidence: text, or a JSON object/array (serialized with sorted keys)"],
                "questions": ["type": "object", "description": "Map of question ID → {type: noul|choice|score, instructions, criteria}: choice criteria are {name: description}, score criteria a list of level descriptions, noul criteria optional {false, true}"],
                "calibrated": ["type": "boolean", "description": "Apply the model's per-type temperature (default true)"],
            ], required: ["state", "questions"])
        ),
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
            "regex":       ["type": "string",  "description": "Constrain the answer to this regular expression (XGrammar syntax)"],
            "grammar":     ["type": "string",  "description": "Constrain the answer to this EBNF/GBNF grammar (XGrammar dialect, root rule \"root\")"],
            "maxTokens":   ["type": "integer", "description": "Max tokens to generate (default 65536)"],
            "temperature": ["type": "number",  "description": "Sampling temperature 0–2 (default: the model's generation_config, else 0.7)"],
            "topP":        ["type": "number",  "description": "Nucleus sampling p (default: the model's generation_config, else 0.9)"],
            "minP":        ["type": "number",  "description": "Min-P 0–1, overrides the server default (0 = off)"],
            "dynamicTemperature": ["type": "boolean", "description": "Entropy-based dynamic temperature (EDT)"],
            "speculativeLookup": ["type": "boolean", "description": "Prompt-lookup speculative decoding for this call; unset takes serve --speculative-lookup"],
            "grammarJumpForward": ["type": "boolean", "description": "Grammar jump-forward constrained decoding for this call; unset takes serve --grammar-jump-forward"],
            "images": [
                "type": "array", "maxItems": ImageInput.maxCount,
                "description": "Images for a vision model, at most 8 of at most 20 MB each: {data, mime} with base64 bytes, or {path} to a file inside a serve --image-dir directory",
                "items": [
                    "type": "object",
                    "properties": [
                        "data": ["type": "string", "description": "Base64 of the encoded image"],
                        "mime": ["type": "string", "enum": ImageInput.mimeTypes.sorted()],
                        "path": ["type": "string", "description": "Absolute path inside a serve --image-dir directory"],
                    ],
                ],
            ],
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
        let introspect = researchTools ? [introspectDescriptor] : []
        return MCPToolsList(tools: list.tools + research.map { MCPToolDescriptor(name: $0.rawValue, description: $0.summary, inputSchema: $0.schema) } + introspect)
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
        case .adapterApply: "Experiment: apply a LoRA adapter live, from an mlx-lm adapter directory or as a rank-1 update of one linear layer (quantized included). Marks the model modified; undo with overlay_undo. With a gate it is kept without expiry only if the gate passes, otherwise undone; without one it expires after 30 min."
        case .gateCheck: "Experiment: the quality gate on the model as it is now against the pristine model (every live adapter and delta lifted): load-probe perplexity and known answers, mean KL and perplexity rise on held-out prompts, and exact-match targets. Each check has its value and limit."
        case .tensorPack: "Experiment: re-pack the weight of one linear layer in memory (affine 2-8 bit with a group size, mxfp4, nf4:32, nf4:64 or nf4:128 (the 16-level NormalFloat codebook with an fp16 absmax scale per group, run by the native nf4 kernels), or ternary absmean scale·{-1,0,1} per group of 64), dequantizing a quantized layer first. The layer is swapped live (the model is marked modified, prompt-cache reuse is off) as an overlay with TTL and bit-exact undo. Returns bytes before/after, RMSE, max abs error and relative error ||W-Ŵ||/||W||."
        case .tensorUnpack: "Experiment: replace one quantized linear layer by a dense float16 or bfloat16 layer holding its dequantized weight, live, as an overlay with TTL and bit-exact undo. Returns bytes before/after and the error of the cast."
        case .trainExperts: "Experiment, long-running: self-training. Collects the model's (and an optional teacher's) answers to checkable tasks, keeps only answers that match the known answer, trains a LoRA on them, saves it under ~/.basalt/adapters, reports loss and held-out accuracy before/after, then applies it through the quality gate as an undoable overlay."
        case .trainJepa: "Experiment, long-running: JEPA training. Trains a LoRA backbone and projector on a JSONL file of multi-view scenes (each line a string array or {\"views\": [...]}) with the LeJEPA loss, AdamW, warmup+cosine and gradient clipping. Saves adapters.safetensors and jepa_projector.safetensors under ~/.basalt/adapters, reports losses, pre-clip gradient norms and the WeightSpectrum rank of the projected embeddings before and after (a falling rank is collapse). The model is left unchanged; attention-only architectures only."
        case .distill: "Experiment, long-running: teacher → student distillation. A teacher behind an OpenAI-compatible or Anthropic endpoint answers the prompts (BASALT_TEACHER_API_KEY goes only to BASALT_TEACHER_URL's scheme, host and port); answers are cached under ~/.basalt/distill so a rerun pays nothing twice, spend stops at budgetUSD, empty, truncated, duplicate and (with filter verify) wrong answers are dropped. A LoRA is trained on the teacher's completions only (the prompt is masked), the teacher's per-token NLL on held-out prompts is reported before and after, and the adapter is applied through the quality gate as an undoable overlay. With onPolicy the student samples (temperature, maxTeacherTokens) and a basalt teacher with the same tokenizer scores those token ids through its /v1/score; the loss is the per-token reverse KL over the teacher's top 20, and the reverse KL on the base student's held-out samples is reported before and after. Mind the teacher's terms of use."
        case .steerSearch: "Experiment: contrastive steering search. The difference of mean residuals (positive minus negative prompts) per layer gives a direction; reports norm and class separation per layer, the best layer's unit direction, the heads whose ablation shrinks the gap most, and KL and projection shift when held-out prompts are steered by each scale. Directed search for activation directions, not self-awareness; leaves no overlay behind."
        case .steerCommit: "Experiment: turns a steering direction into a rank-1 update of the layer's output projection (input = mean module input over the positive prompts) and applies it as an adapter through the quality gate on the held-out prompts; undoable, never written to disk."
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
        case .adapterApply: Self.object(Self.adapterProperties)
        case .gateCheck: Self.object(Self.gateProperties, required: ["heldOut"])
        case .tensorPack: Self.object(Self.packProperties, required: ["tensor"])
        case .tensorUnpack: Self.object(Self.unpackProperties, required: ["tensor"])
        case .trainExperts: Self.object(Self.trainProperties, required: ["heldOut"])
        case .trainJepa: Self.object(Self.trainJepaProperties, required: ["datasetPath"])
        case .distill: Self.object(Self.distillProperties, required: ["prompts", "teacherUrl", "teacherModel", "heldOut"])
        case .steerSearch: Self.object(Self.steerSearchProperties, required: ["positive", "negative", "heldOut"])
        case .steerCommit: Self.object(Self.steerCommitProperties, required: ["layer", "direction", "scale", "positive", "heldOut"])
        }
    }

    static func object(_ properties: [String: Any], required: [String] = []) -> AnyCodable {
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

    static var numbers: [String: Any] { ["type": "array", "items": ["type": "number"], "minItems": 1] }

    private static var tensorProperties: [String: Any] {
        [
            "name": ["type": "string", "description": "Parameter name as model_tensors lists it"],
            "slice": slice,
            "maxValues": ["type": "integer", "minimum": ResearchCall.maxValues.lowerBound, "maximum": ResearchCall.maxValues.upperBound, "description": "Values returned from the slice (default 256; 0 = statistics only)"],
            "spectrum": ["type": "boolean", "description": "Also the spectrum of the dequantized matrix (or of one expert picked by the slice): Marchenko–Pastur noise variance, bulk edges and the eigenvalue mass above the bulk, the Gavish–Donoho rank and the power-law tail exponent alpha. Gram on the GPU, eigvalsh on the CPU; seconds per matrix; smaller side at most 8192; needs serve --research-tools (default false)"],
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

}

extension ResearchTool {
    static var adapterProperties: [String: Any] {
        [
            "directory": ["type": "string", "description": "mlx-lm adapter directory with adapter_config.json and adapters.safetensors; or give module, input and output"],
            "module": ["type": "string", "description": "rank-1: linear layer name, e.g. model.layers.3.mlp.down_proj"],
            "input": numbers.merging(["description": "rank-1: A, one value per input feature of the layer"]) { $1 },
            "output": numbers.merging(["description": "rank-1: B, one value per output feature of the layer"]) { $1 },
            "scale": ["type": "number", "description": "rank-1: factor on the update (default 1)"],
            "gate": ["type": "object", "properties": gateProperties, "required": ["heldOut"], "description": "Run the quality gate after applying; keep the adapter only if it passes"],
        ]
    }

    static var gateProperties: [String: Any] {
        [
            "heldOut": ["type": "array", "items": ["type": "string"], "minItems": ResearchCall.comparePrompts.lowerBound, "maxItems": ResearchCall.comparePrompts.upperBound, "description": "Prompts whose next-token distributions and baseline text are compared with the pristine model"],
            "maxKL": ["type": "number", "minimum": 0, "description": "Limit on the mean KL divergence (default 0.5)"],
            "maxPerplexityRise": ["type": "number", "minimum": 0, "description": "Limit on the relative perplexity rise of the baseline text (default 0.1)"],
            "target": [
                "type": "array", "maxItems": ResearchCall.comparePrompts.upperBound, "description": "Exact-match tasks: the greedy answer must contain expected",
                "items": ["type": "object", "properties": ["prompt": ["type": "string"], "expected": ["type": "string"]], "required": ["prompt", "expected"]],
            ],
        ]
    }

    static var packProperties: [String: Any] {
        [
            "tensor": ["type": "string", "description": "Weight of a linear layer as model_tensors lists it, e.g. model.layers.3.mlp.down_proj.weight"],
            "format": ["type": "string", "enum": PackRequest.packFormats, "description": "Target format (default affine)"],
            "bits": ["type": "integer", "minimum": PackRequest.packBits.lowerBound, "maximum": PackRequest.packBits.upperBound, "description": "affine: bits per weight, 2, 3, 4, 5, 6 or 8 (default 4)"],
            "groupSize": ["type": "integer", "enum": [32, 64, 128], "description": "affine: weights per scale and bias (default 64)"],
        ]
    }

    static var unpackProperties: [String: Any] {
        [
            "tensor": ["type": "string", "description": "Weight of a quantized linear layer as model_tensors lists it"],
            "format": ["type": "string", "enum": PackRequest.unpackFormats, "description": "Float type of the dense layer (default bfloat16)"],
        ]
    }

    static var trainJepaProperties: [String: Any] {
        [
            "datasetPath": ["type": "string", "description": "JSONL file: one scene per line, a string array or {\"views\": [...]} with at least two views"],
            "output": ["type": "string", "description": "Output directory for the adapter and projector (default: ~/.basalt/adapters)"],
            "rank": ["type": "integer", "minimum": 1, "maximum": 64, "description": "LoRA rank (default: 16)"],
            "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Transformer layers to adapt (default: all)"],
            "batchSize": ["type": "integer", "minimum": 1, "maximum": 1024, "description": "Scenes per batch, eight views each (default: 256)"],
            "maxSeqLen": ["type": "integer", "minimum": 4, "maximum": 512, "description": "Maximum tokens per view (default: 64)"],
            "iterations": ["type": "integer", "minimum": 1, "maximum": 10000, "description": "Training iterations (default: 100)"],
            "learningRate": ["type": "number", "description": "Peak learning rate (default: 1e-4)"],
            "weightDecay": ["type": "number", "description": "AdamW weight decay (default: 0.05)"],
            "warmupRatio": ["type": "number", "description": "Warmup fraction of iterations (default: 0.05)"],
            "lambda": ["type": "number", "description": "SIGReg weight (default: 0.02)"],
            "maxGradNorm": ["type": "number", "description": "Gradient clipping max norm (default: 1.0)"],
            "isCausal": ["type": "boolean", "description": "Causal encoding (default: false)"],
        ]
    }

    static var distillProperties: [String: Any] {
        [
            "prompts": ["type": "string", "description": "JSONL file, one {\"prompt\": …, \"answer\"?: …} per line; the last heldOutFraction are held out"],
            "teacherUrl": ["type": "string", "description": "Base URL of the teacher (OpenAI-compatible /v1 or Anthropic /v1); BASALT_TEACHER_API_KEY is sent only when its scheme, host and port match BASALT_TEACHER_URL"],
            "teacherModel": ["type": "string", "description": "Model name sent to the teacher"],
            "provider": ["type": "string", "enum": TeacherProvider.allCases.map(\.rawValue), "description": "Response format of the teacher (default openai)"],
            "maxTeacherTokens": ["type": "integer", "minimum": 16, "maximum": 32_768, "description": "Teacher completion limit; truncated answers are dropped (default 2048)"],
            "temperature": ["type": "number", "minimum": 0, "maximum": 2, "description": "Teacher sampling temperature (default 0.7)"],
            "concurrency": ["type": "integer", "minimum": 1, "maximum": 32, "description": "Parallel teacher calls (default 4)"],
            "reasoningEffort": ["type": "string", "description": "Teacher reasoning effort passed through (reasoning_effort, or output_config.effort on Anthropic), e.g. low; reasoning tokens are billed as output"],
            "inputPricePerMillion": ["type": "number", "minimum": 0, "description": "Teacher price per million prompt tokens, USD"],
            "outputPricePerMillion": ["type": "number", "minimum": 0, "description": "Teacher price per million completion tokens, USD; needed with budgetUSD"],
            "budgetUSD": ["type": "number", "description": "Stop calling the teacher before this spend would be exceeded"],
            "samplesPerPrompt": ["type": "integer", "minimum": 1, "maximum": 8, "description": "Teacher answers per prompt (default 1)"],
            "filter": ["type": "string", "enum": DistillFilter.allCases.map(\.rawValue), "description": "verify keeps only answers that match the prompt's answer (default none)"],
            "heldOutFraction": ["type": "number", "description": "Share of prompts held out for the NLL comparison (default 0.1)"],
            "includeReasoning": ["type": "boolean", "description": "Train on the teacher's reasoning followed by its answer, not the answer alone (default false)"],
            "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layers that get LoRA (default: all)"],
            "rank": ["type": "integer", "minimum": 1, "maximum": 64, "description": "LoRA rank (default 16)"],
            "iterations": ["type": "integer", "minimum": 1, "maximum": 10_000, "description": "Training iterations (default 200)"],
            "batchSize": ["type": "integer", "minimum": 1, "maximum": 64, "description": "Examples per step (default 2)"],
            "learningRate": ["type": "number", "description": "Adam learning rate (default 1e-4)"],
            "maxSeqLen": ["type": "integer", "minimum": 16, "maximum": 32_768, "description": "Longer examples are dropped and counted (default 2048)"],
            "output": ["type": "string", "description": "Adapter directory (default ~/.basalt/adapters/<fingerprint>/<timestamp>); never inside the model directory"],
            "onPolicy": ["type": "boolean", "description": "On-policy distillation: the student samples, a basalt teacher (provider openai) scores its tokens on /v1/score, reverse KL over the teacher's top 20; no cache, budget or filter (default false)"],
            "heldOut": ["type": "array", "items": ["type": "string"], "minItems": ResearchCall.comparePrompts.lowerBound, "maxItems": ResearchCall.comparePrompts.upperBound, "description": "General prompts the quality gate checks the adapter on"],
            "maxKL": ["type": "number", "minimum": 0, "description": "Gate: largest mean KL on heldOut (default 0.5)"],
            "maxPerplexityRise": ["type": "number", "minimum": 0, "description": "Gate: largest relative perplexity rise on heldOut (default 0.1)"],
        ]
    }

    static var trainProperties: [String: Any] {
        [
            "count": ["type": "integer", "minimum": TrainRequest.taskCounts.lowerBound, "maximum": TrainRequest.taskCounts.upperBound, "description": "Generated integer +, -, * problems (default 200); a fifth is held out for accuracy"],
            "seed": ["type": "integer", "minimum": 0, "description": "Seed of the problem generator (default 0)"],
            "file": ["type": "string", "description": "JSONL of {\"prompt\", \"answer\"} instead of generated problems"],
            "teacher": ["type": "string", "description": "Base URL of an OpenAI-compatible server whose answers are collected too; BASALT_TEACHER_API_KEY is sent only when its scheme, host and port match BASALT_TEACHER_URL"],
            "teacherModel": ["type": "string", "description": "Model name sent to the teacher"],
            "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layers that get LoRA (default: all)"],
            "experts": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Refused: the fork has no per-expert LoRA"],
            "rank": ["type": "integer", "minimum": 1, "maximum": 64, "description": "LoRA rank (default 8)"],
            "iterations": ["type": "integer", "minimum": 1, "maximum": 10_000, "description": "Training iterations (default 200)"],
            "learningRate": ["type": "number", "description": "Adam learning rate (default 1e-5)"],
            "output": ["type": "string", "description": "Adapter directory (default ~/.basalt/adapters/<fingerprint>/<timestamp>); never inside the model directory"],
            "onPolicy": ["type": "boolean", "description": "On-policy distillation: the student samples, a basalt teacher (provider openai) scores its tokens on /v1/score, reverse KL over the teacher's top 20; no cache, budget or filter (default false)"],
            "heldOut": ["type": "array", "items": ["type": "string"], "minItems": ResearchCall.comparePrompts.lowerBound, "maxItems": ResearchCall.comparePrompts.upperBound, "description": "General prompts the quality gate checks the adapter on"],
            "maxKL": ["type": "number", "minimum": 0, "description": "Gate: largest mean KL on heldOut (default 0.5)"],
            "maxPerplexityRise": ["type": "number", "minimum": 0, "description": "Gate: largest relative perplexity rise on heldOut (default 0.1)"],
        ]
    }

    private static func prompts(_ range: ClosedRange<Int>, _ description: String) -> [String: Any] {
        ["type": "array", "items": ["type": "string"], "minItems": range.lowerBound, "maxItems": range.upperBound, "description": description]
    }

    static var steerSearchProperties: [String: Any] {
        [
            "positive": prompts(ContrastRequest.classPrompts, "Prompts showing the behaviour to find"),
            "negative": prompts(ContrastRequest.classPrompts, "Prompts without it"),
            "heldOut": prompts(ContrastRequest.heldOutPrompts, "Prompts steered in the trials"),
            "layers": ["type": "array", "items": ["type": "integer", "minimum": 0], "description": "Layers to search (default: all)"],
            "topHeads": ["type": "integer", "minimum": ContrastRequest.headLimit.lowerBound, "maximum": ContrastRequest.headLimit.upperBound, "description": "Heads reported, ranked by ablation effect (default 8; 0 skips the ablations)"],
            "scales": numbers.merging(["maxItems": ContrastRequest.scaleCount.upperBound, "description": "Steering scales tried on the held-out prompts (default [2, 4, 8])"]) { $1 },
        ]
    }

    static var steerCommitProperties: [String: Any] {
        [
            "layer": ["type": "integer", "minimum": 0, "description": "bestLayer from steer_search"],
            "direction": numbers.merging(["description": "direction from steer_search (hidden size)"]) { $1 },
            "scale": ["type": "number", "description": "Scale chosen from the steer_search trials"],
            "positive": prompts(1...ContrastRequest.classPrompts.upperBound, "Positive prompts; their mean module input becomes the rank-1 input"),
            "heldOut": prompts(ContrastRequest.heldOutPrompts, "Held-out prompts for the quality gate"),
            "maxKL": ["type": "number", "minimum": 0, "description": "Gate: largest mean KL against the baseline (default 0.5)"],
            "maxPerplexityRise": ["type": "number", "minimum": 0, "description": "Gate: largest relative perplexity rise (default 0.1)"],
        ]
    }
}
