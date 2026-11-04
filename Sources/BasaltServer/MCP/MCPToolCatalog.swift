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
