import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer
import BasaltHarness
import BasaltCloud

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the Basalt inference server (REST + optional MCP stdio)"
    )

    @Option(
        name: .customLong("model"),
        help: "Model repo ID (e.g. mlx-community/gemma-4-e4b-it-4bit) or local path"
    )
    var model: String?

    @Option(
        name: .customLong("model-path"),
        help: "Explicit local path to model directory (overrides --model cache lookup)"
    )
    var modelPath: String?

    @Option(name: .customLong("quant"), help: "Quantization subfolder inside the repo (e.g. 2bit | 4bit) — for repos shipping several variants")
    var quant: String?

    @Option(
        name: .customLong("embedding-model"),
        help: "Embedding model served on /v1/embeddings next to the chat model: repo ID (resolved from the Hugging Face cache like --model) or local path; BERT, RoBERTa, XLM-RoBERTa, DistilBERT, NomicBERT, Qwen3-Embedding or LFM2.5-Embedding"
    )
    var embeddingModel: String?

    @Option(
        name: .customLong("rerank-model"),
        help: "Reranker served on /v1/rerank next to the chat model: repo ID or local path of an encoder cross-encoder, Qwen3-Reranker or Jina reranker (its name must contain \"rerank\")"
    )
    var rerankModel: String?

    @Option(
        name: .customLong("reasoning-effort"),
        help: "Chat-template reasoning budget (Qwen3.8: xhigh | medium | low), or `none` to disable thinking entirely"
    )
    var reasoningEffort: String?

    @Option(name: .customLong("top-k"), help: "Keep only the k most likely tokens (unset disables)")
    var topK: Int?

    @Option(name: .customLong("min-p"), help: "Drop tokens below this fraction of the top token's probability")
    var minP: Float?

    @Option(name: .customLong("seed"), help: "Seed the sampler so a non-greedy run is reproducible")
    var seed: UInt64?

    @OptionGroup var kvCache: KVCacheOptions

    @OptionGroup var expertStreaming: ExpertStreamingOptions

    @Option(name: .customLong("repetition-penalty"), help: "Penalty on recently emitted tokens, e.g. 1.1 (default: off)")
    var repetitionPenalty: Float?

    @Option(
        name: .customLong("prompt-cache-mb"),
        help: "Memory for the prompt-cache prefix tree shared by all requests, in MB (default: half of what Metal's recommended working set leaves after the weights; 0 disables)"
    )
    var promptCacheMB: Int?

    @OptionGroup var promptCacheDisk: PromptCacheDiskOptions

    @Option(
        name: .customLong("hybrid-anchors"),
        help: "Recurrent-state snapshots kept per conversation at message boundaries (assistant turn starts, tool result and reasoning ends, at least 512 tokens apart) on hybrid models such as Qwen3.5, Nemotron-H or LFM2, so a prompt edited mid-history resumes from the last one before the edit; each costs one recurrent state from the prompt-cache budget (default: 8; 0 disables)"
    )
    var hybridAnchors = 8

    @Flag(
        name: .customLong("cache-blend"),
        help: "Reuse the KV of prompt chunks (split after blank lines) in any order, recomputing only the most deviating tokens (CacheBlend, approximate; Llama and Qwen3 only; takes half the prompt-cache budget)"
    )
    var cacheBlend = false

    @Option(
        name: .customLong("cache-blend-ratio"),
        help: "Share of reused chunk tokens recomputed by --cache-blend, 0...1 (default: 0.3; 0 is fastest, 1 ≈ a full prefill up to numerical noise)"
    )
    var cacheBlendRatio: Float = 0.3

    @Flag(
        name: .customLong("speculative-lookup"),
        help: "Draft up to 16 tokens per step by matching the last 3-5 tokens against the prompt and output, verified in one forward pass (faster when the answer copies its context, slower on free prose; greedy output can differ at floating-point near-ties; attention-only models and dense recurrent hybrids such as Qwen3.5 or Nemotron-H, without --kv-bits)"
    )
    var speculativeLookup = false

    @Flag(
        name: .customLong("speculative-recycling"),
        help: "With speculative lookup on, draft up to 6 tokens from the model's own argmax successors (Token Recycling) when no n-gram matches; measured neutral to 1.2x over lookup alone on reasoning-style text, no gain on prose"
    )
    var speculativeRecycling = false

    @Flag(
        name: .customLong("mtp"),
        help: "Draft with the model's multi-token-prediction head (mtp.safetensors, e.g. MTPLX packs of Qwen3.5/3.8 and Bonsai 2), verified in one forward pass and exact at any temperature (greedy drafts, the target samples every token); requests without JSON constraints, speculative lookup, --kv-bits or --kv-zones and prompts up to 8192 tokens (prefilled in one forward; a follow-up turn reuses the stored prefix from the prompt cache and prefills only the new tail)"
    )
    var mtp = false

    @Option(
        name: .customLong("mtp-block"),
        help: "With --mtp, the largest verify round: one token the target already chose plus up to N-1 drafted by the MTP head. Rounds start at one drafted token, grow by one after four fully accepted rounds and drop back to one on a rejection (default 4; Qwen3.5-family targets rewind up to 7 drafted tokens, others clamp to their limit; 2 drafts one token every round)"
    )
    var mtpBlock = 4

    @Flag(
        name: .customLong("experimental-samplers"),
        help: "Accept the experimental dry_* (DRY repetition penalty) and xtc_* (exclude top choices) request fields. Measured on a 2-bit Qwen3.8, DRY turned a verbatim loop that RepetitionGuard stops into a loop with corrupted numbers that runs to max_tokens; on models that do not loop it changed nothing. Without this flag those fields are refused with 400"
    )
    var experimentalSamplers = false

    @OptionGroup var experimental: ExperimentalServeOptions

    @OptionGroup var thinking: ThinkingOptions

    @Flag(
        name: .customLong("grammar-jump-forward"),
        help: "Feed XGrammar-forced spans in one forward pass during constrained decoding (opt-in; JSON schema and forced tool calls)"
    )
    var grammarJumpForward = false

    @OptionGroup var network: ServeNetworkOptions

    @Option(
        name: .customLong("max-concurrent"),
        help: "Generations decoding at the same time; each holds its own KV cache (default: 1)"
    )
    var maxConcurrent: Int = 1

    @Option(
        name: .customLong("batch-slots"),
        help: "Generations decoded together in one batched forward, on models whose KV cache is plain attention in every layer; needs --max-concurrent 2 or more (default: 1, off)"
    )
    var batchSlots: Int = 1

    @Flag(name: .customLong("wired-limit"), inversion: .prefixedNo, help: "Hold the wired memory limit at Metal's recommended working set while a model is loaded (on Gemma 4 26B-A4B it cut swap by 41% under competing memory pressure and changed nothing without it)")
    var wiredLimit: Bool = true

    @Flag(name: .customLong("require-verified"), help: "Before loading, re-hash every file of the model snapshot against its sidecar or Hub tree API digest and refuse to start when any file is unverified or mismatched; a download it needs refuses files the server sends no digest for") var requireVerified = false

    @Flag(name: .customLong("model-probe"), inversion: .prefixedNo, help: "Probe a model once per fingerprint at load (chat template, tokenizer round trip, stop tokens, finite logits, entropy, perplexity, two greedy known answers; cached in ~/.basalt/model-probes) and refuse to load it when a check fails")
    var modelProbe: Bool = true

    @Option(
        name: .customLong("max-queue"),
        help: "Requests waiting for a generation slot; beyond it the server answers 503 with Retry-After (default: 8)"
    )
    var maxQueue: Int = 8

    @Option(name: .customLong("max-tokens"), help: "Default maximum tokens to generate (default: 65536, range: 2048-128000)")
    var maxTokens: Int = 65536

    @Flag(help: "Start MCP stdio transport in addition to REST")
    var mcp: Bool = false

    @Flag(help: "Start REST only, suppress MCP stdio")
    var rest: Bool = false

    @Option(name: .customLong("adapter"), help: "Serve a LoRA adapter on the loaded model as name=directory (an mlx-lm adapter directory: adapter_config.json and adapters.safetensors), checked against the model at startup; repeatable. A request picks one with the X-Basalt-Adapter header (unknown name: 400); requests for different adapters take turns, the same adapter runs concurrently") var adapters: [String] = []

    @Option(name: .customLong("image-dir"), help: "With --mcp, a directory the generate tool may read image paths from, after resolving symlinks; repeatable (default: none, every path is refused)") var imageDirs: [String] = []

    @Flag(
        name: .customLong("research-tools"),
        help: "With --mcp, also list and accept the experiment tools (intervene, weights_delta, overlays_list, overlay_undo, experiment_compare, adapter_apply, gate_check, tensor_pack, tensor_unpack, train_experts, steer_search, steer_commit, introspect): in-memory overlays with undo and a TTL (an adapter that passed its quality gate has none); steering and ablation apply to research calls only, while a weight delta, adapter or pack changes the live model, marks every response as modified and turns off prompt-cache reuse until it is undone; model files on disk are never written (train_experts saves its adapter to a new directory) (default: off; the read-only X-ray tools are always on)"
    )
    var researchTools = false

    @Flag(name: .customLong("dry-run"), help: "Perform a dry-run memory feasibility assessment and exit without starting the server")
    var dryRun: Bool = false

    @Flag(name: .customLong("stats"), help: "Periodically output inference telemetry (tok/s, TTFT, RAM, MLX GPU memory, active requests)")
    var stats: Bool = false

    @Option(name: .customLong("stats-interval"), help: "Interval in seconds between periodic telemetry stats outputs (default: 5)")
    var statsInterval: Int = 5

    @Option(
        name: .customLong("canary-minutes"),
        help: "Every N minutes, when no request is running or queued, send one tiny known-answer request (17 + 25, greedy, thinking off) at background priority and count pass/fail in the telemetry snapshot (default: off)"
    )
    var canaryMinutes: Int?

    @Flag(
        name: .customLong("no-memory-coordinator"),
        help: "Keep the load-time memory limits: do not shrink the MLX allocator cache and prompt cache, or refuse long prompts, when macOS reports memory pressure"
    )
    var noMemoryCoordinator = false

    @Flag(
        name: .customLong("no-thermal-policy"),
        help: "Keep the full prefill chunk and admit long prompts when macOS reports a serious or critical thermal state"
    )
    var noThermalPolicy = false

    @OptionGroup var cloud: CloudFallbackOptions

    @OptionGroup var sessionLog: SessionLogOptions

    @Option(name: .long, help: "Log level: debug | info | warn | error (default: info)")
    var logLevel: ServerConfig.LogLevel = .info

    static let defaultModel = "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit"

    mutating func run() async throws {
        if wiredLimit { MLXInferenceEngine.renewsResidency() }
        if experimental.smallMKernel { MLXInferenceEngine.enablesSmallMKernel() }
        if dryRun { return await assessMemory() }

        let network = try resolveAccess()

        let resolvedPath = try await resolveModelPath()

        var config = try ServerConfig(
            modelPath: resolvedPath,
            restPort:  network.port,
            host:      network.host,
            apiKey:    network.apiKey,
            access:    .init(
                rateLimitPerMinute: network.rateLimit, rateBurst: network.rateBurst,
                allowedOrigins: network.allowedOrigins, virtualKeys: network.virtualKeys)
        )
        config.modelId = model ?? modelPath
        config.maxTokens = maxTokens
        config.logLevel = logLevel
        config.experimentalZeroCopyBody = experimental.zeroCopyBody

        BasaltLogger.globalLevel = config.logLevel.basaltLevel

        let orchestrator = try await makeOrchestrator(maxTokens: config.maxTokens)

        let embeddings = try await loadEmbeddings()

        try await load(orchestrator, from: resolvedPath)
        if !noMemoryCoordinator { await orchestrator.coordinateMemory(with: MemoryCoordinator()) }
        if !noThermalPolicy { await orchestrator.coordinateThermal() }

        try await attachCloudFallback(to: orchestrator, virtualKeys: config.virtualKeys)

        let startREST = !mcp || rest
        let startMCP  = mcp

        if startREST { log("REST → http://\(config.host):\(config.restPort)") }
        if startMCP  { log("MCP → JSON-RPC 2.0 on stdin/stdout") }

        let mcpServer = startMCP ? MCPServer(
            orchestrator: orchestrator,
            modelId: resolvedPath.split(separator: "/").last.map { String($0) },
            library: HarnessExperienceLibrary(rootDirectory: HarnessExperienceLibrary.defaultRoot)
        ).withResearchTools(researchTools).withImageDirectories(imageDirs.map { URL(fileURLWithPath: $0) }) : nil

        try await Self.runServers(
            ServerLaunch(
                config: config, startREST: startREST, mcpServer: mcpServer, embeddings: embeddings,
                statsInterval: stats ? min(max(1, statsInterval), 86_400) : nil, canaryInterval: canaryMinutes.map { .seconds($0 * 60) }),
            orchestrator: orchestrator)
    }
}

extension ServeCommand {
    private func makeOrchestrator(maxTokens: Int) async throws -> ModelOrchestratorActor {
        let engine = try makeEngine()
        await engine.probeModels(modelProbe)
        await engine.streamExperts(expertStreaming.setting)
        if batchSlots > 1, maxConcurrent < 2 {
            log("\(yellow("[warn]")) --batch-slots \(batchSlots) has no effect with --max-concurrent \(maxConcurrent): only one generation runs at a time")
        }
        let sessionLogWriter = sessionLog.writer()
        if let sessionLogWriter {
            log("Session log: \(sessionLog.mode.rawValue) → \(sessionLogWriter.directory.path) (0700; `--session-log meta` keeps no content, `--session-log off` disables)")
        }
        let orchestrator = ModelOrchestratorActor(
            engine: engine, maxTokens: maxTokens, kv: .init(precision: kvCache.bits?.fixedPrecision, adaptive: kvCache.bits == .auto, zones: kvCache.zones),
            gate: GenerationGate(maxConcurrent: maxConcurrent, maxQueue: maxQueue), sessionLog: sessionLogWriter)
        await orchestrator.setAdaptiveThinkingBudget(thinking.experimentalAdaptiveThinkingBudget)
        return orchestrator
    }

    static func resolveNetwork(_ flags: NetworkSettings.Flags = NetworkSettings.Flags(), configFile: String?) throws -> NetworkSettings {
        try NetworkSettings.resolve(flags, file: ConfigSection.serverConfig(at: configFile))
    }

    static func resolveCloudFallback(
        _ flags: CloudFallbackSettings.Flags = CloudFallbackSettings.Flags(), disabled: Bool = false, configFile: String?,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        savedOpenRouterKey: () -> String? = { CloudCommand.loadConfig()?.apiKey }
    ) throws -> CloudFallbackSettings? {
        guard !disabled else { return nil }
        return try CloudFallbackSettings.resolve(
            flags, section: ConfigSection.cloudFallback(at: configFile), environment: environment, savedOpenRouterKey: savedOpenRouterKey)
    }
}

enum KVBitsSetting: Equatable, ExpressibleByArgument {
    case auto
    case fixed(KVPrecision)

    init?(argument: String) {
        if argument == "auto" {
            self = .auto
        } else if let precision = KVPrecision(argument: argument) {
            self = .fixed(precision)
        } else {
            return nil
        }
    }

    var fixedPrecision: KVPrecision? {
        if case .fixed(let precision) = self { return precision }
        return nil
    }

    var budgetPrecision: KVPrecision? {
        fixedPrecision ?? KVCacheGeometry.adaptiveKVPrecisionLadder.last ?? nil
    }

    static let zonesHelp = ArgumentHelp(
        "With --kv-bits: keep the first \(KVZones.standard.sinkTokens) and the last \(KVZones.standard.windowTokens) tokens of each full-attention layer in fp16 and quantize the tokens between them as they arrive")

    static let zonesAssumption = "assumed to apply; serve checks the model at load and sizes a model whose attention bypasses the zoned cache (Gemma 4) or whose head dimension has no quantization group as uniform --kv-bits, k8v4 as 8-bit"

    static let evictHelp = ArgumentHelp(
        "With --kv-zones: drop this fraction (0 to 0.9) of the middle tokens of each full-attention layer as they leave the recent window, the ones whose keys have the largest L2 norm (default: keep all)")

    static func evictionSizing(_ fraction: Double?) -> String? {
        guard let fraction, fraction > 0 else { return nil }
        return "budget sized with \(Int((fraction * 100).rounded()))% of each full-attention layer's quantized middle evicted; serve keeps every token, and sizes so, on a model that prefills more than the recent window in one pass"
    }

    static func zones(_ enabled: Bool, evicting fraction: Double? = nil) -> KVZones? {
        enabled ? KVZones.standard.evicting(fraction ?? 0) : nil
    }

    static func validate(zones: Bool, bits: KVBitsSetting?, evict: Double? = nil) throws {
        if let conflict = KVPrecision.flagConflict(kvBitsSet: bits != nil, fixed: bits?.fixedPrecision, zones: zones) {
            throw ValidationError(conflict)
        }
        guard let evict else { return }
        guard zones else { throw ValidationError("--kv-evict needs --kv-zones: it drops tokens from the zoned cache's quantized middle") }
        guard KVZones.evictionFractions.contains(evict) else {
            throw ValidationError("--kv-evict takes a fraction between 0 and 0.9, got \(evict)")
        }
    }
}

// The CLI parses the level; the type stays in BasaltCore without depending on ArgumentParser.
extension ServerConfig.LogLevel: ExpressibleByArgument {}

extension SessionLogMode: ExpressibleByArgument {}
