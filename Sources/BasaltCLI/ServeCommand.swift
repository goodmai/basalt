import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer
import BasaltSpeech
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
        name: .customLong("embedding-adapter"),
        help: "LoRA adapter directory (adapter_config.json and adapters.safetensors) applied to --embedding-model at startup for bidirectional mean-pooled embeddings"
    )
    var embeddingAdapter: String?

    @Option(name: .customLong("decision-model"), help: "OpenJev decision model served on /v1/systemone (single-pass noul/choice/score): \"self\" reads option letters from the chat model; an OpenJev head bundle (openjev_config.json, head.safetensors, PEFT adapter) runs on its Qwen3 base; any other repo, such as openjev/OpenJev-Flash-9B-MLX, is loaded beside the chat model and read out by option letters")
    var decisionModel: String?

    @Option(
        name: .customLong("rerank-model"),
        help: "Reranker served on /v1/rerank next to the chat model: repo ID or local path of an encoder cross-encoder, Qwen3-Reranker or Jina reranker (its name must contain \"rerank\")"
    )
    var rerankModel: String?

    @OptionGroup var sampling: SamplingOptions

    @OptionGroup var kvCache: KVCacheOptions

    @OptionGroup var expertStreaming: ExpertStreamingOptions

    @OptionGroup var promptCache: PromptCacheOptions

    @OptionGroup var promptCacheDisk: PromptCacheDiskOptions

    @OptionGroup var speculative: SpeculativeOptions

    @OptionGroup var experimental: ExperimentalServeOptions

    @OptionGroup var thinking: ThinkingOptions

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

        let engine = try makeEngine()
        let orchestrator = try await makeOrchestrator(engine: engine, maxTokens: config.maxTokens)

        let embeddings = try await loadEmbeddings()
        let decisions = try await loadDecisions(chat: engine)

        try await load(orchestrator, from: resolvedPath)
        if !noMemoryCoordinator { await orchestrator.coordinateMemory(with: MemoryCoordinator()) }
        if !noThermalPolicy { await orchestrator.coordinateThermal() }
        await experimental.attachMediaFallbacks(to: orchestrator)

        try await attachCloudFallback(to: orchestrator, virtualKeys: config.virtualKeys)

        let startREST = !mcp || rest
        let startMCP  = mcp

        if startREST { log("REST → http://\(config.host):\(config.restPort)") }
        if startMCP  { log("MCP → JSON-RPC 2.0 on stdin/stdout") }

        let mcpServer = startMCP ? MCPServer(
            orchestrator: orchestrator,
            modelId: resolvedPath.split(separator: "/").last.map { String($0) },
            library: HarnessExperienceLibrary(rootDirectory: HarnessExperienceLibrary.defaultRoot)
        ).withResearchTools(researchTools).withImageDirectories(imageDirs.map { URL(fileURLWithPath: $0) }).withDecisions(decisions) : nil

        try await Self.runServers(
            ServerLaunch(
                config: config, startREST: startREST, mcpServer: mcpServer, embeddings: embeddings, decisions: decisions,
                statsInterval: stats ? min(max(1, statsInterval), 86_400) : nil, canaryInterval: canaryMinutes.map { .seconds($0 * 60) }),
            orchestrator: orchestrator)
    }
}

extension ServeCommand {
    private func makeOrchestrator(engine: MLXInferenceEngine, maxTokens: Int) async throws -> ModelOrchestratorActor {
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
