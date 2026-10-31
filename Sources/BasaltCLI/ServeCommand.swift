import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer
import BasaltHarness
import BasaltCloud

// MARK: — serve (subcommand)

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the Basalt inference server (REST + optional MCP stdio)"
    )

    // MARK: — Arguments

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

    @Option(
        name: .customLong("quant"),
        help: "Quantization subfolder inside the repo (e.g. 2bit | 4bit) — for repos shipping several variants"
    )
    var quant: String?

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

    @Option(
        name: .customLong("kv-bits"),
        help: "Quantize the KV cache to 4 or 8 bits, k8v4 (8-bit keys, 4-bit values; needs --kv-zones; experimental k8v4h also Walsh–Hadamard-rotates the values), or `auto`: fp16 until a request's max_tokens does not fit, then 8-bit, k8v4 with --kv-zones, then 4-bit (default: full precision)"
    )
    var kvBits: KVBitsSetting?

    @Flag(name: .customLong("kv-zones"), help: KVBitsSetting.zonesHelp)
    var kvZones: Bool = false

    @Option(name: .customLong("kv-evict"), help: KVBitsSetting.evictHelp)
    var kvEvict: Double?

    @Option(name: .customLong("repetition-penalty"), help: "Penalty on recently emitted tokens, e.g. 1.1 (default: off)")
    var repetitionPenalty: Float?

    @Option(
        name: .customLong("prompt-cache-mb"),
        help: "Memory for the prompt-cache prefix tree shared by all requests, in MB (default: half of what Metal's recommended working set leaves after the weights; 0 disables)"
    )
    var promptCacheMB: Int?

    @OptionGroup var promptCacheDisk: PromptCacheDiskOptions

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
        name: .customLong("grammar-jump-forward"),
        help: "Feed XGrammar-forced spans in one forward pass during constrained decoding (opt-in; JSON schema and forced tool calls)"
    )
    var grammarJumpForward = false

    // Optional so an explicit `--port 8080` can be told apart from "not passed":
    // precedence is CLI > --config file > default.
    @Option(name: .shortAndLong, help: "REST API port (default: 8080)")
    var port: Int?

    @Option(name: .long, help: "Bind address (default: 127.0.0.1)")
    var host: String?

    @Option(
        name: .customLong("api-key"),
        help: "API key required for authenticating requests, mandatory when binding to a non-loopback address. Prefer the BASALT_API_KEY environment variable: argv is visible to every local user via ps."
    )
    var apiKey: String?

    @Flag(
        name: .customLong("auto-generate-key"),
        help: "Automatically generate a cryptographically strong API key if none is provided on non-loopback bind (a new key on every start — pass a fixed key for clients that reconnect)"
    )
    var autoGenerateKey: Bool = false

    @Option(
        name: .customLong("rate-limit"),
        help: "Server-wide request limit per minute, 429 beyond it (default: off on loopback, \(ServerConfig.defaultRateLimitPerMinute) otherwise; 0 disables)"
    )
    var rateLimit: Int?

    @Option(
        name: .customLong("rate-burst"),
        help: "Maximum burst of requests allowed before 429 (default: min(rate-limit, 10); 0 disables burst capping)"
    )
    var rateBurst: Int?

    @Option(
        name: .customLong("allow-origin"),
        help: "Browser origin allowed to call the API besides loopback ones, e.g. https://app.example:3000 (repeatable; also allowed_origins in ~/.basalt/config.json)"
    )
    var allowOrigins: [String] = []

    @Option(
        name: .customLong("config"),
        help: "Path to JSON server configuration file (default: ~/.basalt/config.json when it exists)"
    )
    var configFile: String?

    @Option(
        name: .customLong("max-concurrent"),
        help: "Generations decoding at the same time; each holds its own KV cache (default: 1)"
    )
    var maxConcurrent: Int = 1

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

    @Flag(name: .customLong("dry-run"), help: "Perform a dry-run memory feasibility assessment and exit without starting the server")
    var dryRun: Bool = false

    @Flag(name: .customLong("stats"), help: "Periodically output inference telemetry (tok/s, TTFT, RAM, MLX GPU memory, active requests)")
    var stats: Bool = false

    @Option(name: .customLong("stats-interval"), help: "Interval in seconds between periodic telemetry stats outputs (default: 5)")
    var statsInterval: Int = 5

    @Option(
        name: .customLong("cloud-fallback"),
        help: "Model on an OpenAI-compatible server that answers when the local model cannot: the request does not fit in free memory, the prompt exceeds --local-context-limit, or the task rates beyond the local model (also cloud_fallback.model in the config; spend is capped by the `basalt cloud` daily/monthly budgets)"
    )
    var cloudFallback: String?

    @Option(
        name: .customLong("cloud-base-url"),
        help: "OpenAI-compatible endpoint for --cloud-fallback, e.g. http://192.168.1.20:8000/v1 (default: https://openrouter.ai/api/v1; also cloud_fallback.base_url)"
    )
    var cloudBaseURL: String?

    @Flag(name: .customLong("no-cloud-fallback"), help: "Never switch to the cloud fallback, even if the config enables it")
    var noCloudFallback = false

    @Option(
        name: .customLong("router-params"),
        help: "JSON with trained Bradley–Terry router parameters (strongEmbedding, weakEmbedding, weights, projection); enables the capability reason of the cloud fallback (also cloud_fallback.router_params)"
    )
    var routerParams: String?

    @Option(
        name: .customLong("cost-threshold"),
        help: "Win probability θ of the cloud model above which a task counts as beyond the local model (default 0.5; also cloud_fallback.cost_threshold)"
    )
    var costThreshold: Double?

    @Option(
        name: .customLong("cache-stickiness"),
        help: "Penalty λ for leaving a warm session's model, scaled by its cached prefix (default 0.1; also cloud_fallback.cache_stickiness); sessions are told apart by the x-session-id header"
    )
    var cacheStickiness: Double?

    @Option(
        name: .customLong("local-context-limit"),
        help: "Prompt tokens beyond which the local model counts as degraded and the cloud fallback answers instead"
    )
    var localContextLimit: Int?

    @Option(
        name: .customLong("session-log"),
        help: "JSONL session log of requests and replies: full (default; prompts and replies are stored on this machine) | meta (metrics, sizes and hashes only) | off"
    )
    var sessionLog: SessionLogMode = .full

    @Option(
        name: .customLong("session-log-redact"),
        help: "Replace private keys and API tokens (sk-, ghp_, AKIA, xox, Bearer) in full-mode records with [REDACTED:*]: on (default) | off; best effort, not a guarantee"
    )
    var sessionLogRedact: Toggle = .on

    @Option(name: .customLong("session-idle-minutes"), help: "Write session_suspend and close a session file after this many idle minutes; the next request resumes it (default: 30)")
    var sessionIdleMinutes: Int = 30

    @Option(name: .customLong("session-log-dir"), help: "Directory of the session log (default: ~/.basalt/sessions)")
    var sessionLogDir: String?

    @Option(name: .customLong("session-log-retention-days"), help: "Delete session files older than this many days at start (default: 14)")
    var sessionLogRetentionDays: Int = 14

    @Option(name: .customLong("session-log-max-total-mb"), help: "Delete the oldest session files beyond this total size at start (default: 1024)")
    var sessionLogMaxTotalMB: Int = 1024

    @Option(name: .long, help: "Log level: debug | info | warn | error (default: info)")
    var logLevel: ServerConfig.LogLevel = .info

    // MARK: — Run

    private static let logger = BasaltLogger(module: "ServeCommand")
    private static let defaultModel = "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit"

    struct CloudFallbackSettings: Equatable {
        let model: String
        let baseURL: URL
        let apiKey: String
        let localContextLimit: Int?
        var routerParams: URL? = nil
        var costThreshold: Double? = nil
        var cacheStickiness: Double? = nil
        var sessionBudget: Double? = nil
    }

    static func resolveCloudFallback(
        cliModel: String?, cliBaseURL: String?, cliDisabled: Bool, cliLocalContextLimit: Int?,
        cliRouterParams: String? = nil, cliCostThreshold: Double? = nil, cliCacheStickiness: Double? = nil,
        configFile: String?,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        savedOpenRouterKey: () -> String? = { CloudCommand.loadConfig()?.apiKey }
    ) throws -> CloudFallbackSettings? {
        guard !cliDisabled else { return nil }
        var section: [String: Any] = [:]
        if let configFile {
            let url = URL(fileURLWithPath: (configFile as NSString).expandingTildeInPath)
            let root = try ServerConfig.load(from: url)
            if let raw = root["cloud_fallback"] {
                guard let dict = raw as? [String: Any] else {
                    throw ValidationError("--config: \"cloud_fallback\" must be an object")
                }
                section = dict
            }
        }
        func value<T>(_ key: String, as _: T.Type) throws -> T? {
            guard let raw = section[key], !(raw is NSNull) else { return nil }
            guard CFGetTypeID(raw as CFTypeRef) != CFBooleanGetTypeID() || T.self == Bool.self, let typed = raw as? T else {
                throw ValidationError("--config: \"cloud_fallback.\(key)\" must be \(T.self), got \(raw)")
            }
            return typed
        }
        if cliModel == nil, try value("enabled", as: Bool.self) == false { return nil }
        guard let model = try cliModel ?? value("model", as: String.self), !model.isEmpty else { return nil }
        let rawURL = try cliBaseURL ?? value("base_url", as: String.self) ?? "https://openrouter.ai/api/v1"
        guard let baseURL = URL(string: rawURL), ["http", "https"].contains(baseURL.scheme?.lowercased()), baseURL.host != nil else {
            throw ValidationError("cloud fallback base URL must be http(s)://host[:port]/path, got \(rawURL)")
        }
        let openRouter = OpenRouterClient.isOpenRouter(baseURL)
        let keyVariable = try value("api_key_env", as: String.self) ?? (openRouter ? "OPENROUTER_API_KEY" : nil)
        let apiKey = try value("api_key", as: String.self)
            ?? keyVariable.flatMap { environment[$0] }
            ?? (openRouter ? savedOpenRouterKey() : nil)
            ?? ""
        if openRouter, apiKey.isEmpty {
            throw ValidationError("cloud fallback on OpenRouter needs OPENROUTER_API_KEY, cloud_fallback.api_key or `basalt cloud configure`")
        }
        let limit = try cliLocalContextLimit ?? value("local_context_limit", as: Int.self)
        if let limit, limit <= 0 {
            throw ValidationError("local_context_limit must be a positive number of tokens, got \(limit)")
        }
        let routerParams = try (cliRouterParams ?? value("router_params", as: String.self))
            .map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        let threshold = try cliCostThreshold ?? value("cost_threshold", as: Double.self)
        let stickiness = try cliCacheStickiness ?? value("cache_stickiness", as: Double.self)
        if routerParams == nil, threshold != nil || stickiness != nil {
            throw ValidationError("cost_threshold and cache_stickiness need router_params (trained Bradley–Terry parameters)")
        }
        if let threshold, !(0...1).contains(threshold) {
            throw ValidationError("cost_threshold must be a probability between 0 and 1, got \(threshold)")
        }
        if let stickiness, stickiness < 0 {
            throw ValidationError("cache_stickiness must not be negative, got \(stickiness)")
        }
        let sessionBudget = try value("session_budget", as: Double.self)
        if let sessionBudget, sessionBudget < 0 {
            throw ValidationError("session_budget must not be negative, got \(sessionBudget)")
        }
        return CloudFallbackSettings(
            model: model, baseURL: baseURL, apiKey: apiKey, localContextLimit: limit,
            routerParams: routerParams, costThreshold: threshold, cacheStickiness: stickiness, sessionBudget: sessionBudget)
    }

    static func cloudFallbackRouter(
        _ settings: CloudFallbackSettings, orchestrator: ModelOrchestratorActor, virtualKeys: [VirtualKey] = [],
        costURL: URL = CloudCommand.costURL()
    ) async throws -> ModelRouter {
        let saved = CloudCommand.loadConfig()
        let budget = CostTracker.Budget(dailyLimit: saved?.dailyBudget ?? 5, monthlyLimit: saved?.monthlyBudget ?? 50)
        let tracker = try CostTracker(storageURL: costURL, budget: budget)
        await tracker.setDefaultSessionBudget(settings.sessionBudget)
        for key in virtualKeys {
            if let limit = key.maxBudget { await tracker.setMaxBudget(limit, for: .virtualKey(key.id)) }
        }
        let costRouting: ModelRouter.CostRouting?
        if let params = settings.routerParams {
            let scorer: BradleyTerryRouter
            do {
                scorer = try BradleyTerryRouter.load(from: params)
            } catch {
                throw ValidationError("router_params \(params.path): \(error.localizedDescription)")
            }
            costRouting = ModelRouter.CostRouting(
                policy: StickyRoutingPolicy(
                    strongModel: settings.model, weakModel: "local",
                    threshold: settings.costThreshold ?? 0.5, cacheStickinessPenalty: settings.cacheStickiness ?? 0.1),
                scorer: scorer)
        } else {
            costRouting = nil
        }
        return ModelRouter(
            localOrchestrator: orchestrator,
            cloudClient: try OpenRouterClient(config: .init(
                apiKey: settings.apiKey, baseURL: settings.baseURL, timeout: 120, maxRetries: 2)),
            strategy: .cloudOnly, costRouting: costRouting, costTracker: tracker,
            fallbackCloudModel: settings.model)
    }

    enum Toggle: String, ExpressibleByArgument, CaseIterable { case on, off }

    static func sessionLogWriter(
        mode: SessionLogMode, directory: String?, retentionDays: Int, maxTotalMB: Int, redacts: Bool = true,
        idleMinutes: Int = 30, defaultSession: String = SessionLogWriter.processSession()
    ) -> SessionLogWriter? {
        guard mode != .off else { return nil }
        let url = directory.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? SessionLogWriter.defaultDirectory
        SessionLogWriter.applyRetention(directory: url, days: retentionDays, maxTotalBytes: Int64(maxTotalMB) << 20)
        let writer = SessionLogWriter(
            mode: mode, directory: url, redacts: redacts, defaultSession: defaultSession, idleTimeout: TimeInterval(max(1, idleMinutes) * 60))
        Task { await writer.startFlushing() }
        return writer
    }

    func validate() throws {
        guard sessionLogRetentionDays >= 0, sessionLogMaxTotalMB >= 0 else {
            throw ValidationError("--session-log-retention-days and --session-log-max-total-mb must not be negative")
        }
        guard maxConcurrent >= 1 else {
            throw ValidationError("--max-concurrent must be at least 1, got \(maxConcurrent)")
        }
        guard maxQueue >= 0 else {
            throw ValidationError("--max-queue must be 0 or more, got \(maxQueue)")
        }
        if let rateLimit, rateLimit < 0 {
            throw ValidationError("--rate-limit must be 0 (off) or a positive number of requests per minute, got \(rateLimit)")
        }
        if let rateBurst, rateBurst < 0 {
            throw ValidationError("--rate-burst must be 0 (off) or a positive number, got \(rateBurst)")
        }
        guard (0...1).contains(cacheBlendRatio) else {
            throw ValidationError("--cache-blend-ratio must be between 0 and 1, got \(cacheBlendRatio)")
        }
        try KVBitsSetting.validate(zones: kvZones, bits: kvBits, evict: kvEvict)
        try promptCacheDisk.validate(sessionLog: sessionLog)
        guard promptCacheDisk.megabytes == 0 || promptCacheMB != 0 else {
            throw ValidationError("--prompt-cache-disk-mb persists the in-memory prompt cache, which --prompt-cache-mb 0 turns off")
        }
    }

    mutating func run() async throws {
        let targetModel = model ?? modelPath ?? Self.defaultModel

        if dryRun {
            let evaluator = MemoryEvaluator()
            log("🔍 Performing dry-run memory assessment for \(bold(targetModel))…")
            let assessment = await evaluator.evaluate(
                modelId: targetModel, kvPrecision: kvBits?.budgetPrecision, kvZones: KVBitsSetting.zones(kvZones, evicting: kvEvict))

            fputs("\n  Model:           \(assessment.modelName) (\(assessment.modelId))\n", stderr)
            fputs("  Required RAM:    \(assessment.requiredRAMFormatted)\n", stderr)
            fputs("  Available RAM:   \(assessment.availableRAMFormatted)\n", stderr)
            fputs("  Physical RAM:    \(assessment.totalRAMFormatted)\n", stderr)
            fputs("  Fit Level:       \(assessment.fitLevel.emoji) \(assessment.fitLevel.rawValue)\n", stderr)
            fputs("  Context Budget:  \(assessment.contextBudgetLine)\n", stderr)
            if kvZones { fputs("  KV zones:        \(KVBitsSetting.zonesAssumption)\n", stderr) }
            if let line = KVBitsSetting.evictionSizing(kvEvict) { fputs("  KV eviction:     \(line)\n", stderr) }
            fputs("\n", stderr)

            if let warning = assessment.warning {
                fputs(red("  \(warning)") + "\n", stderr)
            }
            if let rec = assessment.recommendation {
                fputs(yellow("  💡 Recommendation: \(rec)") + "\n", stderr)
            }

            if assessment.fitsInMemory {
                fputs(green("  ✓ Dry-run passed: Model fits in memory budget.\n\n"), stderr)
            } else {
                fputs(red("  ❌ Dry-run warning: Model exceeds available RAM!\n\n"), stderr)
            }
            return
        }

        let network = try Self.resolveNetwork(
            cliPort: port, cliHost: host,
            cliApiKey: apiKey ?? ProcessInfo.processInfo.environment["BASALT_API_KEY"],
            cliRateLimit: rateLimit,
            cliRateBurst: rateBurst,
            cliAllowedOrigins: allowOrigins,
            configFile: configFile ?? Self.defaultConfigFileIfPresent()
        )
        // Settle the exposure rule before resolveModelPath(), which may download gigabytes.
        let access: (key: String?, generated: Bool)
        do {
            access = try ServerConfig.resolveApiKey(
                host: network.host, apiKey: network.apiKey, autoGenerateKeyIfMissing: autoGenerateKey
            )
        } catch ServerConfigError.missingApiKeyForNonLoopback(let badHost) {
            log("\(red("Security error:")) Refusing to start on non-loopback interface '\(badHost)' without an API key.")
            log("Pass \(bold("--api-key <secret>")) or \(bold("--auto-generate-key")) to enable access.")
            throw ValidationError("Refusing to start on non-loopback interface '\(badHost)' without an API key. Pass --api-key <secret> or --auto-generate-key to protect external access.")
        }
        if access.generated, let generated = access.key {
            log("⚠️  Binding to non-loopback address '\(network.host)' without explicit key.")
            log("🔑 Auto-generated API key: \(bold(generated))")
        }

        let resolvedPath = try await resolveModelPath()

        let config = try ServerConfig(
            modelPath: resolvedPath,
            modelId:   model ?? modelPath,
            restPort:  network.port,
            host:      network.host,
            maxTokens: maxTokens,
            logLevel:  logLevel,
            apiKey:    access.key,
            rateLimitPerMinute: network.rateLimit,
            rateBurst: network.rateBurst,
            allowedOrigins: network.allowedOrigins,
            virtualKeys: network.virtualKeys
        )

        // Apply log level from config
        switch config.logLevel {
        case .debug: BasaltLogger.globalLevel = .debug
        case .info:  BasaltLogger.globalLevel = .info
        case .warn:  BasaltLogger.globalLevel = .warn
        case .error: BasaltLogger.globalLevel = .error
        }

        // Shared inference engine + orchestrator
        let engine       = MLXInferenceEngine(
            reasoningEffort: reasoningEffort,
            promptCacheBytes: promptCacheMB.map { min(max(0, $0), 1 << 30) << 20 },
            kvPrecision: kvBits?.fixedPrecision,
            kvZones: KVBitsSetting.zones(kvZones, evicting: kvEvict),
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            seed: seed,
            cacheBlendRatio: cacheBlend ? min(max(0, cacheBlendRatio), 1) : nil,
            speculativeLookup: speculativeLookup,
            speculativeRecycling: speculativeRecycling,
            grammarJumpForward: grammarJumpForward,
            promptCacheDisk: try promptCacheDisk.store()
        )
        let sessionLogWriter = Self.sessionLogWriter(
            mode: sessionLog, directory: sessionLogDir, retentionDays: sessionLogRetentionDays, maxTotalMB: sessionLogMaxTotalMB,
            redacts: sessionLogRedact == .on, idleMinutes: sessionIdleMinutes)
        if let sessionLogWriter {
            log("Session log: \(sessionLog.rawValue) → \(sessionLogWriter.directory.path) (0700; `--session-log meta` keeps no content, `--session-log off` disables)")
        }
        let orchestrator = ModelOrchestratorActor(
            engine: engine, maxTokens: config.maxTokens, kvPrecision: kvBits?.fixedPrecision, adaptiveKVPrecision: kvBits == .auto,
            kvZones: KVBitsSetting.zones(kvZones, evicting: kvEvict), gate: GenerationGate(maxConcurrent: maxConcurrent, maxQueue: maxQueue), sessionLog: sessionLogWriter)

        log("Loading model from \(bold(resolvedPath))…")
        do {
            try await orchestrator.loadModel(path: resolvedPath)
            // The repo id, not the resolved path, so MCP status reports what an agent asked for.
            await orchestrator.setModelId(model ?? modelPath ?? Self.defaultModel)
            log("Model ready.")
        } catch {
            if case .modelConfigurationInvalid = error {
                log("\(red("[error]")) \(error.localizedDescription)")
                throw ExitCode.failure
            }
            log("\(yellow("[warn]")) \(error.localizedDescription) — running in stub mode.")
        }

        if let fallback = try Self.resolveCloudFallback(
            cliModel: cloudFallback, cliBaseURL: cloudBaseURL, cliDisabled: noCloudFallback,
            cliLocalContextLimit: localContextLimit, cliRouterParams: routerParams, cliCostThreshold: costThreshold,
            cliCacheStickiness: cacheStickiness, configFile: configFile ?? Self.defaultConfigFileIfPresent())
        {
            let router = try await Self.cloudFallbackRouter(fallback, orchestrator: orchestrator, virtualKeys: config.virtualKeys)
            if OpenRouterClient.isOpenRouter(fallback.baseURL) {
                do {
                    try await router.syncCloudModels()
                } catch {
                    log("\(yellow("[warn]")) Could not fetch OpenRouter prices: \(error.localizedDescription)")
                }
            }
            if await !router.knowsPrice(of: fallback.model) {
                log("\(yellow("[warn]")) No price known for \(fallback.model): its cloud spend is counted as $0 and the budget does not limit it")
            }
            await orchestrator.setCloudFallback(router, arbiter: HybridArbiter(localContextLimit: fallback.localContextLimit))
            log("Cloud fallback → \(bold(fallback.model)) at \(fallback.baseURL.absoluteString)")
        }

        let startREST = !mcp || rest
        let startMCP  = mcp

        if startREST { log("REST → http://\(config.host):\(config.restPort)") }
        if startMCP  { log("MCP → JSON-RPC 2.0 on stdin/stdout") }

        let mcpServer = startMCP ? MCPServer(
            orchestrator: orchestrator,
            modelId: resolvedPath.split(separator: "/").last.map(String.init),
            library: HarnessExperienceLibrary(rootDirectory: HarnessExperienceLibrary.defaultRoot)
        ) : nil

        try await Self.runServers(
            orchestrator: orchestrator,
            config: config,
            startREST: startREST,
            mcpServer: mcpServer,
            statsInterval: stats ? min(max(1, statsInterval), 86_400) : nil
        )
    }

    package static func runServers(
        orchestrator: ModelOrchestratorActor,
        config: ServerConfig,
        startREST: Bool,
        mcpServer: MCPServer?,
        statsInterval: Int? = nil,
        shutdownDeadline: Duration = .seconds(5),
        handleSignals: Bool = true
    ) async throws {
        defer {
            await withTaskCancellationShield {
                await orchestrator.shutdown(deadline: shutdownDeadline)
            }
        }
        try await withThrowingTaskGroup(of: Void.self) { group in
            if startREST {
                let restServer = RESTServer(
                    orchestrator: orchestrator,
                    config: config,
                    shutdownDeadline: shutdownDeadline,
                    handleSignals: handleSignals
                )
                group.addTask { try await restServer.run() }
            }
            if let mcpServer {
                if startREST {
                    group.addTask { await mcpServer.run() }
                } else {
                    group.addTask {
                        try await mcpServer.runServiceGroup(shutdownDeadline: shutdownDeadline, handleSignals: handleSignals)
                    }
                }
            }
            if let interval = statsInterval {
                group.addTask {
                    while (try? await Task.sleep(for: .seconds(interval))) != nil {
                        let snapshot = await orchestrator.telemetrySnapshot()
                        fputs("\(snapshot.formattedStatsLine)\n", stderr)
                    }
                }
            }
            try await group.next()
            group.cancelAll()
        }
    }

    static func defaultConfigFileIfPresent() -> String? {
        let url = ServerConfig.defaultConfigFileURL
        return FileManager.default.fileExists(atPath: url.path) ? url.path : nil
    }

    /// Port, host, API key and rate limit with precedence CLI > `--config` file > default.
    /// An unreadable or malformed config file — or a value of the wrong type in it — is an error,
    /// never a silent fallback to defaults.
    static func resolveNetwork(
        cliPort: Int?,
        cliHost: String?,
        cliApiKey: String?,
        cliRateLimit: Int? = nil,
        cliRateBurst: Int? = nil,
        cliAllowedOrigins: [String] = [],
        configFile: String?
    ) throws -> (port: Int, host: String, apiKey: String?, rateLimit: Int?, rateBurst: Int?, allowedOrigins: [String], virtualKeys: [VirtualKey]) {
        var dict: [String: Any] = [:]
        if let configFile {
            let url = URL(fileURLWithPath: (configFile as NSString).expandingTildeInPath)
            do {
                dict = try ServerConfig.load(from: url)
            } catch {
                throw ValidationError("Cannot read server config \(url.path): \(error.localizedDescription)")
            }
        }
        func value<T>(_ key: String, as _: T.Type) throws -> T? {
            guard let raw = dict[key], !(raw is NSNull) else { return nil }
            // JSONSerialization bridges `true` to NSNumber, which `as? Int` would read as 1.
            guard CFGetTypeID(raw as CFTypeRef) != CFBooleanGetTypeID() || T.self == Bool.self,
                  let typed = raw as? T else {
                throw ValidationError("--config: \"\(key)\" must be \(T.self), got \(raw)")
            }
            return typed
        }
        // An empty --api-key / BASALT_API_KEY is "not set", so it cannot mask the config's key.
        let cliKey = cliApiKey.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 }
        let rateLimit = try cliRateLimit ?? value("rate_limit", as: Int.self)
        if let rateLimit, rateLimit < 0 {
            // Same rule as the flag: a negative value is an error, never a silent "off".
            throw ValidationError("rate_limit must be 0 (off) or a positive number of requests per minute, got \(rateLimit)")
        }
        let rateBurst = try cliRateBurst ?? value("rate_burst", as: Int.self)
        if let rateBurst, rateBurst < 0 {
            throw ValidationError("rate_burst must be 0 (off) or a positive number, got \(rateBurst)")
        }
        let allowedOrigins = try cliAllowedOrigins + (value("allowed_origins", as: [String].self) ?? [])
        for origin in allowedOrigins where RequestGuardMiddleware.parseOrigin(origin) == nil {
            throw ValidationError("allowed origin must be scheme://host[:port]: http or https, no path, no trailing slash, no wildcard; got \(origin)")
        }
        let virtualKeys = try (value("virtual_keys", as: [[String: Any]].self) ?? []).map { entry -> VirtualKey in
            guard let key = entry["key"] as? String, key.count >= 16, let id = entry["id"] as? String, !id.isEmpty else {
                throw ValidationError("virtual_keys: every entry needs \"id\" and a \"key\" of at least 16 characters")
            }
            let budget = entry["max_budget"].flatMap { $0 is NSNull ? nil : $0 }
            if let budget, (budget as? NSNumber).map({ CFGetTypeID($0) == CFBooleanGetTypeID() || $0.doubleValue < 0 }) ?? true {
                throw ValidationError("virtual_keys[\(id)].max_budget must be a non-negative number of USD, got \(budget)")
            }
            return VirtualKey(key: key, id: id, maxBudget: (budget as? NSNumber)?.doubleValue)
        }
        if Set(virtualKeys.map(\.id)).count != virtualKeys.count || Set(virtualKeys.map(\.key)).count != virtualKeys.count {
            throw ValidationError("virtual_keys: ids and keys must be unique")
        }
        return (
            port: try cliPort ?? value("port", as: Int.self) ?? 8080,
            host: try cliHost ?? value("host", as: String.self) ?? "127.0.0.1",
            apiKey: try cliKey ?? value("api_key", as: String.self),
            rateLimit: rateLimit,
            rateBurst: rateBurst,
            allowedOrigins: allowedOrigins,
            virtualKeys: virtualKeys
        )
    }

    // MARK: — Path resolution

    private func resolveModelPath() async throws -> String {
        if let explicit = modelPath { return explicit }

        let modelId = model ?? Self.defaultModel
        if model == nil { log("No model specified — using default: \(dim(modelId))") }
        guard modelId.contains("/") else { return modelId }  // treat as local path

        let snapshot = ModelCache.cacheDir(for: modelId)

        if let resolved = ModelCache.resolve(repoId: modelId, quant: quant) {
            if quant == nil, resolved != snapshot {
                log("Auto-detected quantization: \(dim(resolved.lastPathComponent))")
            }
            return resolved.path
        }

        log("Model not in local cache: \(bold(modelId))\(quant.map { " [\($0)]" } ?? ""). Downloading…")
        let hub = HuggingFaceHub()
        do {
            let dest = try await hub.download(repoId: modelId, subfolder: quant, token: nil,
                                              onFile: DownloadProgressReporter().callAsFunction)
            print("\n")
            return dest.path
        } catch {
            log("\(red("Download failed:")) \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    private func log(_ msg: String) {
        fputs("[serve] \(msg)\n", stderr)
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
        fixedPrecision ?? KVCacheGeometry.adaptiveKVPrecisionLadder.last!
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
