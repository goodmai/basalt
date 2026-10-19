import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference
import BasaltServer

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

    @Option(name: .customLong("kv-bits"), help: "Quantize the KV cache to 4 or 8 bits (default: full precision)")
    var kvBits: Int?

    @Option(name: .customLong("repetition-penalty"), help: "Penalty on recently emitted tokens, e.g. 1.1 (default: off)")
    var repetitionPenalty: Float?

    @Option(
        name: .customLong("prompt-cache-mb"),
        help: "Memory for the prompt-cache prefix tree shared by all requests, in MB (default: half of what Metal's recommended working set leaves after the weights; 0 disables)"
    )
    var promptCacheMB: Int?

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
        name: .customLong("config"),
        help: "Path to JSON server configuration file (default: ~/.gem/config.json when it exists)"
    )
    var configFile: String?

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

    @Option(name: .long, help: "Log level: debug | info | warn | error (default: info)")
    var logLevel: ServerConfig.LogLevel = .info

    // MARK: — Run

    private static let logger = GemLogger(module: "ServeCommand")
    private static let defaultModel = "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit"

    func validate() throws {
        if let rateLimit, rateLimit < 0 {
            throw ValidationError("--rate-limit must be 0 (off) or a positive number of requests per minute, got \(rateLimit)")
        }
    }

    mutating func run() async throws {
        let targetModel = model ?? modelPath ?? Self.defaultModel

        if dryRun {
            let evaluator = MemoryEvaluator()
            log("🔍 Performing dry-run memory assessment for \(bold(targetModel))…")
            let assessment = await evaluator.evaluate(modelId: targetModel, kvBits: kvBits)

            fputs("\n  Model:           \(assessment.modelName) (\(assessment.modelId))\n", stderr)
            fputs("  Required RAM:    \(assessment.requiredRAMFormatted)\n", stderr)
            fputs("  Available RAM:   \(assessment.availableRAMFormatted)\n", stderr)
            fputs("  Physical RAM:    \(assessment.totalRAMFormatted)\n", stderr)
            fputs("  Fit Level:       \(assessment.fitLevel.emoji) \(assessment.fitLevel.rawValue)\n", stderr)
            fputs("  Context Budget:  \(assessment.maxContextBudgetTokens) tokens\n\n", stderr)

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
            rateLimitPerMinute: network.rateLimit
        )

        // Apply log level from config
        switch config.logLevel {
        case .debug: GemLogger.globalLevel = .debug
        case .info:  GemLogger.globalLevel = .info
        case .warn:  GemLogger.globalLevel = .warn
        case .error: GemLogger.globalLevel = .error
        }

        // Shared inference engine + orchestrator
        let engine       = MLXInferenceEngine(
            reasoningEffort: reasoningEffort,
            promptCacheBytes: promptCacheMB.map { min(max(0, $0), 1 << 30) << 20 },
            kvBits: kvBits,
            repetitionPenalty: repetitionPenalty,
            topK: topK,
            minP: minP,
            seed: seed
        )
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: config.maxTokens, kvBits: kvBits)

        log("Loading model from \(bold(resolvedPath))…")
        do {
            try await orchestrator.loadModel(path: resolvedPath)
            // The repo id, not the resolved path, so MCP status reports what an agent asked for.
            await orchestrator.setModelId(model ?? modelPath ?? Self.defaultModel)
            log("Model ready.")
        } catch {
            log("\(yellow("[warn]")) \(error.localizedDescription) — running in stub mode.")
        }

        // Default: REST only. MCP stdio can be added with --mcp.
        // When --mcp is set without --rest, MCP still starts alongside REST.
        let startREST = !mcp || rest || (!mcp && !rest)
        let startMCP  = mcp

        if startREST { log("REST → http://\(config.host):\(config.restPort)") }
        if startMCP  { log("MCP → JSON-RPC 2.0 on stdin/stdout") }

        try await withThrowingTaskGroup(of: Void.self) { group in
            if startREST {
                let restServer = RESTServer(orchestrator: orchestrator, config: config)
                group.addTask { try await restServer.run() }
            }
            if startMCP {
                let mcpServer = MCPServer(
                    orchestrator: orchestrator,
                    modelId: resolvedPath.split(separator: "/").last.map(String.init)
                )
                group.addTask { await mcpServer.run() }
            }
            if stats {
                // Clamped to a day: Task.sleep would trap on an interval too large for UInt64 ns.
                let interval = min(max(1, statsInterval), 86_400)
                group.addTask {
                    // Exit quietly on shutdown: a CancellationError from sleep must not surface
                    // as the serve command's error once the group is cancelled.
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

    // MARK: — Network resolution

    /// ~/.gem/config.json is what `onboard` writes; serve reads it without --config so those values apply.
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
        configFile: String?
    ) throws -> (port: Int, host: String, apiKey: String?, rateLimit: Int?) {
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
        return (
            port: try cliPort ?? value("port", as: Int.self) ?? 8080,
            host: try cliHost ?? value("host", as: String.self) ?? "127.0.0.1",
            apiKey: try cliKey ?? value("api_key", as: String.self),
            rateLimit: rateLimit
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

// The CLI parses the level; the type stays in BasaltCore without depending on ArgumentParser.
extension ServerConfig.LogLevel: ExpressibleByArgument {}
