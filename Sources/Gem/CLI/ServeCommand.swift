import Foundation
import ArgumentParser

// MARK: — serve (subcommand)

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the Gemm inference server (REST + optional MCP stdio)"
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
        help: "Chat-template reasoning budget for models that expose one (Qwen3.8: xhigh | medium | low)"
    )
    var reasoningEffort: String?

    @Option(name: .shortAndLong, help: "REST API port (default: 8080)")
    var port: Int = 8080

    @Option(name: .long, help: "Bind address (default: 127.0.0.1)")
    var host: String = "127.0.0.1"

    @Option(name: .customLong("max-tokens"), help: "Default maximum tokens to generate (default: 65536, range: 2048-128000)")
    var maxTokens: Int = 65536

    @Flag(help: "Start MCP stdio transport in addition to REST")
    var mcp: Bool = false

    @Flag(help: "Start REST only, suppress MCP stdio")
    var rest: Bool = false

    @Flag(name: .customLong("dry-run"), help: "Perform a dry-run memory feasibility assessment and exit without starting the server")
    var dryRun: Bool = false

    @Option(name: .long, help: "Log level: debug | info | warn | error (default: info)")
    var logLevel: ServerConfig.LogLevel = .info

    // MARK: — Run

    private static let logger = GemLogger(module: "ServeCommand")
    private static let defaultModel = "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit"

    mutating func run() async throws {
        let targetModel = model ?? modelPath ?? Self.defaultModel

        if dryRun {
            let evaluator = MemoryEvaluator()
            log("🔍 Performing dry-run memory assessment for \(bold(targetModel))…")
            let assessment = await evaluator.evaluate(modelId: targetModel)

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

        let resolvedPath = try await resolveModelPath()

        let config = ServerConfig(
            modelPath: resolvedPath,
            modelId:   model ?? modelPath,
            restPort:  port,
            host:      host,
            maxTokens: maxTokens,
            logLevel:  logLevel
        )

        // Apply log level from config
        switch config.logLevel {
        case .debug: GemLogger.globalLevel = .debug
        case .info:  GemLogger.globalLevel = .info
        case .warn:  GemLogger.globalLevel = .warn
        case .error: GemLogger.globalLevel = .error
        }

        // Shared inference engine + orchestrator
        let engine       = MLXInferenceEngine(reasoningEffort: reasoningEffort)
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: config.maxTokens)

        log("Loading model from \(bold(resolvedPath))…")
        do {
            try await orchestrator.loadModel(path: resolvedPath)
            await orchestrator.setModelId(model ?? modelPath ?? resolvedPath)
            log("Model ready.")
        } catch {
            log("\(yellow("[warn]")) \(error.localizedDescription) — running in stub mode.")
        }

        // Default: REST only. MCP stdio can be added with --mcp.
        // When --mcp is set without --rest, MCP still starts alongside REST.
        let startREST = !mcp || rest || (!mcp && !rest)
        let startMCP  = mcp

        if startREST { log("REST → http://\(host):\(port)") }
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
            try await group.next()
            group.cancelAll()
        }
    }

    // MARK: — Path resolution

    private func resolveModelPath() async throws -> String {
        if let explicit = modelPath { return explicit }

        let modelId = model ?? Self.defaultModel
        if model == nil { log("No model specified — using default: \(dim(modelId))") }
        guard modelId.contains("/") else { return modelId }  // treat as local path

        let snapshot = ModelCache.cacheDir(for: modelId)

        // If quant is explicit, use it
        if let quantFolder = quant {
            let target = snapshot.appendingPathComponent(quantFolder)
            if FileManager.default.fileExists(atPath: target.appendingPathComponent("config.json").path) {
                return target.path
            }
        }

        // If no quant specified, try root first, then search common quant folders
        if FileManager.default.fileExists(atPath: snapshot.appendingPathComponent("config.json").path) {
            return snapshot.path
        }

        // Auto-detect available quant folder if root config.json doesn't exist
        let commonQuantFolders = ["2bit", "4bit", "8bit", "6bit", "bf16", "fp16"]
        for folder in commonQuantFolders {
            let target = snapshot.appendingPathComponent(folder)
            if FileManager.default.fileExists(atPath: target.appendingPathComponent("config.json").path) {
                log("Auto-detected quantization: \(dim(folder))")
                return target.path
            }
        }

        log("Model not in local cache: \(bold(modelId))\(quant.map { " [\($0)]" } ?? ""). Downloading…")
        let hub = HuggingFaceHub()
        do {
            nonisolated(unsafe) var lastFile = ""
            let dest = try await hub.download(repoId: modelId, subfolder: quant, token: nil) { filename, downloaded, total in
                if filename != lastFile {
                    if !lastFile.isEmpty { print() }
                    lastFile = filename
                }
                printFileProgress(filename: filename, downloaded: downloaded, total: total)
            }
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
