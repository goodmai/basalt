import Foundation
import ArgumentParser

// MARK: — serve (subcommand)

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct ServeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Start the Gemma inference server (MCP stdio + REST)"
    )

    // MARK: — Arguments

    @Option(
        name: .customLong("model"),
        help: "Model repo ID (e.g. mlx-community/gemma-4-e2b-it-4bit) or local path"
    )
    var model: String?

    @Option(
        name: .customLong("model-path"),
        help: "Explicit local path to model directory (overrides --model cache lookup)"
    )
    var modelPath: String?

    @Option(name: .shortAndLong, help: "REST API port")
    var port: Int = 8080

    @Option(name: .long, help: "Bind address for REST")
    var host: String = "127.0.0.1"

    @Flag(help: "Start MCP stdio transport only (no REST)")
    var mcp: Bool = false

    @Flag(help: "Start REST server only (no MCP)")
    var rest: Bool = false

    @Option(name: .customLong("max-tokens"), help: "Default maximum tokens to generate (e.g. 65536, max 65536)")
    var maxTokens: Int = 65536

    @Option(name: .long, help: "Log level: debug | info | warn | error")
    var logLevel: ServerConfig.LogLevel = .info

    // MARK: — Run

    mutating func run() async throws {
        let resolvedPath = try await resolveModelPath()

        let config = ServerConfig(
            modelPath: resolvedPath,
            restPort: port,
            host: host,
            maxTokens: maxTokens,
            logLevel: logLevel
        )

        // Shared inference engine + orchestrator
        let engine = MLXInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: config.maxTokens)

        log("Loading model from \(bold(resolvedPath))…")
        do {
            try await orchestrator.loadModel(path: resolvedPath)
            log("Model ready.")
        } catch {
            log("\(yellow("[warn]")) \(error.localizedDescription) — running in stub mode.")
        }

        let startMCP  = mcp || (!mcp && !rest)
        let startREST = rest || (!mcp && !rest)

        if startREST { log("REST (A2A)  → http://\(host):\(port)") }
        if startMCP  { log("MCP (stdio) → JSON-RPC 2.0 on stdin/stdout") }

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
        // Explicit local path wins
        if let explicit = modelPath {
            return explicit
        }

        if let modelId = model {
            // HF-style repo ID (contains "/")
            if modelId.contains("/") {
                let cached = ModelCache.cacheDir(for: modelId)
                if FileManager.default.fileExists(atPath: cached.path) {
                    return cached.path
                }
                
                // Not cached — download it automatically
                log("Model not in local cache: \(bold(modelId)). Downloading...")
                let hub = HuggingFaceHub()
                do {
                    nonisolated(unsafe) var lastFile = ""
                    let dest = try await hub.download(repoId: modelId, token: nil) { filename, downloaded, total in
                        if filename != lastFile {
                            if !lastFile.isEmpty { print() }   // newline after previous file
                            lastFile = filename
                        }
                        printFileProgress(filename: filename, downloaded: downloaded, total: total)
                    }
                    print("\n")
                    return dest.path
                } catch {
                    log("\(red("Failed to download model:")) \(error.localizedDescription)")
                    throw ExitCode.failure
                }
            }
            // Treated as a local path
            return modelId
        }

        // No model specified — try default dev path
        let dev = ServerConfig.development.modelPath
        log("No model specified — using default path: \(dim(dev))")
        return dev
    }

    private func log(_ msg: String) {
        fputs("[serve] \(msg)\n", stderr)
    }
}
