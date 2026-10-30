import Foundation
import ArgumentParser
import BasaltCore
import BasaltInference

// MARK: — chat (subcommand)

@available(macOS 10.15, macCatalyst 13, iOS 13, tvOS 13, watchOS 6, *)
struct ChatCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Start an interactive CLI chat session with the model."
    )

    @Option(name: .shortAndLong, help: "Model identifier (HF repo) or local path")
    var model: String?

    @Option(name: .customLong("model-path"), help: "Explicit local path to model (overrides --model)")
    var modelPath: String?

    @Option(name: .customLong("max-tokens"), help: "Default maximum tokens to generate (default 65536)")
    var maxTokens: Int = 65536

    @Option(name: .shortAndLong, help: "Initial prompt to execute immediately and exit (non-interactive mode)")
    var prompt: String?

    @Option(name: .long, help: "Log level: debug | info | warn | error")
    var logLevel: ServerConfig.LogLevel = .info

    @Option(name: .customLong("resume"), help: "Continue a logged chat: a session id from `basalt sessions list`, or latest")
    var resume: String?

    @Option(name: .customLong("session-log"), help: "Session log for this chat: full (default; needed for --resume) | meta | off")
    var sessionLog: SessionLogMode = .full

    @Option(name: .customLong("session-log-dir"), help: "Directory of the session log (default: ~/.basalt/sessions)")
    var sessionLogDir: String?

    static func request(_ prompt: String, history: [ChatMessageInput], maxTokens: Int, sessionId: String) -> GenerationRequest {
        GenerationRequest(prompt: prompt, maxTokens: maxTokens, messages: history + [ChatMessageInput(role: .user, text: prompt)])
            .withCacheKey(cacheKey(sessionId))
            .withRouting(sessionId: sessionId, budgetKey: nil)
            .withOrigin(RequestOrigin(source: "cli", endpoint: "chat", stream: true))
    }

    static func cacheKey(_ sessionId: String) -> String {
        "cli:" + sessionId
    }

    static func resumed(_ id: String, in directory: URL) throws -> (session: String, history: [ChatMessageInput]) {
        let session = try SessionsCommand.resolve(id, in: directory)
        let lines = SessionLogReader.lines(of: session.file)
        guard SessionLogReader.mode(of: lines) == SessionLogMode.full.rawValue else {
            throw ValidationError("Session \(session.id) was recorded in \(SessionLogReader.mode(of: lines) ?? "an unknown") mode and has no content to resume")
        }
        guard let exchange = SessionLogReader.lastExchange(in: lines) else { return (session.id, []) }
        return (session.id, exchange.messages + (exchange.reply.map { [ChatMessageInput(role: .assistant, text: $0)] } ?? []))
    }

    // MARK: — Run
    private static let logger = GemLogger(module: "ChatCommand")

    mutating func run() async throws {
        Self.logger.trace("ChatCommand started.")
        
        let logDirectory = sessionLogDir.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? SessionLogWriter.defaultDirectory
        var sessionId = "chat-" + ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate, .withTime])
            .replacingOccurrences(of: ":", with: "")
        var history: [ChatMessageInput] = []
        if let resume {
            (sessionId, history) = try Self.resumed(resume, in: logDirectory)
            log("Resuming \(bold(sessionId)) with \(history.count) earlier messages")
        }
        let writer = ServeCommand.sessionLogWriter(
            mode: sessionLog, directory: sessionLogDir, retentionDays: 14, maxTotalMB: 1024, defaultSession: sessionId)
        defer { if writer != nil { print("Resume with: basalt chat --resume \(sessionId)") } }

        let resolvedPath = try await resolveModelPath()

        let engine = MLXInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: maxTokens, sessionLog: writer)

        log("Loading model from \(bold(resolvedPath))…")
        do {
            try await orchestrator.loadModel(path: resolvedPath)
        } catch {
            log("\(red("Failed to load model:")) \(error.localizedDescription)")
            throw ExitCode.failure
        }

        if let initialPrompt = prompt {
            // Non-interactive mode
            let finalPrompt = try await PromptContextBuilder.build(prompt: initialPrompt)
            let request = Self.request(finalPrompt, history: history, maxTokens: maxTokens, sessionId: sessionId)
            
            do {
                let stream = try await orchestrator.generateStream(request: request)
                var stats: GenerationResponse?
                
                for try await chunk in stream {
                    switch chunk {
                    case .toolCall:  // this path declares no tools, so a call has no client to go to
                        break
                    case .text(let t):
                        print(t, terminator: "")
                        fflush(stdout)
                    case .reasoning(let r):
                        print(r, terminator: "")
                        fflush(stdout)
                    case .metadata(let m):
                        stats = m
                    }
                }
                print("\n")
                if let stats = stats {
                    log(dim("Tokens: \(stats.promptTokens) in / \(stats.completionTokens) out | TPS: \(String(format: "%.2f", stats.tokensPerSecond)) | TTFT: \(String(format: "%.3fs", stats.timeToFirstToken))"))
                }
            } catch {
                await writer?.close()
                log("\(red("Error:")) \(error.localizedDescription)")
                throw ExitCode.failure
            }
            await writer?.close()
            return
        }

        log("Model ready. Type 'exit' or 'quit' to stop.")
        print("\n--- Gemma Chat ---")
        
        let controller = ChatController(orchestrator: orchestrator, maxTokens: maxTokens, sessionId: sessionId, history: history)
        await controller.start()
        await writer?.close()
        
        // Ensure raw mode is cleanly disabled when returning to bash
        let terminal = TerminalManager()
        await terminal.disableRawMode()
        print("Goodbye!")
    }

    // MARK: — Path resolution

    private func resolveModelPath() async throws -> String {
        // Explicit local path wins
        if let explicit = modelPath {
            return explicit
        }

        // Check if --model is provided
        if let modelId = model {
            return try await resolveAndDownloadIfNeeded(modelId: modelId)
        }

        // Epic 4.2: Interactive Model Selection if no model is provided
        do {
            let selectedModelId = try await interactiveModelPicker()
            return try await resolveAndDownloadIfNeeded(modelId: selectedModelId)
        } catch {
            log("Interactive picker failed: \(error.localizedDescription)")
            log("Falling back to default: AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit")
            return try await resolveAndDownloadIfNeeded(modelId: "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit")
        }
    }
    
    private func resolveAndDownloadIfNeeded(modelId: String) async throws -> String {
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
                let dest = try await hub.download(repoId: modelId, token: nil,
                                                  onFile: DownloadProgressReporter().callAsFunction)
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

    private func interactiveModelPicker() async throws -> String {
        let models = [
            ("★ Huihui Qwen3.8 27B 4bit", "15.2 GB RAM, 12 TPS — default", "AutisticAF/Huihui-Qwen3.8-27B-abliterated-mlx-4Bit"),
            ("Qwen3.5 4B", "2.3 GB RAM, 92 TPS", "mlx-community/Qwen3.5-4B-4bit"),
            ("Qwen3.5 9B OptiQ", "5.8 GB RAM, 37 TPS", "mlx-community/Qwen3.5-9B-OptiQ-4bit"),
            ("Qwen Coder 7B", "4.1 GB RAM, 60 TPS", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
            ("MYTHOS 26B MoE DQ", "14.5 GB RAM, 14 TPS", "Ex0bit/MYTHOS-26B-A4B-PRISM-PRO-DQ-MLX"),
            ("Qwen3.6 27B", "14.5 GB RAM, 11 TPS", "mlx-community/Qwen3.6-27B-4bit"),
            ("Ornith 1.5 9B", "5.8 GB RAM, coding/agent", "ornith-ai/Ornith-1.5-9B-MLX-4bit"),
            ("Ornith 1.5 35B A3B MoE", "21 GB RAM, coding/agent", "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"),
            ("Gemma 4 2B", "2.7 GB RAM, 60 TPS", "mlx-community/gemma-4-e2b-it-4bit")
        ]
        
        print("\n\(bold("Welcome to Basalt CLI"))")
        print("Select a recommended model to chat with:\n")
        
        for (i, (name, specs, repoId)) in models.enumerated() {
            let cached = ModelCache.isDownloaded(repoId: repoId) ? "✓" : " "
            print("  \(i+1). [\(cached)] \(bold(name)) — \(dim(specs))")
        }
        
        print("\nEnter number (1-\(models.count)): ", terminator: "")
        fflush(stdout)
        
        guard let input = readLine(), let idx = Int(input), (1...models.count).contains(idx) else {
            throw ValidationError("Invalid selection. Please run again and select a valid number.")
        }
        
        let selectedModel = models[idx - 1]
        print("\nSelected: \(green(selectedModel.0))\n")
        return selectedModel.2
    }

    private func log(_ msg: String) {
        fputs("[chat] \(msg)\n", stderr)
    }
}
