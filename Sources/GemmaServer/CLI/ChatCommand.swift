import Foundation
import ArgumentParser
import MLX
import MLXLLM

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

    @Option(name: .long, help: "Log level: debug | info | warn | error")
    var logLevel: ServerConfig.LogLevel = .info

    // MARK: — Run

    mutating func run() async throws {
        let resolvedPath = try await resolveModelPath()

        // Shared inference engine + orchestrator
        let engine = MLXInferenceEngine()
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: maxTokens)

        log("Loading model from \(bold(resolvedPath))…")
        do {
            try await orchestrator.loadModel(path: resolvedPath)
            log("Model ready. Type 'exit' or 'quit' to stop.")
        } catch {
            log("\(red("Failed to load model:")) \(error.localizedDescription)")
            throw ExitCode.failure
        }

        print("\n--- Gemma Chat ---")
        
        while true {
            print("\(green("User:")) ", terminator: "")
            fflush(stdout)
            
            guard let line = readLine(), !line.isEmpty else { continue }
            
            if line.lowercased() == "exit" || line.lowercased() == "quit" {
                break
            }
            
            print("\(blue("Gemma:")) ", terminator: "")
            fflush(stdout)
            
            let request = GenerationRequest(prompt: line, maxTokens: maxTokens)
            
            do {
                let stream = try await orchestrator.generateStream(request: request)
                
                var stats: GenerationResponse?
                
                for await chunk in stream {
                    switch chunk {
                    case .text(let t):
                        print(t, terminator: "")
                        fflush(stdout)
                    case .metadata(let m):
                        stats = m
                    }
                }
                print("\n")
                
                if let stats = stats {
                    log(dim("Tokens: \(stats.promptTokens) in / \(stats.completionTokens) out | TPS: \(String(format: "%.2f", stats.tokensPerSecond)) | TTFT: \(String(format: "%.3fs", stats.timeToFirstToken)) | Memory: \(stats.memory.activeBytes / 1024 / 1024)MB"))
                }
                print("")
            } catch {
                log("\(red("Error:")) \(error.localizedDescription)")
            }
        }
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
            log("Falling back to default config path.")
            return ServerConfig.development.modelPath
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

    private func interactiveModelPicker() async throws -> String {
        let models = [
            ("Qwen3.5 4B", "2.3 GB RAM, 92 TPS", "mlx-community/Qwen3.5-4B-4bit"),
            ("Qwen3.5 9B OptiQ", "5.8 GB RAM, 37 TPS", "mlx-community/Qwen3.5-9B-OptiQ-4bit"),
            ("Qwen Coder 7B", "4.1 GB RAM, 60 TPS", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
            ("Qwen3.6 27B", "14.5 GB RAM, 11 TPS", "mlx-community/Qwen3.6-27B-4bit"),
            ("Gemma 4 2B", "2.7 GB RAM, 60 TPS", "mlx-community/gemma-4-e2b-it-4bit")
        ]
        
        print("\n\(bold("Welcome to GemmaServer CLI"))")
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
