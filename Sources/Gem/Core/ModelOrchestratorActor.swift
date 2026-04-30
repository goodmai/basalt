import Foundation
import os

// MARK: — Orchestrator

/// Единственный «источник истины» для обоих транспортов (MCP + REST).
///
/// Гарантирует FIFO-обслуживание: Swift actors сериализуют доступ
/// автоматически, исключая race conditions между MCP и REST клиентами.
///
/// Теорема устойчивости: λ_mcp + λ_rest < μ_inference
/// гарантирует что очередь не накапливается (система устойчива).
public actor ModelOrchestratorActor {

    // MARK: — State

    private let engine: any InferenceEngine
    private let maxTokens: Int
    private var requestCount: Int = 0
    private var totalTokensGenerated: Int = 0
    private var modelSizeMB: Int = 4000 // default fallback

    private let logger = GemLogger(module: "ModelOrchestrator")

    private var currentModelPath: String = ""

    // MARK: — Init

    public init(engine: any InferenceEngine, maxTokens: Int = 4096) {
        self.engine = engine
        self.maxTokens = maxTokens
    }

    // MARK: — Public API (typed throws — Swift 6)

    /// Загружает модель из указанного пути.
    public func loadModel(path: String) async throws(GemError) {
        self.currentModelPath = path
        // Calculate model size for token budgeting
        let size = try? calculateDirectorySize(url: URL(fileURLWithPath: path))
        if let size {
            self.modelSizeMB = Int(size / 1_048_576)
            logger.info("Model size calculated: \(self.modelSizeMB) MB")
        }
        
        try await engine.load(modelPath: path)
    }

    public var modelInfo: String {
        URL(fileURLWithPath: currentModelPath).lastPathComponent
    }

    /// Генерирует текст. Единственная точка входа для обоих транспортов.
    /// Swift actor гарантирует что вызовы выстраиваются в очередь (FIFO)
    /// — никаких дополнительных локов не требуется.
    public func generate(request: GenerationRequest) async throws(GemError) -> GenerationResponse {
        logger.trace("Incoming generate request: \(request.prompt.prefix(50))...")
        let maxT = calculateDynamicMaxTokens()
        var validated = try request.validated(defaultMaxTokens: maxT)
        
        if let requested = request.maxTokens, requested > maxT {
            logger.warn("Requested tokens (\(requested)) exceed dynamic budget (\(maxT)). Capped to prevent OOM.")
            validated = GenerationRequest(prompt: validated.prompt, maxTokens: maxT, temperature: validated.temperature)
        }
        
        requestCount += 1
        let response = try await engine.generate(request: validated)
        totalTokensGenerated += response.completionTokens
        return response
    }

    public func generateStream(request: GenerationRequest) async throws(GemError) -> AsyncStream<StreamChunk> {
        logger.trace("Incoming generateStream request...")
        let maxT = calculateDynamicMaxTokens()
        var validated = try request.validated(defaultMaxTokens: maxT)
        
        if let requested = request.maxTokens, requested > maxT {
            logger.warn("Requested tokens (\(requested)) exceed dynamic budget (\(maxT)). Capped to prevent OOM.")
            validated = GenerationRequest(prompt: validated.prompt, maxTokens: maxT, temperature: validated.temperature)
        }
        
        requestCount += 1
        
        let stream = try await engine.generateStream(request: validated)
        
        return AsyncStream<StreamChunk> { continuation in
            let task = Task.detached {
                var inThinkBlock = false
                var buffer = ""
                var jsonBatch = ""
                var lastBatchTime = Date()
                
                for await chunk in stream {
                    switch chunk {
                    case .text(let t):
                        buffer += t
                        
                        // State machine to strip reasoning blocks
                        if !inThinkBlock {
                            if let startRange = buffer.range(of: "<think>") ?? buffer.range(of: "Thinking Process:\n") {
                                let before = String(buffer[..<startRange.lowerBound])
                                if !before.isEmpty {
                                    jsonBatch += before
                                }
                                inThinkBlock = true
                                buffer = String(buffer[startRange.upperBound...])
                            }
                        }
                        
                        if inThinkBlock {
                            if let endRange = buffer.range(of: "</think>") {
                                inThinkBlock = false
                                buffer = String(buffer[endRange.upperBound...])
                            } else {
                                // Still inside think block, clear buffer except for the last 20 chars to catch split tags
                                if buffer.count > 20 {
                                    buffer = String(buffer.suffix(20))
                                }
                                continue // Skip yielding
                            }
                        }
                        
                        // If not in think block, yield safely
                        if !inThinkBlock {
                            let safeIndex = max(0, buffer.count - 20)
                            if safeIndex > 0 {
                                let safeString = String(buffer.prefix(safeIndex))
                                jsonBatch += safeString
                                buffer = String(buffer.suffix(buffer.count - safeIndex))
                            }
                        }
                        
                        // Batching: yield every 50ms or 10 characters to reduce UI updates
                        if !jsonBatch.isEmpty && (Date().timeIntervalSince(lastBatchTime) > 0.05 || jsonBatch.count > 10) {
                            continuation.yield(.text(jsonBatch))
                            jsonBatch = ""
                            lastBatchTime = Date()
                        }
                        
                    case .metadata(let m):
                        if !inThinkBlock {
                            let finalStr = jsonBatch + buffer
                            if !finalStr.isEmpty {
                                continuation.yield(.text(finalStr))
                            }
                        }
                        continuation.yield(.metadata(m))
                    }
                }
                continuation.finish()
            }
            
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// Состояние для /health эндпоинта и MCP tool: gemma_status.
    public func healthSnapshot(modelId: String?) async -> HealthResponse {
        let ready = await engine.isLoaded
        return HealthResponse(
            status: ready ? "ok" : "initializing",
            modelId: modelId,
            isReady: ready,
            version: HealthResponse.version
        )
    }

    // Diagnostics — только для логирования, не экспортируется.
    var diagnostics: (requests: Int, tokens: Int) {
        (requestCount, totalTokensGenerated)
    }
    
    // MARK: — Private Helpers
    
    private func calculateDynamicMaxTokens() -> Int {
        let dynamicMax = TokenBudgetCalculator.calculateMaxTokensForSystem(modelSizeMB: modelSizeMB)
        return min(maxTokens, dynamicMax)
    }
    
    private func calculateDirectorySize(url: URL) throws -> UInt64 {
        var size: UInt64 = 0
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return size
        }
        
        for case let fileURL as URL in enumerator {
            let resourceValues = try fileURL.resourceValues(forKeys: [.fileSizeKey])
            if let fileSize = resourceValues.fileSize {
                size += UInt64(fileSize)
            }
        }
        return size
    }
}
