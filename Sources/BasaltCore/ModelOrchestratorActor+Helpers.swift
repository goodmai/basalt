import Foundation

extension ModelOrchestratorActor {
    static func newRequestId() -> String {
        "req_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(24)
    }

    static func logStatus(_ error: BasaltError) -> String {
        if case .inferenceError(let message) = error {
            if message.hasPrefix("Timeout after") { return "timeout" }
            if message == "Cancelled" { return "cancelled" }
        }
        return Task.isCancelled ? "cancelled" : "error"
    }

    static func basaltError(_ error: any Error, _ context: String) -> BasaltError {
        error as? BasaltError ?? .inferenceError("\(context): \(error)")
    }

    static func budgetedText(_ request: GenerationRequest) -> String {
        guard let messages = request.messages, !messages.isEmpty else { return request.prompt }
        return ChatMessageInput.plainText(messages)
    }

    static func estimatedTokens(_ prompt: String) -> Int {
        prompt.utf8.count / 4
    }

    static func watchdogSeconds(maxTokens: Int, promptTokens: Int, floor: UInt64 = 300) -> UInt64 {
        let decodeSeconds = UInt64(max(0, maxTokens)) / 4
        let prefillSeconds = UInt64(max(0, promptTokens)) / 50
        return max(floor, decodeSeconds + prefillSeconds)
    }

    static func timeout(for request: GenerationRequest) -> UInt64 {
        watchdogSeconds(maxTokens: request.maxTokens ?? 0, promptTokens: estimatedTokens(budgetedText(request)))
    }

    static func endingWithErrorMetadata(_ source: AsyncThrowingStream<StreamChunk, Error>) -> AsyncStream<StreamChunk> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    for try await chunk in source { continuation.yield(chunk) }
                } catch {
                    BasaltLogger(module: "ModelOrchestrator").warn("Cloud stream failed mid-way: \(error)")
                    continuation.yield(.metadata(GenerationResponse(
                        generatedText: "", promptTokens: 0, completionTokens: 0, tokensPerSecond: 0,
                        generationTime: 0, timeToFirstToken: 0,
                        memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .error)))
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func resolvedDirectorySize(at url: URL) -> Int64 {
        guard let e = FileManager.default.enumerator(at: url,
                        includingPropertiesForKeys: [.fileSizeKey],
                        options: []) else { return 0 }
        var total: Int64 = 0
        for case let f as URL in e {
            let target = f.resolvingSymlinksInPath()
            if let s = try? target.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(s)
            }
        }
        return total
    }
}

public struct CachedModelInfo: Sendable {
    public let id: String
    public let sizeBytes: Int64
    public let isLoaded: Bool

    public var sizeFormatted: String {
        switch sizeBytes {
        case 0..<1_048_576:    return String(format: "%.1f KB", Double(sizeBytes) / 1_024)
        case 0..<1_073_741_824: return String(format: "%.1f MB", Double(sizeBytes) / 1_048_576)
        default:               return String(format: "%.2f GB", Double(sizeBytes) / 1_073_741_824)
        }
    }
}
