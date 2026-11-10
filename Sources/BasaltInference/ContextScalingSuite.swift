import Foundation
import MLX
import BasaltCore

public struct ContextScalingRow: Codable, Equatable, Sendable {
    public let context: Int
    public let promptTokens: Int
    public let cachedPromptTokens: Int
    public let ttftSeconds: Double
    public let prefillTokensPerSecond: Double
    public let decodeTokensPerSecond: Double
    public let completionTokens: Int
    public let peakMemoryMB: Int

    public init(context: Int, runs: [GenerationResponse]) {
        let count = Double(max(1, runs.count))
        func mean(_ value: (GenerationResponse) -> Double) -> Double { runs.map(value).reduce(0, +) / count }
        self.context = context
        promptTokens = runs.map(\.promptTokens).max() ?? 0
        cachedPromptTokens = runs.map(\.cachedPromptTokens).max() ?? 0
        ttftSeconds = mean(\.timeToFirstToken)
        prefillTokensPerSecond = mean { $0.timeToFirstToken > 0 ? Double($0.promptTokens - $0.cachedPromptTokens) / $0.timeToFirstToken : 0 }
        decodeTokensPerSecond = mean(\.tokensPerSecond)
        completionTokens = Int(mean { Double($0.completionTokens) }.rounded())
        peakMemoryMB = (runs.map(\.memory.peakBytes).max() ?? 0) >> 20
    }

    public var line: String {
        String(format: "context %d prompt %d cached %d ttft %.3f s prefill %.0f tok/s decode %.2f tok/s generated %d peak %d MB",
               context, promptTokens, cachedPromptTokens, ttftSeconds, prefillTokensPerSecond, decodeTokensPerSecond, completionTokens, peakMemoryMB)
    }
}

public struct ContextScalingReport: Codable, Sendable {
    public let model: String
    public let chip: String
    public let fitLevel: String
    public let rows: [ContextScalingRow]
    public let notMeasured: [String]

    public init(model: String, chip: String, fitLevel: String, rows: [ContextScalingRow], notMeasured: [String]) {
        (self.model, self.chip, self.fitLevel, self.rows, self.notMeasured) = (model, chip, fitLevel, rows, notMeasured)
    }

    public var markdown: String {
        (["## fae-scaling: \(model)", "", "\(chip), fit tier \(fitLevel)", ""] + ContextScalingSuite.markdown(rows)
            + (notMeasured.isEmpty ? [] : ["", "Not measured: " + notMeasured.joined(separator: "; ")]))
            .joined(separator: "\n") + "\n"
    }
}

public struct ContextScalingSuite: Sendable {
    public static let faeContexts = [1024, 4096, 16384, 32768, 65536, 131072]

    public let runs: Int
    public let tokens: Int
    public let temperature: Double

    public init(runs: Int, tokens: Int, temperature: Double) {
        (self.runs, self.tokens, self.temperature) = (max(1, runs), tokens, temperature)
    }

    public static func modelContext(at directory: URL) -> Int? {
        guard let data = try? Data(contentsOf: directory.appending(component: "config.json")),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        let text = config["text_config"] as? [String: Any]
        return (text?["max_position_embeddings"] ?? config["max_position_embeddings"]) as? Int
    }

    public static func prompt(_ tag: String, words: Int) -> String {
        let vocabulary = "expert router token cache stream decode prefill memory budget layer shard slot".split(separator: " ")
        return "Run \(tag). Continue this list with as many new entries as you can, in the same format.\n"
            + (0..<words).map { "\(vocabulary[$0 % vocabulary.count])\(($0 * 13) % 997)" }.joined(separator: " ")
    }

    public static func markdown(_ rows: [ContextScalingRow]) -> [String] {
        ["| Context | Prompt tokens | TTFT (s) | Prefill tok/s | Decode tok/s | Generated | Peak MB |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
            + rows.map {
                String(format: "| %d | %d | %.2f | %.0f | %.1f | %d | %d |", $0.context, $0.promptTokens, $0.ttftSeconds,
                       $0.prefillTokensPerSecond, $0.decodeTokensPerSecond, $0.completionTokens, $0.peakMemoryMB)
            }
    }

    public func measure(
        _ orchestrator: ModelOrchestratorActor, contexts: [Int], log: (String, String) -> Void
    ) async throws -> (rows: [ContextScalingRow], failed: [Int]) {
        let calibration = try await orchestrator.generate(request: GenerationRequest(prompt: Self.prompt("calibrate", words: 1000), maxTokens: 1))
        let tokensPerWord = max(0.1, Double(calibration.promptTokens) / 1000)
        log("calibrate", String(format: "%.3f prompt tokens per word", tokensPerWord))
        var rows: [ContextScalingRow] = []
        var failed: [Int] = []
        for context in contexts {
            do {
                let row = try await measure(orchestrator, context: context, words: Int(Double(context) / tokensPerWord))
                rows.append(row)
                log("result", row.line)
            } catch {
                failed.append(context)
                log("FAIL context \(context)", "\(error)")
            }
        }
        return (rows, failed)
    }

    func measure(_ orchestrator: ModelOrchestratorActor, context: Int, words: Int) async throws -> ContextScalingRow {
        Memory.peakMemory = 0
        var responses: [GenerationResponse] = []
        for run in 0..<runs {
            let request = GenerationRequest(prompt: Self.prompt("\(UUID().uuidString)-\(run)", words: words), maxTokens: tokens, temperature: temperature)
            responses.append(try await orchestrator.generate(request: request))
        }
        return ContextScalingRow(context: context, runs: responses)
    }
}
