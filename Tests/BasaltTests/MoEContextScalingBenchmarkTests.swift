import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Decode tok/s and TTFT across context lengths on a real MoE model (#242)", .serialized)
struct MoEContextScalingBenchmarkTests {
    struct Row: Codable, Equatable {
        let context: Int
        let promptTokens: Int
        let cachedPromptTokens: Int
        let ttftSeconds: Double
        let prefillTokensPerSecond: Double
        let decodeTokensPerSecond: Double
        let completionTokens: Int
        let peakMemoryMB: Int

        init(context: Int, runs: [GenerationResponse]) {
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
    }

    static var environment: [String: String] { ProcessInfo.processInfo.environment }

    static var modelPath: String? {
        guard environment["BASALT_E2E"] == "1" else { return nil }
        return environment["BASALT_E2E_MOE_SCALING_MODEL_PATH"]
    }

    static func integers(_ key: String, default fallback: String) -> [Int] {
        (environment[key] ?? fallback).split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }

    static func prompt(_ tag: String, words: Int) -> String {
        let vocabulary = "expert router token cache stream decode prefill memory budget layer shard slot".split(separator: " ")
        return "Run \(tag). Continue this list with as many new entries as you can, in the same format.\n"
            + (0..<words).map { "\(vocabulary[$0 % vocabulary.count])\(($0 * 13) % 997)" }.joined(separator: " ")
    }

    static func markdown(_ rows: [Row]) -> [String] {
        ["| Context | Prompt tokens | TTFT (s) | Prefill tok/s | Decode tok/s | Generated | Peak MB |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
            + rows.map {
                String(format: "| %d | %d | %.2f | %.0f | %.1f | %d | %d |", $0.context, $0.promptTokens, $0.ttftSeconds,
                       $0.prefillTokensPerSecond, $0.decodeTokensPerSecond, $0.completionTokens, $0.peakMemoryMB)
            }
    }

    @Test("Rows average the runs, take prefill over the uncached prompt and render one Markdown line per context")
    func rowArithmetic() {
        func response(prompt: Int, cached: Int, ttft: Double, tps: Double, peak: Int) -> GenerationResponse {
            GenerationResponse(
                generatedText: "", promptTokens: prompt, cachedPromptTokens: cached, completionTokens: 128, tokensPerSecond: tps,
                generationTime: 1, timeToFirstToken: ttft, memory: .init(peakBytes: peak, activeBytes: 0, cacheBytes: 0), finishReason: .length)
        }
        let row = Row(context: 4096, runs: [
            response(prompt: 4100, cached: 0, ttft: 2, tps: 40, peak: 10 << 20),
            response(prompt: 4100, cached: 100, ttft: 4, tps: 60, peak: 12 << 20),
        ])
        #expect(row.promptTokens == 4100)
        #expect(row.cachedPromptTokens == 100)
        #expect(row.ttftSeconds == 3)
        #expect(row.prefillTokensPerSecond == (2050 + 1000) / 2)
        #expect(row.decodeTokensPerSecond == 50)
        #expect(row.completionTokens == 128)
        #expect(row.peakMemoryMB == 12)
        #expect(Self.markdown([row]).last == "| 4096 | 4100 | 3.00 | 1525 | 50.0 | 128 | 12 |")
    }

    @Test("Decode tok/s, TTFT, prefill tok/s and peak memory at 1k/4k/16k/32k tokens of context",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MOE_SCALING_MODEL_PATH (optional BASALT_E2E_MOE_SCALING_CONTEXTS, _RUNS, _TOKENS)"))
    func contextScaling() async throws {
        let path = try #require(Self.modelPath)
        let contexts = Self.integers("BASALT_E2E_MOE_SCALING_CONTEXTS", default: "1024,4096,16384,32768")
        let runs = Self.integers("BASALT_E2E_MOE_SCALING_RUNS", default: "3").first ?? 3
        let tokens = Self.integers("BASALT_E2E_MOE_SCALING_TOKENS", default: "128").first ?? 128
        let log = try await ModelRunLog("moe-context-scaling", modelPath: path)
        defer { log.releaseGPU() }
        let fit = try await fitTier(log, contexts: contexts)
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(wiresWorkingSet: true), maxTokens: tokens, telemetry: TelemetryTracker())
        try await log.step("load") { try await orchestrator.loadModel(path: path) }
        let calibration = try await log.step("calibrate") {
            try await orchestrator.generate(request: GenerationRequest(prompt: Self.prompt("calibrate", words: 1000), maxTokens: 1))
        }
        let tokensPerWord = Double(calibration.promptTokens) / 1000
        log.record("config", "runs \(runs) tokens \(tokens) tokens-per-word \(String(format: "%.3f", tokensPerWord)) context budget \(fit.maxContextBudgetTokens)")
        var rows: [Row] = []
        var failures = 0
        for context in contexts {
            guard context <= fit.maxContextBudgetTokens else {
                log.record("skip", "context \(context) exceeds the fit's context budget \(fit.maxContextBudgetTokens)")
                continue
            }
            do {
                let row = try await log.step("context \(context)") {
                    try await measure(orchestrator, context: context, words: Int(Double(context) / tokensPerWord), runs: runs, tokens: tokens)
                }
                rows.append(row)
                log.record("result", String(format: "context %d prompt %d cached %d ttft %.3f s prefill %.0f tok/s decode %.2f tok/s generated %d peak %d MB",
                                            context, row.promptTokens, row.cachedPromptTokens, row.ttftSeconds, row.prefillTokensPerSecond,
                                            row.decodeTokensPerSecond, row.completionTokens, row.peakMemoryMB))
            } catch {
                failures += 1
            }
        }
        await orchestrator.shutdown()
        try export(rows, log: log)
        #expect(failures == 0)
        #expect(!rows.isEmpty)
    }

    func fitTier(_ log: ModelRunLog, contexts: [Int]) async throws -> MemoryAssessment {
        let total = ModelDefinition.inferTotalParameters(id: log.model, name: "", ramMB: 0)
        let active = ModelDefinition.inferActiveParameters(id: log.model, name: "", total: total)
        let fit = await MemoryEvaluator().evaluate(modelId: log.model, contextLength: contexts.max() ?? 0)
        log.record("fit", "\(fit.fitLevel.rawValue) required \(fit.requiredRAMFormatted) budget \(fit.memoryBudgetFormatted) parameters \(total)B active \(active)B\(fit.expertStreaming.map { ", streaming: \($0.summary)" } ?? "")")
        guard active < total else {
            log.record("FAIL moe", "\(log.model) is not a sparse MoE by its id (total \(total)B, active \(active)B)")
            throw BasaltError.invalidRequestStructure(details: "\(log.model) is not a sparse MoE model")
        }
        return fit
    }

    func measure(_ orchestrator: ModelOrchestratorActor, context: Int, words: Int, runs: Int, tokens: Int) async throws -> Row {
        var responses: [GenerationResponse] = []
        for run in 0..<runs {
            let request = GenerationRequest(prompt: Self.prompt("\(UUID().uuidString)-\(run)", words: words), maxTokens: tokens, temperature: 0.7)
            responses.append(try await orchestrator.generate(request: request))
        }
        return Row(context: context, runs: responses)
    }

    func export(_ rows: [Row], log: ModelRunLog) throws {
        for line in Self.markdown(rows) { log.record("table", line) }
        let slug = log.model.replacingOccurrences(of: "/", with: "--")
        let json = URL(fileURLWithPath: log.url.deletingPathExtension().path + "-\(slug).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(rows).write(to: json)
        log.record("json", json.path)
    }
}
