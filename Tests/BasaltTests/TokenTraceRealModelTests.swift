import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("trace_tokens and conditional ablation: decode overhead per setting on real models (#405)", .serialized)
struct TokenTraceRealModelTests {
    static let models = NF4Run.models("BASALT_E2E_TRACE_MODELS")
    static let prompt = "Write a long, detailed essay on the history of the printing press, from Gutenberg to the steam press."
    static let decodeTokens = 128

    struct Arm: Sendable {
        var name: String
        var tokensPerSecond: Double
        var tokens: Int
    }

    static func settings(_ anatomy: ModelAnatomy) -> [(String, TokenTraceRequest)] {
        let count = anatomy.layers.count
        let direction = (0 ..< anatomy.hiddenSize).map { Float(($0 * 7919) % 13) / 13 - 0.5 }
        func request(_ change: (inout TokenTraceRequest) -> Void) -> TokenTraceRequest {
            var request = TokenTraceRequest(prompt: prompt, maxTokens: decodeTokens)
            change(&request)
            return request
        }
        let full = request { $0.projections = [count / 2, 3 * count / 4].map { TraceDirection(layer: $0, vector: direction, threshold: 0) } }
        var arms = [
            ("heads", request { $0.experts = false }),
            ("heads+experts+projections", full),
            ("+attentionEntropy", request { $0 = full; $0.attentionEntropy = true }),
            ("lens top-5 on 4 layers", request { ($0.heads, $0.experts, $0.lensTopK, $0.layers) = (false, false, 5, [0, count / 3, 2 * count / 3, count - 1]) }),
        ]
        if anatomy.layers.contains(where: { $0.experts != nil }) { arms.insert(("experts", request { $0.heads = false }), at: 1) }
        return arms
    }

    static func timed(_ name: String, _ body: () throws -> Int) rethrows -> Arm {
        var runs: [(Double, Int)] = []
        for _ in 0 ..< 3 {
            let start = ContinuousClock.now
            let tokens = try body()
            runs.append((Double(tokens) / (ContinuousClock.now - start).inSeconds, tokens))
        }
        let median = runs.sorted { $0.0 < $1.0 }[1]
        return Arm(name: name, tokensPerSecond: median.0, tokens: median.1)
    }

    static func arms(_ container: ModelContainer, anatomy: ModelAnatomy) async throws -> [Arm] {
        try await container.perform { context in
            let ids = context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            let stops = MLXInferenceEngine.stopTokenIds(context)
            _ = try MLXInferenceEngine.greedy(ids, context: context, probe: nil, maxTokens: 16, stops: stops)
            var arms = [try timed("off") { try MLXInferenceEngine.greedy(ids, context: context, probe: nil, maxTokens: decodeTokens, stops: stops).count }]
            for (name, request) in settings(anatomy) {
                arms.append(try timed(name) {
                    try MLXInferenceEngine.tracedGreedy(ids, maxTokens: decodeTokens, recorder: TokenTraceRecorder(request, readout: MLXInferenceEngine.readout(of: context.model)), context: context).count
                })
            }
            let attention = anatomy.layers.lastIndex { $0.heads != nil } ?? 0
            let ablation = Intervention.ablateHead(layer: attention, head: 0)
            let trigger = InterventionTrigger.projection(layer: attention / 2, vector: (0 ..< anatomy.hiddenSize).map { Float(($0 * 7919) % 13) / 13 - 0.5 }, above: 0)
            for (name, edits) in [("static ablate head", [ablation]), ("conditional ablate head", [Intervention.conditional(ConditionalIntervention(action: ablation, trigger: trigger))])] {
                arms.append(try timed(name) {
                    try MLXInferenceEngine.greedy(ids, context: context, probe: InterventionProbe(edits, anatomy: anatomy), maxTokens: decodeTokens, stops: stops).count
                })
            }
            return arms
        }
    }

    @Test("Decode tok/s of the lab greedy loop with tracing off, per trace setting, and with static and conditional head ablation; heads+experts+projections stays within 25% of off",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_TRACE_MODELS"), arguments: models)
    func overhead(entry: String) async throws {
        let path = NF4Run.path(entry)
        let log = try await ModelRunLog("trace-overhead", modelPath: path)
        defer { log.releaseGPU() }
        let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 30))
        try await log.step("load") { try await engine.load(modelPath: path) }
        let container = try #require(await engine.container)
        let serving = try await log.step("serving decode") { try await NF4Run.decodeSpeed(engine) }
        log.record("RESULT", String(format: "serving decode %.1f tok/s", serving))
        let anatomy = try await engine.anatomy()
        let arms = try await log.step("lab arms") { try await Self.arms(container, anatomy: anatomy) }
        let off = arms[0].tokensPerSecond
        for arm in arms {
            log.record("RESULT", String(format: "%@ %.1f tok/s over %d tokens, overhead %.1f%%", arm.name, arm.tokensPerSecond, arm.tokens, 100 * (1 - arm.tokensPerSecond / off)))
        }
        let full = try #require(arms.first { $0.name == "heads+experts+projections" })
        #expect(full.tokensPerSecond >= 0.75 * off, "heads+experts+projections \(full.tokensPerSecond) tok/s against \(off) off")
    }
}
