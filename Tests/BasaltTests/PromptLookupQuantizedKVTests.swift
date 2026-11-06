import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference

@Suite("Prompt lookup with quantized KV, --kv-bits (#163)", .serialized)
struct PromptLookupQuantizedKVTests {
    static func model(hybrid: Bool) throws -> any LanguageModel {
        guard hybrid else { return try TinyModel.make(.qwen3HeadDim32) }
        let family = TinyModel.Family.qwen35
        let config = family.config.replacingOccurrences(of: "\"head_dim\": 16", with: "\"head_dim\": 32")
        let model = try withRandomState(MLXRandom.RandomState(seed: family.seed)) { try family.model(Data(config.utf8)) }
        try model.prepare()
        eval(model)
        return model
    }

    @Test("Greedy lookup on a quantized cache emits the plain quantized decoder's tokens with drafts accepted, and rejected drafts leave every quantized layer at the prompt plus every emitted token but the last, also when conversion starts mid-decode",
          arguments: [(false, 8), (false, 4), (true, 8), (true, 4)], [0, LookupFixture.prompt.count + 12])
    func greedyEquivalence(variant: (Bool, Int), start: Int) throws {
        let (hybrid, bits) = variant
        let model = try Self.model(hybrid: hybrid)
        let prompt = LookupFixture.prompt
        let parameters = GenerateParameters(maxTokens: 64, kvBits: bits, quantizedKVStart: start, temperature: 0)
        let input = LMInput(tokens: MLXArray(prompt.map(Int32.init)))
        let expected = Array(try TokenIterator(input: input, model: model, parameters: parameters))
        var iterator = try PromptLookupTokenIterator(
            prompt: PromptLookupTokenIterator.Prompt(input: input, history: prompt), model: model,
            cache: try model.newCache(parameters: parameters), decoding: DecodingSettings(processor: nil, parameters: parameters),
            lookup: LookupOptions(speculative: true, kvQuantization: parameters))
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        expectTokenSequence(tokens, expected)
        #expect(try #require(iterator.speculativeDecodingTelemetry).acceptedDraftTokenCount > 0)
        let quantized = iterator.cache.compactMap { $0 as? QuantizedKVCache }
        #expect(!quantized.isEmpty)
        #expect(quantized.allSatisfy { $0.bits == bits && $0.offset == prompt.count + tokens.count - 1 })
    }

    @Test("Through the engine at --kv-bits 8 and 4, greedy output is unchanged with prompt lookup on and the response reports accepted drafts",
          arguments: [KVPrecision.q8, .q4])
    func engineEquivalence(precision: KVPrecision) async throws {
        let plain = MLXInferenceEngine(EngineConfiguration(quantizedKVStart: 0))
        await plain.attach(try TinyModel.container(.qwen3HeadDim32))
        let speculative = MLXInferenceEngine(EngineConfiguration(quantizedKVStart: 0, speculativeLookup: true))
        await speculative.attach(try TinyModel.container(.qwen3HeadDim32))
        let request = GenerationRequest(prompt: LookupFixture.code, maxTokens: 64, temperature: 0).withKVPrecision(precision)
        let expected = try await plain.generate(request: request)
        let actual = try await speculative.generate(request: request)
        #expect(actual.generatedText == expected.generatedText)
        #expect(expected.draftTokens == nil)
        #expect(try #require(actual.acceptedDraftTokens) > 0)
    }
}

@Suite("Prompt lookup with quantized KV on real models: greedy parity and speed at 16k context (#163)", .serialized)
struct PromptLookupQuantizedKVRealModelTests {
    static var modelPaths: [String] {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1", !_isDebugAssertConfiguration() else { return [] }
        return (env["BASALT_E2E_LOOKUP_KV_MODELS"] ?? "").split(separator: ",").map(String.init)
    }

    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let runs = 2

    static func prompt(_ workload: String) throws -> String {
        let sources = root.appendingPathComponent("Sources/BasaltInference")
        let files = try FileManager.default.contentsOfDirectory(atPath: sources.path).filter { $0.hasSuffix(".swift") }.sorted()
        var context = ""
        for file in files where context.count < 60_000 {
            context += try String(contentsOf: sources.appendingPathComponent(file), encoding: .utf8)
        }
        let task = try String(contentsOf: root.appendingPathComponent("benchmarks/speculative-lookup/\(workload).txt"), encoding: .utf8)
        return "Reference code from the project:\n```swift\n\(context)\n```\n\n\(task)"
    }

    struct Cell {
        var plain: [Double] = []
        var lookup: [Double] = []
        var identical = true
        var drafts = (accepted: 0, drafted: 0)
        var promptTokens = 0

        static func best(_ values: [Double]) -> Double { values.max() ?? 0 }
        var ratio: Double { Self.best(lookup) / Self.best(plain) }
    }

    static func measure(_ engine: MLXInferenceEngine, _ request: GenerationRequest) async throws -> Cell {
        var cell = Cell()
        var reference: String?
        for _ in 0 ..< runs {
            let plain = try await engine.generate(request: request)
            let lookup = try await engine.generate(request: request.withSpeculativeLookup(true))
            reference = reference ?? plain.generatedText
            cell.plain.append(plain.tokensPerSecond)
            cell.lookup.append(lookup.tokensPerSecond)
            cell.identical = cell.identical && plain.generatedText == reference && lookup.generatedText == reference
            cell.drafts = (lookup.acceptedDraftTokens ?? 0, lookup.draftTokens ?? 0)
            cell.promptTokens = plain.promptTokens
        }
        return cell
    }

    @Test("Greedy output is identical with lookup on and off at --kv-bits 4 and 8; lookup decodes code edits at least as fast and prose at no less than 0.97x",
          .enabled(if: !modelPaths.isEmpty, "release build only: set BASALT_E2E=1 and BASALT_E2E_LOOKUP_KV_MODELS"), arguments: modelPaths)
    func realModel(path: String) async throws {
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: path)
        _ = try await engine.generate(request: GenerationRequest(prompt: "Say hi.", maxTokens: 8, temperature: 0))
        for workload in ["code", "prose"] {
            let base = GenerationRequest(prompt: try Self.prompt(workload), maxTokens: 300, temperature: 0).with { $0.reasoningEffort = "none" }
            for precision in [KVPrecision.q4, .q8] {
                let cell = try await Self.measure(engine, base.withKVPrecision(precision))
                let applied = cell.drafts.drafted > 0
                let bar = workload == "code" && applied ? 1.0 : 0.97
                print("[#163] \(path) kv\(precision) \(workload) prompt \(cell.promptTokens) tok: plain \(String(format: "%.2f", Cell.best(cell.plain))) tok/s, lookup \(String(format: "%.2f", Cell.best(cell.lookup))) tok/s (\(String(format: "%.3f", cell.ratio))x, bar \(bar)x), drafts \(cell.drafts.accepted)/\(cell.drafts.drafted)\(applied ? "" : " (lookup refused, plain decoding)"), identical \(cell.identical)")
                #expect(cell.identical, "\(path) kv\(precision) \(workload)")
                #expect(cell.ratio >= bar, "\(path) kv\(precision) \(workload): \(cell.ratio)x")
            }
        }
    }
}
