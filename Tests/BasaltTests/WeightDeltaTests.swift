import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Weight deltas: validated, float-only, marked, cache-safe and undone bit for bit (#328)", .serialized)
struct WeightDeltaTests {
    static let norm = "model.norm.weight"

    static func engine(promptCacheBytes: Int? = nil) async throws -> MLXInferenceEngine {
        let engine = try await TinyModel.engine(promptCacheBytes: promptCacheBytes)
        await engine.skipJournalForDeltaTests()
        return engine
    }

    static func cast(_ name: String, to dtype: DType, in container: ModelContainer) async throws {
        try await container.perform { context in
            let weight = try MLXInferenceEngine.writableTensor(name, in: context.model)
            context.model.update(parameters: ModuleParameters.unflattened([(name, weight.asType(dtype))]))
            eval(context.model)
        }
    }

    static func bits(of name: String, in container: ModelContainer) async throws -> [UInt32] {
        try await container.perform { context in
            try MLXInferenceEngine.writableTensor(name, in: context.model).asArray(Float.self).map(\.bitPattern)
        }
    }

    static func logits(_ container: ModelContainer) async throws -> [UInt32] {
        try await container.perform { context in
            let input = MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init)).reshaped(1, -1)
            return context.model(input, cache: try context.model.newCache(parameters: nil)).asArray(Float.self).map(\.bitPattern)
        }
    }

    @Test("A delta changes the logits, and undo restores the tensor and the logits bit for bit", arguments: [DType.float32, .bfloat16])
    func undoIsBitExact(dtype: DType) async throws {
        let engine = try await Self.engine()
        let container = try #require(await engine.container)
        try await Self.cast(Self.norm, to: dtype, in: container)
        let weights = try await Self.bits(of: Self.norm, in: container)
        let logits = try await Self.logits(container)
        let info = try await engine.applyDelta(WeightDelta(tensor: Self.norm, slice: [0 ..< 16], values: [0.5]))
        #expect(try await Self.bits(of: Self.norm, in: container) != weights)
        #expect(try await Self.logits(container) != logits)
        #expect(try await engine.undoDelta(info.id))
        #expect(try await Self.bits(of: Self.norm, in: container) == weights)
        #expect(try await Self.logits(container) == logits)
        #expect(await engine.activeDeltas.isEmpty)
        #expect(try await engine.undoDelta(info.id) == false)
    }

    @Test("A delta on a quantized layer is refused, its packed weight and its float scales alike, as is an unknown tensor")
    func quantizedRefused() async throws {
        let model = try TinyModel.make()
        quantize(model: model, groupSize: 32, bits: 4)
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-q4"), model: model, processor: ByteInputProcessor(), tokenizer: ByteTokenizer())))
        for name in ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.scales", "model.layers.0.self_attn.q_proj.biases"] {
            let error = await #expect(throws: BasaltError.self) { try await engine.applyDelta(WeightDelta(tensor: name, values: [0.1])) }
            guard case .invalidRequestStructure(let details)? = error else { Issue.record("unexpected \(String(describing: error))"); continue }
            #expect(details.contains("quantized write path is not implemented"), "\(details)")
        }
        await #expect(throws: BasaltError.self) { try await engine.applyDelta(WeightDelta(tensor: "model.nope.weight", values: [0.1])) }
        #expect(await !engine.modelModified)
    }

    @Test("A delta is checked at the boundary: slice in range, value count, finite values, |scale| at most 10", arguments: [
        WeightDelta(tensor: "t", slice: [0 ..< 5], values: [1]),
        WeightDelta(tensor: "t", slice: [0 ..< 1, 0 ..< 1, 0 ..< 1], values: [1]),
        WeightDelta(tensor: "t", slice: [2 ..< 2], values: [1]),
        WeightDelta(tensor: "t", slice: [0 ..< 1], values: [1, 2]),
        WeightDelta(tensor: "t", values: nil),
        WeightDelta(tensor: "t", values: [.nan]),
        WeightDelta(tensor: "t", values: [1], scale: 10.5),
        WeightDelta(tensor: "t", values: [1], scale: -.infinity),
    ])
    func refusedAtTheBoundary(delta: WeightDelta) {
        #expect(throws: BasaltError.self) { try MLXInferenceEngine.deltaRanges(delta, shape: [4, 8]) }
    }

    @Test("A partial slice covers the trailing dimensions whole and takes one value per element or one to broadcast")
    func acceptedSlices() throws {
        let row = WeightDelta(tensor: "t", slice: [1 ..< 3], values: Array(repeating: 1, count: 16))
        #expect(try MLXInferenceEngine.deltaRanges(row, shape: [4, 8]) == [1 ..< 3, 0 ..< 8])
        #expect(try MLXInferenceEngine.deltaRanges(WeightDelta(tensor: "t", values: [1], scale: -10), shape: [4, 8]) == [0 ..< 4, 0 ..< 8])
    }

    @Test("While a delta is active responses are marked, telemetry lists it, a second delta on the tensor is refused and no prompt cache is reused; after undo reuse resumes")
    func cacheAndMarking() async throws {
        let engine = try await Self.engine(promptCacheBytes: 1 << 30)
        let orchestrator = ModelOrchestratorActor(engine: engine, maxTokens: 16, telemetry: TelemetryTracker())
        let before = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(!before.modelModified)
        let info = try await engine.applyDelta(WeightDelta(tensor: Self.norm, values: [0.25]))
        #expect(await orchestrator.telemetrySnapshot().activeDeltas?.map(\.id) == [info.id])
        await #expect(throws: BasaltError.self) { try await engine.applyDelta(WeightDelta(tensor: Self.norm, values: [1])) }
        let during = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        let again = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        #expect(during.modelModified && again.modelModified)
        #expect(during.cachedPromptTokens == 0 && again.cachedPromptTokens == 0)
        #expect(try await engine.undoDelta(info.id))
        let cleared = try await engine.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        let reused = try await engine.generate(request: greedy(Conversation.afterToolCall, key: "ns"))
        #expect(!cleared.modelModified && !reused.modelModified)
        #expect(cleared.cachedPromptTokens == 0)
        #expect(reused.cachedPromptTokens > 0)
        #expect(await orchestrator.telemetrySnapshot().activeDeltas?.isEmpty == true)
    }
}

extension MLXInferenceEngine {
    func skipJournalForDeltaTests() { experiments.registry.journal = nil }
}
