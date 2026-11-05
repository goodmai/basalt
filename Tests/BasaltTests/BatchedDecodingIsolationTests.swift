import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXLLM
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltCLI

enum BatchFixture {
    static let lengths = [5, 17, 8, 11]

    static func prompt(_ length: Int, seed: Int) -> [Int] {
        (0 ..< length).map { ($0 * 31 + seed * 17) % 250 + 1 }
    }

    static var prompts: [[Int]] { lengths.enumerated().map { prompt($1, seed: $0) } }

    static func row(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map(Int32.init)).reshaped(1, -1)
    }

    static func prefilled(_ model: any LanguageModel, _ prompt: [Int]) throws -> [KVCacheSimple] {
        let layers = try model.newCache(parameters: nil).compactMap { $0 as? KVCacheSimple }
        eval(model(row(prompt), cache: layers))
        return layers
    }

    static func soloGreedy(_ model: any LanguageModel, _ prompt: [Int], steps: Int) throws -> [Int] {
        let cache = try model.newCache(parameters: nil)
        var next = argMax(model(row(prompt), cache: cache)[0, -1], axis: -1)
        var tokens: [Int] = []
        for _ in 0 ..< steps {
            tokens.append(next.item(Int.self))
            next = argMax(model(next.reshaped(1, 1), cache: cache)[0, -1], axis: -1)
        }
        return tokens
    }

    static func text(_ tokens: [Int]) -> String {
        ByteTokenizer().decode(tokenIds: tokens, skipSpecialTokens: false)
    }

    static func launch(
        _ core: BatchCore, _ model: any LanguageModel, _ prompt: [Int], maxTokens: Int, stop: Set<Int> = []
    ) throws -> (stream: AsyncStream<Generation>, task: Task<Void, Never>) {
        let seat = try #require(core.decoder.seat(on: core))
        let iterator = try BatchedTokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), model: model,
            layers: try #require(BatchedTokenIterator.batchableLayers(try model.newCache(parameters: nil))),
            decoding: DecodingSettings(
                processor: nil, sampler: GenerateParameters(temperature: 0).sampler(), prefill: PrefillParameters(), maxTokens: maxTokens),
            stopTokens: stop, seat: seat)
        var configuration = ModelConfiguration(id: "tiny-qwen3")
        configuration.eosTokenIds = stop
        let (stream, task, _) = MLXInferenceEngine.handOff(
            promptTokenCount: prompt.count,
            context: ModelContext(configuration: configuration, model: model, processor: ByteInputProcessor(), tokenizer: ByteTokenizer()),
            iterator: iterator, tools: nil, toolCallPolicy: ToolCallPolicy())
        return (stream, task)
    }

    static func collect(_ stream: AsyncStream<Generation>, upTo limit: Int = .max) async -> String {
        var text = ""
        for await generation in stream {
            if case .chunk(let chunk) = generation { text += chunk }
            if text.unicodeScalars.count >= limit { break }
        }
        return text
    }

    static func engine(_ model: any LanguageModel, id: String, batchSlots: Int, processor: any UserInputProcessor = ByteInputProcessor())
        async -> MLXInferenceEngine
    {
        let engine = MLXInferenceEngine(batchSlots: batchSlots)
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: id), model: model, processor: processor, tokenizer: ByteTokenizer())))
        return engine
    }

    static func request(_ index: Int, maxTokens: Int = 32) -> GenerationRequest {
        let text = String(String.UnicodeScalarView(prompts[index].compactMap(UnicodeScalar.init)))
        return greedy(text, key: index.isMultiple(of: 2) ? "batch-\(index)" : nil, maxTokens: maxTokens)
    }
}

@Suite("BatchKVCache keeps every row's attention equal to its own cache (#240)", .serialized)
struct BatchKVCacheTests {

    final class Solo {
        let cache: [KVCache]
        let model: any LanguageModel

        init(_ model: any LanguageModel, _ prompt: [Int]) throws {
            self.model = model
            cache = try model.newCache(parameters: nil)
            eval(model(BatchFixture.row(prompt), cache: cache))
        }

        func step(_ token: Int) -> MLXArray {
            model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)[0, -1]
        }
    }

    static func expectRowsMatch(_ batched: MLXArray, _ solo: [MLXArray]) {
        for (index, alone) in solo.enumerated() {
            let row = batched[index]
            #expect(allClose(row, alone, atol: 1e-4).item(Bool.self), "row \(index) logits differ")
            #expect(argMax(row).item(Int.self) == argMax(alone).item(Int.self), "row \(index) picks another token")
        }
    }

    static func batchStep(_ model: any LanguageModel, _ layers: [KVCache], _ tokens: [Int]) -> MLXArray {
        model(MLXArray(tokens.map(Int32.init)).reshaped(-1, 1), cache: layers)[0..., -1]
    }

    @Test("Rows of different lengths merged into one batch give the logits each row gets alone")
    func mergedRowsMatchSolo() throws {
        let model = try TinyModel.make()
        let prompts = BatchFixture.prompts
        let fed = [7, 42, 99, 3]
        let solo = try prompts.map { try Solo(model, $0) }
        let rows = try prompts.map { try BatchFixture.prefilled(model, $0) }
        let layers = rows[0].indices.map { layer in
            let batch = BatchKVCache()
            batch.append(rows.map { $0[layer] })
            return batch
        }
        #expect(layers[0].leftPadding == BatchFixture.lengths.map { 17 - $0 })
        Self.expectRowsMatch(Self.batchStep(model, layers, fed), zip(solo, fed).map { $0.step($1) })
    }

    @Test("Retiring the longest row trims the padding every remaining row shares, and a row joining later still matches")
    func filterJoinTrim() throws {
        let model = try TinyModel.make()
        let prompts = BatchFixture.prompts
        let solo = try prompts.map { try Solo(model, $0) }
        let first = try prompts.prefix(3).map { try BatchFixture.prefilled(model, $0) }
        let layers = first[0].indices.map { layer in
            let batch = BatchKVCache()
            batch.append(first.map { $0[layer] })
            return batch
        }
        Self.expectRowsMatch(Self.batchStep(model, layers, [11, 12, 13]), [solo[0].step(11), solo[1].step(12), solo[2].step(13)])

        for layer in layers { layer.filter(rows: [0, 2]) }
        #expect(layers[0].leftPadding == [3, 0])
        #expect(layers[0].offset == 9)

        let joining = try BatchFixture.prefilled(model, prompts[3])
        for (index, layer) in layers.enumerated() { layer.append([joining[index]]) }
        #expect(layers[0].leftPadding == [5, 2, 0])
        Self.expectRowsMatch(Self.batchStep(model, layers, [21, 23, 24]), [solo[0].step(21), solo[2].step(23), solo[3].step(24)])

        for layer in layers { layer.filter(rows: [2]) }
        #expect(layers[0].leftPadding == [0])
        #expect(layers[0].offset == 12)
        Self.expectRowsMatch(Self.batchStep(model, layers, [31]), [solo[3].step(31)])
    }
}

@Suite("BatchRotatingKVCache keeps every row's sliding-window attention equal to its own cache (#255)", .serialized)
struct BatchRotatingKVCacheTests {
    static func prefilledLikeTheIterator(_ model: any LanguageModel, _ prompt: [Int]) throws -> [KVCache] {
        let layers = try BatchProbe.prefilled(model, MLXArray(prompt.dropLast().map(Int32.init)))
        eval([model(MLXArray([Int32(prompt[prompt.count - 1])]).reshaped(1, 1), cache: layers)] + layers.flatMap { $0.innerState() })
        return layers
    }

    @Test("Rows joining and retiring before, while and after the window wraps keep the logits each row gets alone",
          arguments: [TinyModel.Family.gemma4, .gemma4Moe])
    func joinAndRetireAcrossWrap(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let prompts = [3, 6, 13, 2].enumerated().map { BatchFixture.prompt($1, seed: $0) }
        let solo = try prompts.map { try BatchKVCacheTests.Solo(model, $0) }
        let rows = try prompts.map { try Self.prefilledLikeTheIterator(model, $0) }
        let layers = rows[0].map(BatchCore.batchLayer)
        let ring = try #require(layers.first as? BatchRotatingKVCache)
        var fed = 0
        func join(_ joining: [Int]) {
            for (index, layer) in layers.enumerated() { BatchCore.join(joining.map { rows[$0][index] }, into: layer) }
        }
        func retire(keeping kept: [Int]) {
            for layer in layers {
                (layer as? BatchKVCache)?.filter(rows: kept)
                (layer as? BatchRotatingKVCache)?.filter(rows: kept)
            }
        }
        func decode(_ live: [Int], steps: Int) {
            for _ in 0 ..< steps {
                fed += 1
                let tokens = live.map { ($0 * 37 + fed * 11) % 250 + 1 }
                BatchKVCacheTests.expectRowsMatch(
                    BatchKVCacheTests.batchStep(model, layers, tokens), zip(live, tokens).map { solo[$0].step($1) })
            }
        }

        join([0, 1])
        #expect(ring.leftPadding == [3, 0] && !ring.wrapped)
        decode([0, 1], steps: 4)
        #expect(ring.leftPadding == [1, 0] && ring.wrapped && ring.idx == 2)

        join([2])
        #expect(ring.leftPadding == [1, 0, 0] && !ring.wrapped && ring.idx == TinyModel.slidingWindow)
        decode([0, 1, 2], steps: 2)
        retire(keeping: [0, 2])
        #expect(ring.leftPadding == [0, 0] && ring.wrapped && ring.idx == 2)

        join([3])
        #expect(ring.leftPadding == [0, 0, 6])
        decode([0, 2, 3], steps: 3)
        #expect(ring.leftPadding == [0, 0, 3] && ring.wrapped)
        retire(keeping: [2])
        #expect(ring.leftPadding == [0] && !ring.wrapped && ring.idx == 5)
        decode([3], steps: 5)
        #expect(ring.positions == [prompts[3].count + 8])
    }
}

@Suite("Batched decode gives each row the tokens it gets alone (#240)", .serialized)
struct BatchedDecoderTests {

    @Test("Four prompts of different lengths decoded together match their solo greedy decode for 32 steps, on plain attention, recurrent hybrids and sliding-window models",
          .timeLimit(.minutes(10)), arguments: [TinyModel.Family.qwen3Small, .qwen35, .nemotronH, .lfm2, .gemma4, .gemma4Moe])
    func fourRowsMatchSolo(family: TinyModel.Family) async throws {
        let model = try TinyModel.make(family)
        let expected = try BatchFixture.prompts.map { BatchFixture.text(try BatchFixture.soloGreedy(model, $0, steps: 32)) }
        let core = BatchCore(model: model, slots: 4)
        let launches = try BatchFixture.prompts.map { try BatchFixture.launch(core, model, $0, maxTokens: 32) }
        var produced: [String] = []
        for launch in launches { produced.append(await BatchFixture.collect(launch.stream)) }
        #expect(produced == expected)
        #expect(core.decoder.admittedRows == 4)
        #expect(core.decoder.checkInvariants() == [])
    }

    @Test("A row joining mid-flight and rows retiring on EOS or max tokens leave the other rows' tokens unchanged, on plain attention, recurrent hybrids and sliding-window models",
          .timeLimit(.minutes(10)), arguments: [TinyModel.Family.qwen3Small, .qwen35, .nemotronH, .lfm2, .gemma4, .gemma4Moe])
    func joinAndRetire(family: TinyModel.Family) async throws {
        let model = try TinyModel.make(family)
        let prompts = BatchFixture.prompts
        let solo = try prompts.map { try BatchFixture.soloGreedy(model, $0, steps: 48) }
        let stop = solo[2][3]
        let core = BatchCore(model: model, slots: 4)

        let long = try BatchFixture.launch(core, model, prompts[0], maxTokens: 48)
        let short = try BatchFixture.launch(core, model, prompts[1], maxTokens: 5)
        let stopping = try BatchFixture.launch(core, model, prompts[2], maxTokens: 48, stop: [stop])
        var longIterator = long.stream.makeAsyncIterator()
        var head = ""
        while head.unicodeScalars.count < 8, let generation = await longIterator.next() {
            if case .chunk(let chunk) = generation { head += chunk }
        }
        let late = try BatchFixture.launch(core, model, prompts[3], maxTokens: 32)
        #expect(core.decoder.checkInvariants() == [])

        var rest = ""
        while let generation = await longIterator.next() {
            if case .chunk(let chunk) = generation { rest += chunk }
        }
        #expect(head + rest == BatchFixture.text(solo[0]))
        #expect(await BatchFixture.collect(short.stream) == BatchFixture.text(Array(solo[1].prefix(5))))
        let stopAt = try #require(solo[2].firstIndex(of: stop))
        #expect(await BatchFixture.collect(stopping.stream) == BatchFixture.text(Array(solo[2].prefix(stopAt))))
        #expect(await BatchFixture.collect(late.stream) == BatchFixture.text(Array(solo[3].prefix(32))))
        #expect(core.decoder.checkInvariants() == [])
    }

    @Test("Cancelling one row stops only that row", .timeLimit(.minutes(10)), arguments: [TinyModel.Family.qwen3Small, .qwen35])
    func cancellingOneRow(family: TinyModel.Family) async throws {
        let model = try TinyModel.make(family)
        let prompts = BatchFixture.prompts
        let expected = BatchFixture.text(try BatchFixture.soloGreedy(model, prompts[0], steps: 40))
        let core = BatchCore(model: model, slots: 2)
        let kept = try BatchFixture.launch(core, model, prompts[0], maxTokens: 40)
        let cancelled = try BatchFixture.launch(core, model, prompts[1], maxTokens: 1_000_000)

        #expect(await BatchFixture.collect(cancelled.stream, upTo: 4).unicodeScalars.count == 4)
        cancelled.task.cancel()
        await cancelled.task.value
        #expect(core.decoder.checkInvariants() == [])
        #expect(await BatchFixture.collect(kept.stream) == expected)
        await kept.task.value
        let seats = [core.decoder.seat(on: core), core.decoder.seat(on: core)]
        #expect(seats.allSatisfy { $0 != nil })
        #expect(core.decoder.checkInvariants() == [])
    }
}

@Suite("Engine and orchestrator route plain-attention and recurrent-hybrid requests to the batch (#240, #255)", .serialized)
struct BatchedRoutingTests {

    static func solo(_ count: Int, family: TinyModel.Family = .qwen3Small) async throws -> [String] {
        let engine = await BatchFixture.engine(try TinyModel.make(family), id: "tiny-\(family)", batchSlots: 1)
        var texts: [String] = []
        for index in 0 ..< count { texts.append(try await engine.generate(request: BatchFixture.request(index)).generatedText) }
        return texts
    }

    struct Reply: Sendable {
        let text: String
        let metadata: GenerationResponse
    }

    static func concurrently(_ orchestrator: ModelOrchestratorActor, count: Int) async throws -> [Reply] {
        try await withThrowingTaskGroup(of: (Int, Reply).self) { group in
            for index in 0 ..< count {
                group.addTask {
                    let request = BatchFixture.request(index)
                    guard index.isMultiple(of: 2) else {
                        let response = try await orchestrator.generate(request: request)
                        return (index, Reply(text: response.generatedText, metadata: response))
                    }
                    var text = ""
                    var metadata: GenerationResponse?
                    for await chunk in try await orchestrator.generateStream(request: request) {
                        switch chunk {
                        case .text(let piece): text += piece
                        case .metadata(let final): metadata = final
                        case .reasoning, .toolCall: break
                        }
                    }
                    return (index, Reply(text: text, metadata: try #require(metadata)))
                }
            }
            var replies: [Int: Reply] = [:]
            for try await (index, reply) in group { replies[index] = reply }
            return (0 ..< count).compactMap { replies[$0] }
        }
    }

    @Test("With 4 batch slots and 4 concurrent generations, streaming or not, each request gets its solo output and telemetry, on plain attention and on recurrent hybrids",
          .timeLimit(.minutes(10)), arguments: [TinyModel.Family.qwen3Small, .qwen35, .nemotronH, .lfm2])
    func orchestratorBatchesFourRequests(family: TinyModel.Family) async throws {
        let expected = try await Self.solo(4, family: family)
        let engine = await BatchFixture.engine(try TinyModel.make(family), id: "tiny-\(family)", batchSlots: 4)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 4))
        let responses = try await Self.concurrently(orchestrator, count: 4)
        #expect(responses.map(\.text) == expected)
        #expect(responses.allSatisfy { $0.metadata.completionTokens == 32 && $0.metadata.tokensPerSecond > 0 && $0.metadata.timeToFirstToken > 0 })
        #expect(await engine.batchedRows == 4)
        #expect(await engine.checkInvariants(idle: true) == [])
        #expect(await orchestrator.gate.checkInvariants(idle: true) == [])
    }

    @Test("With 1 batch slot, concurrent generations decode on their own as before; on a model with compiled traces they no longer race each other",
          .timeLimit(.minutes(10)), arguments: [TinyModel.Family.qwen3Small, .qwen35])
    func singleSlotIsUnchanged(family: TinyModel.Family) async throws {
        let expected = try await Self.solo(4, family: family)
        let engine = await BatchFixture.engine(try TinyModel.make(family), id: "tiny-\(family)", batchSlots: 1)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 4))
        #expect(try await Self.concurrently(orchestrator, count: 4).map(\.text) == expected)
        #expect(await engine.batchedRows == 0)
    }

    @Test("serve --batch-slots defaults to 1 and refuses 0")
    func batchSlotsOption() throws {
        #expect(try ServeCommand.parse([]).batchSlots == 1)
        #expect(try ServeCommand.parse(["--batch-slots", "4", "--max-concurrent", "4"]).batchSlots == 4)
        #expect(throws: (any Error).self) { _ = try ServeCommand.parse(["--batch-slots", "0"]) }
    }

    @Test("A request with a response format keeps the serialized path; the same engine batches a plain one")
    func grammarStaysSolo() async throws {
        let engine = MLXInferenceEngine(batchSlots: 4)
        await engine.attach(try ConstrainedModel.container())
        let constrained = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema)))
        #expect(try ConstrainedModel.isValid(constrained.generatedText))
        #expect(await engine.batchedRows == 0)
        _ = try await engine.generate(request: ConstrainedModel.request(format: nil, maxTokens: 4))
        #expect(await engine.batchedRows == 1)
    }

    @Test("Under tight memory pressure a plain request decodes on its own; back at relaxed it batches again")
    func pressureStopsJoining() async throws {
        let engine = MLXInferenceEngine(batchSlots: 4)
        await engine.attach(try ConstrainedModel.container())
        _ = await engine.apply(.tight)
        _ = try await engine.generate(request: ConstrainedModel.request(format: nil, maxTokens: 4))
        #expect(await engine.batchedRows == 0)
        _ = await engine.apply(.relaxed)
        _ = try await engine.generate(request: ConstrainedModel.request(format: nil, maxTokens: 4))
        #expect(await engine.batchedRows == 1)
    }
}
