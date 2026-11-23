import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Inkling 2 P2a: DSpark speculative decoding rewinds window KV and conv states and decodes exactly like plain greedy (#650, #228)", .serialized)
struct InklingDSparkTests {
    static let draftConfig = #""mtp_config": {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "intermediate_size": 64, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "local_layer_ids": [0], "sliding_window_size": 4, "block_size": 4, "markov_rank": 8, "markov_head_type": "rnn_hidden", "target_layer_ids": [1, 4, 7, 10]}, "#

    static var prompt: LMInput { LMInput(tokens: MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init))) }

    static func target() throws -> Inkling2Model {
        try #require(try TinyModel.make(.inkling2Loop, adding: draftConfig) as? Inkling2Model)
    }

    static func parameters(_ temperature: Float) -> GenerateParameters {
        GenerateParameters(maxTokens: 48, temperature: temperature, topP: temperature > 0 ? 0.9 : 1, seed: 11)
    }

    static func plain(_ model: any LanguageModel, _ parameters: GenerateParameters) throws -> [Int] {
        Array(IteratorSequence(try TokenIterator(input: prompt, model: model, parameters: parameters)))
    }

    static func speculate(_ model: any LanguageModel, _ drafter: any MTPDrafterModel, _ parameters: GenerateParameters) throws -> (tokens: [Int], iterator: MTPSpeculativeTokenIterator) {
        var iterator = try MTPSpeculativeTokenIterator(input: prompt, mainModel: model, drafter: drafter, parameters: parameters, blockSize: 4)
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        return (tokens, iterator)
    }

    static func checkpoint(for target: Inkling2Model) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(path: "inkling-dspark-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let config = "{" + draftConfig + TinyModel.Family.inkling2Loop.config.dropFirst()
        try Data(config.utf8).write(to: directory.appending(component: "config.json"))
        let configuration = try JSONDecoder().decode(Inkling2DSparkConfiguration.self, from: Data(config.utf8))
        let weights = withRandomState(MLXRandom.RandomState(seed: 650)) {
            let fresh = Inkling2DSparkDraftModel(configuration, target: target.config)
            var weights = Dictionary(uniqueKeysWithValues: fresh.parameters().flattened().filter { !$0.0.contains(".fold.") })
            for key in ["fuse_weight", "mask_embed"] { weights[key] = MLXRandom.normal(weights[key]!.shape) * 0.5 }
            weights["markov_head.hidden_out.weight"] = MLXRandom.normal([64, 8]) * 0.2
            weights["markov_head.hidden_out.bias"] = MLXRandom.normal([64]) * 0.1
            return weights
        }
        try save(arrays: weights, url: directory.appending(component: "mtp.safetensors"))
        return directory
    }

    @Test("Window KV and conv states rewind exactly: drafts that are partly accepted give the plain decoder's tokens, greedy and seeded at T = 0.7",
          arguments: [Float(0), 0.7])
    func rewindParity(temperature: Float) throws {
        let model = try TinyModel.make(.inkling2Loop)
        let expected = try Self.plain(model, Self.parameters(temperature))
        let run = try Self.speculate(model, SuccessorDrafter(following: expected), Self.parameters(temperature))
        expectTokenSequence(run.tokens, expected)
        #expect(run.iterator.passthroughReason == nil)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(run.iterator.acceptedCount > 0)
        #expect(telemetry.rejectedDraftTokenCount > 0)
    }

    @Test("A rejected write of three positions trims back to the conv states and next logits of a one-token decode, and a longer write keeps no windows")
    func convolutionStatesRewind() throws {
        let model = try TinyModel.make(.inkling2Loop)
        let tokens = MLXArray(TinyModel.tokens(Conversation.firstTurn).prefix(12).map(Int32.init))[.newAxis]
        let rewound = try model.newCache(parameters: nil)
        let kept = try model.newCache(parameters: nil)
        _ = model(tokens[0..., ..<9], cache: rewound)
        _ = model(tokens[0..., ..<9], cache: kept)
        _ = model(tokens[0..., 9 ..< 12], cache: rewound)
        _ = model(tokens[0..., 9 ..< 10], cache: kept)
        for (live, reference) in zip(rewound, kept) {
            #expect(live.trim(2) == 2 || !(live is ConvolutionStatesCache))
            if let live = live as? ConvolutionStatesCache, let reference = reference as? ConvolutionStatesCache {
                for slot in 0 ..< 4 { #expect(abs(try #require(live[slot]) - (try #require(reference[slot]))).max().item(Float.self) < 1e-4) }
            }
        }
        let next = MLXArray([Int32(7)]).reshaped(1, 1)
        #expect(abs(model(next, cache: rewound) - model(next, cache: kept)).max().item(Float.self) < 1e-3)
        let prefill = try #require(try model.newCache(parameters: nil)[1] as? ConvolutionStatesCache)
        prefill.commit(states: (0 ..< 4).map { _ in MLXArray.zeros([1, 3, 2]) }, windows: (0 ..< 4).map { _ in MLXArray.zeros([1, 23, 2]) }, count: 20)
        #expect(prefill.rewindablePositions == 0 && prefill.trim(1) == 0)
    }

    @Test("The emit flag returns the tap features, the last tap is the final hidden state, and a chunked prompt gives the plain last-row logits")
    func tapFeatures() throws {
        let model = try Self.target()
        let tokens = MLXArray((0 ..< 600).map { Int32(($0 * 37 + 5) % 250) })[.newAxis]
        var emit = LMOutput.State()
        emit[mtpEmitFlagKey] = true
        let short = model(LMInput.Text(tokens: tokens[0..., ..<20]), cache: try model.newCache(parameters: nil), state: emit)
        let features = try #require(short.state?[mtpLastHiddenStatesKey])
        #expect(features.shape == [1, 20, 4 * 64])
        let final = model.model.languageModel(tokens[0..., ..<20], cache: try model.newCache(parameters: nil))
        #expect(abs(features[.ellipsis, (3 * 64)...] - final).max().item(Float.self) == 0)
        let long = model(LMInput.Text(tokens: tokens), cache: try model.newCache(parameters: nil), state: emit)
        #expect(try #require(long.state?[mtpLastHiddenStatesKey]).shape == [1, 600, 4 * 64])
        let plain = model(tokens, cache: try model.newCache(parameters: nil))[0..., (-1)..., 0...]
        #expect(long.logits.shape == plain.shape)
        #expect(abs(long.logits - plain).max().item(Float.self) < 2e-2)
    }

    @Test("The sidecar loads DSpark from mtp.safetensors beside an Inkling 2 target and folds its head into the target's vocabulary head")
    func loadsAndFolds() throws {
        let target = try Self.target()
        let directory = try Self.checkpoint(for: target)
        defer { try? FileManager.default.removeItem(at: directory) }
        let drafter = try #require(try MTPSidecar.load(directory: directory, target: target) as? GreedyDraftingMTPDrafter)
        let dspark = try #require(drafter.inner as? Inkling2DSparkDraftModel)
        #expect(drafter.draftWidthCost == 0.27 && drafter.maximumBlockSize == 4)
        #expect(MTPSidecar.defaultBlockSize(for: drafter) == 4 && MTPSidecar.defaultBlockSize(for: SuccessorDrafter(following: [1, 2])) == 4)
        let raw = try MLX.loadArrays(url: directory.appending(component: "mtp.safetensors"))
        let fold = matmul(target.lmHead.weight.asType(.float32), try #require(raw["markov_head.hidden_out.weight"]).asType(.float32))
        #expect(abs(dspark.markovHead.fold.weight.asType(.float32) - fold).max().item(Float.self) < 2e-2)
        #expect(throws: (any Error).self) { try MTPSidecar.load(directory: directory, target: try TinyModel.make(.inkling2Loop)) }
    }

    @Test("DSpark drafts past one token a round under the depth policy, and its drafts change nothing in the greedy or seeded output", arguments: [Float(0), 0.7])
    func dsparkParity(temperature: Float) throws {
        let target = try Self.target()
        let directory = try Self.checkpoint(for: target)
        defer { try? FileManager.default.removeItem(at: directory) }
        let drafter = try #require(try MTPSidecar.load(directory: directory, target: target))
        let expected = try Self.plain(target, Self.parameters(temperature))
        let run = try Self.speculate(target, drafter, Self.parameters(temperature))
        expectTokenSequence(run.tokens, expected)
        #expect(run.iterator.passthroughReason == nil)
        let telemetry = try #require(run.iterator.speculativeDecodingTelemetry)
        #expect(telemetry.roundCount > 0)
        #expect(telemetry.draftTokenCount > telemetry.roundCount)
    }

    @Test("Through the engine an Inkling 2 request drafts with DSpark, answers like one without it, and the capability table admits --mtp for Inkling 2")
    func engineDrafts() async throws {
        #expect(ModelCapabilities.of(modelType: "inkling2_loop")?.allows(.mtp) == true)
        let target = try Self.target()
        let directory = try Self.checkpoint(for: target)
        defer { try? FileManager.default.removeItem(at: directory) }
        func reply(mtp: Bool) async throws -> GenerationResponse {
            let engine = MLXInferenceEngine(EngineConfiguration(seed: 11, mtp: mtp))
            let container = ModelContainer(context: ModelContext(
                configuration: ModelConfiguration(id: "tiny-inkling2-dspark"), model: try Self.target(), processor: ByteInputProcessor(), tokenizer: ByteTokenizer()))
            await engine.attach(container)
            if mtp { await engine.useMTPDrafter(in: container) { try? MTPSidecar.load(directory: directory, target: $0.model) } }
            return try await engine.generate(request: GenerationRequest(prompt: Conversation.firstTurn, maxTokens: 32, temperature: 0))
        }
        let plain = try await reply(mtp: false)
        let drafted = try await reply(mtp: true)
        #expect((drafted.draftTokens ?? 0) > 0)
        #expect(drafted.generatedText == plain.generatedText)
    }

    @Test("With a prompt cache the second identical request reuses the prefix and DSpark state and still answers like a fresh engine without --mtp")
    func cachedSecondTurn() async throws {
        let target = try Self.target()
        let directory = try Self.checkpoint(for: target)
        defer { try? FileManager.default.removeItem(at: directory) }
        func engine(mtp: Bool) async throws -> MLXInferenceEngine {
            let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: 1 << 26, seed: 11, mtp: mtp))
            let container = ModelContainer(context: ModelContext(
                configuration: ModelConfiguration(id: "tiny-inkling2-dspark"), model: try Self.target(), processor: ByteInputProcessor(), tokenizer: ByteTokenizer()))
            await engine.attach(container)
            if mtp { await engine.useMTPDrafter(in: container) { try? MTPSidecar.load(directory: directory, target: $0.model) } }
            return engine
        }
        let request = greedy(Conversation.firstTurn, key: "agent", maxTokens: 32)
        let plain = try await engine(mtp: false).generate(request: request)
        let drafting = try await engine(mtp: true)
        let first = try await drafting.generate(request: request)
        let second = try await drafting.generate(request: request)
        #expect(first.generatedText == plain.generatedText)
        #expect(second.cachedPromptTokens > 0)
        #expect((second.draftTokens ?? 0) > 0)
        #expect(second.generatedText == plain.generatedText)
    }

    @Test("The depth policy starts at two drafts at DSpark's width cost, widens to the block when every draft is accepted, and drops to one when none is")
    func depthPolicy() {
        var policy = MTPDraftDepthPolicy(widthCost: 0.27, maximumDepth: 7)
        #expect(policy.depth(maximum: 7) == 2)
        var greedy = policy
        for _ in 0 ..< 40 { greedy.record(drafted: greedy.depth(maximum: 7), accepted: greedy.depth(maximum: 7)) }
        #expect(greedy.depth(maximum: 7) == 7)
        #expect(greedy.depth(maximum: 3) == 3)
        for _ in 0 ..< 40 { policy.record(drafted: policy.depth(maximum: 7), accepted: 0) }
        #expect(policy.depth(maximum: 7) == 1)
        #expect(policy.acceptance(at: 0) < 0.1)
    }

    @Test("Repeated target_layer_ids and a block_size the conv states cannot rewind are refused; a null mtp_config loads the target without taps")
    func configurationEdges() throws {
        #expect(throws: (any Error).self) { _ = try TinyModel.make(.inkling2Loop, adding: Self.draftConfig.replacingOccurrences(of: "[1, 4, 7, 10]", with: "[1, 4, 4, 10]")) }
        let target = try Self.target()
        let config = "{" + Self.draftConfig.replacingOccurrences(of: #""block_size": 4"#, with: #""block_size": 18"#) + TinyModel.Family.inkling2Loop.config.dropFirst()
        let draft = try JSONDecoder().decode(Inkling2DSparkConfiguration.self, from: Data(config.utf8))
        #expect(throws: Inkling2Configuration.Invalid.self) { _ = try draft.validated(for: target.config) }
        let plain = try #require(try TinyModel.make(.inkling2Loop, adding: #""mtp_config": null, "#) as? Inkling2Model)
        #expect(plain.config.draftTaps == nil)
    }

    @Test("A 600-row prompt fills the drafter's context in chunks with the same keys and values as one append")
    func chunkedContext() throws {
        let target = try Self.target()
        let directory = try Self.checkpoint(for: target)
        defer { try? FileManager.default.removeItem(at: directory) }
        let drafter = try Inkling2DSparkDraftModel.load(url: directory.appending(component: "mtp.safetensors"), directory: directory, target: target)
        let features = MLXRandom.normal([1, 600, 4 * 64], key: MLXRandom.key(7)) * 0.3
        var chunked = drafter.makeState(parameters: nil)
        drafter.prepareDrafterState(target: target, input: MTPPrefillInput(promptTokens: MLXArray.zeros([600], dtype: .int32), targetHidden: features, firstBonus: nil), state: &chunked, sampler: ArgMaxSampler())
        var whole = drafter.makeState(parameters: nil)
        for start in [0, 300] {
            drafter.prepareDrafterState(target: target, input: MTPPrefillInput(promptTokens: MLXArray.zeros([300], dtype: .int32), targetHidden: features[0..., start ..< start + 300], firstBonus: nil), state: &whole, sampler: ArgMaxSampler())
        }
        #expect(chunked.nextPosition == 600 && whole.nextPosition == 600)
        for (a, b) in zip(chunked.cache, whole.cache) {
            for (x, y) in zip(a.state, b.state) { #expect(abs(x - y).max().item(Float.self) < 1e-5) }
        }
    }
}
