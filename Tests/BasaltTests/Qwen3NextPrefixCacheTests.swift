import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Hybrid prefix cache on the tiny qwen3_next fixture: a restored anchor against a prefill cut at the same token, and two-turn reuse on the engine with resident and streamed experts (#642 P3)", .serialized)
struct Qwen3NextPrefixCacheTests {
    static let boundary = TinyModel.tokens(Conversation.firstTurn).count - 1
    static let zones = KVZones(sinkTokens: 4, windowTokens: 16, migrationStep: 8)
    static let zonedFormats: [(KVPrecision, String)] = [
        (.k8v4, ""),
        (KVPrecision.polar4.resolvingPolar(rotated: 32), #""head_dim": 128, "#),
    ]

    static func bitwise(_ a: MLXArray?, _ b: MLXArray?) -> Bool {
        guard let a, let b else { return false }
        return a.shape == b.shape && a.dtype == b.dtype && arrayEqual(a, b).item(Bool.self)
    }

    static func mismatches(_ restored: [any KVCache], _ cut: [any KVCache]) -> [String] {
        guard restored.count == cut.count else { return ["\(restored.count) layers against \(cut.count)"] }
        return zip(restored, cut).enumerated().flatMap { index, pair -> [String] in
            let (a, b) = pair
            if let a = a as? MambaCache, let b = b as? MambaCache {
                return (bitwise(a[0], b[0]) ? [] : ["layer \(index) conv state"])
                    + (bitwise(a[1], b[1]) ? [] : ["layer \(index) recurrent state"])
            }
            guard a.offset == b.offset else { return ["layer \(index) offset \(a.offset) against \(b.offset)"] }
            let (left, right) = (a.state, b.state)
            return left.count == right.count && zip(left, right).allSatisfy { bitwise($0, $1) } ? [] : ["layer \(index) attention K/V"]
        }
    }

    @Test("On plain fp32 K/V, a cache leased from the anchor at token r holds the conv state, the recurrent state and the attention K/V of a prefill cut at r bitwise; its tail logits equal the cut prefill's bitwise and one uninterrupted prefill's within 1e-3",
          arguments: [48, boundary])
    func restoredAnchorEqualsCutPrefill(restorePoint: Int) throws {
        let model = try TinyModel.make(.qwen3Next)
        let first = TinyModel.tokens(Conversation.firstTurn)
        let second = TinyModel.tokens(Conversation.afterToolCall)
        let prefill = PrefillParameters(stepSize: 16)
        let head = try HybridPrefixSnapshot.prefillHead(
            MLXArray(first.map(Int32.init)), into: PromptCache(try model.newCache(parameters: nil)),
            plan: CachedPrompt(cuts: [restorePoint], boundary: Self.boundary), model: model, prefill: prefill)
        let anchor = try #require(head.anchors.first ?? head.kept)
        #expect(anchor.isSnapshot(of: restorePoint))
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        tree.checkedStore("ns", tokens: Array(first.prefix(restorePoint)), cache: anchor)

        let cut = PromptCache(try model.newCache(parameters: nil))
        _ = try HybridPrefixSnapshot.prefillHead(
            MLXArray(second.map(Int32.init)), into: cut, plan: CachedPrompt(boundary: restorePoint), model: model, prefill: prefill)
        let lease = try tree.required("ns", for: second)
        #expect(lease.reusedTokens == restorePoint)
        #expect(lease.cache.layers.map { $0 is MambaCache } == [true, true, true, false])
        #expect(Self.mismatches(lease.cache.layers, cut.layers).isEmpty, "\(Self.mismatches(lease.cache.layers, cut.layers))")

        let tail = TinyHybrid.row(Array(second[restorePoint...]))
        let restoredLogits = model(tail, cache: lease.cache.layers)
        let cutLogits = model(tail, cache: cut.layers)
        let full = model(TinyHybrid.row(second), cache: try model.newCache(parameters: nil))[0..., restorePoint..., 0...]
        eval(restoredLogits, cutLogits, full)
        #expect(Self.mismatches(lease.cache.layers, cut.layers).isEmpty, "\(Self.mismatches(lease.cache.layers, cut.layers))")
        _ = consume lease
        #expect(Self.bitwise(restoredLogits, cutLogits))
        let gap = abs(restoredLogits - full).max().item(Float.self)
        #expect(gap <= 1e-3, "max logit gap to one uninterrupted prefill \(gap)")
        #expect(tree.checkInvariants(leases: 0) == [])
    }

    @Test("With a zoned k8v4 or polar4 attention cache, a whole-cache copy taken at token r holds the cut prefill's conv state, recurrent state and exact and quantized K/V bitwise and gives its tail logits bitwise; the prefix tree does not take that cache, because a zoned layer is not a whole-context layer",
          arguments: zonedFormats)
    func zonedCopyEqualsCutPrefill(precision: KVPrecision, extraConfig: String) throws {
        let model = try TinyModel.make(.qwen3Next, adding: extraConfig)
        let prompt = TinyModel.tokens(Conversation.afterToolCall)
        let cut = ZonedKVCache.replacingFullAttention(try model.newCache(parameters: nil), zones: Self.zones, precision: precision)
        _ = model(TinyHybrid.row(Array(prompt[..<Self.boundary])), cache: cut)
        eval(cut.flatMap { $0.innerState() })
        let zoned = try #require(cut.compactMap { $0 as? ZonedKVCache }.first)
        #expect(zoned.headsQuantize == true && zoned.quantizedCount > 0)
        #expect((zoned.state[2].dtype == .uint8) == precision.isPolar)
        if let rotated = precision.polarRotated {
            #expect(zoned.state[0].dim(3) == 128 && zoned.state[2].dim(-1) == rotated / 2)
        }

        let restored = cut.map { $0.copy() }
        #expect(restored.map { $0 is MambaCache } == [true, true, true, false])
        #expect(Self.mismatches(restored, cut) == [])
        var tree = PromptCacheTree(byteBudget: ByteCount(1 << 30))
        tree.checkedStore("ns", tokens: Array(prompt[..<Self.boundary]), cache: PromptCache(restored))
        #expect(!PromptCache(restored).isHybrid && tree.leaves().isEmpty)

        let tail = TinyHybrid.row(Array(prompt[Self.boundary...]))
        let restoredLogits = model(tail, cache: restored)
        let cutLogits = model(tail, cache: cut)
        eval(restoredLogits, cutLogits)
        #expect(Self.bitwise(restoredLogits, cutLogits))
        #expect(Self.mismatches(restored, cut) == [])
    }

    static func secondTurnReusesThePrefix(_ engine: MLXInferenceEngine, fresh: MLXInferenceEngine) async throws {
        let first = AgentTrace.messages(thinking: false)
        let second = first + [
            ChatMessageInput(role: .assistant, text: AgentTrace.text(8, 100)),
            ChatMessageInput(role: .user, text: AgentTrace.text(10, 100)),
        ]
        let turn1 = try await engine.generate(request: AgentTrace.request(first, key: "ns"))
        #expect(turn1.cachedPromptTokens == 0)
        #expect(await engine.hybridAnchorUsage?.count == 3)

        let turn2 = try await engine.generate(request: AgentTrace.request(second, key: "ns"))
        let cold = try await fresh.generate(request: AgentTrace.request(second, key: nil))
        #expect(turn2.cachedPromptTokens == turn1.promptTokens - 1)
        #expect(turn2.promptTokens == cold.promptTokens)
        #expect(cold.cachedPromptTokens == 0)
        #expect(!turn2.generatedText.isEmpty)
        #expect(turn2.generatedText == cold.generatedText)
        #expect(await engine.checkInvariants(idle: true) == [])
    }

    @Test("On the engine, a second turn that extends the first keeps three message-boundary anchors, prefills only the appended tokens and generates what a fresh engine without a cache generates")
    func engineReusesThePrefix() async throws {
        try await Self.secondTurnReusesThePrefix(
            try await TinyModel.engine(.qwen3Next, promptCacheBytes: 1 << 30),
            fresh: try await TinyModel.engine(.qwen3Next, promptCacheBytes: 0))
    }

    @Test("With experts streamed from the fixture's safetensors, and the capabilities the engine resolves for model_type qwen3_next, the second turn reuses the prefix the same way and generates what a fresh streamed engine without a cache generates")
    func streamedEngineReusesThePrefix() async throws {
        let directory = try ExpertStreamingParityTests.source("qwen3_next")
        defer { ExpertStreamingParityTests.removeGenerated(directory) }
        func streamed(promptCacheBytes: Int) async throws -> MLXInferenceEngine {
            let engine = MLXInferenceEngine(EngineConfiguration(promptCacheBytes: promptCacheBytes))
            let capabilities = try #require(ModelCapabilities.of(modelType: "qwen3_next"))
            await engine.record(.loadStarted(path: directory.path))
            await engine.attach(
                try await ExpertStreamingParityTests.streamed(directory, cache: ByteCount(1 << 30)),
                facts: ModelFacts(capabilities: capabilities), epoch: engine.modelEpoch)
            return engine
        }
        let engine = try await streamed(promptCacheBytes: 1 << 30)
        try await Self.secondTurnReusesThePrefix(engine, fresh: try await streamed(promptCacheBytes: 0))
        let cache = try #require(await engine.expertCacheStats())
        #expect(cache.stats.misses > 0 && cache.stats.hits > 0)
    }
}
