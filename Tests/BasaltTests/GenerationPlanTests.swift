import Testing
import MLXLMCommon
import MLXLLM
import BasaltCore
@testable import BasaltInference

@Suite("Per-request decisions come from one pure GenerationPlan, so MTP, lookup, batching, blending and the KV cache cannot collide")
struct GenerationPlanTests {
    enum Decision: Sendable {
        case drafts, batchable, findsAnchors, blends

        func read(_ plan: GenerationPlan) -> Bool {
            switch self {
            case .drafts: plan.drafts
            case .batchable: plan.batchable
            case .findsAnchors: plan.findsAnchors
            case .blends: plan.blends
            }
        }
    }

    struct Case: Sendable, CustomTestStringConvertible {
        let testDescription: String
        var request = GenerationRequest(prompt: "p")
        var settings = GenerationPlan.Settings()
        var facts = GenerationPlan.Facts(promptTokens: 100)
        let decision: Decision
        let expected: Bool
    }

    static let zoned = GenerationPlan.Settings(kvZones: .standard, attendsThroughZones: true, prefillsInChunks: true)
    static let drafter = GenerationPlan.Facts(promptTokens: 100, hasMTPDrafter: true)
    static let cached = GenerationPlan.Facts(namespace: "chat", promptTokens: 100)

    static let mtp: [Case] = [
        Case(testDescription: "MTP drafts on a plain prompt when the model has a drafter", facts: drafter, decision: .drafts, expected: true),
        Case(testDescription: "no drafter, no MTP", decision: .drafts, expected: false),
        Case(testDescription: "lookup from the request wins over MTP",
             request: GenerationRequest(prompt: "p").withSpeculativeLookup(true), facts: drafter, decision: .drafts, expected: false),
        Case(testDescription: "lookup from serve wins over MTP",
             settings: GenerationPlan.Settings(speculativeLookup: true), facts: drafter, decision: .drafts, expected: false),
        Case(testDescription: "a request that turns lookup off gets MTP back",
             request: GenerationRequest(prompt: "p").withSpeculativeLookup(false),
             settings: GenerationPlan.Settings(speculativeLookup: true), facts: drafter, decision: .drafts, expected: true),
        Case(testDescription: "a grammar constraint turns MTP off",
             facts: GenerationPlan.Facts(promptTokens: 100, constrained: true, hasMTPDrafter: true), decision: .drafts, expected: false),
        Case(testDescription: "KV zones turn MTP off", settings: zoned, facts: drafter, decision: .drafts, expected: false),
        Case(testDescription: "--kv-bits turns MTP off",
             settings: GenerationPlan.Settings(kvPrecision: .q8), facts: drafter, decision: .drafts, expected: false),
        Case(testDescription: "a request kv precision turns MTP off",
             request: GenerationRequest(prompt: "p").withKVPrecision(.q4), facts: drafter, decision: .drafts, expected: false),
        Case(testDescription: "a prompt at the MTP limit still drafts",
             facts: GenerationPlan.Facts(promptTokens: MLXInferenceEngine.mtpPromptLimit, hasMTPDrafter: true), decision: .drafts, expected: true),
        Case(testDescription: "a prompt over the MTP limit does not draft",
             facts: GenerationPlan.Facts(promptTokens: MLXInferenceEngine.mtpPromptLimit + 1, hasMTPDrafter: true), decision: .drafts, expected: false),
    ]

    static let batch: [Case] = [
        Case(testDescription: "a plain request joins the batch", decision: .batchable, expected: true),
        Case(testDescription: "a grammar constraint joins the batch with its own mask",
             facts: GenerationPlan.Facts(promptTokens: 100, constrained: true), decision: .batchable, expected: true),
        Case(testDescription: "a forced tool joins the batch",
             request: GenerationRequest(prompt: "p").with { $0.toolChoice = ToolChoice(forcedTool: "f") }, decision: .batchable, expected: true),
        Case(testDescription: "grammar jump-forward keeps the request out of the batch",
             facts: GenerationPlan.Facts(promptTokens: 100, constrained: true, jumpsForward: true), decision: .batchable, expected: false),
        Case(testDescription: "lookup keeps the request out of the batch",
             request: GenerationRequest(prompt: "p").withSpeculativeLookup(true), decision: .batchable, expected: false),
        Case(testDescription: "MTP keeps the request out of the batch", facts: drafter, decision: .batchable, expected: false),
        Case(testDescription: "cache blending keeps the request out of the batch",
             facts: GenerationPlan.Facts(namespace: "chat", promptTokens: 100, blendAvailable: true, modelIsHybrid: false),
             decision: .batchable, expected: false),
        Case(testDescription: "a hybrid model does not blend, so it still batches",
             facts: GenerationPlan.Facts(namespace: "chat", promptTokens: 100, blendAvailable: true, modelIsHybrid: true),
             decision: .batchable, expected: true),
        Case(testDescription: "--kv-bits keeps the request out of the batch",
             settings: GenerationPlan.Settings(kvPrecision: .q8), decision: .batchable, expected: false),
        Case(testDescription: "zoned precision keeps the request out of the batch",
             settings: GenerationPlan.Settings(kvPrecision: .k8v4, kvZones: .standard, attendsThroughZones: true, prefillsInChunks: true),
             decision: .batchable, expected: false),
        Case(testDescription: "zones without a precision leave batching on", settings: zoned, decision: .batchable, expected: true),
        Case(testDescription: "media or a masked prompt keeps the request out of the batch",
             facts: GenerationPlan.Facts(promptTokens: 100, carriesMoreThanTokens: true), decision: .batchable, expected: false),
    ]

    static let cache: [Case] = [
        Case(testDescription: "a recurrent model with a prompt cache looks for anchors",
             facts: GenerationPlan.Facts(namespace: "chat", promptTokens: 100, modelHasRecurrentState: true), decision: .findsAnchors, expected: true),
        Case(testDescription: "no prompt cache, no anchors",
             facts: GenerationPlan.Facts(promptTokens: 100, modelHasRecurrentState: true), decision: .findsAnchors, expected: false),
        Case(testDescription: "MTP keeps its own prefix, so no anchors",
             facts: GenerationPlan.Facts(namespace: "chat", promptTokens: 100, hasMTPDrafter: true, modelHasRecurrentState: true),
             decision: .findsAnchors, expected: false),
        Case(testDescription: "a model not yet probed for recurrent state gets no anchors", facts: cached, decision: .findsAnchors, expected: false),
        Case(testDescription: "a cached prompt on a non-hybrid model blends",
             facts: GenerationPlan.Facts(namespace: "chat", promptTokens: 100, blendAvailable: true, modelIsHybrid: false), decision: .blends, expected: true),
        Case(testDescription: "a model not yet probed for hybrid layers does not blend",
             facts: GenerationPlan.Facts(namespace: "chat", promptTokens: 100, blendAvailable: true), decision: .blends, expected: false),
        Case(testDescription: "without --cache-blend nothing blends",
             facts: GenerationPlan.Facts(namespace: "chat", promptTokens: 100, modelIsHybrid: false), decision: .blends, expected: false),
        Case(testDescription: "without a prompt cache nothing blends",
             facts: GenerationPlan.Facts(promptTokens: 100, blendAvailable: true, modelIsHybrid: false), decision: .blends, expected: false),
    ]

    @Test("Each decision follows its inputs", arguments: mtp + batch + cache)
    func decision(_ row: Case) {
        #expect(row.decision.read(GenerationPlan.make(row.request, row.settings, row.facts)) == row.expected)
    }

    @Test("Zones apply only to a model that attends through them, and eviction only to one that prefills in chunks")
    func zones() {
        let evicting = GenerationPlan.Settings(kvPrecision: .k8v4, kvZones: KVZones.standard.evicting(0.5), attendsThroughZones: true)
        let request = GenerationRequest(prompt: "p")
        #expect(GenerationPlan.make(request, evicting, GenerationPlan.Facts()).kvZones == KVZones.standard.evicting(0))
        #expect(GenerationPlan.make(request, evicting, GenerationPlan.Facts()).kvPrecision == .k8v4)
        var chunked = evicting
        chunked.prefillsInChunks = true
        #expect(GenerationPlan.make(request, chunked, GenerationPlan.Facts()).kvZones == KVZones.standard.evicting(0.5))
        var flat = evicting
        flat.attendsThroughZones = false
        #expect(GenerationPlan.make(request, flat, GenerationPlan.Facts()) == GenerationPlan(batchable: false))
        var throughUpdate = flat
        throughUpdate.evictsThroughUpdate = true
        #expect(GenerationPlan.make(request, throughUpdate, GenerationPlan.Facts()).kvZones == KVZones.standard.evicting(0.5).keepingMiddleFullPrecision())
        throughUpdate.capabilities = .conservative
        #expect(GenerationPlan.make(request, throughUpdate, GenerationPlan.Facts()).kvZones == nil)
    }

    @Test("Head split needs the full-precision unzoned cache; Quest steps aside for head split")
    func headSplitAndQuest() {
        let mask = KVHeadMask(retrieval: [[0, 1]])
        let request = GenerationRequest(prompt: "p")
        let both = GenerationPlan.Settings(kvHeadMask: mask, questTopK: 16)
        #expect(GenerationPlan.make(request, both, GenerationPlan.Facts()) == GenerationPlan(kvHeadMask: mask, batchable: true))
        #expect(GenerationPlan.make(request, GenerationPlan.Settings(questTopK: 16), GenerationPlan.Facts()) == GenerationPlan(questTopK: 16, batchable: true))
        #expect(GenerationPlan.make(request.withKVPrecision(.q8), both, GenerationPlan.Facts()) == GenerationPlan(batchable: false))
        var zonedBoth = Self.zoned
        zonedBoth.kvHeadMask = mask
        zonedBoth.questTopK = 16
        #expect(GenerationPlan.make(request, zonedBoth, GenerationPlan.Facts()) == GenerationPlan(kvZones: .standard, batchable: true))
    }

    @Test("A prompt-cache namespace needs a budget, the full-precision cache, a token-only prompt and the same model")
    func namespace() {
        let request = GenerationRequest(prompt: "p").with { $0.cacheKey = "chat" }
        let settings = GenerationPlan.Settings()
        #expect(GenerationPlan.cacheNamespace(request, settings, cacheEnabled: true, media: .text, sameEpoch: true) == "chat")
        #expect(GenerationPlan.cacheNamespace(request, settings, cacheEnabled: false, media: .text, sameEpoch: true) == nil)
        #expect(GenerationPlan.cacheNamespace(request, settings, cacheEnabled: true, media: .uncacheable, sameEpoch: true) == nil)
        #expect(GenerationPlan.cacheNamespace(request, settings, cacheEnabled: true, media: .text, sameEpoch: false) == nil)
        #expect(GenerationPlan.cacheNamespace(request.withKVPrecision(.q8), settings, cacheEnabled: true, media: .text, sameEpoch: true) == nil)
        #expect(GenerationPlan.cacheNamespace(
            request, GenerationPlan.Settings(kvPrecision: .q4), cacheEnabled: true, media: .text, sameEpoch: true) == nil)
    }

    @Test("Grammar jump-forward is allowed only on a full-precision cache and a token-only prompt")
    func plainPrompt() {
        let request = GenerationRequest(prompt: "p")
        #expect(GenerationPlan.plainPrompt(request, GenerationPlan.Settings(), carriesMoreThanTokens: false))
        #expect(!GenerationPlan.plainPrompt(request, GenerationPlan.Settings(), carriesMoreThanTokens: true))
        #expect(!GenerationPlan.plainPrompt(request, GenerationPlan.Settings(kvPrecision: .q8), carriesMoreThanTokens: false))
        #expect(!GenerationPlan.plainPrompt(request.withKVPrecision(.k8v4), GenerationPlan.Settings(), carriesMoreThanTokens: false))
    }

    static let layouts = [
        zoned, GenerationPlan.Settings(kvHeadMask: KVHeadMask(retrieval: [[0, 1]]), questTopK: 16), GenerationPlan.Settings(questTopK: 16),
    ]

    @Test("Every known family keeps today's plan for the facts its caches produce", arguments: mtp + batch + cache)
    func knownFamiliesKeepTodaysPlan(_ row: Case) {
        for (modelType, capabilities) in ModelCapabilities.families {
            for base in [row.settings] + Self.layouts {
                var (today, facts) = (base, row.facts)
                if !capabilities.allows(.mtp) { facts.hasMTPDrafter = false }
                if !capabilities.allows(.cacheBlend) { facts.modelIsHybrid = true }
                if !capabilities.allows(.quest) { today.questTopK = nil }
                var verified = today
                verified.capabilities = capabilities
                var expected = GenerationPlan.make(row.request, today, facts)
                if !capabilities.allows(.batchedDecode) { expected.batchable = false }
                #expect(GenerationPlan.make(row.request, verified, facts) == expected, "\(modelType)")
                #expect(GenerationPlan.plainPrompt(row.request, verified, carriesMoreThanTokens: facts.carriesMoreThanTokens)
                    == GenerationPlan.plainPrompt(row.request, today, carriesMoreThanTokens: facts.carriesMoreThanTokens), "\(modelType)")
                #expect(GenerationPlan.cacheNamespace(row.request, verified, cacheEnabled: true, media: .text, sameEpoch: true)
                    == GenerationPlan.cacheNamespace(row.request, today, cacheEnabled: true, media: .text, sameEpoch: true), "\(modelType)")
            }
        }
    }

    @Test("Each feature a known family leaves out is one the runtime already refuses for that family's caches",
          arguments: [
              (TinyModel.Family.qwen3Small, "qwen3"), (.qwen3, "qwen3"), (.qwen3Moe, "qwen3_moe"), (.qwen35, "qwen3_5"), (.qwen35Moe, "qwen3_5_moe"),
              (.prism, PrismHadamard.modelType), (.nemotronH, "nemotron_h"), (.sigmoidMoe, "nemotron_h"), (.lfm2, "lfm2"), (.lfm2Moe, "lfm2_moe"),
              (.gemma3, "gemma3_text"), (.gemma4, "gemma4_text"), (.gemma4Moe, "gemma4"),
          ])
    func exclusionsMatchTheRuntime(family: TinyModel.Family, modelType: String) throws {
        let capabilities = try #require(ModelCapabilities.of(modelType: modelType))
        let model = try TinyModel.make(family)
        let cache = try model.newCache(parameters: nil)
        if !capabilities.allows(.mtp) { #expect(!(model is MLXLLM.Qwen35Model || model is MLXLLM.Qwen35TextModel)) }
        if !capabilities.allows(.cacheBlend) { #expect(PromptCache(cache).isHybrid) }
        if !capabilities.allows(.batchedDecode), family != .gemma3 { #expect(BatchedTokenIterator.batchableLayers(cache) == nil) }
        if !capabilities.allows(.quest) { #expect(try !QuestKVCache.modelAttendsThroughCache(model)) }
    }

    @Test("An unknown model_type gets the conservative plan and one warning naming it and every feature that stays off")
    func unknownFamily() {
        #expect(ModelCapabilities.of(modelType: "deepseek_v3") == nil)
        #expect(ModelCapabilities.of(modelType: nil) == nil)
        #expect(ModelCapabilities.of(modelType: "Qwen3_5") != nil)
        let everything = GenerationPlan.Settings(
            kvZones: .standard, attendsThroughZones: true, prefillsInChunks: true, kvHeadMask: KVHeadMask(retrieval: [[0, 1]]),
            questTopK: 16, speculativeLookup: true, capabilities: .conservative)
        let request = GenerationRequest(prompt: "p").with { $0.cacheKey = "chat" }.withSpeculativeLookup(true)
        let namespace = GenerationPlan.cacheNamespace(request, everything, cacheEnabled: true, media: .text, sameEpoch: true)
        #expect(namespace == nil)
        #expect(!GenerationPlan.plainPrompt(request, everything, carriesMoreThanTokens: false))
        let facts = GenerationPlan.Facts(
            namespace: namespace, promptTokens: 100, hasMTPDrafter: true, blendAvailable: true, modelIsHybrid: false, modelHasRecurrentState: true)
        #expect(GenerationPlan.make(request, everything, facts) == GenerationPlan())
        let warning = ModelCapabilities.unverifiedWarning(modelType: "deepseek_v3")
        #expect(warning.contains("deepseek_v3"))
        #expect(ModelCapabilities.Feature.allCases.allSatisfy { warning.contains($0.rawValue) })
    }
}
