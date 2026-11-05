import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Feature matrix: on every tiny family a feature leaves greedy output unchanged, or reports itself off and falls back (#327)", .serialized)
struct FeatureMatrixTests {
    enum Feature: String, CaseIterable, Sendable {
        case lookup, mtp, cacheBlend, kvZones, headSplit, quest

        func applies(to family: TinyModel.Family) -> Bool {
            switch self {
            case .lookup: ![.qwen35Moe, .lfm2Moe, .sigmoidMoe].contains(family)
            case .mtp: [.qwen35, .prism].contains(family)
            case .cacheBlend: [.qwen3, .qwen3HeadDim32].contains(family)
            case .kvZones, .headSplit: family == .qwen3HeadDim32
            case .quest: ![.gemma4, .gemma4Moe].contains(family)
            }
        }

        func makeEngine() -> MLXInferenceEngine {
            let bytes = FeatureMatrixTests.promptCacheBytes
            return switch self {
            case .lookup: MLXInferenceEngine(promptCacheBytes: bytes, speculativeLookup: true)
            case .mtp: MLXInferenceEngine(promptCacheBytes: bytes, mtp: true)
            case .cacheBlend: MLXInferenceEngine(promptCacheBytes: bytes, cacheBlendRatio: 0.15)
            case .kvZones: MLXInferenceEngine(promptCacheBytes: bytes, kvPrecision: .q8, kvZones: .standard)
            case .headSplit: MLXInferenceEngine(promptCacheBytes: bytes, kvHeadSplit: 0.5)
            case .quest: MLXInferenceEngine(promptCacheBytes: bytes, questTopK: 2)
            }
        }

        func prepare(_ engine: MLXInferenceEngine, _ container: ModelContainer, _ family: TinyModel.Family) async throws {
            switch self {
            case .mtp where applies(to: family):
                await engine.useMTPDrafter(in: container) { _ in try? GreedyDraftingMTPDrafter(MTPPromptCacheTests.head()) }
            case .kvZones: await engine.probeKVZones()
            case .headSplit: try await FeatureMatrixTests.probeHeads(engine, storingScores: applies(to: family))
            case .quest: await engine.probeQuest()
            case .lookup, .mtp, .cacheBlend: break
            }
        }

        func reportsOn(_ engine: MLXInferenceEngine, _ replies: [GenerationResponse]) async -> Bool {
            switch self {
            case .lookup, .mtp: return replies.contains { $0.draftTokens != nil }
            case .cacheBlend:
                let store = await engine.cacheBlend
                return (store?.chunkCount ?? 0) > 0
            case .kvZones: return await engine.supportsKVZones
            case .headSplit: return await engine.kvHeadMask != nil
            case .quest: return await engine.questApplies
            }
        }
    }

    static let promptCacheBytes = 1 << 26
    static let families = TinyModel.Family.allCases.filter { $0 != .qwen3Small }
    static let slidingWindowFamilies: [TinyModel.Family] = [.gemma3, .gemma4, .gemma4Moe]

    static var turns: [GenerationRequest] {
        [greedy(Conversation.firstTurn, key: nil, maxTokens: 16), greedy(Conversation.firstTurn, key: "matrix", maxTokens: 16),
         greedy(Conversation.afterToolCall, key: "matrix")]
    }

    static func run(_ family: TinyModel.Family, _ feature: Feature?) async throws -> (engine: MLXInferenceEngine, replies: [GenerationResponse]) {
        let container = try TinyModel.container(family)
        let engine = feature?.makeEngine() ?? MLXInferenceEngine(promptCacheBytes: promptCacheBytes)
        await engine.attach(container)
        try await feature?.prepare(engine, container, family)
        var replies: [GenerationResponse] = []
        for request in turns { replies.append(try await engine.generate(request: request)) }
        return (engine, replies)
    }

    static func probeHeads(_ engine: MLXInferenceEngine, storingScores: Bool) async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: "feature-matrix-\(UUID().uuidString)")
        let scores = KVHeadClassifier.storedScores(for: directory)
        defer { try? FileManager.default.removeItem(at: scores) }
        if storingScores {
            try FileManager.default.createDirectory(at: scores.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(Array(repeating: [Float(1), 0], count: 4)).write(to: scores)
        }
        await engine.probeKVHeads(directory: directory)
    }

    @Test("A plain first turn, a cached first turn and a second turn reusing it decode the same greedy text with the feature on; the feature reports itself on exactly where it applies",
          .timeLimit(.minutes(1)), arguments: families, Feature.allCases)
    func featureLeavesGreedyOutputUnchanged(family: TinyModel.Family, feature: Feature) async throws {
        try await withKnownIssue("Nemotron-H diverges from the plain decoder intermittently in full parallel runs (#250)", isIntermittent: true) {
            let plain = try await Self.run(family, nil).replies
            let (engine, replies) = try await Self.run(family, feature)
            #expect(replies.map(\.generatedText) == plain.map(\.generatedText))
            let applies = feature.applies(to: family)
            let reported = await feature.reportsOn(engine, replies)
            if feature == .lookup {
                #expect(applies || !reported, "lookup drafted on \(family), which it must decode plainly")
            } else {
                #expect(reported == applies, "\(feature) on \(family) reported \(reported ? "on" : "off")")
            }
            if feature == .cacheBlend {
                #expect(replies.map(\.cachedPromptTokens) == plain.map(\.cachedPromptTokens))
            }
        } when: {
            family == .nemotronH && feature == .lookup
        }
    }

    @Test("Four concurrent requests decoded in one batch each get their solo greedy text; sliding-window families decode alone and report no batched rows (#319)",
          .timeLimit(.minutes(1)), arguments: families)
    func batchedDecodeMatchesSolo(family: TinyModel.Family) async throws {
        let expected = try await BatchedRoutingTests.solo(4, family: family)
        let engine = await BatchFixture.engine(try TinyModel.make(family), id: "tiny-\(family)", batchSlots: 4)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 4))
        #expect(try await BatchedRoutingTests.concurrently(orchestrator, count: 4).map(\.text) == expected)
        #expect(await engine.batchedRows == (Self.slidingWindowFamilies.contains(family) ? 0 : 4))
    }
}
