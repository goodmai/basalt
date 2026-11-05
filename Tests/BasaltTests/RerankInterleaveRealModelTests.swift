import Foundation
import Synchronization
import Testing
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("A large rerank next to a streaming chat on real models (#355)", .serialized)
struct RerankInterleaveRealModelTests {
    struct Run {
        let scores: [Double]
        let chunks: Int
        let split: Int
    }

    static var models: [String] {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_RERANK_MODELS"] ?? "").split(separator: ",").map {
            GoldenLogitsRealModelTests.path($0.trimmingCharacters(in: .whitespaces))
        }
    }

    static let query = "Which mill does the river pass before dawn?"

    static let documents = (0 ..< 128).map { document in
        (0 ..< 2 + document % 9).map { "Note \($0) of file \(document): the river carries silt past mill \($0 * 7 + document) before dawn." }
            .joined(separator: " ")
    }

    static func reranker(_ path: String, _ log: ModelRunLog) async throws -> EmbeddingService {
        try await log.step("load reranker") {
            try await EmbeddingService(embeddingModel: nil, rerankModel: (id: "reranker", directory: URL(fileURLWithPath: path)))
        }
    }

    static func counted(_ reranker: RerankerContainer, gate: GenerationGate) async throws -> Run {
        let counts = Mutex((chunks: 0, split: 0))
        var options = EmbeddingService.rerankOptions(yieldingTo: gate)
        let ask = options.beforeMicroBatch
        options.beforeMicroBatch = {
            let split = await ask?() ?? false
            counts.withLock { $0 = (chunks: $0.chunks + 1, split: $0.split + (split ? 1 : 0)) }
            return split
        }
        let scores = try await reranker.scores(query: query, documents: documents, instruction: nil, options: options).results.map(\.score)
        return counts.withLock { Run(scores: scores, chunks: $0.chunks, split: $0.split) }
    }

    static func reranking(_ reranker: RerankerContainer, gate: GenerationGate, rounds: Int) async throws -> [Run] {
        var runs: [Run] = []
        for _ in 0 ..< rounds {
            runs.append(try await counted(reranker, gate: gate))
        }
        return runs
    }

    @Test("Scores are bit-identical without a gate and with an idle gate; a busy gate scores one document per step",
          .enabled(if: models.count == 2, "set BASALT_E2E=1 and BASALT_E2E_RERANK_MODELS=<chat>,<reranker>"))
    func chunkedScoresAreBitIdentical() async throws {
        let log = try await ModelRunLog("rerank-chunks", modelPath: Self.models[1])
        defer { log.releaseGPU() }
        let service = try await Self.reranker(Self.models[1], log)
        let reranker = try #require(service.reranker)
        let whole = try await service.rerank(query: Self.query, documents: Self.documents, yieldingTo: nil)
        let idle = try await service.rerank(query: Self.query, documents: Self.documents, yieldingTo: GenerationGate())
        let idleRun = try await Self.counted(reranker, gate: GenerationGate())
        let clock = Mutex(Duration.zero)
        let busyGate = GenerationGate(now: { clock.withLock { $0 } }, sleep: { step in clock.withLock { $0 += step } })
        _ = try await busyGate.admit()
        let busy = try await Self.counted(reranker, gate: busyGate)
        let drift = zip(busy.scores, whole).map { abs($0 - $1) }.max() ?? 0
        log.record("result", String(
            format: "RERANK-CHUNKS %d documents: idle gate %d chunks, %d split, bit-identical %@; busy gate %d chunks, %d split, max |score drift| %.3g",
            Self.documents.count, idleRun.chunks, idleRun.split, String(idle == whole && idleRun.scores == whole),
            busy.chunks, busy.split, drift))
        #expect(idle == whole)
        #expect(idleRun.scores == whole)
        #expect(idleRun.split == 0)
        #expect(busy.chunks == Self.documents.count)
    }

    @Test("Chat keeps ≥ 0.9× its solo tok/s while 128-document reranks run in a loop; chunks and splits per request are logged",
          .enabled(if: models.count == 2, "set BASALT_E2E=1 and BASALT_E2E_RERANK_MODELS=<chat>,<reranker>"))
    func rerankingNextToChat() async throws {
        let (chatPath, rerankerPath) = (Self.models[0], Self.models[1])
        let log = try await ModelRunLog("rerank-interleave", modelPath: chatPath)
        defer { log.releaseGPU() }
        log.record("reranker", rerankerPath)
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(), telemetry: TelemetryTracker())
        try await log.step("load chat") { try await orchestrator.loadModel(path: chatPath) }
        let reranker = try #require(try await Self.reranker(rerankerPath, log).reranker)
        _ = try await Self.counted(reranker, gate: orchestrator.gate)
        _ = try await EmbeddingInterleaveRealModelTests.chat(orchestrator)
        let solo = try await EmbeddingInterleaveRealModelTests.chat(orchestrator)
        let soloStart = ContinuousClock.now
        let soloRuns = try await Self.reranking(reranker, gate: orchestrator.gate, rounds: 3)
        let soloRound = (ContinuousClock.now - soloStart).inSeconds / Double(soloRuns.count)
        let rounds = Int((solo.seconds / soloRound * 1.5).rounded(.up)) + 1
        async let busyRuns = Self.reranking(reranker, gate: orchestrator.gate, rounds: rounds)
        let busy = try await EmbeddingInterleaveRealModelTests.chat(orchestrator)
        let during = try await busyRuns
        log.record("result", String(
            format: "RERANK-INTERLEAVE chat solo %.1f tok/s (decode %.1f, %d tokens), during reranking %.1f tok/s (decode %.1f) = %.2fx; rerank solo %.1f documents/s, %d chunks/request; with the chat %d requests, chunks %@, split %@",
            solo.rate, solo.decode, solo.tokens, busy.rate, busy.decode, busy.rate / solo.rate,
            Double(Self.documents.count) / soloRound, soloRuns.first?.chunks ?? 0, during.count,
            during.map { String($0.chunks) }.joined(separator: "/"), during.map { String($0.split) }.joined(separator: "/")))
        #expect(soloRuns.allSatisfy { $0.split == 0 })
        #expect(busy.rate >= 0.9 * solo.rate, "chat during reranking \(busy.rate) tok/s vs solo \(solo.rate)")
    }
}
