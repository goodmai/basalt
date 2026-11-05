import Foundation
import Testing
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("An indexing job on an embedder next to a streaming chat on real models (#233)", .serialized)
struct EmbeddingInterleaveRealModelTests {
    struct Chat {
        let tokens: Int
        let seconds: Double
        let decode: Double

        var rate: Double { Double(tokens) / seconds }
    }

    static var models: [String] {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_INTERLEAVE_MODELS"] ?? "").split(separator: ",").map {
            GoldenLogitsRealModelTests.path($0.trimmingCharacters(in: .whitespaces))
        }
    }

    static let texts = 32

    static let indexingBody: String = {
        let input = (0 ..< texts).map { document in
            (0 ..< 15).map { "Note \($0) of file \(document): the river carries silt past mill \($0 * 7 + document) before dawn." }
                .joined(separator: " ")
        }
        return (try? JSONEncoder().encode(["input": input])).map { String(decoding: $0, as: UTF8.self) } ?? ""
    }()

    static let chatRequest = GenerationRequest(
        prompt: "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.", maxTokens: 256,
        temperature: 0
    ).with { $0.enableThinking = false }

    static func chat(_ orchestrator: ModelOrchestratorActor) async throws -> Chat {
        let start = ContinuousClock.now
        let response = try await orchestrator.generate(request: chatRequest)
        return Chat(tokens: response.completionTokens, seconds: (ContinuousClock.now - start).inSeconds, decode: response.tokensPerSecond)
    }

    static func index(_ client: some TestClientProtocol) async throws -> Int {
        try await client.execute(
            uri: "/v1/embeddings", method: .post, headers: [.contentType: "application/json"], body: ByteBuffer(string: indexingBody)
        ) { response in
            #expect(response.status == .ok)
            let body = try JSONSerialization.jsonObject(with: Data(String(buffer: response.body).utf8)) as? [String: Any]
            return (body?["usage"] as? [String: Any])?["prompt_tokens"] as? Int ?? 0
        }
    }

    static func indexing(_ client: some TestClientProtocol, rounds: Int) async throws -> [ContinuousClock.Instant] {
        var ends: [ContinuousClock.Instant] = []
        for _ in 0 ..< rounds {
            _ = try await index(client)
            ends.append(.now)
        }
        return ends
    }

    @Test("Chat keeps ≥ 0.9× its solo tok/s while 32 × ~240-token texts index in a loop; embedding texts/s solo and during the chat are logged",
          .enabled(if: models.count == 2, "set BASALT_E2E=1 and BASALT_E2E_INTERLEAVE_MODELS=<chat>,<embedder>"))
    func indexingNextToChat() async throws {
        let (chatPath, embedderPath) = (Self.models[0], Self.models[1])
        let log = try await ModelRunLog("embedding-interleave", modelPath: chatPath)
        defer { log.releaseGPU() }
        log.record("embedder", embedderPath)
        let orchestrator = ModelOrchestratorActor(engine: MLXInferenceEngine(), telemetry: TelemetryTracker())
        try await log.step("load chat") { try await orchestrator.loadModel(path: chatPath) }
        let embeddings = try await log.step("load embedder") {
            try await EmbeddingService(embeddingModel: (id: "embedder", directory: URL(fileURLWithPath: embedderPath)), rerankModel: nil)
        }
        let router = RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: chatPath), embeddings: embeddings).makeRouter()
        try await Application(router: router).test(.router) { (client) async throws in
            let tokens = try await Self.index(client)
            _ = try await Self.chat(orchestrator)
            let solo = try await Self.chat(orchestrator)
            let soloStart = ContinuousClock.now
            let soloRounds = try await Self.indexing(client, rounds: 4)
            let soloSeconds = (soloRounds[soloRounds.count - 1] - soloStart).inSeconds
            let soloIndex = Double(Self.texts * soloRounds.count) / soloSeconds
            let rounds = Int((solo.seconds / (soloSeconds / Double(soloRounds.count)) * 1.5).rounded(.up)) + 1
            let busyStart = ContinuousClock.now
            async let busyRounds = Self.indexing(client, rounds: rounds)
            let busy = try await Self.chat(orchestrator)
            let chatEnd = ContinuousClock.now
            let during = try await busyRounds.filter { $0 <= chatEnd }
            let busyIndex = during.last.map { Double(Self.texts * during.count) / ($0 - busyStart).inSeconds } ?? 0
            log.record("result", String(
                format: "INTERLEAVE chat solo %.1f tok/s (decode %.1f, %d tokens), during indexing %.1f tok/s (decode %.1f) = %.2fx; embedding solo %.1f texts/s, during chat %.1f texts/s = %.2fx; %d tokens/text",
                solo.rate, solo.decode, solo.tokens, busy.rate, busy.decode, busy.rate / solo.rate, soloIndex, busyIndex,
                busyIndex / soloIndex, tokens / Self.texts))
            #expect(busy.rate >= 0.9 * solo.rate, "chat during indexing \(busy.rate) tok/s vs solo \(solo.rate)")
        }
    }
}
