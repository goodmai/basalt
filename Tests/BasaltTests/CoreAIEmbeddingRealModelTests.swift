import Foundation
import Testing
import Synchronization
@testable import BasaltCore
@testable import BasaltInference

@Suite("Core AI embeddings on the Neural Engine against the PyTorch reference (#223)", .serialized, .timeLimit(.minutes(5)))
struct CoreAIEmbeddingRealModelTests {
    struct Reference: Decodable {
        let texts: [String]
        let vectors: [[Float]]
    }

    static let env = ProcessInfo.processInfo.environment
    static let bundle = env["BASALT_E2E"] == "1" ? env["BASALT_E2E_COREAI_EMBEDDING_BUNDLE"] : nil
    static let reference = env["BASALT_E2E_COREAI_EMBEDDING_REFERENCE"]

    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = (a.reduce(0) { $0 + $1 * $1 } * b.reduce(0) { $0 + $1 * $1 }).squareRoot()
        return norms > 0 ? dot / norms : 0
    }

    @Test("Every vector has cosine ≥ 0.999 to sentence-transformers fp32, Matryoshka 256 stays unit length, and throughput is logged",
          .enabled(if: bundle != nil && reference != nil, "set BASALT_E2E=1, BASALT_E2E_COREAI_EMBEDDING_BUNDLE (coreai.embedding.export --platform iOS --text-only) and BASALT_E2E_COREAI_EMBEDDING_REFERENCE"))
    func matchesReference() async throws {
        let directory = URL(fileURLWithPath: try #require(Self.bundle))
        let expected = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: URL(fileURLWithPath: try #require(Self.reference))))
        let service = try await CoreAIEmbeddingService(directory: directory)
        let start = ContinuousClock.now
        let result = try await service.embed(expected.texts.map(EmbeddingInput.text), dimensions: nil, yieldingTo: nil)
        let elapsed = ContinuousClock.now - start
        let cosines = zip(result.vectors, expected.vectors).map { Self.cosine($0, $1) }
        let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        print("RESULT coreai-embedding \(directory.lastPathComponent): \(cosines.count) texts, min cosine \(cosines.min() ?? .nan), mean \(cosines.reduce(0, +) / Float(max(cosines.count, 1))), \(String(format: "%.1f", Double(result.promptTokens) / seconds)) tokens/s, \(String(format: "%.1f", Double(cosines.count) / seconds)) texts/s")
        #expect(cosines.count == expected.vectors.count && cosines.allSatisfy { $0 >= 0.999 }, "\(cosines)")
        let short = try await service.embed([.text(expected.texts[0])], dimensions: 256, yieldingTo: nil).vectors[0]
        #expect(short.count == 256 && abs(short.reduce(0) { $0 + $1 * $1 } - 1) < 1e-3)
    }

    @Test("Given the gate, rows wait for a running generation; requests past the in-flight limit get serverBusy(1); without the gate an embedding runs beside it (#399)",
          .enabled(if: bundle != nil, "set BASALT_E2E=1 and BASALT_E2E_COREAI_EMBEDDING_BUNDLE"))
    func gateAndBackpressure() async throws {
        let service = try await CoreAIEmbeddingService(directory: URL(fileURLWithPath: try #require(Self.bundle)))
        let polls = Mutex(0)
        let gate = GenerationGate(now: { .zero }, sleep: { step in
            polls.withLock { $0 += 1 }
            try await Task.sleep(for: step)
        })
        let generation = try await gate.admit()
        let beside = try await service.embed([.text("beside decode")], dimensions: nil, yieldingTo: nil)
        #expect(beside.vectors.count == 1 && polls.withLock { $0 } == 0)
        let waiting = (0 ..< CoreAIEmbeddingService.maxInFlight).map { index in
            Task { try await service.embed([.text("chunk \(index)"), .text("next")], dimensions: nil, yieldingTo: gate) }
        }
        while await service.admission.running < CoreAIEmbeddingService.maxInFlight || polls.withLock({ $0 }) < CoreAIEmbeddingService.maxInFlight {
            try Task.checkCancellation()
            await Task.yield()
        }
        let refused = await #expect(throws: BasaltError.self) { try await service.embed([.text("one too many")], dimensions: nil, yieldingTo: nil) }
        if case .serverBusy(1)? = refused {} else { Issue.record("expected serverBusy(1), got \(String(describing: refused))") }
        await gate.release(generation)
        for task in waiting { #expect(try await task.value.vectors.count == 2) }
        #expect(await service.admission.running == 0)
    }
}
