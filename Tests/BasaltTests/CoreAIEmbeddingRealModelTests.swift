import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Core AI embeddings on the Neural Engine against the PyTorch reference (#223)", .serialized)
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
}
