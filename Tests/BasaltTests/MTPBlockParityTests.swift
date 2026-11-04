import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("MTP drafting deeper than one token on a real hybrid model keeps greedy output identical to plain decoding (#207)", .serialized)
struct MTPBlockParityTests {
    static var modelPath: String? {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return nil }
        return env["BASALT_E2E_MTP_MODEL_PATH"]
    }

    static let prompt = """
        Rename the parameter `count` to `limit` everywhere in this function and return only the code:

        func firstWords(_ text: String, count: Int) -> [String] {
            let words = text.split(separator: " ").map(String.init)
            guard count > 0 else { return [] }
            return Array(words.prefix(count))
        }
        """

    static func generate(mtp: Bool, blockSize: Int = 2) async throws -> GenerationResponse {
        let engine = MLXInferenceEngine(mtp: mtp, mtpBlockSize: blockSize)
        try await engine.load(modelPath: try #require(modelPath))
        return try await engine.generate(request: GenerationRequest(
            prompt: prompt, maxTokens: 160, temperature: 0).with { $0.reasoningEffort = "none" })
    }

    @Test("Greedy text with MTP blocks 2, 3 and 4 equals the text without MTP, and each block drafts",
          .enabled(if: modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_MTP_MODEL_PATH"))
    func deeperBlocksMatchPlainDecoding() async throws {
        let plain = try await Self.generate(mtp: false)
        for blockSize in [2, 3, 4] {
            let drafted = try await Self.generate(mtp: true, blockSize: blockSize)
            print("MTPBlockParity block=\(blockSize) drafted=\(drafted.draftTokens ?? 0) accepted=\(drafted.acceptedDraftTokens ?? 0) calls=\(drafted.modelCalls ?? 0)")
            #expect((drafted.draftTokens ?? 0) > 0)
            #expect(drafted.generatedText == plain.generatedText)
        }
    }
}
