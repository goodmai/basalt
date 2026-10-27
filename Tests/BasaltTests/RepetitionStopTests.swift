import Testing
import Foundation
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

struct ChantingTokenizer: MLXLMCommon.Tokenizer {
    let bytes = ByteTokenizer()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { bytes.encode(text: text, addSpecialTokens: addSpecialTokens) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { String(repeating: "na ", count: tokenIds.count) }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { "na " }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try bytes.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

final class ReentrantTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    var store: CacheBlendStore?
    var decodes = 0

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        decodes += 1
        _ = store?.chunkCount
        return tokenIds == [0] ? "\n\n" : "x"
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

@Suite("Repetition stop in the generation loop (#87)", .serialized)
struct RepetitionStopTests {

    @Test("A verbatim loop ends the stream early with finish reason repetition, also without dynamic temperature")
    func loopStopsEarly() async throws {
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-llama"),
            model: try TinyModel.make(), processor: ByteInputProcessor(), tokenizer: ChantingTokenizer())))
        let response = try await engine.generate(request: GenerationRequest(prompt: "sing", maxTokens: 2_000, temperature: 0))
        #expect(response.finishReason == .repetition)
        #expect(response.completionTokens < 400)
        #expect(response.generatedText.hasPrefix("na na na"))
    }

    @Test("A repetition stop reads as truncated in both APIs, like the token limit")
    func repetitionMapsToTruncation() {
        #expect(OpenAIController.finishReason(.repetition, calledTools: false) == "length")
        #expect(AnthropicController.stopReason(.repetition, calledTools: false) == "max_tokens")
        #expect(OpenAIController.finishReason(.stop, calledTools: false) == "stop")
    }

    @Test("Records that differ only in an id, tables and code are not cut as loops")
    func structuredOutputIsNotALoop() {
        let outputs: [[String]] = [
            ["["] + (0..<60).map { "{\"id\": \($0), \"type\": \"string\", \"required\": true, \"nullable\": false, \"default\": null},\n" },
            (0..<60).map { "| item\($0) | ✓ | ✓ | ✓ | ✓ | n/a | n/a | n/a | yes | yes |\n" },
            (0..<60).map { "    case .value\($0):\n        return nil\n    }\n    switch other {\n" },
        ]
        for chunks in outputs {
            var guard_ = RepetitionGuard()
            #expect(!chunks.contains { guard_.shouldStop(after: $0) })
        }
    }

    @Test("The cycle check needs the tail to repeat verbatim with one period")
    func verbatimCycle() {
        #expect(RepetitionGuard.endsInVerbatimCycle(Array(String(repeating: "abcdefgh", count: 40).utf8), probeSize: 16, repeats: 4))
        #expect(!RepetitionGuard.endsInVerbatimCycle(Array((0..<40).map { "abcdefg\($0)" }.joined().utf8), probeSize: 16, repeats: 4))
        #expect(!RepetitionGuard.endsInVerbatimCycle(Array("abcabc".utf8), probeSize: 16, repeats: 4))
    }

    @Test("Token texts are decoded outside the store lock and only once per token")
    func decodesOutsideTheLock() {
        let store = CacheBlendStore(ratio: 0.2, byteBudget: 1 << 20)
        let tokenizer = ReentrantTokenizer()
        tokenizer.store = store
        let tokens = [1, 2, 0, 3, 1, 0, 2]
        #expect(store.chunks(tokens, tokenizer: tokenizer) == [0..<3, 3..<6, 6..<7])
        #expect(tokenizer.decodes == 4)
        #expect(store.chunks(tokens, tokenizer: tokenizer) == [0..<3, 3..<6, 6..<7])
        #expect(tokenizer.decodes == 4)
    }
}
