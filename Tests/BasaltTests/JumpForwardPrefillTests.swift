import Testing
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Grammar jump-forward with an assistant prefill (#205)", .serialized)
struct JumpForwardPrefillTests {
    @Test("A prefill whose tokens split across the forced span differently is accepted with jump-forward on, and decodes token for token as with it off (#205)")
    func prefillAcrossForcedSpanBoundary() async throws {
        let engine = try await ConstrainedModel.engine(tokenizer: BoundaryMergeTokenizer())
        let prefill = #"x{"name"#
        #expect(BoundaryMergeTokenizer().encode(text: prefill, addSpecialTokens: false) == [120, 128, 143])
        #expect(BoundaryMergeTokenizer().encode(text: #"{"name":""#, addSpecialTokens: false).prefix(2) == [123, 142])
        let request = GenerationRequest(
            prompt: "Name it.", maxTokens: 12, temperature: 0,
            messages: [ChatMessageInput(role: .user, text: "Name it."), ChatMessageInput(role: .assistant, text: prefill)]
        ).with { $0.responseFormat = .regex(#"x\{"name":"[a-z]+"\}"#) }
        let off = try await engine.generate(request: request.withGrammarJumpForward(false))
        let on = try await engine.generate(request: request.withGrammarJumpForward(true))
        #expect(on.generatedText == off.generatedText)
        #expect(on.completionTokens == off.completionTokens)
        #expect(on.forcedTokens == nil)
    }
}

struct BoundaryMergeTokenizer: MLXLMCommon.Tokenizer {
    static let pieces = JSONPieceTokenizer.pieces.merging([142: "\"name", 143: "name"]) { $1 }
    let base = JSONPieceTokenizer()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            let fits = Self.pieces.filter { rest.hasPrefix($0.value) && !(ids.isEmpty && $0.value == "{\"") }
            guard let (id, piece) = fits.max(by: { $0.value.count < $1.value.count }) else { return ids }
            ids.append(id)
            rest = rest.dropFirst(piece.count)
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { Self.pieces[$0] ?? base.decode(tokenIds: [$0], skipSpecialTokens: skipSpecialTokens) }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { (0..<256).contains(id) ? decode(tokenIds: [id], skipSpecialTokens: false) : nil }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try base.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext)
    }
}
