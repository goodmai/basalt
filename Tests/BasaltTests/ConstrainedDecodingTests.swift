import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct JSONPieceTokenizer: MLXLMCommon.Tokenizer {
    static let open = 250
    static let close = 251
    static let endOfText = 252
    static let endOfUserTurn = 255
    static let markers = ["<think>": open, "</think>": close, "<eos>": endOfText]
    static let pieces: [Int: String] = {
        var pieces: [Int: String] = [:]
        for byte in 0..<128 { pieces[byte] = String(UnicodeScalar(UInt8(byte))) }
        for (offset, piece) in ["{\"", "\":", "\",", "\"}", "true", "false", "null", "fast", "slow", "},", "}]", "]}", "\": [", "\": {"]
            .enumerated() { pieces[128 + offset] = piece }
        return pieces
    }()

    static let vocabulary = TokenVocabulary(tokenBytes: (0..<256).map { Array((pieces[$0] ?? "").utf8) })

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while let first = rest.first {
            if let (marker, id) = Self.markers.first(where: { rest.hasPrefix($0.key) }) {
                ids.append(id)
                rest = rest.dropFirst(marker.count)
            } else {
                ids.append(first.unicodeScalars.first.map { $0.value < 128 ? Int($0.value) : 63 } ?? 63)
                rest = rest.dropFirst()
            }
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in Self.markers.first { $0.value == id }?.key ?? Self.pieces[id] ?? "" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { Self.markers[token] }
    func convertIdToToken(_ id: Int) -> String? { decode(tokenIds: [id], skipSpecialTokens: false) }
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { encode(text: $0["content"] as? String ?? "", addSpecialTokens: false) } + [Self.endOfUserTurn]
    }
}

struct JSONPieceInputProcessor: UserInputProcessor {
    let opensThinking: Bool

    func prepare(input: UserInput) throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let tokens = try JSONPieceTokenizer().applyChatTemplate(messages: messages, tools: nil, additionalContext: nil)
        return LMInput(tokens: MLXArray((tokens + (opensThinking ? [JSONPieceTokenizer.open] : [])).map(Int32.init)))
    }
}

enum ConstrainedModel {
    static let schema = #"""
        {"type": "object", "properties": {
            "mode": {"enum": ["fast", "slow"]},
            "flags": {"type": "array", "items": {"type": "boolean"}},
            "inner": {"type": "object", "properties": {"on": {"type": "boolean"}, "note": {"type": "null"}}, "required": ["on", "note"]}},
         "required": ["mode", "flags", "inner"]}
        """#

    static func engine(opensThinking: Bool = false, profile: any ModelProfile = GenericProfile()) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-json", reasoningConfig: QwenReasoningProtocol.tagged),
            model: try withRandomState(MLXRandom.RandomState(seed: 56)) { try TinyModel.make() },
            processor: JSONPieceInputProcessor(opensThinking: opensThinking),
            tokenizer: JSONPieceTokenizer())), profile: profile, vocabulary: JSONPieceTokenizer.vocabulary)
        return engine
    }

    static func request(
        format: ResponseFormat?, thinking: Bool = false, budget: Int? = nil, maxTokens: Int = 300,
        tools: [ToolDefinition]? = nil, choice: ToolChoice? = nil
    ) -> GenerationRequest {
        GenerationRequest(prompt: "Report the mode.", maxTokens: maxTokens, temperature: 0, enableThinking: thinking,
                          tools: tools, toolChoice: choice, thinkingBudget: budget, responseFormat: format)
    }

    static func isValid(_ text: String, _ schema: String = schema) throws -> Bool {
        var matcher = JSONGrammarMatcher(grammar: try JSONGrammar.compile(schemaJSON: schema))
        return matcher.accept(Array(text.utf8)) && matcher.isAccepting
            && (try? JSONSerialization.jsonObject(with: Data(text.utf8))) is [String: Any]
    }
}

@Suite("Grammar-constrained decoding in the engine (#42)", .serialized)
struct ConstrainedDecodingEngineTests {

    @Test("A random model writes invalid JSON unconstrained; with response_format json_schema it writes a schema-valid document and stops at its closing brace")
    func constrainsARandomModel() async throws {
        let engine = try await ConstrainedModel.engine()
        let free = try await engine.generate(request: ConstrainedModel.request(format: nil, maxTokens: 120))
        #expect(try !ConstrainedModel.isValid(free.generatedText))

        let constrained = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema)))
        #expect(try ConstrainedModel.isValid(constrained.generatedText), "\(constrained.generatedText)")
        #expect(constrained.finishReason == .stop)
        #expect(constrained.completionTokens < 300)
    }

    @Test("json_object yields a JSON object")
    func jsonObject() async throws {
        let engine = try await ConstrainedModel.engine()
        let result = try await engine.generate(request: ConstrainedModel.request(format: .jsonObject, maxTokens: 400))
        #expect((try? JSONSerialization.jsonObject(with: Data(result.generatedText.utf8))) is [String: Any], "\(result.generatedText)")
    }

    @Test("Reasoning stays free under a thinking budget; the constraint starts after </think>")
    func reasoningIsLeftFree() async throws {
        let engine = try await ConstrainedModel.engine(opensThinking: true, profile: QwenProfile())
        let result = try await engine.generate(request: ConstrainedModel.request(
            format: .jsonSchema(ConstrainedModel.schema), thinking: true, budget: 8, maxTokens: 400))
        let reasoning = try #require(result.reasoningText)
        #expect(reasoning.hasSuffix(ThinkingModel.transition))
        #expect(try !ConstrainedModel.isValid(String(reasoning.dropLast(ThinkingModel.transition.count))))
        #expect(try ConstrainedModel.isValid(result.generatedText.trimmingCharacters(in: .whitespacesAndNewlines)),
                "\(result.generatedText)")
    }

    @Test("A tool_choice that forces a function constrains its arguments and returns them as that tool call")
    func forcedToolArguments() async throws {
        let engine = try await ConstrainedModel.engine()
        let tool = ToolDefinition(name: "set_mode", parametersJSON: ConstrainedModel.schema)
        let result = try await engine.generate(request: ConstrainedModel.request(
            format: nil, tools: [tool], choice: ToolChoice(forcedTool: "set_mode")))
        let call = try #require(result.toolCalls?.first)
        #expect(call.name == "set_mode")
        #expect(try ConstrainedModel.isValid(call.arguments), "\(call.arguments)")
        #expect(result.generatedText.isEmpty)
    }

    @Test("An unsupported response_format schema is a request error; a forced tool with one falls back to unconstrained")
    func unsupportedSchemas() async throws {
        let engine = try await ConstrainedModel.engine()
        let unsupported = #"{"type": "object", "properties": {"s": {"type": "string", "pattern": "^a"}}}"#
        let error = await #expect(throws: BasaltError.self) {
            _ = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(unsupported)))
        }
        guard case .invalidRequestStructure(let details)? = error else { Issue.record("\(String(describing: error))"); return }
        #expect(details == "response_format: unsupported JSON Schema keyword 'pattern' at #/properties/s")
        let tool = ToolDefinition(name: "f", parametersJSON: unsupported)
        let fallback = try await engine.generate(request: ConstrainedModel.request(
            format: nil, maxTokens: 20, tools: [tool], choice: ToolChoice(forcedTool: "f")))
        #expect(fallback.toolCalls == nil)
    }
}

@Suite("JSON constraint processor phases (#42)")
struct JSONConstraintProcessorTests {
    static let masker = GrammarMasker(
        grammar: try! JSONGrammar.compile(schemaJSON: ConstrainedModel.schema), vocabulary: JSONPieceTokenizer.vocabulary)
    static let markers = ReasoningMarkers(openers: [JSONPieceTokenizer.open], closers: [[JSONPieceTokenizer.close]])

    @Test("Before the answer the opener or the grammar's first bytes are allowed; inside reasoning nothing is masked; after the closer only the grammar is")
    func phases() {
        var processor = JSONConstraintProcessor(
            masker: Self.masker, stopTokens: [JSONPieceTokenizer.endOfText], markers: Self.markers, phase: .awaitingAnswer)
        let brace = Int(UInt8(ascii: "{"))
        let letter = Int(UInt8(ascii: "q"))
        #expect(processor.allowedTokens().contains(JSONPieceTokenizer.open))
        #expect(processor.allowedTokens().contains(brace))
        #expect(!processor.allowedTokens().contains(letter))

        processor.didSample(token: MLXArray(Int32(JSONPieceTokenizer.open)))
        #expect(processor.phase == .reasoning)
        let logits = MLXArray.zeros([1, 256])
        #expect(processor.process(logits: logits).min().item(Float.self) == 0)

        processor.didSample(token: MLXArray(Int32(letter)))
        processor.didSample(token: MLXArray(Int32(JSONPieceTokenizer.close)))
        #expect(processor.phase == .constrained)
        #expect(!processor.allowedTokens().contains(JSONPieceTokenizer.open))
        let masked = processor.process(logits: logits)
        #expect(masked[0, brace].item(Float.self) == 0)
        #expect(masked[0, letter].item(Float.self) == -Float.infinity)
    }

    @Test("Once the top-level value closes only the stop token is left")
    func stopsAtTheClosingBrace() {
        var processor = JSONConstraintProcessor(
            masker: Self.masker, stopTokens: [JSONPieceTokenizer.endOfText], markers: ReasoningMarkers(), phase: .constrained)
        for byte in Array(#"{"mode":"fast","flags":[],"inner":{"on":true,"note":null}}"#.utf8) {
            processor.didSample(token: MLXArray(Int32(byte)))
        }
        let allowed = processor.allowedTokens()
        #expect(allowed.contains(JSONPieceTokenizer.endOfText))
        #expect((0..<256).filter(allowed.contains) == [JSONPieceTokenizer.endOfText])
    }
}
