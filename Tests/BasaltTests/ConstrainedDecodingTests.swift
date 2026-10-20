import Foundation
import Testing
import Synchronization
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

struct JSONPieceTokenizer: MLXLMCommon.Tokenizer {
    var endsText = true
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
    var eosToken: String? { endsText ? "<eos>" : nil }
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
    static let renderedTools = Mutex<[String]>([])

    func prepare(input: UserInput) throws -> LMInput {
        Self.renderedTools.withLock { $0 = (input.tools ?? []).map { "\($0)" } }
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

    static func engine(
        opensThinking: Bool = false, profile: any ModelProfile = GenericProfile(), endsText: Bool = true
    ) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-json", reasoningConfig: QwenReasoningProtocol.tagged),
            model: try withRandomState(MLXRandom.RandomState(seed: 56)) { try TinyModel.make() },
            processor: JSONPieceInputProcessor(opensThinking: opensThinking),
            tokenizer: JSONPieceTokenizer(endsText: endsText))), profile: profile, vocabulary: JSONPieceTokenizer.vocabulary)
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

    @Test("An unsupported strict response_format schema is a request error; a forced tool with one falls back to unconstrained")
    func unsupportedSchemas() async throws {
        let engine = try await ConstrainedModel.engine()
        let unsupported = #"{"type": "object", "properties": {"s": {"type": "string", "pattern": "^a"}}}"#
        let error = await #expect(throws: BasaltError.self) {
            _ = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(unsupported, strict: true)))
        }
        guard case .invalidRequestStructure(let details)? = error else { Issue.record("\(String(describing: error))"); return }
        #expect(details == "response_format: unsupported JSON Schema keyword 'pattern' at #/properties/s")
        let tool = ToolDefinition(name: "f", parametersJSON: #"{"allOf": [{"type": "object"}]}"#)
        let fallback = try await engine.generate(request: ConstrainedModel.request(
            format: nil, maxTokens: 20, tools: [tool], choice: ToolChoice(forcedTool: "f")))
        #expect(fallback.toolCalls == nil)
    }

    @Test("Dynamic temperature with a narrow top_p filters the already-masked distribution, so the answer stays schema-valid")
    func nucleusAfterTheMask() async throws {
        let engine = try await ConstrainedModel.engine()
        for _ in 0..<3 {
            let result = try await engine.generate(request: GenerationRequest(
                prompt: "Report the mode.", maxTokens: 300, temperature: 1.0, topP: 0.05, dynamicTemperature: true,
                responseFormat: .jsonSchema(ConstrainedModel.schema)))
            #expect(try ConstrainedModel.isValid(result.generatedText), "\(result.generatedText)")
        }
    }

    @Test("A forced tool call cut by max_tokens comes back as text with finish reason length, not as a tool call")
    func truncatedForcedToolIsText() async throws {
        let engine = try await ConstrainedModel.engine()
        let tool = ToolDefinition(name: "set_mode", parametersJSON: ConstrainedModel.schema)
        let result = try await engine.generate(request: ConstrainedModel.request(
            format: nil, maxTokens: 6, tools: [tool], choice: ToolChoice(forcedTool: "set_mode")))
        #expect(result.toolCalls == nil)
        #expect(!result.generatedText.isEmpty)
        #expect(result.finishReason == .length)
    }

    @Test("A forced tool_choice still renders the tool, with its description, into the prompt")
    func forcedToolKeepsItsDescription() async throws {
        let engine = try await ConstrainedModel.engine()
        let tool = ToolDefinition(name: "set_mode", description: "Switches the reactor mode", parametersJSON: ConstrainedModel.schema)
        _ = try await engine.generate(request: ConstrainedModel.request(
            format: nil, tools: [tool], choice: ToolChoice(forcedTool: "set_mode")))
        let rendered = JSONPieceInputProcessor.renderedTools.withLock { tools in tools }
        #expect(rendered.contains { $0.contains("Switches the reactor mode") }, "\(rendered)")
    }

    @Test("Without any stop token generation ends at the closing brace instead of running to max_tokens")
    func endsWithoutStopTokens() async throws {
        let engine = try await ConstrainedModel.engine(endsText: false)
        let result = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema)))
        #expect(try ConstrainedModel.isValid(result.generatedText), "\(result.generatedText)")
        #expect(result.completionTokens < 300)
        #expect(result.finishReason == .stop)
    }

    @Test("The grammar compiled by the controller is the one the engine enforces; the schema is not compiled again")
    func usesTheCompiledGrammar() async throws {
        let engine = try await ConstrainedModel.engine()
        let booleans = try JSONGrammar.compile(schemaJSON: #"{"type": "array", "items": {"type": "boolean"}}"#)
        let result = try await engine.generate(request: GenerationRequest(
            prompt: "Report the mode.", maxTokens: 300, temperature: 0,
            responseFormat: .jsonSchema(ConstrainedModel.schema), responseGrammar: booleans))
        #expect((try? JSONSerialization.jsonObject(with: Data(result.generatedText.utf8))) is [Bool], "\(result.generatedText)")
    }

    @Test("An unreadable tokenizer.json is read once per model, not on every request")
    func brokenTokenizerIsReadOnce() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("basalt-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("tokenizer.json")
        try Data("{}".utf8).write(to: file)
        let engine = MLXInferenceEngine()
        await engine.pointTokenizer(at: directory)
        #expect(await engine.loadedVocabulary() == nil)
        try Data(#"{"model": {"vocab": {"a": 0}}}"#.utf8).write(to: file)
        #expect(await engine.loadedVocabulary() == nil)
    }
}

extension MLXInferenceEngine {
    func pointTokenizer(at directory: URL) { tokenizerDirectory = directory }
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

    @Test("A closer spelled over several tokens ends reasoning; the text after it starts the document")
    func closerAcrossTokens() {
        var processor = JSONConstraintProcessor(
            masker: Self.masker, stopTokens: [JSONPieceTokenizer.endOfText], markers: ReasoningMarkers(), phase: .reasoning)
        for byte in Array("plan</thi".utf8) { processor.didSample(token: MLXArray(Int32(byte))) }
        #expect(processor.phase == .reasoning)
        for byte in Array("nk>\n".utf8) { processor.didSample(token: MLXArray(Int32(byte))) }
        #expect(processor.phase == .constrained)
        #expect(processor.allowedTokens().contains(Int(UInt8(ascii: "{"))))
    }

    @Test("A token the grammar rejects does not release the constraint, and a closed document without stop tokens leaves the logits alone")
    func neverReleases() {
        var processor = JSONConstraintProcessor(
            masker: Self.masker, stopTokens: [], markers: ReasoningMarkers(), phase: .constrained)
        processor.didSample(token: MLXArray(Int32(UInt8(ascii: "q"))))
        #expect(processor.phase == .constrained)
        let logits = MLXArray.zeros([1, 256])
        #expect(processor.process(logits: logits)[0, Int(UInt8(ascii: "q"))].item(Float.self) == -Float.infinity)
        for byte in Array(#"{"mode":"fast","flags":[],"inner":{"on":true,"note":null}}"#.utf8) {
            processor.didSample(token: MLXArray(Int32(byte)))
        }
        #expect(processor.matcher.isClosed)
        #expect(processor.process(logits: logits).min().item(Float.self) == 0)
    }
}
