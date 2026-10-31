import Foundation
import Testing
import Synchronization
import MLX
import MLXLMCommon
import MLXLLM
import MLXGuidedGeneration
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

    static let markerVocabulary = try! GrammarVocabulary(
        pieces: (0..<256).map { JSONPieceTokenizer().convertIdToToken($0)! }, vocabType: .raw, stopTokens: [endOfText],
        markers: ReasoningMarkers(openers: [open], closers: [[close]]))

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
    func convertIdToToken(_ id: Int) -> String? { (0..<256).contains(id) ? decode(tokenIds: [id], skipSpecialTokens: false) : nil }
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
        opensThinking: Bool = false, profile: any ModelProfile = GenericProfile(), endsText: Bool = true,
        tokenizer: (any MLXLMCommon.Tokenizer)? = nil
    ) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine()
        await engine.attach(try container(opensThinking: opensThinking, tokenizer: tokenizer ?? JSONPieceTokenizer(endsText: endsText)),
                            profile: profile)
        return engine
    }

    static func container(opensThinking: Bool = false, tokenizer: any MLXLMCommon.Tokenizer = JSONPieceTokenizer()) throws -> ModelContainer {
        ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-json", reasoningConfig: QwenReasoningProtocol.tagged),
            model: try withRandomState(MLXRandom.RandomState(seed: 56)) { try TinyModel.make() },
            processor: JSONPieceInputProcessor(opensThinking: opensThinking), tokenizer: tokenizer))
    }

    static func request(
        format: ResponseFormat?, thinking: Bool = false, budget: Int? = nil, maxTokens: Int = 300,
        tools: [ToolDefinition]? = nil, choice: ToolChoice? = nil
    ) -> GenerationRequest {
        GenerationRequest(prompt: "Report the mode.", maxTokens: maxTokens, temperature: 0, enableThinking: thinking,
                          tools: tools, toolChoice: choice, thinkingBudget: budget, responseFormat: format)
    }

    static func isValid(_ text: String, _ schema: String = schema) throws -> Bool {
        try SchemaCheck.accepts(schema, text) && (try? JSONSerialization.jsonObject(with: Data(text.utf8))) is [String: Any]
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
        #expect(try SchemaCheck.accepts(#"{"type": "object"}"#, result.generatedText), "\(result.generatedText.debugDescription)")
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

    @Test("A strict schema with a keyword XGrammar ignores, or one XGrammar rejects, is a request error; a forced tool with one falls back to unconstrained")
    func unsupportedSchemas() async throws {
        let engine = try await ConstrainedModel.engine()
        let unsupported = #"{"type": "object", "properties": {"n": {"type": "integer", "multipleOf": 2}}}"#
        let error = await #expect(throws: BasaltError.self) {
            _ = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(unsupported, strict: true)))
        }
        guard case .invalidRequestStructure(let details)? = error else { Issue.record("\(String(describing: error))"); return }
        #expect(details == "response_format: unsupported JSON Schema keyword 'multipleOf' at #/properties/n")
        let rejected = await #expect(throws: BasaltError.self) {
            _ = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(#"{"type": 42}"#)))
        }
        guard case .invalidRequestStructure(let reason)? = rejected else { Issue.record("\(String(describing: rejected))"); return }
        #expect(reason.hasPrefix("response_format: "), "\(reason)")
        let tool = ToolDefinition(name: "f", parametersJSON: #"{"$ref": "https://example.com/s.json"}"#)
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

    @Test("A stream the client abandons does not return its grammar to the pool; one that runs to its end does")
    func abandonedStreamIsNotPooled() async throws {
        let config = #"{"hidden_size": 1024, "num_hidden_layers": 12, "intermediate_size": 4096, "num_attention_heads": 8, "num_key_value_heads": 8, "rms_norm_eps": 1e-5, "vocab_size": 256, "rope_theta": 10000.0, "tie_word_embeddings": false}"#
        let model = try withRandomState(MLXRandom.RandomState(seed: 56)) { LlamaModel(try JSONDecoder().decode(LlamaConfiguration.self, from: Data(config.utf8))) }
        model.update(parameters: model.parameters().mapValues { $0 * 4 })
        eval(model)
        let engine = MLXInferenceEngine()
        await engine.attach(ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "slow-json"), model: model,
            processor: JSONPieceInputProcessor(opensThinking: false), tokenizer: JSONPieceTokenizer())))
        let long = #"{"type": "string", "minLength": 300}"#
        let schema = try JSONSchemaGate.prepare(long, strict: false).schema
        let consumer = Task {
            for await _ in try await engine.generateStream(request: ConstrainedModel.request(format: .jsonSchema(long), maxTokens: 400)) { break }
        }
        try await consumer.value
        try await Task.sleep(for: .seconds(1))
        #expect(await engine.idleGrammars[schema, default: []].isEmpty)
        _ = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(long), maxTokens: 400))
        #expect(await engine.idleGrammars[schema]?.count == 1)
    }

    @Test("A model without any stop token cannot end a constrained document, so response_format is a request error")
    func refusesWithoutStopTokens() async throws {
        let engine = try await ConstrainedModel.engine(endsText: false)
        await #expect(throws: BasaltError.self) {
            _ = try await engine.generate(request: ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema)))
        }
    }

    @Test("Grammar jump forward reduces model calls, validates against schema, and records forced tokens")
    func grammarJumpForwardExecution() async throws {
        let engine = try await ConstrainedModel.engine()
        let baseRequest = ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema), maxTokens: 300)
        let plain = try await engine.generate(request: baseRequest)
        #expect(try ConstrainedModel.isValid(plain.generatedText))

        let ffRequest = baseRequest.withGrammarJumpForward(true)
        let fast = try await engine.generate(request: ffRequest)
        #expect(try ConstrainedModel.isValid(fast.generatedText))
        #expect(fast.forcedTokens ?? 0 > 0)
        #expect((fast.modelCalls ?? Int.max) < plain.completionTokens)
    }

    @Test("Grammar jump forward interplay with presence penalty preserves schema validity and records tokens")
    func grammarJumpForwardWithPresencePenalty() async throws {
        let engine = try await ConstrainedModel.engine()
        let baseReq = ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema), thinking: false, maxTokens: 300)
        let req = GenerationRequest(
            prompt: baseReq.prompt, maxTokens: baseReq.maxTokens, temperature: 0,
            enableThinking: false, presencePenalty: 0.5, responseFormat: baseReq.responseFormat
        ).withGrammarJumpForward(true)
        let res = try await engine.generate(request: req)
        #expect(try ConstrainedModel.isValid(res.generatedText), "Text: \(res.generatedText), finish: \(res.finishReason)")
        #expect(res.forcedTokens ?? 0 > 0)
    }

    @Test("When grammarJumpForward is false, execution is bit-identical to default unflagged request")
    func grammarJumpForwardDefaultEquivalence() async throws {
        let engine = try await ConstrainedModel.engine()
        let req1 = ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema), maxTokens: 200)
        let req2 = req1.withGrammarJumpForward(false)
        let res1 = try await engine.generate(request: req1)
        let res2 = try await engine.generate(request: req2)
        #expect(res1.generatedText == res2.generatedText)
        #expect(res1.forcedTokens == nil)
        #expect(res2.forcedTokens == nil)
    }
}

enum SchemaCheck {
    static let bytes = try! GrammarTokenizer(
        vocab: (0..<256).map { String(format: "<0x%02X>", $0) } + ["</s>"], vocabType: .byteFallback, eosTokenId: 256)

    static func accepts(_ schema: String, _ text: String, strict: Bool = false) throws -> Bool {
        let grammar = try GrammarConstraint(tokenizer: bytes, jsonSchema: try JSONSchemaGate.prepare(schema, strict: strict).schema)
        for byte in text.utf8 {
            guard (try? grammar.commitToken(Int32(byte))) != nil else { return false }
        }
        return try grammar.computeMask().mask.allows(256)
    }
}

@Suite("JSON constraint processor phases (#42)")
struct JSONConstraintProcessorTests {
    static func processor(_ phase: JSONConstraintProcessor.Phase, markers: Bool = true) throws -> JSONConstraintProcessor {
        let vocabulary = markers ? JSONPieceTokenizer.markerVocabulary : try GrammarVocabulary(
            pieces: (0..<256).map { JSONPieceTokenizer().convertIdToToken($0)! }, vocabType: .raw,
            stopTokens: [JSONPieceTokenizer.endOfText], markers: ReasoningMarkers())
        let schema = try JSONSchemaGate.prepare(ConstrainedModel.schema, strict: true).schema
        let grammar = try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: schema)
        return JSONConstraintProcessor(session: GrammarSession(schema: schema, grammar: grammar), vocabulary: vocabulary, phase: phase)
    }

    static let document = #"{"mode":"fast","flags":[],"inner":{"on":true,"note":null}}"#

    @Test("Before the answer the opener or the grammar's first bytes are allowed; inside reasoning nothing is masked; after the closer only the grammar is")
    func phases() throws {
        var processor = try Self.processor(.awaitingAnswer)
        let brace = Int(UInt8(ascii: "{"))
        let letter = Int(UInt8(ascii: "q"))
        let allowed = try #require(processor.allowedTokens())
        #expect(allowed.allows(JSONPieceTokenizer.open))
        #expect(allowed.allows(brace))
        #expect(!allowed.allows(letter))

        processor.didSample(token: MLXArray(Int32(JSONPieceTokenizer.open)))
        #expect(processor.phase == .reasoning)
        let logits = MLXArray.zeros([1, 256])
        #expect(processor.process(logits: logits).min().item(Float.self) == 0)

        processor.didSample(token: MLXArray(Int32(letter)))
        processor.didSample(token: MLXArray(Int32(JSONPieceTokenizer.close)))
        #expect(processor.phase == .constrained)
        #expect(processor.allowedTokens()?.allows(JSONPieceTokenizer.open) == false)
        let masked = processor.process(logits: logits)
        #expect(masked[0, brace].item(Float.self) == 0)
        #expect(masked[0, letter].item(Float.self) == -Float.infinity)
    }

    @Test("Once the top-level value closes only the stop token is left")
    func stopsAtTheClosingBrace() throws {
        var processor = try Self.processor(.constrained, markers: false)
        for byte in Array(Self.document.utf8) {
            processor.didSample(token: MLXArray(Int32(byte)))
        }
        let allowed = try #require(processor.allowedTokens())
        #expect((0..<256).filter(allowed.allows) == [JSONPieceTokenizer.endOfText])
    }

    @Test("A closer spelled over several tokens ends reasoning and the grammar starts at the document")
    func closerAcrossTokens() throws {
        var processor = try Self.processor(.reasoning, markers: false)
        for byte in Array("plan</thi".utf8) { processor.didSample(token: MLXArray(Int32(byte))) }
        #expect(processor.phase == .reasoning)
        for byte in Array("nk>".utf8) { processor.didSample(token: MLXArray(Int32(byte))) }
        #expect(processor.phase == .constrained)
        #expect(processor.allowedTokens()?.allows(Int(UInt8(ascii: "{"))) == true)
    }

    @Test("A token the grammar rejects does not release the constraint")
    func neverReleases() throws {
        var processor = try Self.processor(.constrained, markers: false)
        processor.didSample(token: MLXArray(Int32(UInt8(ascii: "q"))))
        #expect(processor.phase == .constrained)
        let logits = MLXArray.zeros([1, 256])
        let masked = processor.process(logits: logits)
        #expect(masked[0, Int(UInt8(ascii: "q"))].item(Float.self) == -Float.infinity)
        #expect(masked[0, Int(UInt8(ascii: "{"))].item(Float.self) == 0)
    }

    @Test("A commit that lands after the session was rewound into the pool leaves the pooled grammar untouched, and the retired session masks nothing")
    func lateCommitAfterRewind() throws {
        let vocabulary = JSONPieceTokenizer.markerVocabulary
        let schema = try JSONSchemaGate.prepare(ConstrainedModel.schema, strict: true).schema
        let grammar = try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: schema)
        let pristine = try grammar.computeMask().mask
        let session = GrammarSession(schema: schema, grammar: grammar)
        for byte in Array(#"{"mode"#.utf8) { session.commit(Int(byte)) }
        let pooled = try #require(session.rewound())
        #expect(try pooled.computeMask().mask == pristine)
        session.commit(Int(UInt8(ascii: "{")))
        #expect(try pooled.computeMask().mask == pristine)
        #expect(session.mask() == nil)
        #expect(session.rewound() == nil)
    }

    @Test("A session whose grammar refused a sampled token is not pooled")
    func refusedCommitIsNotPooled() throws {
        let vocabulary = JSONPieceTokenizer.markerVocabulary
        let schema = try JSONSchemaGate.prepare(ConstrainedModel.schema, strict: true).schema
        let session = GrammarSession(schema: schema, grammar: try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: schema))
        session.commit(Int(UInt8(ascii: "{")))
        session.commit(Int(UInt8(ascii: "q")))
        #expect(session.rewound() == nil)
    }

    @Test("Fast-forward rewound: pooled grammar after fast-forward matches freshly compiled grammar mask")
    func fastForwardRewind() throws {
        let tokenizer = JSONPieceTokenizer()
        let vocabulary = try GrammarVocabulary(
            pieces: (0..<256).map { tokenizer.convertIdToToken($0)! }, vocabType: .raw, stopTokens: [JSONPieceTokenizer.endOfText],
            markers: ReasoningMarkers(openers: [JSONPieceTokenizer.open], closers: [[JSONPieceTokenizer.close]]),
            hostTokenizer: tokenizer)
        let schema = try JSONSchemaGate.prepare(ConstrainedModel.schema, strict: true).schema
        let grammar = try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: schema, fastForward: true, hostTokenizer: tokenizer)
        let pristine = try grammar.computeMask().mask
        let session = GrammarSession(schema: schema, grammar: grammar, poolKey: schema + ":ff")
        session.commit(Int(UInt8(ascii: "{")))
        let forced = session.takeForcedTokens()
        #expect(session.totalForcedTokens == forced.count)
        let pooled = try #require(session.rewound())
        #expect(try pooled.computeMask().mask == pristine)
    }
}

struct ListTokenizer: MLXLMCommon.Tokenizer {
    final class Lookups: Sendable {
        let count = Mutex(0)
    }

    let pieces: [Int: String]
    var eosToken: String?
    var special: Set<Int> = []
    let lookups = Lookups()

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            guard let (id, piece) = pieces.filter({ !$0.value.isEmpty && rest.hasPrefix($0.value) }).max(by: { $0.value.count < $1.value.count })
            else { return ids }
            ids.append(id)
            rest = rest.dropFirst(piece.count)
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.filter { !skipSpecialTokens || !special.contains($0) }.compactMap { pieces[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.first { $0.value == token }?.key }
    func convertIdToToken(_ id: Int) -> String? {
        lookups.count.withLock { $0 += 1 }
        return pieces[id]
    }
    var bosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }

    static func bytes(_ extra: [Int: String]) -> [Int: String] {
        Dictionary(uniqueKeysWithValues: (0..<128).map { ($0, String(UnicodeScalar(UInt8($0)))) }).merging(extra) { $1 }
    }
}

@Suite("Grammar vocabulary from the loaded tokenizer (#42)")
struct GrammarVocabularyTests {
    static func vocabulary(_ tokenizer: ListTokenizer, extraEOSTokens: Set<String> = []) throws -> GrammarVocabulary {
        try GrammarVocabulary(context: ModelContext(
            configuration: ModelConfiguration(id: "list", extraEOSTokens: extraEOSTokens), model: try TinyModel.make(),
            processor: JSONPieceInputProcessor(opensThinking: false), tokenizer: tokenizer))
    }

    static func processor(
        _ vocabulary: GrammarVocabulary, _ schema: String = #"{"type": "string"}"#, phase: JSONConstraintProcessor.Phase = .constrained
    ) throws -> JSONConstraintProcessor {
        let prepared = try JSONSchemaGate.prepare(schema, strict: true).schema
        let grammar = try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: prepared)
        return JSONConstraintProcessor(session: GrammarSession(schema: prepared, grammar: grammar), vocabulary: vocabulary, phase: phase)
    }

    static func allowed(_ processor: JSONConstraintProcessor) throws -> [Int] {
        let mask = try #require(processor.allowedTokens())
        return (0..<processor.vocabulary.size).filter(mask.allows)
    }

    @Test("Byte-level BPE: Ġ is a space, Ċ a raw newline that a string refuses, Ã© the bytes of é; the stop token only ends the document")
    func byteLevel() throws {
        let vocabulary = try Self.vocabulary(ListTokenizer(
            pieces: [0: "{", 1: "Ġ\"", 2: "Ċ", 3: "Ã©", 4: "\"", 5: "}", 6: "<|im_end|>", 7: "a"], eosToken: "<|im_end|>"))
        var processor = try Self.processor(vocabulary)
        processor.didSample(token: MLXArray(Int32(4)))
        let inside = try Self.allowed(processor)
        #expect(inside == [0, 1, 3, 4, 5, 7], "\(inside)")
        processor.didSample(token: MLXArray(Int32(1)))
        #expect(try Self.allowed(processor) == [6])
    }

    @Test("SentencePiece: ▁ is a space, <0x0A> a raw newline, <0x22> a quote that closes the string")
    func sentencePiece() throws {
        let vocabulary = try Self.vocabulary(ListTokenizer(
            pieces: [0: "<0x0A>", 1: "▁a", 2: "é", 3: "\"", 4: "</s>", 5: "<0x22>"], eosToken: "</s>"))
        var processor = try Self.processor(vocabulary)
        processor.didSample(token: MLXArray(Int32(3)))
        #expect(try Self.allowed(processor) == [1, 2, 3, 5])
        processor.didSample(token: MLXArray(Int32(1)))
        processor.didSample(token: MLXArray(Int32(5)))
        #expect(try Self.allowed(processor) == [4])
    }

    @Test("Stop tokens and reasoning markers past a gap in the ids are kept, never fit inside a value, and every stop token ends the document")
    func reservedTokensPastAGap() throws {
        let tokenizer = ListTokenizer(
            pieces: ListTokenizer.bytes([250: "<think>", 251: "</think>", 252: "<eos>", 253: "<|im_end|>"]), eosToken: "<eos>")
        let vocabulary = try Self.vocabulary(tokenizer, extraEOSTokens: ["<|im_end|>"])
        #expect(vocabulary.size == 254)
        #expect(vocabulary.stopTokens == [252, 253])
        var processor = try Self.processor(vocabulary)
        processor.didSample(token: MLXArray(Int32(UInt8(ascii: "\""))))
        let inside = try Self.allowed(processor)
        #expect(inside.contains(Int(UInt8(ascii: "a"))))
        #expect(inside.filter { $0 >= 128 }.isEmpty, "\(inside.filter { $0 >= 128 })")
        processor.didSample(token: MLXArray(Int32(UInt8(ascii: "\""))))
        #expect(try Self.allowed(processor) == [252, 253])
    }

    @Test("A stop token beyond the last piece still ends a closed document: logits wider or narrower than the vocabulary are masked to it")
    func closedDocumentAllowsOnlyStops() throws {
        let vocabulary = try GrammarVocabulary(
            pieces: (0..<128).map { String(UnicodeScalar(UInt8($0))) }, vocabType: .raw, stopTokens: [200], markers: ReasoningMarkers())
        #expect(vocabulary.size == 201)
        var processor = try Self.processor(vocabulary)
        for byte in Array(#""ok""#.utf8) { processor.didSample(token: MLXArray(Int32(byte))) }
        let wide = processor.process(logits: MLXArray.zeros([1, 300]))
        #expect(wide.dim(-1) == 300)
        #expect((0..<300).filter { wide[0, $0].item(Float.self) == 0 } == [200])
        let narrow = processor.process(logits: MLXArray.zeros([1, 150]))
        #expect(narrow.dim(-1) == 150)
        #expect(narrow.max().item(Float.self) == -Float.infinity)
    }

    @Test("The vocabulary is read from the tokenizer once per model, and a model switch drops it and the pooled grammars")
    func readOncePerModel() async throws {
        let tokenizer = ListTokenizer(pieces: JSONPieceTokenizer.pieces.merging([252: "<eos>"]) { $1 }, eosToken: "<eos>")
        let engine = try await ConstrainedModel.engine(tokenizer: tokenizer)
        let request = ConstrainedModel.request(format: .jsonSchema(ConstrainedModel.schema), maxTokens: 120)
        _ = try await engine.generate(request: request)
        let lookups = tokenizer.lookups.count.withLock { $0 }
        _ = try await engine.generate(request: request)
        #expect(tokenizer.lookups.count.withLock { $0 } == lookups)
        #expect(await !engine.idleGrammars.isEmpty)

        await engine.attach(try ConstrainedModel.container(tokenizer: JSONPieceTokenizer(endsText: false)))
        #expect(await engine.idleGrammars.isEmpty)
        await #expect(throws: BasaltError.self) { _ = try await engine.generate(request: request) }
    }

    @Test("An opener spelled as a special token plus text, like Gemma 4's <|channel>thought, opens reasoning on its special token; a plain-text one does not")
    func multiTokenOpener() throws {
        let channel = ListTokenizer(pieces: ListTokenizer.bytes([260: "<|channel>", 261: "<channel|>", 262: "<eos>"]), eosToken: "<eos>", special: [260, 261, 262])
        let markers = ReasoningMarkers(tokenizer: channel)
        #expect(markers.openers == [260])
        #expect(markers.closers.contains([261]))
        let plain = ListTokenizer(pieces: ListTokenizer.bytes([260: "<|channel>", 262: "<eos>"]), eosToken: "<eos>", special: [262])
        #expect(ReasoningMarkers(tokenizer: plain).openers.isEmpty)

        var processor = try Self.processor(try Self.vocabulary(channel), phase: .awaitingAnswer)
        #expect(try Self.allowed(processor).contains(260))
        processor.didSample(token: MLXArray(Int32(260)))
        #expect(processor.phase == .reasoning)
        for byte in Array("thought plan".utf8) { processor.didSample(token: MLXArray(Int32(byte))) }
        processor.didSample(token: MLXArray(Int32(261)))
        #expect(processor.phase == .constrained)
        #expect(try Self.allowed(processor).contains(Int(UInt8(ascii: "\""))))
    }

    @Test("A special token that also closes reasoning, like harmony's <|channel|>, never opens it, so <|channel|>final stays constrained")
    func harmonyChannelIsNotAnOpener() throws {
        let harmony = ListTokenizer(pieces: ListTokenizer.bytes([260: "<|channel|>", 262: "<eos>"]), eosToken: "<eos>", special: [260, 262])
        let markers = ReasoningMarkers(tokenizer: harmony)
        #expect(!markers.openers.contains(260))
        #expect(markers.closers.contains([260]))
    }
}

