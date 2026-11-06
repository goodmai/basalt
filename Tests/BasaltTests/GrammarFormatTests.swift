import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Regex, EBNF and structural-tag response formats (#346)", .serialized)
struct GrammarFormatTests {
    static let structuralTag = #"""
        {"type": "structural_tag", "format": {"type": "sequence", "elements": [
            {"type": "const_string", "value": "ID: "}, {"type": "regex", "pattern": "[0-9]{3}"}]}}
        """#

    static func format(_ body: [String: Any]) throws -> ResponseFormat? {
        try OpenAIController.responseFormat(in: body, data: JSONSerialization.data(withJSONObject: body))
    }

    static let grammarA = "root ::= \"a\""

    @Test("OpenAI response_format accepts regex and grammar, and vLLM's guided_regex and guided_grammar")
    func parses() throws {
        #expect(try Self.format(["response_format": ["type": "regex", "regex": "[a-z]+"]]) == .regex("[a-z]+"))
        #expect(try Self.format(["response_format": ["type": "grammar", "grammar": Self.grammarA]]) == .grammar(Self.grammarA))
        #expect(try Self.format(["guided_regex": "[0-9]{2}"]) == .regex("[0-9]{2}"))
        #expect(try Self.format(["guided_grammar": Self.grammarA]) == .grammar(Self.grammarA))
    }

    @Test("OpenAI response_format structural_tag keeps the whole tag object as the source")
    func parsesStructuralTag() throws {
        let object = try #require(JSONSerialization.jsonObject(with: Data(Self.structuralTag.utf8)) as? [String: Any])
        guard case .structuralTag(let source)? = try Self.format(["response_format": object]) else {
            Issue.record("expected a structural tag")
            return
        }
        #expect(source.contains("const_string") && source.contains("structural_tag"))
    }

    static var malformed: [[String: Any]] { [
        ["response_format": ["type": "regex"]],
        ["response_format": ["type": "regex", "regex": ""]],
        ["response_format": ["type": "grammar", "grammar": 3]],
        ["guided_regex": "a", "guided_grammar": grammarA],
        ["guided_regex": "a", "response_format": ["type": "json_object"]],
    ] }

    @Test("A malformed grammar request is refused before generation", arguments: 0..<5)
    func refusesMalformed(index: Int) {
        #expect(throws: OpenAIController.InvalidResponseFormat.self) { try Self.format(Self.malformed[index]) }
    }

    @Test("MCP generate takes regex or grammar, not both")
    func mcpArguments() throws {
        #expect(try MCPArguments.generation(["prompt": "p", "regex": "[a-z]+"]).responseFormat == .regex("[a-z]+"))
        #expect(try MCPArguments.generation(["prompt": "p", "grammar": #"root ::= "a""#]).responseFormat == .grammar(#"root ::= "a""#))
        #expect(try MCPArguments.generation(["prompt": "p"]).responseFormat == nil)
        #expect(throws: MCPFailure.self) { try MCPArguments.generation(["prompt": "p", "regex": "a", "grammar": #"root ::= "a""#]) }
    }

    @Test("Each format has its own pool key prefix, and an oversized or empty source is refused")
    func sources() throws {
        #expect(Set([ResponseFormat.regex("a").kind, ResponseFormat.grammar("a").kind, ResponseFormat.structuralTag("a").kind, ResponseFormat.jsonObject.kind]).count == 4)
        #expect(try ResponseFormat.regex("[a-z]").grammarSource() == "[a-z]")
        #expect(throws: UnsupportedJSONSchema.self) { try ResponseFormat.regex("").grammarSource() }
        #expect(throws: UnsupportedJSONSchema.self) { try ResponseFormat.grammar(String(repeating: "a", count: ResponseFormat.maxGrammarBytes + 1)).grammarSource() }
    }

    @Test("A random model's answer matches the regex, the EBNF grammar and the structural tag it was constrained to", arguments: [
        (ResponseFormat.regex("[A-Z]{3}-[0-9]{4}"), "[A-Z]{3}-[0-9]{4}"),
        (ResponseFormat.grammar(#"root ::= ("yes" | "no") " because " [a-z]{1,8}"#), "(yes|no) because [a-z]{1,8}"),
        (ResponseFormat.structuralTag(structuralTag), "ID: [0-9]{3}"),
    ])
    func constrains(format: ResponseFormat, pattern: String) async throws {
        let engine = try await ConstrainedModel.engine()
        let response = try await engine.generate(request: ConstrainedModel.request(format: format, maxTokens: 60))
        #expect(response.generatedText.wholeMatch(of: try Regex(pattern)) != nil, "\(format.kind): \(response.generatedText)")
        #expect(response.finishReason == .stop)
    }

    @Test("With thinking off on a template that still opens a thinking block, the constrained answer is the answer, not reasoning", arguments: [
        ResponseFormat.regex("[A-Z]{3}-[0-9]{4}"), ResponseFormat.jsonObject,
    ])
    func constrainedAnswerIsNotReasoning(format: ResponseFormat) async throws {
        let engine = try await ConstrainedModel.engine(opensThinking: true)
        let response = try await engine.generate(request: ConstrainedModel.request(format: format, thinking: false, maxTokens: 60))
        #expect(!response.generatedText.isEmpty, "\(format.kind): the answer landed in reasoning: \(response.reasoningText ?? "")")
        #expect(response.reasoningText?.isEmpty ?? true)
    }

    @Test("A regex or grammar XGrammar rejects is a request error that names the format", arguments: [
        ResponseFormat.regex("[a-"), ResponseFormat.grammar("root ::= ("), ResponseFormat.structuralTag(#"{"format": 1}"#),
    ])
    func refusesUncompilable(format: ResponseFormat) async throws {
        let engine = try await ConstrainedModel.engine()
        let error = await #expect(throws: BasaltError.self) { try await engine.generate(request: ConstrainedModel.request(format: format)) }
        guard case .invalidRequestStructure(let details)? = error else {
            Issue.record("unexpected \(String(describing: error))")
            return
        }
        #expect(details.contains(format.kind), "\(details)")
    }
}

@Suite("Constrained requests in the batched decoder (#346)", .serialized)
struct BatchedGrammarTests {
    static let formats: [ResponseFormat?] = [
        .regex("[A-Z]{3}-[0-9]{4}"), .jsonSchema(ConstrainedModel.schema), nil, .grammar(#"root ::= ("yes" | "no") " because " [a-z]{1,8}"#),
    ]

    static func engine(slots: Int) async throws -> MLXInferenceEngine {
        let engine = MLXInferenceEngine(EngineConfiguration(batchSlots: slots))
        await engine.attach(try ConstrainedModel.container())
        return engine
    }

    @Test("Two constrained, one free and one EBNF request decode in one batch, and each row gets exactly its solo output", .timeLimit(.minutes(5)))
    func batchedMatchesSolo() async throws {
        let solo = try await Self.engine(slots: 1)
        var expected: [String] = []
        for format in Self.formats { expected.append(try await solo.generate(request: ConstrainedModel.request(format: format, maxTokens: 60)).generatedText) }
        let batched = try await Self.engine(slots: 4)
        let orchestrator = ModelOrchestratorActor(engine: batched, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 4))
        let texts = try await withThrowingTaskGroup(of: (Int, String).self) { group in
            for (index, format) in Self.formats.enumerated() {
                group.addTask { (index, try await orchestrator.generate(request: ConstrainedModel.request(format: format, maxTokens: 60)).generatedText) }
            }
            var replies: [Int: String] = [:]
            for try await (index, text) in group { replies[index] = text }
            return Self.formats.indices.compactMap { replies[$0] }
        }
        #expect(texts == expected)
        #expect(await batched.batchedRows == Self.formats.count)
    }
}
