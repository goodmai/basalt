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

    static func body(_ json: String) -> Data { Data(json.utf8) }

    @Test("OpenAI response_format accepts regex, grammar and structural_tag, and vLLM's guided_regex and guided_grammar")
    func parses() throws {
        #expect(try OpenAIController.responseFormat(fromRequestBody: Self.body(#"{"response_format": {"type": "regex", "regex": "[a-z]+"}}"#)) == .regex("[a-z]+"))
        #expect(try OpenAIController.responseFormat(fromRequestBody: Self.body(#"{"response_format": {"type": "grammar", "grammar": "root ::= \"a\""}}"#)) == .grammar(#"root ::= "a""#))
        #expect(try OpenAIController.responseFormat(fromRequestBody: Self.body(#"{"guided_regex": "[0-9]{2}"}"#)) == .regex("[0-9]{2}"))
        #expect(try OpenAIController.responseFormat(fromRequestBody: Self.body(#"{"guided_grammar": "root ::= \"b\""}"#)) == .grammar(#"root ::= "b""#))
    }

    @Test("OpenAI response_format structural_tag keeps the whole tag object as the source")
    func parsesStructuralTag() throws {
        let tag = try OpenAIController.responseFormat(fromRequestBody: Self.body(#"{"response_format": \#(Self.structuralTag)}"#))
        guard case .structuralTag(let source)? = tag else {
            Issue.record("expected a structural tag, got \(String(describing: tag))")
            return
        }
        #expect(source.contains("const_string") && source.contains("structural_tag"))
    }

    @Test("A malformed grammar request is refused before generation", arguments: [
        #"{"response_format": {"type": "regex"}}"#,
        #"{"response_format": {"type": "regex", "regex": ""}}"#,
        #"{"response_format": {"type": "grammar", "grammar": 3}}"#,
        #"{"guided_regex": "a", "guided_grammar": "root ::= \"a\""}"#,
        #"{"guided_regex": "a", "response_format": {"type": "json_object"}}"#,
    ])
    func refusesMalformed(json: String) {
        #expect(throws: OpenAIController.InvalidResponseFormat.self) { try OpenAIController.responseFormat(fromRequestBody: Self.body(json)) }
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
