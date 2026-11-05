import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Regex, EBNF and structural-tag formats on real models (#346)", .serialized)
struct GrammarFormatRealModelTests {
    static var models: [String] {
        let env = ProcessInfo.processInfo.environment
        guard env["BASALT_E2E"] == "1" else { return [] }
        return (env["BASALT_E2E_GRAMMAR_MODELS"] ?? "").split(separator: ",").map(String.init)
    }

    static let diffGrammar = #"""
        root ::= "--- a/" path "\n+++ b/" path "\n@@ -" num "," num " +" num "," num " @@\n" line line? line? line? line?
        path ::= [a-zA-Z0-9_./-]+
        num ::= [0-9]+
        line ::= [ +-] [^\n]* "\n"
        """#

    static let tag = #"""
        {"type": "structural_tag", "format": {"type": "sequence", "elements": [
            {"type": "const_string", "value": "ANSWER: "}, {"type": "regex", "pattern": "(yes|no)"}]}}
        """#

    static let cases: [(ResponseFormat, String, [String])] = [
        (.regex("[A-Z]{3}-[0-9]{4}"), "[A-Z]{3}-[0-9]{4}", [
            "Give a ticket ID in the form ABC-1234 for a broken printer.",
            "Invent an order number like XYZ-0042.",
            "Reply with a flight code such as LHR-2210.",
            "Make up a part number in the format AAA-9999.",
            "Write a booking reference, three capital letters, a dash and four digits.",
        ]),
        (.grammar(diffGrammar), #"--- a/[a-zA-Z0-9_./-]+\n\+\+\+ b/[a-zA-Z0-9_./-]+\n@@ -[0-9]+,[0-9]+ \+[0-9]+,[0-9]+ @@\n(?:[ +-][^\n]*\n){1,5}"#, [
            "Write a unified diff that renames the variable x to count in main.swift.",
            "Show a patch that fixes a typo 'recieve' to 'receive' in README.md.",
            "Produce a diff adding a newline at the end of config.yaml.",
            "Give a unified diff that changes the port 8080 to 9090 in server.py.",
            "Write a diff that removes a debug print from app.js.",
        ]),
        (.structuralTag(tag), "ANSWER: (yes|no)", [
            "Is water wet?",
            "Is 17 a prime number?",
            "Can penguins fly?",
            "Is Paris the capital of France?",
            "Is the sun a planet?",
        ]),
    ]

    @Test("Every answer matches the regex, the unified-diff EBNF grammar and the structural tag it was constrained to",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_GRAMMAR_MODELS"), arguments: models)
    func matches(entry: String) async throws {
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: entry)
        var mismatches: [String] = []
        for (format, pattern, prompts) in Self.cases {
            let expected = try Regex(pattern)
            for prompt in prompts {
                var request = GenerationRequest(prompt: prompt, maxTokens: 256, temperature: 0)
                request.responseFormat = format
                request.enableThinking = false
                request.thinkingBudget = 96
                let response = try await engine.generate(request: request)
                if response.generatedText.wholeMatch(of: expected) == nil { mismatches.append("\(format.kind) '\(prompt)': \(response.generatedText)") }
            }
        }
        print("[#346] \(entry): \(15 - mismatches.count)/15 matched")
        #expect(mismatches.isEmpty, "\(mismatches)")
    }

    static func constrained(_ format: ResponseFormat?, _ prompt: String) -> GenerationRequest {
        var request = GenerationRequest(prompt: prompt, maxTokens: 256, temperature: 0)
        request.responseFormat = format
        request.enableThinking = false
        request.thinkingBudget = 96
        return request
    }

    @Test("With --batch-slots 4, three constrained rows and a free one decode together, and every constrained answer matches its own grammar",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_GRAMMAR_MODELS"), arguments: models)
    func batched(entry: String) async throws {
        let engine = MLXInferenceEngine(batchSlots: 4)
        try await engine.load(modelPath: entry)
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker(), gate: GenerationGate(maxConcurrent: 4))
        let rows: [(ResponseFormat?, String?, String)] = Self.cases.map { ($0.0, $0.1, $0.2[0]) } + [(nil, nil, "Say hello in one short sentence.")]
        let texts = try await withThrowingTaskGroup(of: (Int, String).self) { group in
            for (index, row) in rows.enumerated() {
                group.addTask { (index, try await orchestrator.generate(request: Self.constrained(row.0, row.2)).generatedText) }
            }
            var replies: [Int: String] = [:]
            for try await (index, text) in group { replies[index] = text }
            return rows.indices.compactMap { replies[$0] }
        }
        let mismatches = try zip(rows, texts).filter { row, text in try row.1.map { text.wholeMatch(of: try Regex($0)) == nil } ?? false }
        let batchedRows = await engine.batchedRows
        print("[#346] batched \(entry): \(rows.count - mismatches.count)/\(rows.count) matched, batched rows \(batchedRows)")
        #expect(mismatches.isEmpty, "\(mismatches.map { "\($0.0.0?.kind ?? "free"): \($0.1)" })")
    }

    @Test("A JSON-schema request with an assistant prefill answers the same with --grammar-jump-forward as without it (#205)",
          .enabled(if: !models.isEmpty, "set BASALT_E2E=1 and BASALT_E2E_GRAMMAR_MODELS"), arguments: models)
    func prefillUnderJumpForward(entry: String) async throws {
        let engine = MLXInferenceEngine()
        try await engine.load(modelPath: entry)
        let schema = #"{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}"#
        let messages = [ChatMessageInput(role: .user, text: "Invent a person and give their name and age as JSON."),
                        ChatMessageInput(role: .assistant, text: #"{"name":"#)]
        var request = GenerationRequest(prompt: "", maxTokens: 64, temperature: 0, messages: messages)
        request.responseFormat = .jsonSchema(schema)
        request.enableThinking = false
        let off = try await engine.generate(request: request.withGrammarJumpForward(false))
        let on = try await engine.generate(request: request.withGrammarJumpForward(true))
        print("[#205] \(entry): off \(off.generatedText) | on \(on.generatedText)")
        #expect(on.generatedText == off.generatedText)
    }
}
