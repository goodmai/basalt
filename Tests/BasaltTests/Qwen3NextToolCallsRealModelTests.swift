import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Qwen3-Coder-Next with streamed experts on a real checkpoint: every scripted Read, Edit and Bash turn parses to the requested tool call with schema-valid arguments and a byte-exact multi-line string (#642 P4)", .serialized)
struct Qwen3NextToolCallsRealModelTests {
    struct Script: Sendable {
        let tool: Qwen3NextTools.Tool
        let ask: String
        let arguments: [String: JSONValue]
    }

    struct Outcome: Sendable {
        var called = false
        var violations: [String] = []
        var exact: Bool?
        var valuesEqual = false
        var mostlyCached = false
        var failed = false

        var passed: Bool { called && violations.isEmpty && exact != false }
    }

    static var env: [String: String] { ExpertStreamingRealModelTests.env }

    static var turns: Int { Int(env["BASALT_E2E_QWEN3NEXT_TOOL_TURNS"] ?? "") ?? 50 }

    static var tokens: Int { Int(env["BASALT_E2E_QWEN3NEXT_TOOL_TOKENS"] ?? "") ?? 192 }

    static let system = ChatMessageInput(
        role: .system, text: "You are a coding agent working in a Swift repository. Answer every request with exactly one tool call and no other text.")

    static func code(_ turn: Int) -> String {
        let indent = turn % 2 == 0 ? "    " : "\t"
        return "\(indent)func budget\(turn)(_ items: [Int]) -> Int {\n\(indent)\(indent)let total = items.reduce(0, +)\n\n"
            + "\(indent)\(indent)return total * \(turn + 2)\n\(indent)}"
    }

    static func script(_ turn: Int) -> Script {
        let file = "Sources/Module\(turn)/File\(turn).swift"
        let flagged = turn % 2 == 0
        switch turn % 3 {
        case 0:
            let (offset, limit) = (10 + 7 * turn, 20 + turn)
            var arguments: [String: JSONValue] = ["file_path": .string(file), "offset": .int(offset)]
            if flagged { arguments["limit"] = .int(limit) }
            return Script(
                tool: Qwen3NextTools.read,
                ask: "Call the Read tool once with file_path \(file) and offset \(offset)" + (flagged ? " and limit \(limit)." : ". Do not pass limit."),
                arguments: arguments)
        case 1:
            let text = code(turn)
            var arguments: [String: JSONValue] = ["file_path": .string(file), "old_string": .string("return total"), "new_string": .string(text)]
            if flagged { arguments["replace_all"] = .bool(true) }
            return Script(
                tool: Qwen3NextTools.edit,
                ask: "Call the Edit tool once with file_path \(file) and old_string equal to: return total\n"
                    + "new_string is exactly the lines between the two ``` lines below. Keep every space, tab and blank line, and leave the ``` lines out:\n"
                    + "```\n\(text)\n```\n" + (flagged ? "Set replace_all to true." : "Do not pass replace_all."),
                arguments: arguments)
        default:
            let (command, timeout) = ("swift test --filter Budget\(turn)Tests", 60000 + 1000 * turn)
            return Script(
                tool: Qwen3NextTools.bash, ask: "Call the Bash tool once with command: \(command)\nSet timeout to \(timeout).",
                arguments: ["command": .string(command), "timeout": .int(timeout)])
        }
    }

    static func history(_ script: Script, turn: Int) -> [ChatMessageInput] {
        let arguments = (try? JSONEncoder().encode(script.arguments)).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
        return [
            ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: "call_\(turn)", name: script.tool.name, arguments: arguments)]),
            ChatMessageInput(role: .tool, text: "ok", toolCallId: "call_\(turn)", toolName: script.tool.name),
        ]
    }

    static func request(_ messages: [ChatMessageInput]) -> GenerationRequest {
        GenerationRequest(prompt: "", maxTokens: tokens, temperature: 0, tools: Qwen3NextTools.all.map(\.definition), messages: messages)
            .withCacheKey("tool-turns")
    }

    static func grade(_ response: GenerationResponse, _ script: Script) -> Outcome {
        var outcome = Outcome()
        outcome.mostlyCached = response.cachedPromptTokens * 2 >= response.promptTokens
        if script.arguments["new_string"] != nil { outcome.exact = false }
        guard let call = response.toolCalls?.first, call.name == script.tool.name else { return outcome }
        outcome.called = true
        guard let arguments = try? JSONDecoder().decode([String: JSONValue].self, from: Data(call.arguments.utf8)) else {
            outcome.violations = ["arguments are not a JSON object"]
            return outcome
        }
        outcome.violations = Qwen3NextTools.violations(arguments, of: script.tool)
        outcome.valuesEqual = arguments == script.arguments
        if let expected = script.arguments["new_string"] { outcome.exact = arguments["new_string"] == expected }
        return outcome
    }

    static func describe(_ outcome: Outcome, _ response: GenerationResponse, _ script: Script) -> String {
        let called = (response.toolCalls ?? []).map(\.name)
        let schema = outcome.violations.isEmpty ? "valid" : "violations \(outcome.violations)"
        let exact = outcome.exact.map { $0 ? "byte-exact" : "differs" } ?? "not asked"
        return [
            "asked \(script.tool.name)", "called \(called)", "parsed \(outcome.called ? "yes" : "no")",
            "repaired \(response.toolCallRepairs)", "rejected \(response.toolCallRejections)", "schema \(schema)",
            "multi-line string \(exact)", "all values \(outcome.valuesEqual ? "equal" : "differ")",
            "prompt \(response.promptTokens)", "from prefix cache \(response.cachedPromptTokens)",
            String(format: "ttft %.2f s, decode %.2f tok/s", response.timeToFirstToken, response.tokensPerSecond),
            "completion \(response.completionTokens)", "finish \(response.finishReason.rawValue)",
        ].joined(separator: ", ")
    }

    static func runTurn(_ engine: MLXInferenceEngine, _ messages: [ChatMessageInput], script: Script, turn: Int, log: ModelRunLog) async -> Outcome {
        do {
            let response = try await Qwen3NextAgentTurnsRealModelTests.turn(engine, request(messages), step: "turn \(turn)", log: log)
            let outcome = grade(response, script)
            log.record(outcome.passed ? "turn \(turn)" : "FAIL turn \(turn)", describe(outcome, response, script))
            if !outcome.passed || !outcome.valuesEqual {
                let got = (response.toolCalls ?? []).map(\.arguments)
                let expected = history(script, turn: turn)[0].toolCalls?.map(\.arguments) ?? []
                log.record("diff turn \(turn)", "text \(response.generatedText.debugDescription) calls \(got) expected \(expected)")
            }
            return outcome
        } catch {
            return Outcome(failed: true)
        }
    }

    static func summary(_ outcomes: [Outcome]) -> String {
        let edits = outcomes.filter { $0.exact != nil }
        let violations = outcomes.reduce(0) { $0 + $1.violations.count }
        return [
            "turns \(outcomes.count)", "parsed to the requested tool \(outcomes.count(where: \.called))",
            "turns with schema violations \(outcomes.count { !$0.violations.isEmpty }) (\(violations) violations)",
            "Edit turns \(edits.count) with the multi-line string byte-exact in \(edits.count { $0.exact == true })",
            "all values equal in \(outcomes.count(where: \.valuesEqual))",
            "served mostly from the prefix cache \(outcomes.count(where: \.mostlyCached))",
            "generation errors \(outcomes.count(where: \.failed))",
        ].joined(separator: ", ")
    }

    @Test("One conversation of scripted tool turns on one engine: 100% of turns parse to a call of the requested tool, no argument violates the tool's schema, and every Edit turn's multi-line new_string equals the requested text byte for byte",
          .enabled(if: Qwen3NextAgentTurnsRealModelTests.modelPath != nil, "set BASALT_E2E=1 and BASALT_E2E_QWEN3NEXT_MODEL_PATH"))
    func scriptedToolTurns() async throws {
        let path = try #require(Qwen3NextAgentTurnsRealModelTests.modelPath)
        let log = try await ModelRunLog("qwen3next-tool-calls", modelPath: path)
        defer { log.releaseGPU() }
        Memory.clearCache()
        let engine = MLXInferenceEngine(EngineConfiguration(expertStreaming: Qwen3NextAgentTurnsRealModelTests.streaming))
        try await log.step("load") { try await engine.load(modelPath: path) }
        var messages = [Self.system]
        var outcomes: [Outcome] = []
        for turn in 0 ..< Self.turns {
            let script = Self.script(turn)
            messages.append(GoldenPrompt.user(script.ask))
            outcomes.append(await Self.runTurn(engine, messages, script: script, turn: turn, log: log))
            messages += Self.history(script, turn: turn)
        }
        await engine.unloadCurrentModel()
        Memory.clearCache()
        log.record(outcomes.allSatisfy(\.passed) ? "gate" : "FAIL gate", Self.summary(outcomes))
        #expect(outcomes.count == Self.turns)
        let uncalled = outcomes.indices.filter { !outcomes[$0].called }
        let violating = outcomes.indices.filter { !outcomes[$0].violations.isEmpty }
        let inexact = outcomes.indices.filter { outcomes[$0].exact == false }
        #expect(uncalled.isEmpty, "turns without a call of the requested tool: \(uncalled)")
        #expect(violating.isEmpty, "turns with schema violations: \(violating)")
        #expect(inexact.isEmpty, "Edit turns whose new_string differs: \(inexact)")
    }
}
