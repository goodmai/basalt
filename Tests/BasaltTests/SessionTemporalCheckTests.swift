import Testing
import Foundation
@testable import BasaltCore

@Suite("Past-time LTL monitor over session tool calls (#359)")
struct SessionTemporalCheckTests {
    struct OperatorCase: Sendable, CustomTestStringConvertible {
        let name: String
        let formula: PastFormula
        let trace: [UInt64]
        let expected: [Bool]
        var testDescription: String { name }
    }

    static let a = PastFormula.atom(0)
    static let b = PastFormula.atom(1)

    static let operatorCases = [
        OperatorCase(name: "not / or", formula: .or(a, .not(b)), trace: [0, 3, 2], expected: [true, true, false]),
        OperatorCase(name: "and", formula: .and(a, b), trace: [1, 3, 2], expected: [false, true, false]),
        OperatorCase(name: "Y is false at the first step", formula: .yesterday(a), trace: [1, 0, 1, 1], expected: [false, true, false, true]),
        OperatorCase(name: "O latches", formula: .once(a), trace: [0, 1, 0], expected: [false, true, true]),
        OperatorCase(name: "H holds until the first miss", formula: .historically(a), trace: [1, 1, 0, 1], expected: [true, true, false, false]),
        OperatorCase(name: "a S b", formula: .since(a, b), trace: [1, 2, 1, 1, 0, 2], expected: [false, true, true, true, false, true]),
    ]

    @Test("Each operator follows its recurrence on a hand-written trace", arguments: operatorCases)
    func operators(_ testCase: OperatorCase) {
        var monitor = PastMonitor(testCase.formula)
        #expect(testCase.trace.map { monitor.step($0) } == testCase.expected)
    }

    static func call(_ name: String, _ arguments: String) -> ChatMessageInput {
        ChatMessageInput(role: .assistant, text: "", toolCalls: [ToolCallOutput(id: UUID().uuidString, name: name, arguments: arguments)])
    }

    static func result(_ text: String) -> ChatMessageInput {
        ChatMessageInput(role: .tool, text: text, toolCallId: "id", toolName: "Read")
    }

    static func fired(_ messages: [ChatMessageInput]) -> Set<String> {
        Set(SessionTemporalCheck.check(messages).map { "\($0.property.rawValue)@\($0.messageIndex)" })
    }

    @Test("P1 fires on the third identical call, with argument key order ignored")
    func identicalTriple() {
        let user = ChatMessageInput(role: .user, text: "go")
        let twice = [user, Self.call("Read", #"{"a":1,"b":2}"#), Self.result("ok"), Self.call("Read", #"{"b":2,"a":1}"#), Self.result("ok")]
        #expect(Self.fired(twice).isEmpty)
        #expect(Self.fired(twice + [Self.call("Read", #"{"a": 1, "b": 2}"#)]) == ["P1@5"])
        #expect(Self.fired(twice + [Self.call("Read", #"{"a":1,"b":3}"#)]).isEmpty)
    }

    @Test("P2 fires on a call repeated unchanged after its result was an error")
    func retryAfterError() {
        #expect(Self.fired([Self.call("Bash", "{}"), Self.result("Error: no such file"), Self.call("Bash", "{}")]) == ["P2@2"])
        #expect(Self.fired([Self.call("Bash", "{}"), Self.result("<tool_use_error>x</tool_use_error>"), Self.call("Bash", "{}")]) == ["P2@2"])
        #expect(Self.fired([Self.call("Bash", "{}"), Self.result("fine"), Self.call("Bash", "{}")]).isEmpty)
        #expect(Self.fired([Self.call("Bash", "{}"), Self.result("Error: x"), Self.call("Bash", #"{"x":1}"#)]).isEmpty)
    }

    @Test("P3 fires on the first retry of a tool the gate rejected")
    func retryAfterRejection() {
        let rejected = ChatMessageInput(role: .assistant, text: "\n[tool call to Bash rejected: undeclared]")
        #expect(Self.fired([rejected, Self.call("Bash", "{}"), Self.result("ok"), Self.call("Bash", #"{"n":2}"#)]) == ["P3@1"])
        #expect(Self.fired([rejected, Self.call("Read", "{}")]).isEmpty)
    }

    @Test("P4 fires after eight tool rounds with no assistant text, and text resets it")
    func silentRounds() {
        let rounds = (0..<SessionTemporalCheck.silentRoundLimit).flatMap { [Self.call("Read", #"{"n":\#($0)}"#), Self.result("ok")] }
        #expect(Self.fired(rounds) == ["P4@14"])
        #expect(Self.fired(Array(rounds.dropLast(2))).isEmpty)
        let spoken = Array(rounds.prefix(8)) + [ChatMessageInput(role: .assistant, text: "progress")] + rounds.suffix(8)
        #expect(Self.fired(spoken).isEmpty)
    }

    @Test("Feeding a history in two parts gives the verdicts of feeding it whole")
    func incrementalMatchesWhole() {
        let history = [Self.call("Bash", "{}"), Self.result("Error: x"), Self.call("Bash", "{}"), Self.result("Error: x"), Self.call("Bash", "{}")]
        var monitor = SessionTemporalCheck()
        let first = history.prefix(2).enumerated().flatMap { monitor.feed($1, at: $0) }
        let rest = history.enumerated().dropFirst(2).flatMap { monitor.feed($1, at: $0) }
        #expect(Set(first + rest) == SessionTemporalCheck.check(history))
    }

    @Test("The reader rebuilds a fixture log through the parent chain, adds the last response's calls and counts malformed lines")
    func fixtureLog() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("ptltl-\(UUID().uuidString).jsonl")
        defer { try? FileManager.default.removeItem(at: file) }
        let call = #"{"id":"c","name":"Read","arguments":"{\"path\":\"a\"}"}"#
        let log = """
        {"v":1,"type":"session_start","ts":"t","session":"s","mode":"full"}
        {"v":1,"type":"request","ts":"t","session":"s","id":"r1","model_requested":"gemma","content":{"messages":[{"role":"user","text":"go"},{"role":"assistant","text":"","toolCalls":[\(call)]},{"role":"tool","text":"ok","toolCallId":"c","toolName":"Read"}]}}
        {"v":1,"type":"response","ts":"t","session":"s","request_id":"r1","status":"ok","output":{"chars":0,"sha256":"x","tool_calls":[\(call)]}}
        not json
        {"v":1,"type":"request","ts":"t","session":"s","id":"r2","content":{"parent":"r1","prefix_count":3,"messages":[{"role":"assistant","text":"","toolCalls":[\(call)]},{"role":"tool","text":"Error: busy","toolCallId":"c","toolName":"Read"}]}}
        {"v":1,"type":"response","ts":"t","session":"s","request_id":"r2","status":"ok","output":{"chars":0,"sha256":"x","tool_calls":[\(call)]}}
        """
        try log.write(to: file, atomically: true, encoding: .utf8)
        let read = SessionLogReader.read(file)
        #expect(read.malformed == 1)
        #expect(read.lines.compactMap(\.modelRequested) == ["gemma"])
        let violations = SessionTemporalCheck.check(read.lines)
        #expect(Set(violations.map { "\($0.property.rawValue)@\($0.messageIndex)(\($0.tool))" }) == ["P1@5(Read)", "P2@5(Read)"])
    }
}
