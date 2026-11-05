import Foundation
import Testing
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("Generation relay (#284)")
struct GenerationRelayTests {
    typealias Setup = GenerationRelay.Setup

    struct Run {
        var relay: GenerationRelay
        var transcript: [String]
        var steps: Int
        var calls: [ToolCallOutput]

        var metadata: GenerationResponse {
            relay.response(
                generationTime: 1, timeToFirstToken: 0.5, memory: GenerationResponse.MemoryUsage(peakBytes: 0, activeBytes: 0, cacheBytes: 0),
                reasoningTokens: nil, forcedTokens: nil, modelModified: false)
        }
    }

    static let tools = [ToolDefinition(name: "shell"), ToolDefinition(name: "read")]

    static func run(_ setup: Setup, _ script: [Generation]) -> Run {
        var relay = GenerationRelay(setup)
        var chunks: [StreamChunk] = []
        var steps = 0
        for generation in script {
            steps += 1
            let step = relay.step(generation)
            chunks += step.chunks
            if step.stops { break }
        }
        chunks += relay.finish()
        let calls = chunks.compactMap { chunk -> ToolCallOutput? in
            guard case .toolCall(let call) = chunk else { return nil }
            return call
        }
        return Run(relay: relay, transcript: transcript(chunks), steps: steps, calls: calls)
    }

    static func transcript(_ chunks: [StreamChunk]) -> [String] {
        chunks.reduce(into: [String]()) { lines, chunk in
            let (kind, body): (String, String) = switch chunk {
            case .text(let text): ("text", text)
            case .reasoning(let text): ("reasoning", text)
            case .toolCall(let call): ("call", call.name)
            case .metadata: ("metadata", "")
            }
            if kind == "text" || kind == "reasoning", let last = lines.last, last.hasPrefix(kind + ": ") {
                lines[lines.count - 1] += body
            } else {
                lines.append("\(kind): \(body)")
            }
        }
    }

    static func call(_ name: String) -> Generation {
        let arguments: [String: any Sendable] = ["command": "ls"]
        return .toolCall(ToolCall(function: ToolCall.Function(name: name, arguments: arguments)))
    }

    @Test("Answer chunks pass through as text and a stream without completion info counts its own steps")
    func plainAnswer() {
        let run = Self.run(Setup(request: GenerationRequest(prompt: "p"), promptTokens: 10, reusedTokens: 4), [.chunk("Hello"), .chunk(", world")])
        #expect(run.transcript == ["text: Hello, world"])
        #expect(run.metadata.finishReason == .stop)
        #expect(run.metadata.completionTokens == 2)
        #expect(run.metadata.promptTokens == 10)
        #expect(run.metadata.cachedPromptTokens == 4)
        #expect(run.metadata.degenerate == nil)
    }

    @Test("Completion info supplies the counts, and reaching max_tokens finishes with length")
    func completionInfo() {
        let info = GenerateCompletionInfo(promptTokenCount: 6, generationTokenCount: 3, promptTime: 0.1, generationTime: 0.5)
        let run = Self.run(Setup(request: GenerationRequest(prompt: "p", maxTokens: 3), promptTokens: 99, reusedTokens: 4), [.chunk("abc"), .info(info)])
        #expect(run.metadata.promptTokens == 10)
        #expect(run.metadata.completionTokens == 3)
        #expect(run.metadata.finishReason == .length)
    }

    @Test("Thinking markers split reasoning from the answer and the reasoning text is kept for counting")
    func thinking() {
        let run = Self.run(Setup(request: GenerationRequest(prompt: "p"), startsThinking: true), [.chunk("plan"), .chunk("</think>"), .chunk("\nanswer")])
        #expect(run.transcript == ["reasoning: plan", "text: answer"])
        #expect(run.relay.reasoningText == "plan")
        #expect(run.metadata.degenerate == nil)
    }

    @Test("Reasoning that never reaches an answer reports empty-after-thinking")
    func emptyAfterThinking() {
        let run = Self.run(Setup(request: GenerationRequest(prompt: "p"), startsThinking: true), [.chunk("plan")])
        #expect(run.transcript == ["reasoning: plan"])
        #expect(run.metadata.degenerate == .emptyAfterThinking)
    }

    @Test("Reasoning cut off at max_tokens finishes with length and is not reported as degenerate")
    func thinkingCutOffAtMaxTokens() {
        let info = GenerateCompletionInfo(promptTokenCount: 6, generationTokenCount: 2, promptTime: 0.1, generationTime: 0.5)
        let run = Self.run(Setup(request: GenerationRequest(prompt: "p", maxTokens: 2), startsThinking: true), [.chunk("plan"), .chunk(" more"), .info(info)])
        #expect(run.metadata.finishReason == .length)
        #expect(run.metadata.degenerate == nil)
    }

    @Test("A tool call flushes the held text first, and with parallel calls off the stream stops after it")
    func singleToolCallStops() {
        let request = GenerationRequest(prompt: "p", tools: Self.tools).with { $0.toolChoice = ToolChoice(parallel: false) }
        let run = Self.run(Setup(request: request), [.chunk("Let me look."), Self.call("shell"), .chunk("never")])
        #expect(run.steps == 2)
        #expect(run.transcript == ["text: Let me look.", "call: shell"])
        #expect(run.calls.first?.arguments == #"{"command":"ls"}"#)
    }

    @Test("A call to an undeclared tool becomes rejection text and the stream goes on")
    func undeclaredToolCall() {
        let request = GenerationRequest(prompt: "p", tools: Self.tools)
        let run = Self.run(Setup(request: request), [Self.call("rm"), .chunk("ok")])
        #expect(run.steps == 2)
        #expect(run.transcript == ["text: \n[tool call to rm rejected: undeclared]ok"])
    }

    @Test("A malformed rejected call is repaired and counted; any other rejection is text and counted")
    func rejectedToolCalls() {
        let malformed = RejectedToolCall(
            reason: .malformedSyntax, format: .json,
            rawText: "<tool_call>{'name': 'shell', 'arguments': {'command': ['ls', '-la'], 'timeout_ms': 1000,}}</tool_call>")
        let undeclared = RejectedToolCall(reason: .undeclaredTool, format: .json, toolName: "rm", rawText: "<tool_call>rm</tool_call>")
        let request = GenerationRequest(prompt: "p", tools: Self.tools)
        let run = Self.run(Setup(request: request), [.rejectedToolCall(malformed), .rejectedToolCall(undeclared)])
        #expect(run.transcript == ["call: shell", "text: \n[tool call to rm rejected: undeclared_tool]"])
        #expect(run.calls.first?.arguments == #"{"command":["ls","-la"],"timeout_ms":1000}"#)
        #expect(run.metadata.toolCallRepairs == 1)
        #expect(run.metadata.toolCallRejections == 1)
    }

    @Test("Tool calls written as text are scanned out of the answer")
    func scannedToolCall() {
        let request = GenerationRequest(prompt: "p", tools: Self.tools)
        let text = #"Sure.<TOOLCALL>[{"name": "shell", "arguments": {"command": "ls"}}]</TOOLCALL>"#
        let run = Self.run(Setup(request: request, scansToolCallText: true), [.chunk(text)])
        #expect(run.transcript == ["text: Sure.", "call: shell"])
    }

    @Test("A verbatim loop stops the stream with finish reason repetition, unless a constraint turns the guard off")
    func repetition() {
        let line = Generation.chunk("the same sentence over and over again, forever and ever and ever.\n")
        let script = Array(repeating: line, count: 20)
        let guarded = Self.run(Setup(request: GenerationRequest(prompt: "p")), script)
        #expect(guarded.steps < script.count)
        #expect(guarded.metadata.finishReason == .repetition)
        let constrained = Self.run(Setup(request: GenerationRequest(prompt: "p"), guardsRepetition: false), script)
        #expect(constrained.steps == script.count)
        #expect(constrained.metadata.finishReason == .stop)
    }

    @Test("A tripped non-finite alarm stops at the next chunk and finishes with error")
    func alarm() {
        let alarm = NonFiniteLogitAlarm()
        alarm.trip()
        let run = Self.run(Setup(request: GenerationRequest(prompt: "p"), alarm: alarm), [.chunk("a"), .chunk("b")])
        #expect(run.steps == 1)
        #expect(run.metadata.finishReason == .error)
    }

    @Test("A forced tool collects the answer after the prefill into one call when it parses as an object")
    func forcedToolCall() {
        let request = GenerationRequest(prompt: "p", tools: Self.tools).with { $0.toolChoice = ToolChoice(forcedTool: "read") }
        let run = Self.run(
            Setup(request: request, forcedTool: "read", prefixAnswer: #"{"path":"#, guardsRepetition: false), [.chunk(#" "a.txt"}"#)])
        #expect(run.transcript == ["call: read"])
        #expect(run.calls.first?.arguments == #"{"path": "a.txt"}"#)
    }

    @Test("A forced tool whose arguments do not parse returns the generated continuation as text")
    func forcedToolFallsBackToText() {
        let request = GenerationRequest(prompt: "p", tools: Self.tools).with { $0.toolChoice = ToolChoice(forcedTool: "read") }
        let run = Self.run(
            Setup(request: request, forcedTool: "read", prefixAnswer: #"{"path":"#, guardsRepetition: false), [.chunk(#" "a.txt""#)])
        #expect(run.transcript == [#"text: "a.txt""#])
    }
}
