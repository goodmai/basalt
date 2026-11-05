import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltServer

actor ScriptedResearchEngine: InferenceEngine, ModelResearch {
    enum Turn: Sendable {
        case call(String, String)
        case answer(String)
        case fail
        case hang
    }

    private var script: [Turn]
    private let hanging: AsyncStream<Void>.Continuation?
    private let stall = AsyncStream.makeStream(of: Void.self)
    private var active: [OverlayInfo] = []
    private(set) var requests: [GenerationRequest] = []
    private(set) var traces: [TraceRequest] = []
    private(set) var undone: [OverlayID] = []

    init(_ script: [Turn], hanging: AsyncStream<Void>.Continuation? = nil) {
        self.script = script
        self.hanging = hanging
    }

    var isLoaded: Bool { true }

    func load(modelPath: String) async throws(BasaltError) {}

    func generate(request: GenerationRequest) async throws(BasaltError) -> GenerationResponse {
        requests.append(request)
        switch script.isEmpty ? .answer("done") : script.removeFirst() {
        case .call(let name, let arguments):
            return Self.response("", calls: [ToolCallOutput(id: "call-\(requests.count)", name: name, arguments: arguments)])
        case .answer(let text):
            return Self.response(text)
        case .fail:
            throw .inferenceError("scripted failure")
        case .hang:
            hanging?.yield()
            for await _ in stall.stream {}
            throw .cancelled
        }
    }

    func generateStream(request: GenerationRequest) async throws(BasaltError) -> AsyncStream<StreamChunk> {
        throw .inferenceError("the scripted engine does not stream")
    }

    static func response(_ text: String, calls: [ToolCallOutput]? = nil) -> GenerationResponse {
        GenerationResponse(
            generatedText: text, promptTokens: 1, completionTokens: 1, tokensPerSecond: 1, generationTime: 0, timeToFirstToken: 0,
            memory: .init(peakBytes: 0, activeBytes: 0, cacheBytes: 0), finishReason: .stop, toolCalls: calls)
    }

    func anatomy() async throws(BasaltError) -> ModelAnatomy { throw .invalidRequestStructure(details: "unscripted") }
    func tensors(matching filter: String?) async throws(BasaltError) -> [TensorInfo] { [] }
    func tensor(_ request: TensorRequest) async throws(BasaltError) -> TensorReport { throw .invalidRequestStructure(details: "unscripted") }
    func promptCacheReport(namespace: String?) async -> PromptCacheReport { PromptCacheReport(budget: 0, bytes: 0, evictions: 0, namespaces: []) }
    func applyDelta(_ delta: WeightDelta) async throws(BasaltError) -> OverlayInfo { throw .invalidRequestStructure(details: "unscripted") }
    func compare(_ request: CompareRequest) async throws(BasaltError) -> CompareReport { CompareReport(rows: []) }
    func overlays() async -> [OverlayInfo] { active }

    func trace(_ request: TraceRequest) async throws(BasaltError) -> TraceReport {
        traces.append(request)
        return TraceReport(tokens: Array(repeating: "token", count: 100), position: 0, layers: [], final: [])
    }

    func intervene(_ intervention: Intervention) async throws(BasaltError) -> OverlayInfo {
        let info = OverlayInfo(id: OverlayID((active.map(\.id.rawValue).max() ?? 0) + 1), description: "scripted", createdAt: Date())
        active.append(info)
        return info
    }

    func undo(_ overlay: OverlayID) async throws(BasaltError) {
        guard let index = active.firstIndex(where: { $0.id == overlay }) else { throw .invalidRequestStructure(details: "no overlay") }
        active.remove(at: index)
        undone.append(overlay)
    }
}

@Suite("Introspection: the served model studies itself through the X-ray tools (#332)")
struct IntrospectorTests {
    static let ablation = #"{"kind": "ablate_head", "layer": 0, "head": 1}"#

    static func introspector(_ engine: ScriptedResearchEngine, interventions: Bool = false, journal: URL? = nil) -> Introspector {
        Introspector(orchestrator: ModelOrchestratorActor(engine: engine), interventions: interventions, journal: journal)
    }

    @Test("A model that calls trace_run and then answers yields a two-step report; the truncated result is fed back and journaled")
    func traceThenAnswer() async throws {
        let engine = ScriptedResearchEngine([
            .call("trace_run", #"{"prompt": "The capital of France is", "topK": 3}"#),
            .answer("**Hypothesis:** the last layer predicts Paris\nTest: trace_run on the prompt\nOutcome: confirmed, p = 0.9\nAnswer: layer 3"),
        ])
        let journal = FileManager.default.temporaryDirectory.appending(path: "introspect-\(UUID().uuidString)")
        var request = IntrospectRequest(goal: "Where does Paris appear?")
        request.resultBudget = 200
        let report = try await Self.introspector(engine, journal: journal).introspect(request)
        #expect(report.steps.count == 2 && report.stop == .answered && report.answer?.hasSuffix("Answer: layer 3") == true)
        #expect(report.steps[0].tool == "trace_run" && report.steps[1].tool == nil)
        let result = try #require(report.steps[0].result)
        #expect(result.contains(#""tokens""#) && result.hasSuffix("characters]"))
        #expect(await engine.traces == [TraceRequest(prompt: "The capital of France is", topK: 3)])
        #expect(report.hypotheses == [Hypothesis(statement: "the last layer predicts Paris", test: "trace_run on the prompt", outcome: "confirmed, p = 0.9")])
        let fedBack = try #require(await engine.requests.last?.messages?.last)
        #expect(fedBack.role == .tool && fedBack.toolName == "trace_run" && fedBack.text == result)
        #expect(await engine.requests.first?.tools?.map(\.name) == ["model_describe", "model_tensors", "model_tensor", "kv_inspect", "trace_run"])
        let lines = try String(contentsOf: journal.appending(path: "introspect.jsonl"), encoding: .utf8).split(separator: "\n")
        #expect(lines.count == 2 && lines.allSatisfy { $0.contains(#""event":"introspect""#) })
    }

    @Test("Repeating a call, even with its keys reordered, stops the loop before the call runs again")
    func repeatedCall() async throws {
        let engine = ScriptedResearchEngine([
            .call("trace_run", #"{"prompt": "hi", "topK": 3}"#),
            .call("trace_run", #"{"topK":3,"prompt":"hi"}"#),
            .answer("unreachable"),
        ])
        let report = try await Self.introspector(engine).introspect(IntrospectRequest(goal: "g"))
        #expect(report.stop == .repeatedCall && report.steps.count == 2 && report.steps[1].result == nil)
        #expect(await engine.traces.count == 1)
        #expect(await engine.requests.count == 2)
    }

    @Test("An intervention the loop created is undone when generation fails")
    func undoneOnError() async throws {
        let engine = ScriptedResearchEngine([.call("intervene", Self.ablation), .fail])
        let introspector = Self.introspector(engine, interventions: true)
        await #expect(throws: BasaltError.self) {
            try await introspector.introspect(IntrospectRequest(goal: "g", allowInterventions: true))
        }
        #expect(await engine.undone == [1])
        #expect(await engine.overlays().isEmpty)
    }

    @Test("An intervention the loop created is undone when the loop is cancelled mid-generation")
    func undoneOnCancellation() async throws {
        let (hanging, signal) = AsyncStream.makeStream(of: Void.self)
        let engine = ScriptedResearchEngine([.call("intervene", Self.ablation), .hang], hanging: signal)
        let introspector = Self.introspector(engine, interventions: true)
        let task = Task { try await introspector.introspect(IntrospectRequest(goal: "g", allowInterventions: true)) }
        for await _ in hanging { break }
        task.cancel()
        await #expect(throws: (any Error).self) { try await task.value }
        #expect(await engine.undone == [1])
        #expect(await engine.overlays().isEmpty)
    }

    @Test("Interventions are not offered without allowInterventions, and a bad call is fed back as an error, not thrown")
    func refusedCalls() async throws {
        let engine = ScriptedResearchEngine([.call("intervene", Self.ablation), .call("model_tensor", "{}"), .answer("Answer: none")])
        let report = try await Self.introspector(engine).introspect(IntrospectRequest(goal: "g"))
        #expect(report.stop == .answered && report.steps.count == 3)
        #expect(report.steps[0].result?.hasPrefix("error: intervene is not one of the tools") == true)
        #expect(report.steps[1].result?.contains("Missing required argument: name") == true)
        #expect(await engine.overlays().isEmpty)
        #expect(await engine.requests.first?.tools?.contains { $0.name == "intervene" } == false)
    }

    @Test("Hypotheses parse leniently from noisy text, and a restatement replaces the earlier one")
    func hypotheses() {
        let text = """
            Let me think.
            Test: a test line before any hypothesis is ignored
            1. **Hypothesis 1:** head 3 of layer 2 reads the tool call
               - *Test*: trace_run with attention at the last position
               - Outcome: open
            ## Hypothesis:
            Hypothesis: the router prefers expert 5
            Result: refuted — expert 1 wins
            Some trailing prose: with a colon.
            """
        var all = Hypothesis.parse(text)
        #expect(all == [
            Hypothesis(statement: "head 3 of layer 2 reads the tool call", test: "trace_run with attention at the last position", outcome: "open"),
            Hypothesis(statement: "the router prefers expert 5", outcome: "refuted — expert 1 wins"),
        ])
        Hypothesis.merge([Hypothesis(statement: "head 3 of layer 2 reads the tool call", outcome: "confirmed")], into: &all)
        #expect(all.count == 2 && all[0].outcome == "confirmed")
        #expect(Hypothesis.parse("no fields here\njust prose").isEmpty)
    }

    @Test("introspect is listed and accepted only with --research-tools and parses its arguments")
    func mcpTool() async throws {
        #expect(MCPTool(name: "introspect") == .introspect)
        #expect(!MCPToolCatalog.catalog(researchTools: false).tools.contains { $0.name == "introspect" })
        #expect(MCPToolCatalog.catalog(researchTools: true).tools.contains { $0.name == "introspect" })
        var expected = IntrospectRequest(goal: "g", maxSteps: 3, allowInterventions: true)
        expected.resultBudget = 500
        #expect(try IntrospectRequest(mcp: ["goal": "g", "maxSteps": 3, "allowInterventions": true, "resultBudget": 500]) == expected)
        #expect(throws: MCPFailure.self) { try IntrospectRequest(mcp: ["goal": "g", "maxSteps": 0]) }
        #expect(throws: MCPFailure.self) { try IntrospectRequest(mcp: [:]) }
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: ModelOrchestratorActor(engine: ScriptedResearchEngine([]))) { capturer.capture($0) }
        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"introspect","arguments":{"goal":"g"}}}"#)
        #expect(capturer.lines.first?.contains("-32601") == true)
    }
}
