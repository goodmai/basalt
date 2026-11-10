import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Trigger gates turn a per-token condition into the tokens an edit applies to (#405)")
struct TriggerGateTests {
    @Test("Without a window the edit applies exactly where the condition holds, across forwards")
    func perToken() {
        var gate = TriggerGate(window: nil)
        #expect(gate.step([false, true, false]) == [false, true, false])
        #expect(gate.step([true]) == [true])
        #expect(gate.fired == [1, 3])
        #expect(gate.first == 1 && gate.tokens == 4 && gate.rate == 0.5)
    }

    @Test("A window keeps the edit on for that many tokens from the first trigger, then off even if it fires again")
    func window() {
        var gate = TriggerGate(window: 3)
        #expect(gate.step([false, false, true, false]) == [false, false, true, true])
        #expect(gate.step([false]) == [true])
        #expect(gate.step([true, false]) == [false, false])
        #expect(gate.fired == [2, 5] && gate.first == 2)
        #expect(TriggerGate(window: 2).rate == nil)
    }
}

@Suite("intervene with when: triggers fire on the expected tokens and leave other tokens untouched (#405)", .serialized)
struct ConditionalInterventionTests {
    static let prompt = "the cat sat on the mat and looked at the dog"

    struct Run: Sendable {
        var expected: [Int] = []
        var fired: [Int] = []
        var unchangedBeforeFirst = true
        var changedAtFirst = false
        var silentIdentical = false
    }

    static func run(_ trigger: @escaping @Sendable (TokenTraceTable) -> (InterventionTrigger, [Int])) async throws -> Run {
        let engine = try await TinyModel.engine(.lfm2Moe)
        let anatomy = try await engine.anatomy()
        let container = try TinyModel.container(.lfm2Moe)
        return try await container.perform { context in
            let ids = context.tokenizer.encode(text: Self.prompt, addSpecialTokens: true)
            var request = TokenTraceRequest(prompt: Self.prompt, maxTokens: 0, layers: [1])
            request.projections = [TraceDirection(layer: 1, vector: (0 ..< anatomy.hiddenSize).map { Float($0 % 3) - 1 }, threshold: 0)]
            let recorder = TokenTraceRecorder(request, readout: nil)
            _ = try MLXInferenceEngine.tracedGreedy(ids, maxTokens: 0, recorder: recorder, context: context)
            let (floats, ints) = try #require(recorder.table())
            let table = TokenTraceTable(layout: try #require(recorder.layout), tokens: ids.count, floats: floats.asArray(Float.self), ints: ints.asArray(Int32.self))
            let (condition, expected) = trigger(table)
            let rule = ConditionalIntervention(action: .ablateHead(layer: 3, head: 1), trigger: condition)
            let edited = InterventionProbe([.conditional(rule)], anatomy: anatomy)
            let path = ids
            let base = try MLXInferenceEngine.logProbabilities(path, from: 0, context: context, probe: InterventionProbe([], anatomy: anatomy))
            let after = try MLXInferenceEngine.logProbabilities(path, from: 0, context: context, probe: edited)
            var silentRule = rule
            silentRule.trigger = .expertRouted(layer: 1, expert: 0, gateAbove: 10)
            let silent = try MLXInferenceEngine.logProbabilities(path, from: 0, context: context, probe: InterventionProbe([.conditional(silentRule)], anatomy: anatomy))
            var result = Run(expected: expected, fired: edited.conditions[0].gate.fired)
            let first = result.fired.first ?? ids.count
            result.unchangedBeforeFirst = first == 0 || base[..<first].asArray(Float.self) == after[..<first].asArray(Float.self)
            result.changedAtFirst = first < ids.count - 1 && base[first].asArray(Float.self) != after[first].asArray(Float.self)
            result.silentIdentical = base.asArray(Float.self) == silent.asArray(Float.self)
            return result
        }
    }

    @Test("An expert-routed trigger fires on exactly the tokens the trace shows routing that expert; earlier tokens are bit-identical, the first firing token changes")
    func expertTrigger() async throws {
        let result = try await Self.run { table in
            let routed = (0 ..< table.tokens).filter { table.row(.experts, layer: 1, token: $0)?.contains(2) == true }
            return (.expertRouted(layer: 1, expert: 2, gateAbove: 0), routed)
        }
        #expect(!result.expected.isEmpty)
        #expect(result.fired == result.expected)
        #expect(result.unchangedBeforeFirst)
        #expect(result.changedAtFirst)
        #expect(result.silentIdentical)
    }

    @Test("A projection trigger fires on exactly the tokens whose traced projection is above the threshold")
    func projectionTrigger() async throws {
        let result = try await Self.run { table in
            let values = table.column(.projection, layer: 1) ?? []
            let threshold = values.sorted()[values.count / 2]
            let vector = (0 ..< 64).map { Float($0 % 3) - 1 }
            return (.projection(layer: 1, vector: vector, above: threshold), values.indices.filter { values[$0] > threshold })
        }
        #expect(!result.expected.isEmpty)
        #expect(result.fired == result.expected)
        #expect(result.unchangedBeforeFirst)
        #expect(result.changedAtFirst)
    }

    static let rule = ConditionalIntervention(action: .ablateExpert(layer: 3, expert: 1), trigger: .expertRouted(layer: 1, expert: 2, gateAbove: 0), window: 4)

    @Test("A conditional overlay is listed, reports its trigger rate in experiment_compare, and after undo is gone and the comparison is exact")
    func overlayLifecycle() async throws {
        let engine = try await TinyModel.engine(.lfm2Moe)
        let info = try await engine.intervene(.conditional(Self.rule))
        #expect(info.description.contains("when expert 2 of layer 1 is routed"))
        #expect(await engine.overlays().map(\.id) == [info.id])
        let report = try await engine.compare(CompareRequest(prompts: [Self.prompt], overlays: [info.id], maxTokens: 4))
        let rate = try #require(report.rows[0].triggerRate)
        #expect(rate > 0 && rate <= 1)
        try await engine.undo(info.id)
        #expect(await engine.overlays().isEmpty)
        await #expect(throws: BasaltError.self) { try await engine.compare(CompareRequest(prompts: [Self.prompt], overlays: [info.id])) }
        let clean = try await engine.compare(CompareRequest(prompts: [Self.prompt], overlays: [], maxTokens: 4))
        #expect(clean.rows[0].klDivergence == 0 && clean.rows[0].overlayText == clean.rows[0].baselineText && clean.rows[0].triggerRate == nil)
    }

    @Test("A conditional overlay expires like any other")
    func expiry() {
        var registry = OverlayRegistry(journal: nil)
        let start = Date(timeIntervalSince1970: 1_000)
        _ = registry.add(.conditional(Self.rule), description: "c", at: start, lifetime: 60)
        #expect(registry.expire(at: start.addingTimeInterval(30)).isEmpty)
        #expect(registry.expire(at: start.addingTimeInterval(61)).map(\.intervention) == [Intervention.conditional(Self.rule)] as [Intervention?])
        #expect(registry.overlays.isEmpty)
    }

    @Test("Actions at or before the trigger layer, nested conditions, bad windows, directions and experts are refused", arguments: [
        ConditionalIntervention(action: .ablateHead(layer: 1, head: 0), trigger: .expertRouted(layer: 1, expert: 0, gateAbove: 0)),
        ConditionalIntervention(action: .conditional(ConditionalInterventionTests.rule), trigger: .expertRouted(layer: 1, expert: 0, gateAbove: 0)),
        ConditionalIntervention(action: .ablateHead(layer: 3, head: 0), trigger: .expertRouted(layer: 1, expert: 0, gateAbove: 0), window: 0),
        ConditionalIntervention(action: .ablateHead(layer: 3, head: 0), trigger: .projection(layer: 1, vector: [1], above: 0)),
        ConditionalIntervention(action: .ablateHead(layer: 3, head: 0), trigger: .expertRouted(layer: 1, expert: 9, gateAbove: 0)),
        ConditionalIntervention(action: .ablateHead(layer: 3, head: 7), trigger: .expertRouted(layer: 1, expert: 0, gateAbove: 0)),
        ConditionalIntervention(action: .ablateHead(layer: 3, head: 0), trigger: .expertRouted(layer: 8, expert: 0, gateAbove: 0)),
    ])
    func refused(rule: ConditionalIntervention) async throws {
        let engine = try await TinyModel.engine(.lfm2Moe)
        await #expect(throws: BasaltError.self) { try await engine.intervene(.conditional(rule)) }
        #expect(await engine.overlays().isEmpty)
    }

    @Test("intervene parses when into a conditional overlay and names a bad trigger")
    func parsing() throws {
        let args: [String: Any] = ["kind": "ablate_head", "layer": 3, "head": 1, "when": ["kind": "expert", "layer": 1, "expert": 2, "gate": 0.25, "window": 8]]
        let expected = ConditionalIntervention(action: .ablateHead(layer: 3, head: 1), trigger: .expertRouted(layer: 1, expert: 2, gateAbove: 0.25), window: 8)
        #expect(try Intervention(mcp: args) == .conditional(expected))
        let projection: [String: Any] = ["kind": "steer", "layer": 2, "vector": [1.0], "scale": -2.0, "when": ["kind": "projection", "layer": 0, "vector": [0.5], "threshold": 1.5]]
        #expect(try Intervention(mcp: projection) == .conditional(ConditionalIntervention(
            action: .steer(layer: 2, vector: [1], scale: -2), trigger: .projection(layer: 0, vector: [0.5], above: 1.5))))
        for (when, fragment) in [("x", "when must be an object"), (["kind": "x", "layer": 1], "when.kind must be one of"), (["kind": "projection", "layer": 1, "vector": [1.0]], "when.threshold")] as [(Any, String)] {
            let failure = #expect(throws: MCPFailure.self) { try Intervention(mcp: ["kind": "ablate_head", "layer": 3, "head": 1, "when": when]) }
            #expect(failure?.message.contains(fragment) == true, "\(failure?.message ?? "")")
        }
    }
}
