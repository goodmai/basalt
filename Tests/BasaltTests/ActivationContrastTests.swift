import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("activation_contrast ranks heads and experts that separate two prompt sets (#405)", .serialized)
struct ActivationContrastTests {
    static let layout = TokenTraceLayout(
        floats: [TraceSegment(layer: 0, field: .heads, width: 2), TraceSegment(layer: 1, field: .gates, width: 1)],
        ints: [TraceSegment(layer: 1, field: .experts, width: 1)])

    static func sample(heads: [(Float, Float)], experts: [Int32], prompt: Int) -> TracedSample {
        let floats: [Float] = heads.flatMap { [$0.0, $0.1, 0.5] }
        return TracedSample(table: TokenTraceTable(layout: layout, tokens: heads.count, floats: floats, ints: experts), promptTokens: prompt)
    }

    @Test("Cohen's d uses the pooled standard deviation; AUROC counts ties as half", arguments: [
        ([1, 2, 3], [1, 2, 3], Float(0), Float(0.5)),
        ([2, 3, 4], [0, 1, 2], Float(2), Float(8.5 / 9)),
        ([5, 5], [5, 5], Float(0), Float(0.5)),
    ] as [([Float], [Float], Float, Float)])
    func statistics(a: [Float], b: [Float], d: Float, auroc: Float) {
        #expect(abs(ActivationStatistics.cohensD(a, b) - d) < 1e-5)
        #expect(abs(ActivationStatistics.auroc(a, b) - auroc) < 1e-5)
    }

    @Test("A head louder in set a and an expert routed only in set a rank first, with rates, means and the onset window before the generation")
    func ranking() throws {
        let a = [
            Self.sample(heads: [(1, 4), (1, 6), (1, 8)], experts: [3, 3, 3], prompt: 2),
            Self.sample(heads: [(1, 5), (1, 7), (1, 9)], experts: [3, 1, 3], prompt: 2),
        ]
        let b = [
            Self.sample(heads: [(1, 1), (1, 1), (1, 1)], experts: [1, 1, 1], prompt: 2),
            Self.sample(heads: [(1, 2), (1, 1), (1, 2)], experts: [1, 1, 1], prompt: 2),
        ]
        let ranked = ActivationStatistics.rank(a: a, b: b, onsetWindow: 1, topK: 10)
        #expect(ranked.count == 4)
        let head = try #require(ranked.first { $0.kind == .head && $0.index == 1 })
        #expect(head.auroc == 1)
        #expect(head.cohensD > 2)
        #expect(head.meanA == 6.5 && head.onsetA == 6.5 && head.onsetB == 1)
        #expect(head.rateA == 1 && head.rateB == 0)
        let expert = try #require(ranked.first { $0.kind == .expert && $0.index == 3 })
        #expect(abs(expert.rateA - 5.0 / 6) < 1e-6 && expert.rateB == 0)
        #expect(abs(expert.meanA - 0.5 * 5 / 6) < 1e-6)
        let silent = try #require(ranked.first { $0.kind == .head && $0.index == 0 })
        #expect(silent.cohensD == 0 && silent.auroc == 0.5)
        #expect(ranked.last == silent)
        #expect(ActivationStatistics.rank(a: a, b: b, onsetWindow: 1, topK: 1).count == 1)
    }

    @Test("On a tiny MoE fixture the report ranks both kinds, gives a direction per traced layer and the best layer's unit direction")
    func endToEnd() async throws {
        let engine = try await TinyModel.engine(.lfm2Moe)
        let anatomy = try await engine.anatomy()
        var request = ActivationContrastRequest(a: LabSteerTests.positive, b: LabSteerTests.negative, maxTokens: 3, layers: [3, 1, 2])
        request.topK = 100
        let report = try await engine.activationContrast(request)
        #expect(report.layers.map(\.index) == [1, 2, 3])
        #expect([1, 2, 3].contains(report.bestLayer))
        #expect(report.direction.count == anatomy.hiddenSize)
        #expect(abs(report.direction.reduce(0) { $0 + $1 * $1 } - 1) < 1e-3)
        #expect(Set(report.components.map(\.kind)) == [.head, .expert])
        #expect(report.components.allSatisfy { (0 ... 1).contains($0.auroc) && (0 ... 1).contains($0.rateA) && $0.cohensD.isFinite })
        let effects = report.components.map { abs($0.cohensD) }
        #expect(effects == effects.sorted(by: >))
    }

    @Test("Out-of-range sets, windows and layers are refused", arguments: [
        ActivationContrastRequest(a: ["x"], b: ["y", "z"]),
        ActivationContrastRequest(a: ["x", "y"], b: ["y", "z"], maxTokens: 257),
        ActivationContrastRequest(a: ["x", "y"], b: ["y", "z"], layers: [7]),
        ActivationContrastRequest(a: ["x", "y"], b: ["y", "z"], layers: []),
    ])
    func refused(request: ActivationContrastRequest) async throws {
        let engine = try await TinyModel.engine(.lfm2Moe)
        await #expect(throws: BasaltError.self) { try await engine.activationContrast(request) }
    }
}
