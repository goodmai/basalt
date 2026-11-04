import Foundation
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Trace summaries rank probability rows and measure attention spread (#328)")
struct TraceSummaryTests {

    @Test("Top-k keeps the k largest values in descending order, earlier index first on ties, and skips NaN",
          arguments: [
            ([0.1, 0.5, 0.2, 0.2], 2, [1, 2]),
            ([0.1, 0.5, 0.2, 0.2], 3, [1, 2, 3]),
            ([0.3, 0.7], 5, [1, 0]),
            ([Float.nan, 0.4, 0.6], 2, [2, 1]),
            ([0.9, 0.1], 0, []),
            ([], 3, []),
          ] as [([Float], Int, [Int])])
    func top(row: [Float], k: Int, indices: [Int]) {
        let ranked = TraceSummary.top(row, k: k)
        #expect(ranked.map(\.index) == indices)
        #expect(ranked.map(\.value) == indices.map { row[$0] })
    }

    @Test("Entropy is ln n for a uniform row, zero for a one-hot row, and ignores zero entries",
          arguments: [
            ([0.25, 0.25, 0.25, 0.25], Float(log(4.0))),
            ([0, 1, 0], 0),
            ([0.5, 0.5, 0], Float(log(2.0))),
          ] as [([Float], Float)])
    func entropy(row: [Float], expected: Float) {
        #expect(abs(TraceSummary.entropy(row) - expected) < 1e-6)
    }

    @Test("Heads split a flat row per head and report attended positions shifted by the first key's position")
    func heads() {
        let heads = TraceSummary.heads([0.7, 0.2, 0.1, 0.0, 0.5, 0.5], heads: 2, k: 2, firstPosition: 4)
        #expect(heads.map(\.head) == [0, 1])
        #expect(heads.map { $0.top.map(\.position) } == [[4, 5], [5, 6]])
        #expect(abs(heads[1].entropy - Float(log(2.0))) < 1e-6)
        #expect(TraceSummary.heads([0.5, 0.5, 0.5], heads: 2, k: 1, firstPosition: 0).isEmpty)
    }
}

@Suite("Trace runs one forward with the probe on tiny hybrid and MoE fixtures (#328)", .serialized)
struct XRayTraceTests {
    static let prompt = "the cat sat on the mat"

    @Test("One layer trace per layer, probabilities sum to at most 1, the last layer's lens is the final top-k, and router entries appear only on MoE layers",
          arguments: [(.qwen35, []), (.lfm2Moe, [1, 2, 3])] as [(TinyModel.Family, Set<Int>)])
    func endToEnd(family: TinyModel.Family, moeLayers: Set<Int>) async throws {
        let engine = try await TinyModel.engine(family)
        let report = try await engine.trace(TraceRequest(prompt: Self.prompt))
        let tokens = ByteTokenizer().encode(text: Self.prompt, addSpecialTokens: true)
        #expect(report.tokens.count == tokens.count)
        #expect(report.position == tokens.count - 1)
        #expect(report.layers.map(\.index) == [0, 1, 2, 3])
        #expect(report.final.count == 10)
        let rows = [report.final] + report.layers.compactMap(\.lens)
        #expect(rows.count == 5)
        #expect(rows.allSatisfy { $0.map(\.probability).reduce(0, +) <= 1 + 1e-4 })
        let last = try #require(report.layers.last?.lens)
        #expect(last.map(\.id) == report.final.map(\.id))
        #expect(zip(last, report.final).allSatisfy { abs($0.probability - $1.probability) < 1e-4 })
        #expect(Set(report.layers.filter { $0.experts != nil }.map(\.index)) == moeLayers)
        #expect(report.layers.allSatisfy { $0.hiddenNorm.isFinite && $0.hiddenNorm > 0 })
        for heads in report.layers.compactMap(\.attention) {
            #expect(heads.allSatisfy { $0.entropy.isFinite && $0.top.allSatisfy { (0 ..< tokens.count).contains($0.position) } })
        }
    }

    @Test("Only the requested layers are traced, and a request without lens, attention or router leaves them out")
    func requestedLayers() async throws {
        let engine = try await TinyModel.engine(.lfm2Moe)
        var request = TraceRequest(prompt: Self.prompt, position: 2, layers: [3, 1, 3], topK: 3)
        (request.attention, request.router, request.logitLens) = (false, false, false)
        let report = try await engine.trace(request)
        #expect(report.position == 2)
        #expect(report.layers.map(\.index) == [1, 3])
        #expect(report.final.count == 3)
        #expect(report.layers.allSatisfy { $0.lens == nil && $0.attention == nil && $0.experts == nil })
    }

    @Test("Out-of-bounds requests are refused before or after the forward",
          arguments: [
            TraceRequest(prompt: XRayTraceTests.prompt, topK: 0),
            TraceRequest(prompt: XRayTraceTests.prompt, topK: 51),
            TraceRequest(prompt: XRayTraceTests.prompt, layers: []),
            TraceRequest(prompt: XRayTraceTests.prompt, layers: [9]),
            TraceRequest(prompt: XRayTraceTests.prompt, position: 99),
            TraceRequest(prompt: String(repeating: "a", count: 2049)),
          ])
    func refused(request: TraceRequest) async throws {
        let engine = try await TinyModel.engine(.qwen35)
        await #expect(throws: BasaltError.self) { try await engine.trace(request) }
    }
}
