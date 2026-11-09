import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

final class RawHookCapture: ForwardProbe {
    var heads: [Int: [MLXArray]] = [:]
    var routers: [Int: [MLXArray]] = [:]
    var experts: [Int: [(indices: MLXArray, gates: MLXArray)]] = [:]

    func residual(layer: Int, _ hidden: MLXArray) -> MLXArray { hidden }
    func attentionWeights(layer: Int, _ weights: MLXArray) {}
    func wantsAttention(layer: Int) -> Bool { false }
    func headMask(layer: Int) -> MLXArray? { nil }
    func expertMask(layer: Int) -> MLXArray? { nil }
    func headOutputs(layer: Int, _ perHead: MLXArray) { heads[layer, default: []].append(perHead) }
    func routerLogits(layer: Int, _ logits: MLXArray) { routers[layer, default: []].append(logits.reshaped(-1, logits.dim(-1))) }
    func routerTopK(layer: Int, indices: MLXArray, gates: MLXArray) {
        experts[layer, default: []].append((indices.reshaped(-1, indices.dim(-1)), gates.reshaped(-1, gates.dim(-1))))
    }

    func headNorms(_ layer: Int) -> [[Float]] {
        heads[layer, default: []].flatMap { chunk -> [[Float]] in
            let (count, positions, width) = (chunk.dim(1), chunk.dim(2), chunk.dim(3))
            let values = chunk.asType(.float32).asArray(Float.self)
            return (0 ..< positions).map { position in
                (0 ..< count).map { head in
                    let start = (head * positions + position) * width
                    return Float(values[start ..< start + width].reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot())
                }
            }
        }
    }

    func rows(_ chunks: [MLXArray]) -> [[Float]] {
        chunks.flatMap { chunk in
            let values = chunk.asType(.float32).asArray(Float.self)
            return (0 ..< chunk.dim(0)).map { Array(values[$0 * chunk.dim(1) ..< ($0 + 1) * chunk.dim(1)]) }
        }
    }
}

struct TraceParity: Sendable {
    var traced: [Int] = []
    var untraced: [Int] = []
    var headLayers = 0
    var headError: Float = 0
    var expertLayers = 0
    var idsMatch = true
    var gateError: Float = 0
    var routerTopKMatches = true
    var tokens = 0
}

@Suite("trace_tokens records heads, experts and projections per token with the stock graph's output (#405)", .serialized)
struct TokenTraceParityTests {
    static let prompt = "the cat sat on the mat"
    static let families: [TinyModel.Family] = [.qwen3, .qwen3Moe, .qwen35, .qwen35Moe, .gemma4, .gemma4Moe, .lfm2, .lfm2Moe, .nemotronH, .sigmoidMoe]
    static let softmaxRouters: Set<TinyModel.Family> = [.qwen3Moe, .qwen35Moe, .gemma4Moe]

    static func parity(_ family: TinyModel.Family, maxTokens: Int = 8) async throws -> TraceParity {
        let container = try TinyModel.container(family)
        return try await container.perform { context in
            let prompt = context.tokenizer.encode(text: Self.prompt, addSpecialTokens: true)
            var result = TraceParity()
            result.untraced = try MLXInferenceEngine.greedy(
                prompt, context: context, probe: nil, maxTokens: maxTokens, stops: MLXInferenceEngine.stopTokenIds(context))
            let capture = RawHookCapture()
            let recorder = TokenTraceRecorder(TokenTraceRequest(prompt: Self.prompt, maxTokens: maxTokens), readout: nil, edits: capture)
            result.traced = try MLXInferenceEngine.tracedGreedy(prompt, maxTokens: maxTokens, recorder: recorder, context: context)
            let layout = try #require(recorder.layout)
            let (floats, ints) = try #require(recorder.table())
            let table = TokenTraceTable(layout: layout, tokens: floats.dim(0), floats: floats.asArray(Float.self), ints: ints.asArray(Int32.self))
            result.tokens = table.tokens
            for layer in capture.heads.keys.sorted() {
                result.headLayers += 1
                for (token, manual) in capture.headNorms(layer).enumerated() {
                    let reported = try #require(table.floatRow(.heads, layer: layer, token: token))
                    result.headError = max(result.headError, zip(manual, reported).map { abs($0 - $1) }.max() ?? .infinity)
                }
            }
            for layer in capture.experts.keys.sorted() {
                result.expertLayers += 1
                let ids = capture.rows(capture.experts[layer, default: []].map(\.indices))
                let gates = capture.rows(capture.experts[layer, default: []].map(\.gates))
                let logits = capture.rows(capture.routers[layer, default: []])
                for token in 0 ..< table.tokens {
                    let reportedIds = try #require(table.row(.experts, layer: layer, token: token)).map(Int.init)
                    let reportedGates = try #require(table.floatRow(.gates, layer: layer, token: token))
                    result.idsMatch = result.idsMatch && reportedIds == ids[token].map { Int($0) }
                    result.gateError = max(result.gateError, zip(gates[token], reportedGates).map { abs($0 - $1) }.max() ?? .infinity)
                    let topK = Set(TraceSummary.top(logits[token], k: reportedIds.count).map(\.index))
                    result.routerTopKMatches = result.routerTopKMatches && topK == Set(reportedIds)
                }
            }
            return result
        }
    }

    @Test("Token ids are identical with tracing on and off, head norms match a manual per-head L2 within 1e-4, expert ids match the router's top-k exactly and gates within 1e-5",
          arguments: TokenTraceParityTests.families)
    func parity(family: TinyModel.Family) async throws {
        let result = try await Self.parity(family)
        #expect(result.traced == result.untraced)
        #expect(result.tokens == ByteTokenizer().encode(text: Self.prompt, addSpecialTokens: true).count + result.traced.count)
        #expect(result.headLayers > 0)
        #expect(result.headError < 1e-4)
        let moe: Set<TinyModel.Family> = [.qwen3Moe, .qwen35Moe, .gemma4Moe, .lfm2Moe, .sigmoidMoe]
        #expect((result.expertLayers > 0) == moe.contains(family))
        #expect(result.idsMatch)
        #expect(result.gateError < 1e-5)
        if Self.softmaxRouters.contains(family) { #expect(result.routerTopKMatches) }
    }
}

@Suite("trace_tokens writes a bounded 0600 trace file and a summary through the engine (#405)", .serialized)
struct TokenTraceEngineTests {
    static func directory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "basalt-trace-\(UUID().uuidString)")
    }

    static func parse(_ data: Data) throws -> (header: TokenTraceFile.Header, records: [TokenTraceFile.Record], floats: [Float]) {
        var lines: [Data] = []
        var start = data.startIndex
        let header = try #require(data[start...].firstIndex(of: 0x0A))
        let decoded = try JSONDecoder().decode(TokenTraceFile.Header.self, from: data[start ..< header])
        start = header + 1
        for _ in 0 ..< decoded.tokens {
            let end = try #require(data[start...].firstIndex(of: 0x0A))
            lines.append(data[start ..< end])
            start = end + 1
        }
        let records = try lines.map { try JSONDecoder().decode(TokenTraceFile.Record.self, from: $0) }
        let floats = MLXArray(Data(data[start...]), [decoded.tokens, decoded.floatsPerToken], type: Float.self).asArray(Float.self)
        return (decoded, records, floats)
    }

    @Test("The file holds a header, one record per prompt and generated token and tokens x floatsPerToken floats that agree with the summary; it is mode 0600 in a 0700 directory")
    func writesFile() async throws {
        let engine = try await TinyModel.engine(.lfm2Moe)
        let anatomy = try await engine.anatomy()
        var request = TokenTraceRequest(prompt: TokenTraceParityTests.prompt, maxTokens: 4)
        request.directory = Self.directory()
        request.lensTopK = 3
        request.projections = [TraceDirection(layer: 2, vector: Array(repeating: 1, count: anatomy.hiddenSize), threshold: -1e9)]
        defer { try? FileManager.default.removeItem(at: request.directory) }
        let report = try await engine.traceTokens(request)
        let data = try Data(contentsOf: URL(fileURLWithPath: report.file))
        #expect(data.count == report.bytes)
        let (header, records, floats) = try Self.parse(data)
        #expect(header.tokens == report.promptTokens + report.generated.count)
        #expect(records.map(\.generated) == Array(repeating: false, count: report.promptTokens) + Array(repeating: true, count: report.generated.count))
        #expect(records.suffix(report.generated.count).map(\.id) == report.generated)
        #expect(floats.count == header.tokens * header.floatsPerToken)
        #expect(records.allSatisfy { $0.ints.count == header.intsPerToken })
        let table = TokenTraceTable(layout: header.layout, tokens: header.tokens, floats: floats, ints: records.flatMap(\.ints))
        #expect(TokenTraceSummary.layers(table, topK: request.topK) == report.layers)
        #expect(report.crossings.map(\.count) == [header.tokens])
        #expect(report.layers.filter { !$0.experts.isEmpty }.map(\.index) == [1, 2, 3])
        #expect(report.layers.flatMap(\.experts).allSatisfy { $0.tokens <= header.tokens && $0.meanGate > 0 })
        #expect(table.row(.lens, layer: 0, token: 0)?.count == 3)
        let attributes = try FileManager.default.attributesOfItem(atPath: report.file)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
        let folder = try FileManager.default.attributesOfItem(atPath: request.directory.path)
        #expect((folder[.posixPermissions] as? NSNumber)?.intValue == 0o700)
    }

    @Test("Caps are refused before any file is written",
          arguments: [
            { (r: inout TokenTraceRequest) in r.maxFileBytes = 1024 },
            { (r: inout TokenTraceRequest) in r.maxTokens = 1025 },
            { (r: inout TokenTraceRequest) in r.layers = [9] },
            { (r: inout TokenTraceRequest) in r.layers = [] },
            { (r: inout TokenTraceRequest) in r.projections = [TraceDirection(layer: 1, vector: [1], threshold: 0)] },
            { (r: inout TokenTraceRequest) in (r.heads, r.experts) = (false, false) },
            { (r: inout TokenTraceRequest) in r.prompt = String(repeating: "a", count: 2040) },
          ] as [@Sendable (inout TokenTraceRequest) -> Void])
    func refused(change: @Sendable (inout TokenTraceRequest) -> Void) async throws {
        let engine = try await TinyModel.engine(.qwen35)
        var request = TokenTraceRequest(prompt: TokenTraceParityTests.prompt, maxTokens: 16)
        request.directory = Self.directory()
        change(&request)
        await #expect(throws: BasaltError.self) { try await engine.traceTokens(request) }
        #expect(!FileManager.default.fileExists(atPath: request.directory.path))
    }
}

@Suite("Token trace summaries rank heads and experts and find projection crossings (#405)")
struct TokenTraceSummaryTests {
    static let layout = TokenTraceLayout(
        floats: [
            TraceSegment(layer: 0, field: .heads, width: 2), TraceSegment(layer: 1, field: .gates, width: 2),
            TraceSegment(layer: 1, field: .routerEntropy, width: 1), TraceSegment(layer: 1, field: .projection, width: 1),
        ],
        ints: [TraceSegment(layer: 1, field: .experts, width: 2)])

    static let table = TokenTraceTable(
        layout: layout, tokens: 3,
        floats: [1, 4, 0.6, 0.4, 1, -1, 3, 2, 0.7, 0.3, 2, 5, 2, 0, 0.5, 0.5, 3, 0.5],
        ints: [7, 2, 7, 3, 2, 7])

    @Test("Heads rank by mean norm, experts by tokens routed with their mean gate, the router entropy is averaged")
    func layers() {
        let layers = TokenTraceSummary.layers(Self.table, topK: 5)
        #expect(layers.map(\.index) == [0, 1])
        #expect(layers[0].heads == [HeadActivity(head: 0, meanNorm: 2, maxNorm: 3), HeadActivity(head: 1, meanNorm: 2, maxNorm: 4)])
        #expect(layers[1].experts.map(\.expert) == [7, 2, 3])
        #expect(layers[1].experts.map(\.tokens) == [3, 2, 1])
        #expect(abs((layers[1].experts.first?.meanGate ?? 0) - (0.6 + 0.7 + 0.5) / 3) < 1e-6)
        #expect(layers[1].routerEntropy == 2)
        #expect(TokenTraceSummary.layers(Self.table, topK: 1).map(\.experts.count) == [0, 1])
    }

    @Test("Crossings list the positions strictly above the threshold and the maximum")
    func crossings() {
        let crossings = TokenTraceSummary.crossings(Self.table, directions: [TraceDirection(layer: 1, vector: [], threshold: 0)])
        #expect(crossings == [ProjectionCrossing(layer: 1, threshold: 0, count: 2, positions: [1, 2], max: 5)])
        #expect(TokenTraceSummary.crossings(Self.table, directions: [TraceDirection(layer: 0, vector: [], threshold: 0)]).isEmpty)
    }

    @Test("The estimate grows with heads, experts, lens and projections, and maxFileBytes bounds the run")
    func estimate() {
        var anatomy = ModelAnatomy(modelType: "t", architecture: "t", layers: [], parameterCount: 0, bytes: 0)
        var attention = LayerAnatomy(index: 0, kind: "full attention", cache: "")
        attention.heads = 8
        var moe = LayerAnatomy(index: 1, kind: "moe", cache: "")
        (moe.experts, moe.expertsPerToken) = (16, 2)
        anatomy.layers = [attention, moe]
        var request = TokenTraceRequest(prompt: "p")
        let base = request.estimatedBytes(anatomy: anatomy, tokens: 10)
        #expect(base == 10 * ((8 + 2 + 2) * 4 + 2 * 11 + TokenTraceRequest.recordOverhead) + TokenTraceRequest.headerOverhead)
        request.lensTopK = 5
        #expect(request.estimatedBytes(anatomy: anatomy, tokens: 10) > base)
        request.layers = [1]
        request.lensTopK = 0
        #expect(request.estimatedBytes(anatomy: anatomy, tokens: 10) == 10 * (4 * 4 + 2 * 11 + TokenTraceRequest.recordOverhead) + TokenTraceRequest.headerOverhead)
    }
}
