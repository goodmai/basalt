import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("GGUF RawSpan dequantization and reader (#490)")
struct RawSpanGGUFDequantTests {
    static let goldenTypes: [String: GGMLType] = [
        "f16": .f16, "bf16": .bf16, "q8_0": .q8_0, "q2K": .q2K, "q4K": .q4K, "q6K": .q6K,
        "iq4NL": .iq4NL, "iq4XS": .iq4XS, "iq3S": .iq3S, "iq3XXS": .iq3XXS,
        "iq2S": .iq2S, "iq2XS": .iq2XS, "iq2XXS": .iq2XXS, "iq1M": .iq1M,
    ]

    static func hexBytes(_ hex: String) -> [UInt8] {
        stride(from: 0, to: hex.count, by: 2).map { offset in
            let start = hex.index(hex.startIndex, offsetBy: offset)
            return UInt8(hex[start ..< hex.index(start, offsetBy: 2)], radix: 16) ?? 0
        }
    }

    @Test("RawSpan and MutableSpan decode every quantized and half GGML type to gguf-py golden values",
          arguments: Array(goldenTypes.keys).sorted())
    func goldenViaRawSpan(_ name: String) throws {
        let (hex, expected) = try #require(ggufDequantGolden[name])
        let type = try #require(Self.goldenTypes[name])
        let raw = Self.hexBytes(hex)
        let (elements, blockBytes) = type.block
        #expect(raw.count == 2 * blockBytes)

        let decoded = GGUFDequant.floats(raw.span.bytes, type: type, count: 2 * elements)
        let fromData = GGUFDequant.floats(Data(raw), type: type, count: 2 * elements)
        #expect(decoded.count == expected.count)
        #expect(decoded.map(\.bitPattern) == fromData.map(\.bitPattern))

        var manual: [Float] = []
        manual.reserveCapacity(2 * elements)
        var scratch = InlineArray<256, Float>(repeating: 0)
        for block in 0 ..< 2 {
            let slice = raw.span.bytes.extracting(block * blockBytes ..< (block + 1) * blockBytes)
            var out = scratch.mutableSpan
            GGUFDequant.decode(slice, type: type, into: &out)
            for index in 0 ..< elements { manual.append(scratch[index]) }
        }
        #expect(manual.map(\.bitPattern) == decoded.map(\.bitPattern))

        let worst = zip(decoded, expected).map { abs($0 - $1) / max(1, abs($1)) }.max() ?? 0
        #expect(worst <= 1e-6, "\(name): worst relative gap \(worst)")
    }

    @Test("RawSpan decodes little-endian f32 blocks and extracted sub-spans")
    func f32ViaRawSpan() {
        let inputs: [Float] = [-3.5, 0.0, 1.25, 42.125, -0.0078125, 1024.5]
        var bytes: [UInt8] = []
        for value in inputs {
            let bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: bits) { bytes.append(contentsOf: $0) }
        }
        let all = GGUFDequant.floats(bytes.span.bytes, type: .f32, count: inputs.count)
        #expect(all == inputs)

        let middle = GGUFDequant.floats(bytes.span.bytes.extracting(4 ..< 20), type: .f32, count: 4)
        #expect(middle == Array(inputs[1 ..< 5]))
    }

    @Test("GGUFReader.integer loads unaligned little-endian values from RawSpan and rejects truncated bounds")
    func unalignedIntegerReader() throws {
        let bytes: [UInt8] = [0xAA, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]
        let u16Value: UInt16 = try GGUFReader.integer(bytes.span.bytes, at: 1, path: "mem.gguf")
        let u32Value: UInt32 = try GGUFReader.integer(bytes.span.bytes, at: 3, path: "mem.gguf")
        let u64Value: UInt64 = try GGUFReader.integer(bytes.span.bytes, at: 7, path: "mem.gguf")
        #expect(u16Value == 0x1234)
        #expect(u32Value == 0x1234_5678)
        #expect(u64Value == 0x0123_4567_89AB_CDEF)

        #expect(throws: BasaltError.self) {
            let _: UInt32 = try GGUFReader.integer(bytes.span.bytes, at: -1, path: "mem.gguf")
        }
        #expect(throws: BasaltError.self) {
            let _: UInt64 = try GGUFReader.integer(bytes.span.bytes, at: 8, path: "mem.gguf")
        }
    }
}

private struct FixedForcedSource: LogitProcessor, ForcedTokenSource {
    var queued: [Int]
    var forcedObserved: [Int] = []

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray { logits }

    mutating func didSample(token: MLXArray) {}

    mutating func takeForcedTokens(limit: Int) -> [Int] {
        let count = min(limit, queued.count)
        let prefix = Array(queued.prefix(count))
        queued.removeFirst(count)
        return prefix
    }

    mutating func didSampleForced(_ token: MLXArray) {
        forcedObserved.append(token.item(Int.self))
    }
}

@Suite("Concrete LogitProcessorChain and sampling pipeline (#490)")
struct LogitProcessorChainTests {
    static func isConcrete(_ stage: LogitStage) -> Bool {
        if case .other = stage { return false }
        return true
    }

    @Test("Standard sampling pipeline builds unboxed LogitStage entries and preserves penalty order")
    func unboxedPipelineAndPenaltyOrder() async throws {
        let engine = MLXInferenceEngine(EngineConfiguration(seed: 7))
        let request = GenerationRequest(prompt: "p", temperature: 0.7, topP: 1.0).with {
            $0.presencePenalty = 1.5
            $0.dryMultiplier = 0.8
            $0.xtcProbability = 0.5
            $0.xtcThreshold = 0.9
            $0.dynamicTemperature = true
        }
        let (sampledParams, base, shaping, exclusion) = await engine.sampling(for: request, breakers: [])
        var params = sampledParams
        params.repetitionPenalty = 2.0

        let withNgram = base.appendingBasaltProcessor { NoRepeatNGram(size: 3, window: 8, whitelist: []) }
        let withXtc = try #require(exclusion.map { withNgram.appendingBasaltProcessor($0) })
        let withEdt = try #require(shaping.map { withXtc.appendingBasaltProcessor($0) })
        let (guarded, _) = NonFiniteLogitGuard.guarding(withEdt, temperature: params.temperature)

        let built = try #require(guarded.makeProcessor(parameters: params) as? LogitProcessorChain)
        #expect(built.stages.count == 7)
        #expect(built.stages.allSatisfy { Self.isConcrete($0) })
        #expect(!built.hasForcedSource)
        #expect(built.processors.last is NonFiniteLogitGuard)

        var chain = built
        #expect(chain.takeForcedTokens(limit: 4).isEmpty)
        chain.prompt(MLXArray([Int32(1), 2]))
        chain.didSample(token: MLXArray(Int32(3)))
        let logits = MLXArray([Float(4), 4, 4, 4, 4]).reshaped(1, -1)
        let processed = chain.process(logits: logits).asArray(Float.self)
        #expect(processed.count == 5)
        #expect(processed[3] < processed[2] && processed[2] < processed[0])
    }

    @Test("copy() isolates mutable stage state across speculative verification branches")
    func copyIsolatesState() {
        var base = LogitProcessorChain()
        base.appendStage(.presence(GeneratedTokensPresencePenalty(penalty: 2.0, window: 16)))
        base.appendStage(.dry(DRYPenalty(multiplier: 1.0, base: 2.0, allowedLength: 2, breakers: [])))
        for id in [Int32(1), 2, 1] { base.didSample(token: MLXArray(id)) }

        var branch = base.copy()
        branch.didSample(token: MLXArray(Int32(4)))

        let zero = MLXArray.zeros([1, 6], type: Float32.self)
        let baseOut = base.process(logits: zero).asArray(Float.self)
        let branchOut = branch.process(logits: zero).asArray(Float.self)
        #expect(baseOut[2] < 0)
        #expect(baseOut[4] == 0)
        #expect(branchOut[4] == -2.0)
    }

    @Test("ForcedTokenSource stages emit bounded runs and sync non-forced stages via PromptAnchoredProcessor")
    func forcedTokensAndPromptAnchor() {
        let components = GenerationComponents()
            .appendingBasaltProcessor { GeneratedTokensPresencePenalty(penalty: 1.25, window: 16) }
            .appendingBasaltProcessor { FixedForcedSource(queued: [2, 3, 4]) }
        guard let built = components.logitProcessorFactory?() else {
            Issue.record("Expected chained processor")
            return
        }
        var anchored = PromptAnchoredProcessor(base: built, prompt: MLXArray([Int32(5)]))
        #expect(anchored.chain.hasForcedSource)
        anchored.prompt(MLXArray([Int32(0)]))

        let firstBatch = anchored.takeForcedTokens(limit: 2)
        #expect(firstBatch == [2, 3])
        for id in firstBatch { anchored.didSampleForced(MLXArray(Int32(id))) }

        let secondBatch = anchored.takeForcedTokens(limit: 4)
        #expect(secondBatch == [4])
        anchored.didSampleForced(MLXArray(Int32(4)))
        #expect(anchored.takeForcedTokens(limit: 4).isEmpty)

        let out = anchored.process(logits: MLXArray.zeros([1, 6], type: Float32.self)).asArray(Float.self)
        #expect(out == [0, 0, -1.25, -1.25, -1.25, 0])
        let source = anchored.chain.stages[1].erased as? FixedForcedSource
        #expect(source?.forcedObserved == [2, 3, 4])
    }
}
