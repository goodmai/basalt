import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import BasaltInference

@Suite("Single-pass Metal logit confidence kernel against stock MLX log-sum-exp, top-2 margin and Shannon entropy (#490, #228)")
struct FusedLogitConfidenceKernelTests {
    static func assertParity(
        _ logits: MLXArray,
        logprobTolerance: Float = 1e-4,
        marginTolerance: Float = 1e-5,
        entropyTolerance: Float = 2e-4,
        label: String = ""
    ) {
        let actual = FusedLogitConfidence.compute(logits)
        let expected = FusedLogitConfidence.reference(logits)
        eval(
            actual.top1TokenID, actual.top1Logprob, actual.top2Margin, actual.entropy,
            expected.top1TokenID, expected.top1Logprob, expected.top2Margin, expected.entropy)

        #expect(actual.top1TokenID.shape == expected.top1TokenID.shape, "\(label) id shape")
        #expect(actual.top1Logprob.shape == expected.top1Logprob.shape, "\(label) logprob shape")
        #expect(actual.top2Margin.shape == expected.top2Margin.shape, "\(label) margin shape")
        #expect(actual.entropy.shape == expected.entropy.shape, "\(label) entropy shape")
        #expect(actual.top1TokenID.dtype == .int32)
        #expect(actual.top1Logprob.dtype == .float32 && actual.top2Margin.dtype == .float32 && actual.entropy.dtype == .float32)

        let idsMatch = (actual.top1TokenID .== expected.top1TokenID).all().item(Bool.self)
        #expect(idsMatch, "\(label) top1TokenID mismatch")

        let logprobErr = abs(actual.top1Logprob - expected.top1Logprob).max().item(Float.self)
        let marginDiff = MLX.where(
            isInf(actual.top2Margin) .&& isInf(expected.top2Margin) .&& (actual.top2Margin .== expected.top2Margin),
            MLXArray(Float(0)),
            abs(actual.top2Margin - expected.top2Margin))
        let marginErr = marginDiff.max().item(Float.self)
        let entropyErr = abs(actual.entropy - expected.entropy).max().item(Float.self)

        #expect(logprobErr <= logprobTolerance, "\(label) logprob error \(logprobErr)")
        #expect(marginErr <= marginTolerance, "\(label) margin error \(marginErr)")
        #expect(entropyErr <= entropyTolerance, "\(label) entropy error \(entropyErr)")
    }

        @Test(
        "Parity across float32, float16 and bfloat16 on tiny setBytes (3) and device vocab sizes 256, 4096, 151936 (Qwen) and 262144 (Gemma 4) at 1 and 8 rows",
        arguments: [3, 256, 4096, 151_936, 262_144], [DType.float32, .float16, .bfloat16]
    )
    func vocabAndDTypeParity(vocab: Int, dtype: DType) {
        for rows in [1, 8] {
            let seed = UInt64(vocab * 13 + rows * 7)
            let logits = (MLXRandom.normal([rows, vocab], key: MLXRandom.key(seed)) * 3.5).asType(dtype)
            eval(logits)
            Self.assertParity(logits, label: "V=\(vocab) rows=\(rows) dtype=\(dtype)")
        }
    }

    @Test("Tied maxima across simdgroups pick the lowest token index with zero margin, and uniform rows give log(V) entropy")
    func tiesAndUniformDistributions() {
        var row0 = [Float](repeating: -2.0, count: 256)
        row0[19] = 4.25
        row0[211] = 4.25
        let row1 = [Float](repeating: 1.75, count: 256)
        var row2 = [Float](repeating: -5.0, count: 256)
        row2[33] = 6.0
        row2[129] = 6.0
        row2[254] = 6.0
        let batch = MLXArray(row0 + row1 + row2, [3, 256])
        for dtype in [DType.float32, .float16, .bfloat16] {
            let typed = batch.asType(dtype)
            Self.assertParity(typed, label: "ties-\(dtype)")
            let out = FusedLogitConfidence(typed)
            #expect(out.top1TokenID.asArray(Int32.self) == [19, 0, 33])
            let margins = out.top2Margin.asArray(Float.self)
            #expect(margins == [0, 0, 0])
            let entropies = out.entropy.asArray(Float.self)
            #expect(abs(entropies[1] - log(Float(256))) < 1e-4)
        }
    }

    @Test("Cross-simdgroup ties on Gemma 4 262144-token vocabulary preserve lowest index and zero margin")
    func largeVocabTie() {
        let indices = MLXArray(0 ..< Int32(262_144))
        let base = MLX.where((indices .== 7) .|| (indices .== 260_003), MLXArray(Float(8.0)), MLXArray(Float(-4.0))).reshaped(1, 262_144)
        Self.assertParity(base.asType(.bfloat16), label: "262k-tie")
        let out = FusedLogitConfidence(base.asType(.bfloat16))
        #expect(out.top1TokenID.item(Int32.self) == 7)
        #expect(out.top2Margin.item(Float.self) == 0)
    }

    @Test("Masked -inf logits, large shift invariance, 1D/3D shape preservation, non-finite rows and scalar statistics helper")
    func edgeCasesAndShapes() throws {
        let masked = MLXArray([
            -Float.infinity, 2.0, -Float.infinity, 0.5,
            -Float.infinity, -Float.infinity, 3.25, -Float.infinity,
        ], [2, 4])
        Self.assertParity(masked, label: "masked")
        let singleUnmasked = FusedLogitConfidence(masked[1])
        #expect(singleUnmasked.top1TokenID.ndim == 0 && singleUnmasked.top1TokenID.item(Int32.self) == 2)
        #expect(singleUnmasked.top1Logprob.item(Float.self) == 0)
        #expect(singleUnmasked.top2Margin.item(Float.self).isInfinite)
        #expect(singleUnmasked.entropy.item(Float.self) == 0)

        let unshifted = MLXRandom.normal([2, 4, 256], key: MLXRandom.key(91)).asType(.float16).asType(.float32)
        let shifted = unshifted + MLXArray(Float(512.0))
        Self.assertParity(shifted, label: "3d-shifted")
        let a = FusedLogitConfidence(unshifted)
        let b = FusedLogitConfidence(shifted)
        #expect(a.entropy.shape == [2, 4])
        #expect(abs(a.entropy - b.entropy).max().item(Float.self) < 1e-5)
        #expect(abs(a.top1Logprob - b.top1Logprob).max().item(Float.self) < 1e-5)

        let nonFinite = MLXArray([
            Float.nan, 1.0, 0.0, 2.0,
            Float.infinity, 1.0, 0.0, 2.0,
            -Float.infinity, -Float.infinity, -Float.infinity, -Float.infinity,
        ], [3, 4])
        let bad = FusedLogitConfidence(nonFinite)
        #expect(bad.top1Logprob.asArray(Float.self).allSatisfy { $0.isNaN })
        #expect(bad.top2Margin.asArray(Float.self).allSatisfy { $0.isNaN })
        #expect(bad.entropy.asArray(Float.self).allSatisfy { $0.isNaN })
        #expect(FusedLogitConfidence.statistics(logits: nonFinite[0]) == nil)
        #expect(FusedLogitConfidence.statistics(logits: nonFinite[1]) == nil)
        #expect(FusedLogitConfidence.statistics(logits: nonFinite[2]) == nil)

        let sampleRow = MLXRandom.normal([4096], key: MLXRandom.key(490))
        let fusedStats = try #require(ConfidenceProbeRecorder.statistics(logits: sampleRow))
        let refOut = FusedLogitConfidence.reference(sampleRow)
        let refProb = Foundation::exp(Double(refOut.top1Logprob.item(Float.self)))
        let refSecond = Foundation::exp(Double((refOut.top1Logprob - refOut.top2Margin).item(Float.self)))
        #expect(abs(fusedStats.probability - refProb) < 1e-5)
        #expect(abs(fusedStats.entropy - Double(refOut.entropy.item(Float.self))) < 1e-4)
        #expect(abs(fusedStats.margin - (refProb - refSecond)) < 1e-5)
        let edtEntropy = EntropyDynamicTemperature.entropy(sampleRow.reshaped(1, -1))
        #expect(edtEntropy.shape == [1, 1])
        #expect(abs(edtEntropy.item(Float.self) - refOut.entropy.item(Float.self)) < 1e-4)
    }

    @Test(
        "End-to-end forward logits from TinyModel Qwen3 and Gemma 4 match stock MLX across float32, float16 and bfloat16 (#228)",
        arguments: [TinyModel.Family.qwen3, .gemma4]
    )
    func tinyModelEndToEndParity(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let prompt = MLXArray([Int32(3), 17, 42, 88, 129, 204], [1, 6])
        let cache = try model.newCache(parameters: nil)
        let prefillLogits = model(prompt, cache: cache)
        let decodeLogits = model(MLXArray([Int32(55)], [1, 1]), cache: cache)[0..., -1, 0...]
        eval(prefillLogits, decodeLogits)

        for dtype in [DType.float32, .float16, .bfloat16] {
            Self.assertParity(prefillLogits.asType(dtype), label: "\(family.rawValue)-prefill-\(dtype)")
            Self.assertParity(decodeLogits.asType(dtype), label: "\(family.rawValue)-decode-\(dtype)")
        }
    }
}
