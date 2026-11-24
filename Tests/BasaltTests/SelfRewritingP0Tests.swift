import Foundation
import Hummingbird
import HummingbirdTesting
import HTTPTypes
import MLX
import MLXLMCommon
import MLXNN
import NIOCore
import Testing
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer

@Suite("Self-rewriting P0: WeightCopy isolation, EditAdapter 5 ops, validator, save/load, and #675 serving (#676)", .serialized)
struct SelfRewritingP0Tests {
    static let oProj = "model.layers.1.self_attn.o_proj"
    static let downProj = "model.layers.1.mlp.down_proj"
    static let qkvProj = "model.layers.0.linear_attn.in_proj_qkv"
    static let dtBias = "model.layers.0.linear_attn.dt_bias"
    static let aLog = "model.layers.0.linear_attn.A_log"

    static var promptTokens: MLXArray {
        MLXArray(TinyModel.tokens(Conversation.firstTurn).map(Int32.init)).reshaped(1, -1)
    }

    static func makeQuantizedQwen35(dtype: DType = .float16) throws -> any LanguageModel {
        let model = try TinyModel.make(.qwen35)
        model.update(parameters: model.parameters().mapValues { $0.asType(dtype) })
        quantize(model: model, groupSize: 32, bits: 4)
        try model.prepare()
        eval(model)
        return model
    }

    static func logitsFloat(_ model: any LanguageModel) throws -> [Float] {
        let cache = try model.newCache(parameters: nil)
        let out = model(promptTokens, cache: cache).asType(.float32)
        eval(out)
        return out.asArray(Float.self)
    }

    static func logitsBits(_ model: any LanguageModel) throws -> [UInt32] {
        try logitsFloat(model).map(\.bitPattern)
    }

    static func unitVector(size: Int, seed: UInt64 = 676) -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let v = MLXRandom.normal([size]).asType(.float32)
            let norm = sqrt((v * v).sum())
            let unit = v / norm
            eval(unit)
            return unit
        }
    }

    static func temporaryDirectory(_ prefix: String) -> URL {
        FileManager.default.temporaryDirectory.appending(path: "\(prefix)-\(UUID().uuidString)")
    }
}

extension SelfRewritingP0Tests {
    @Test("WeightCopy matches original logits bit for bit before edits, isolates every op from the original, and leaves original parameter fingerprints unchanged")
    func weightCopyIsolatesAllFiveOps() throws {
        let original = try TinyModel.make(.qwen35)
        let originalLogits = try Self.logitsBits(original)
        var copy = try WeightCopy.makeSync(from: original)

        let preEditCopyLogits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        #expect(preEditCopyLogits == originalLogits)

        let u = Self.unitVector(size: 64, seed: 101)
        let rOut = Self.unitVector(size: 64, seed: 102)
        let rIn = Self.unitVector(size: 128, seed: 103)
        let shiftVals = MLXArray([Float(0.5), Float(-0.25)])
        let replacementRows = withRandomState(MLXRandom.RandomState(seed: 104)) {
            MLXRandom.normal([2, 64]).asType(.float32) * 0.1
        }

        let ops: [(EditProgram, [String: MLXArray])] = [
            (EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u", alpha: 0.25)]), ["u": u]),
            (EditProgram(ops: [.rank1(module: Self.downProj, outputKey: "o", inputKey: "i", scale: 0.2)]), ["o": rOut, "i": rIn]),
            (EditProgram(ops: [.shift(tensor: Self.dtBias, start: 1, count: 2, valueKey: "s")]), ["s": shiftVals]),
            (EditProgram(ops: [.clip(tensor: "model.layers.3.mlp.down_proj.weight", factor: 0.5)]), [:]),
            (EditProgram(ops: [.rows(tensor: "model.layers.3.mlp.up_proj.weight", rowIndices: [0, 3], valuesKey: "r")]), ["r": replacementRows]),
        ]

        var prevCopyLogits = preEditCopyLogits
        for (program, tensors) in ops {
            _ = try copy.apply(program, tensors: tensors, verifying: original)
            try copy.verifyOriginal(original)
            #expect(try Self.logitsBits(original) == originalLogits)
            let nextCopyLogits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
            #expect(nextCopyLogits != originalLogits)
            #expect(nextCopyLogits != prevCopyLogits)
            prevCopyLogits = nextCopyLogits
        }

        try copy.release(verifying: original)
        #expect(try Self.logitsBits(original) == originalLogits)
    }

    @Test("A deliberate Module.update on a shared array handle in WeightCopy mutates the original and trips the fingerprint check")
    func deliberateSharedMutationTripsFingerprint() throws {
        let original = try TinyModel.make(.qwen35)
        let copy = try WeightCopy.makeSync(from: original)
        let target = Self.oProj + ".weight"
        let copyModel = try copy.model
        let sharedWeight = try #require(copyModel.parameters().flattened().first(where: { $0.0 == target })?.1)
        copyModel.update(parameters: ModuleParameters.unflattened([(target, sharedWeight + 1)]))
        eval(copyModel)

        let error = #expect(throws: WeightCopyError.self) {
            try copy.verifyOriginal(original)
        }
        guard case .originalFingerprintMismatch(let mismatches)? = error else {
            Issue.record("expected originalFingerprintMismatch, got \(String(describing: error))")
            return
        }
        #expect(mismatches.contains { $0.name == target && $0.reason == "content hash changed" })
    }

    @Test("WeightCopy extra byte accounting equals edited tensors plus fused projections on float and 4-bit Qwen3.5, and refuses edits past budget")
    func extraBytesAccountingAndBudgetEnforcement() throws {
        let floatOriginal = try TinyModel.make(.qwen35)
        var floatCopy = try WeightCopy.makeSync(from: floatOriginal)
        #expect(floatCopy.accounting.editedBytes == 0)
        #expect(floatCopy.accounting.fusedProjectionBytes == 0)
        #expect(floatCopy.accounting.totalExtraBytes == 0)

        let dtBytes = Int64(try #require(floatOriginal.parameters().flattened().first(where: { $0.0 == Self.dtBias })?.1.nbytes))
        let shiftProgram = EditProgram(ops: [.shift(tensor: Self.dtBias, start: 0, count: 4, valueKey: "v")])
        let plannedShift = try floatCopy.planBudget(for: shiftProgram)
        _ = try floatCopy.apply(shiftProgram, tensors: ["v": MLXArray([Float(0.5)])], verifying: floatOriginal)
        #expect(floatCopy.accounting.editedBytes == dtBytes)
        #expect(floatCopy.accounting.totalExtraBytes == plannedShift.totalExtraBytes)

        let dampProgram = EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u", alpha: 0.1)])
        let plannedDamp = try floatCopy.planBudget(for: dampProgram)
        _ = try floatCopy.apply(dampProgram, tensors: ["u": Self.unitVector(size: 64)], verifying: floatOriginal)
        #expect(floatCopy.accounting.editedBytes == dtBytes + Int64((64 + 64) * 4))
        #expect(floatCopy.accounting.totalExtraBytes == plannedDamp.totalExtraBytes)
        try floatCopy.release(verifying: floatOriginal)

        let q4Original = try Self.makeQuantizedQwen35()
        var q4Copy = try WeightCopy.makeSync(from: q4Original)
        let q4FusedBefore = q4Copy.accounting.fusedProjectionBytes
        #expect(q4FusedBefore > 0 && q4Copy.accounting.editedBytes == 0)

        let downModule = try #require(q4Original.namedModules().first(where: { $0.0 == Self.downProj })?.1)
        let downBytes = Int64(TensorPacker.bytes(downModule))
        let clipProgram = EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 2.0)])
        let plannedClip = try q4Copy.planBudget(for: clipProgram)
        _ = try q4Copy.apply(clipProgram, tensors: [:], verifying: q4Original)
        #expect(q4Copy.accounting.editedBytes == downBytes)
        #expect(q4Copy.accounting.fusedProjectionBytes == q4FusedBefore)
        #expect(q4Copy.accounting.totalExtraBytes == plannedClip.totalExtraBytes)
        try q4Copy.release(verifying: q4Original)

        var tightCopy = try WeightCopy.makeSync(from: floatOriginal, budgetBytes: ByteCount(100))
        #expect(throws: EditValidationError.self) {
            _ = try tightCopy.apply(EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 2.0)]), tensors: [:], verifying: floatOriginal)
        }
        #expect(tightCopy.accounting.editedBytes == 0)
        try tightCopy.release(verifying: floatOriginal)
    }
}

extension SelfRewritingP0Tests {
    @Test("Consuming release() drops all references to the copy module tree and its owned arrays")
    func consumingReleaseDropsReferences() throws {
        let original = try TinyModel.make(.qwen35)
        weak var weakModel: Module?
        weak var weakEditedTensor: MLXArray?

        do {
            var copy = try WeightCopy.makeSync(from: original)
            _ = try copy.apply(
                EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 1.5)]),
                tensors: [:],
                verifying: original
            )
            let copyModel = try copy.model
            weakModel = copyModel
            weakEditedTensor = copyModel.parameters().flattened().first(where: { $0.0 == Self.downProj + ".weight" })?.1
            #expect(weakModel != nil)
            #expect(weakEditedTensor != nil)
            try copy.release(verifying: original)
        }

        #expect(weakModel == nil)
        #expect(weakEditedTensor == nil)
    }

    @Test("WeightCopy builds from a ModelContainer and clones an already-adapted model cleanly")
    func weightCopyFromContainerAndAdaptedModel() async throws {
        let container = try TinyModel.container(.qwen35)
        let copy = try await WeightCopy.make(from: container)
        let containerLogits = try await container.perform { context in
            try Self.logitsBits(context.model)
        }
        let copyLogits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        #expect(copyLogits == containerLogits)
        try await copy.verifyOriginal(in: container)
        try copy.release()

        let adaptedOriginal = try TinyModel.make(.qwen35)
        let prior = try EditAdapter(
            program: EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u", alpha: 0.2)]),
            tensors: ["u": Self.unitVector(size: 64, seed: 701)]
        )
        try adaptedOriginal.load(adapter: prior)
        let adaptedLogits = try Self.logitsBits(adaptedOriginal)
        var copyFromAdapted = try WeightCopy.makeSync(from: adaptedOriginal)
        let clonedLogits = try copyFromAdapted.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        #expect(clonedLogits == adaptedLogits)
        _ = try copyFromAdapted.apply(
            EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u2", alpha: 0.1)]),
            tensors: ["u2": Self.unitVector(size: 64, seed: 702)],
            verifying: adaptedOriginal
        )
        #expect(try Self.logitsBits(adaptedOriginal) == adaptedLogits)
        try copyFromAdapted.release(verifying: adaptedOriginal)
    }

    @Test("rank1Damp matches dense W - alpha * u * (u^T W) within 1e-5 on float weights and is exact against itself on 4-bit weights")
    func rank1DampMathParityFloatAnd4Bit() throws {
        let original = try TinyModel.make(.qwen35)
        var copy = try WeightCopy.makeSync(from: original)
        let alpha: Float = 0.35
        let u = Self.unitVector(size: 64, seed: 201)
        let program = EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u", alpha: alpha)])
        _ = try copy.apply(program, tensors: ["u": u], verifying: original)
        let copyLogits = try copy.logits(for: Self.promptTokens).asType(.float32)

        let denseRef = try TinyModel.make(.qwen35)
        let origLinear = try #require(denseRef.namedModules().first(where: { $0.0 == Self.oProj })?.1 as? Linear)
        let w = origLinear.weight.asType(.float32)
        let expectedW = (w - alpha * matmul(u.reshaped(64, 1), matmul(u.reshaped(1, 64), w))).asType(origLinear.weight.dtype)
        try MLXInferenceEngine.replace(Self.oProj, with: Linear(weight: expectedW, bias: origLinear.bias), in: denseRef)
        let refLogits = try Self.logitsFloat(denseRef)
        let maxDiff = zip(copyLogits.asArray(Float.self), refLogits).map { abs($0 - $1) }.max() ?? .infinity
        #expect(maxDiff < 1e-5, "maxDiff=\(maxDiff)")
        try copy.release(verifying: original)

        let q4Orig = try Self.makeQuantizedQwen35()
        var q4Copy = try WeightCopy.makeSync(from: q4Orig)
        let q4LayerBefore = try #require(q4Orig.namedModules().first(where: { $0.0 == Self.oProj })?.1 as? QuantizedLinear)
        let x = withRandomState(MLXRandom.RandomState(seed: 202)) { MLXRandom.normal([1, 4, 64]).asType(.float16) }
        let baseY = q4LayerBefore(x).asType(.float32)
        let expectedY = baseY - alpha * (baseY * u.reshaped(1, 1, 64)).sum(axis: -1, keepDims: true) * u.reshaped(1, 1, 64)
        eval(expectedY)

        _ = try q4Copy.apply(program, tensors: ["u": u], verifying: q4Orig)
        let q4CopyLayer = try #require((try q4Copy.model).namedModules().first(where: { $0.0 == Self.oProj })?.1 as? Linear)
        let actualY = q4CopyLayer(x).asType(.float32)
        eval(actualY)
        #expect(abs(actualY - expectedY).max().item(Float.self) < 1e-2)

        var q4CopySecond = try WeightCopy.makeSync(from: q4Orig)
        _ = try q4CopySecond.apply(program, tensors: ["u": u], verifying: q4Orig)
        let logitsA = try q4Copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        let logitsB = try q4CopySecond.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        #expect(logitsA == logitsB)
        try q4Copy.release(verifying: q4Orig)
        try q4CopySecond.release(verifying: q4Orig)
    }
}

extension SelfRewritingP0Tests {
    @Test("shift mutates only its targeted slice on Qwen3.5 dt_bias and A_log, leaving other entries bit-identical")
    func shiftMutatesOnlySliceOnQwen35() throws {
        let original = try TinyModel.make(.qwen35)
        var copy = try WeightCopy.makeSync(from: original)

        let origDt = try #require(original.parameters().flattened().first(where: { $0.0 == Self.dtBias })?.1.asArray(Float.self))
        let origALog = try #require(original.parameters().flattened().first(where: { $0.0 == Self.aLog })?.1.asArray(Float.self))
        #expect(origDt.count == 4 && origALog.count == 4)

        let program = EditProgram(ops: [
            .shift(tensor: Self.dtBias, start: 1, count: 2, valueKey: "dt_delta"),
            .shift(tensor: Self.aLog, start: 2, count: 1, valueKey: "a_delta"),
        ])
        _ = try copy.apply(
            program,
            tensors: ["dt_delta": MLXArray([Float(0.5), Float(-0.25)]), "a_delta": MLXArray([Float(0.75)])],
            verifying: original
        )

        let copyModel = try copy.model
        let editedDt = try #require(copyModel.parameters().flattened().first(where: { $0.0 == Self.dtBias })?.1.asArray(Float.self))
        let editedALog = try #require(copyModel.parameters().flattened().first(where: { $0.0 == Self.aLog })?.1.asArray(Float.self))

        #expect(editedDt[0].bitPattern == origDt[0].bitPattern)
        #expect(abs(editedDt[1] - (origDt[1] + 0.5)) < 1e-6)
        #expect(abs(editedDt[2] - (origDt[2] - 0.25)) < 1e-6)
        #expect(editedDt[3].bitPattern == origDt[3].bitPattern)
        #expect(editedALog[0].bitPattern == origALog[0].bitPattern)
        #expect(editedALog[1].bitPattern == origALog[1].bitPattern)
        #expect(abs(editedALog[2] - (origALog[2] + 0.75)) < 1e-6)
        #expect(editedALog[3].bitPattern == origALog[3].bitPattern)
        try copy.release(verifying: original)
    }

    @Test("clip median handles odd and even counts exactly, and unsorted rows replace only their target indices")
    func medianAbsAndUnsortedRowsExactness() throws {
        let oddMedian = try EditOpExecutor.medianAbs(MLXArray([Float(-10), Float(2), Float(-5), Float(1), Float(8)]))
        let evenMedian = try EditOpExecutor.medianAbs(MLXArray([Float(-10), Float(2), Float(-6), Float(1)]))
        eval(oddMedian, evenMedian)
        #expect(abs(oddMedian.item(Float.self) - 5.0) < 1e-6)
        #expect(abs(evenMedian.item(Float.self) - 4.0) < 1e-6)

        let original = try TinyModel.make(.qwen35)
        var copy = try WeightCopy.makeSync(from: original)
        let origWeight = try #require((original.namedModules().first(where: { $0.0 == Self.oProj })?.1 as? Linear)?.weight.asArray(Float.self))
        let rowA = Array(repeating: Float(0.25), count: 64)
        let rowB = Array(repeating: Float(-0.75), count: 64)
        let program = EditProgram(ops: [.rows(tensor: Self.oProj + ".weight", rowIndices: [5, 2], valuesKey: "r")])
        _ = try copy.apply(program, tensors: ["r": MLXArray(rowA + rowB, [2, 64])], verifying: original)

        let editedWeight = try #require(((try copy.model).namedModules().first(where: { $0.0 == Self.oProj })?.1 as? Linear)?.weight.asArray(Float.self))
        #expect(Array(editedWeight[5 * 64 ..< 6 * 64]) == rowA)
        #expect(Array(editedWeight[2 * 64 ..< 3 * 64]) == rowB)
        #expect(Array(editedWeight[0 ..< 64]) == Array(origWeight[0 ..< 64]))
        #expect(Array(editedWeight[3 * 64 ..< 4 * 64]) == Array(origWeight[3 * 64 ..< 4 * 64]))
        try copy.release(verifying: original)
    }

    @Test("clip and rows on 4-bit weights report rmse within requantization bound, preserve scale dtype, and no-op round-trip control gives baseline error")
    func clipRowsAndNoOpControlOn4Bit() throws {
        let q4Orig = try Self.makeQuantizedQwen35(dtype: .float16)
        let origLogits = try Self.logitsFloat(q4Orig)
        var copy = try WeightCopy.makeSync(from: q4Orig)

        let noopMetrics = try copy.apply(EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 1000.0)]), tensors: [:], verifying: q4Orig)
        #expect(noopMetrics.count == 1)
        #expect(noopMetrics[0].rmse.isFinite && noopMetrics[0].rmse >= 0 && noopMetrics[0].rmse < 1e-3)
        let noopLogits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self)
        let noopMaxLogitDelta = zip(origLogits, noopLogits).map { abs($0 - $1) }.max() ?? .infinity
        try copy.undoAll(verifying: q4Orig)

        let clipMetrics = try copy.apply(EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 0.75)]), tensors: [:], verifying: q4Orig)
        let clipLogits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self)
        let clipMaxLogitDelta = zip(origLogits, clipLogits).map { abs($0 - $1) }.max() ?? .infinity
        #expect(clipMetrics[0].rmse.isFinite && clipMetrics[0].rmse > noopMetrics[0].rmse && clipMetrics[0].relativeError < 0.25)
        #expect(clipMaxLogitDelta > noopMaxLogitDelta)
        let clippedLayer = try #require((try copy.model).namedModules().first(where: { $0.0 == Self.downProj })?.1 as? QuantizedLinear)
        #expect(clippedLayer.bits == 4 && clippedLayer.groupSize == 32 && clippedLayer.scales.dtype == .float16)

        let rowsData = withRandomState(MLXRandom.RandomState(seed: 303)) { MLXRandom.normal([3, 128]).asType(.float32) * 0.2 }
        let rowsMetrics = try copy.apply(EditProgram(ops: [.rows(tensor: Self.downProj + ".weight", rowIndices: [0, 5, 11], valuesKey: "rows")]), tensors: ["rows": rowsData], verifying: q4Orig)
        #expect(rowsMetrics[0].rmse.isFinite && rowsMetrics[0].rmse > 0 && rowsMetrics[0].relativeError < 0.25)
        try copy.release(verifying: q4Orig)
    }
}

extension SelfRewritingP0Tests {
    @Test("Undo restores rank-1, float shift, 4-bit requantized clip/rows, and rank-1-then-clip bit for bit on WeightCopy and EditAdapter")
    func undoIsBitExactAcrossAllOps() throws {
        let q4Orig = try Self.makeQuantizedQwen35(dtype: .float16)
        let origLogits = try Self.logitsBits(q4Orig)
        let origFingerprint = ParameterFingerprint.capture(of: q4Orig)
        var copy = try WeightCopy.makeSync(from: q4Orig)

        let u = Self.unitVector(size: 64, seed: 401)
        let rowsData = withRandomState(MLXRandom.RandomState(seed: 402)) { MLXRandom.normal([2, 128]).asType(.float32) * 0.1 }
        let program = EditProgram(ops: [
            .rank1Damp(module: Self.oProj, vectorKey: "u", alpha: 0.2),
            .shift(tensor: Self.dtBias, start: 0, count: 2, valueKey: "dt"),
            .clip(tensor: Self.downProj + ".weight", factor: 1.0),
            .rows(tensor: "model.layers.3.mlp.down_proj.weight", rowIndices: [1, 4], valuesKey: "rows"),
        ])
        let tensors: [String: MLXArray] = ["u": u, "dt": MLXArray([Float(0.5), Float(-0.5)]), "rows": rowsData]

        _ = try copy.apply(program, tensors: tensors, verifying: q4Orig)
        let editedLogits = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        #expect(editedLogits != origLogits)

        _ = try copy.apply(EditProgram(ops: [.clip(tensor: Self.oProj + ".weight", factor: 1.0)]), tensors: [:], verifying: q4Orig)
        try copy.undoLast(verifying: q4Orig)
        #expect(try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == editedLogits)

        try copy.undoLast(verifying: q4Orig)
        #expect(try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == origLogits)
        try copy.release(verifying: q4Orig)

        let adapter = try EditAdapter(program: program, tensors: tensors)
        let directModel = try Self.makeQuantizedQwen35(dtype: .float16)
        let directBefore = try Self.logitsBits(directModel)
        try directModel.load(adapter: adapter)
        #expect(try Self.logitsBits(directModel) == editedLogits)
        #expect(adapter.lastMetrics(for: directModel).count == 4)
        directModel.unload(adapter: adapter)
        #expect(try Self.logitsBits(directModel) == directBefore)
        #expect(ParameterFingerprint.capture(of: directModel).aggregateHash == origFingerprint.aggregateHash)
    }

    static func makeSampleEditAdapter(scale: Float = 5.0) throws -> (adapter: EditAdapter, rOut: MLXArray, rIn: MLXArray) {
        let rOut = Self.unitVector(size: 64, seed: 501)
        let rIn = Self.unitVector(size: 64, seed: 502)
        let dtDelta = MLXArray([Float(0.5), Float(-0.25)])
        let program = EditProgram(
            sessionId: "sess-p0-001",
            baseModelFingerprint: "fp-qwen35",
            approval: .approved,
            pairedTest: PairedTestSummary(
                problemCount: 100,
                baseAccuracy: 0.50,
                editedAccuracy: 0.57,
                standardError: 0.02,
                wrongToRight: 9,
                rightToWrong: 2
            ),
            ops: [
                .rank1(module: Self.oProj, outputKey: "out", inputKey: "in", scale: scale),
                .shift(tensor: Self.dtBias, start: 1, count: 2, valueKey: "dt"),
            ]
        )
        let adapter = try EditAdapter(program: program, tensors: ["out": rOut, "in": rIn, "dt": dtDelta])
        return (adapter, rOut, rIn)
    }

    @Test("EditAdapter saves with 0700 directory and 0600 file permissions, round-trips edit.json + edit.safetensors, and fuses permanently")
    func saveLoadRoundTripAndPermissions() throws {
        let dir = Self.temporaryDirectory("edit-adapter-save")
        defer { try? FileManager.default.removeItem(at: dir) }

        let (adapter, _, _) = try Self.makeSampleEditAdapter()
        try adapter.save(to: dir)

        let dirPerms = try #require(try FileManager.default.attributesOfItem(atPath: dir.path)[.posixPermissions] as? Int)
        let jsonPerms = try #require(try FileManager.default.attributesOfItem(atPath: dir.appending(component: EditAdapter.programFile).path)[.posixPermissions] as? Int)
        #expect((dirPerms & 0o777) == 0o700)
        #expect((jsonPerms & 0o777) == 0o600)
        #expect(ServedAdapterSource.isAdapterDirectory(dir))
        #expect(ServedAdapterSource.hasAdapterFiles(["edit.json", "edit.safetensors"]))
        #expect(EditAdapter.isEditAdapterDirectory(dir))

        let reloaded = try EditAdapter.from(directory: dir)
        #expect(reloaded.program == adapter.program)

        let model = try TinyModel.make(.qwen35)
        #expect(throws: BasaltError.self) {
            _ = try MLXInferenceEngine.attachAnyAdapterDirectory(dir, to: model)
        }
        try reloaded.fuse(with: model)
        #expect(MLXInferenceEngine.loraModules(of: model).isEmpty)
    }
}

extension SelfRewritingP0Tests {
    @Test("Served EditAdapter via #675 registry and X-Basalt-Adapter matches overlay logits, isolates cache, refuses in-flight switch, and restores base")
    func servedEditAdapterMatchesOverlayLogits() async throws {
        let dir = Self.temporaryDirectory("edit-adapter-serve")
        defer { try? FileManager.default.removeItem(at: dir) }

        let (adapter, rOut, rIn) = try Self.makeSampleEditAdapter(scale: 5.0)
        try adapter.save(to: dir)

        let overlayEngine = try await TinyModel.engine(.qwen35)
        await overlayEngine.skipJournalForDeltaTests()
        let overlayContainer = try #require(await overlayEngine.container)
        _ = try await overlayEngine.applyAdapter(.rankOne(module: Self.oProj, input: rIn.asArray(Float.self), output: rOut.asArray(Float.self), scale: 5.0), gate: nil)
        _ = try await overlayEngine.applyDelta(WeightDelta(tensor: Self.dtBias, slice: [1 ..< 3], values: [0.5, -0.25], scale: 1.0))
        let overlayLogits = try await WeightDeltaTests.logits(overlayContainer)
        let overlayResponse = try await overlayEngine.generate(request: greedy(Conversation.firstTurn, key: nil))

        let servedEngine = try await TinyModel.engine(.qwen35, promptCacheBytes: 1 << 30)
        let servedContainer = try #require(await servedEngine.container)
        let baseLogits = try await WeightDeltaTests.logits(servedContainer)
        let orchestrator = ModelOrchestratorActor(engine: servedEngine, maxTokens: 32, telemetry: TelemetryTracker())
        try await orchestrator.loadAdapters(["fix": dir])
        #expect(await orchestrator.adapterNames == ["fix"])

        try await servedEngine.useAdapter("fix")
        #expect(try await WeightDeltaTests.logits(servedContainer) == overlayLogits)

        let app = Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "tiny-qwen35")).makeRouter())
        let body = #"{"model": "tiny-qwen35", "messages": [{"role": "user", "content": "Hello"}], "temperature": 0, "max_tokens": 8}"#
        try await app.test(.router) { client in
            let headers: HTTPFields = [.contentType: "application/json", GenerationRequest.adapterHeader: "fix"]
            try await client.execute(uri: "/v1/chat/completions", method: .post, headers: headers, body: ByteBuffer(string: body)) { res in
                #expect(res.status == .ok)
            }
        }
        let servedFirst = try await orchestrator.generate(request: greedy(Conversation.firstTurn, key: "ns").withAdapter("fix"))
        let baseFirst = try await orchestrator.generate(request: greedy(Conversation.firstTurn, key: "ns"))
        #expect(try await WeightDeltaTests.logits(servedContainer) == baseLogits)
        let servedSecond = try await orchestrator.generate(request: greedy(Conversation.firstTurn, key: "ns").withAdapter("fix"))
        #expect(servedFirst.generatedText == overlayResponse.generatedText)
        #expect(baseFirst.generatedText != servedFirst.generatedText)
        #expect(baseFirst.cachedPromptTokens == 0 && servedSecond.cachedPromptTokens > 0)

        await servedEngine.markAdapterSwitchInFlight()
        await #expect(throws: BasaltError.self) { try await servedEngine.useAdapter(nil) }
    }

    @Test("Clip-only EditAdapter with zero operand tensors saves, loads, and produces distinct cache fingerprints when clip factor changes")
    func clipOnlyEditAdapterRoundTripAndFingerprint() throws {
        let dirA = Self.temporaryDirectory("edit-clip-a")
        let dirB = Self.temporaryDirectory("edit-clip-b")
        defer {
            try? FileManager.default.removeItem(at: dirA)
            try? FileManager.default.removeItem(at: dirB)
        }
        let adapterA = try EditAdapter(program: EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 1.0)]))
        let adapterB = try EditAdapter(program: EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 2.0)]))
        try adapterA.save(to: dirA)
        try adapterB.save(to: dirB)

        let loadedA = try EditAdapter.from(directory: dirA)
        #expect(loadedA.tensors.isEmpty)
        #expect(loadedA.program == adapterA.program)
        #expect(PromptCacheDiskStore.fingerprint(modelDirectory: dirA) != PromptCacheDiskStore.fingerprint(modelDirectory: dirB))
    }
}

extension SelfRewritingP0Tests {
    @Test("Validator refuses embeddings, final norm across LLM and VLM prefixes, and lm_head, while allowing per-layer linear_attn.norm slices")
    func validatorRefusesForbiddenTargets() throws {
        let forbiddenCases: [(String, EditForbiddenTargetKind)] = [
            ("model.embed_tokens.weight", .embedding),
            ("language_model.model.embed_tokens.weight", .embedding),
            ("word_embeddings.weight", .embedding),
            ("tok_embeddings.weight", .embedding),
            ("model.norm.weight", .finalNorm),
            ("norm.weight", .finalNorm),
            ("language_model.model.norm.weight", .finalNorm),
            ("model.language_model.final_layernorm.weight", .finalNorm),
            ("lm_head", .lmHead),
            ("lm_head.weight", .lmHead),
            ("language_model.lm_head.weight", .lmHead),
        ]

        for (target, expectedKind) in forbiddenCases {
            let program = EditProgram(ops: [.shift(tensor: target, start: 0, count: 1, valueKey: "v")])
            let error = #expect(throws: EditValidationError.self) {
                try EditProgramValidator.validate(program)
            }
            #expect(error == .forbiddenTarget(target: target, kind: expectedKind), "\(target)")
        }

        let perLayerNorm = EditProgram(ops: [.shift(tensor: "model.layers.0.linear_attn.norm.weight", start: 0, count: 4, valueKey: "v")])
        try EditProgramValidator.validate(
            perLayerNorm,
            tensorShapes: ["model.layers.0.linear_attn.norm.weight": [16]],
            tensorValues: ["v": [0.1, 0.1, 0.1, 0.1]]
        )
    }

    @Test("Validator refuses whole-tensor rescales, |alpha| > 1, non-finite values, max ops exceeded, extra tensors, and path traversal keys")
    func validatorRefusesNumericAndBoundaryViolations() {
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.shift(tensor: Self.oProj + ".scales", start: 0, count: 1, valueKey: "v")]))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(
                EditProgram(ops: [.shift(tensor: "model.layers.0.input_layernorm.weight", start: 0, count: 64, valueKey: "v")]),
                tensorShapes: ["model.layers.0.input_layernorm.weight": [64]]
            )
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(
                EditProgram(ops: [.rows(tensor: Self.downProj + ".weight", rowIndices: Array(0 ..< 64), valuesKey: "r")]),
                tensorShapes: [Self.downProj + ".weight": [64, 128]]
            )
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u", alpha: 1.01)]))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u", alpha: .nan)]))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.rank1(module: Self.oProj, outputKey: "o", inputKey: "i", scale: .infinity)]))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.shift(tensor: Self.dtBias, start: 0, count: 2, valueKey: "v")]), tensorValues: ["v": [0.1, .nan]])
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 0)]))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: []))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: Array(repeating: .clip(tensor: Self.downProj + ".weight", factor: 2.0), count: 5)), maxOps: 4)
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.rows(tensor: Self.downProj + ".weight", rowIndices: [1, 1], valuesKey: "r")]))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.shift(tensor: Self.dtBias, start: 3, count: 2, valueKey: "v")]), tensorShapes: [Self.dtBias: [4]])
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "../secret", alpha: 0.1)]))
        }
        #expect(throws: EditValidationError.self) {
            try EditProgramValidator.validate(EditProgram(ops: [.clip(tensor: Self.downProj + ".weight", factor: 2.0)]), tensorValues: ["smuggled": [1.0]])
        }
        #expect(throws: EditValidationError.self) {
            let badPaired = PairedTestSummary(problemCount: 10, baseAccuracy: .nan, editedAccuracy: 0.5, standardError: 0.1, wrongToRight: 1, rightToWrong: 0)
            try EditProgramValidator.validate(EditProgram(pairedTest: badPaired, ops: [.clip(tensor: Self.downProj + ".weight", factor: 2.0)]))
        }
    }

    @Test("Multiple stacked rank-1 ops on the same module accumulate accurately and undo back to the prior state")
    func stackedRankOneOpsAndIncrementalUndo() throws {
        let original = try TinyModel.make(.qwen35)
        var copy = try WeightCopy.makeSync(from: original)
        let u1 = Self.unitVector(size: 64, seed: 601)
        let u2 = Self.unitVector(size: 64, seed: 602)

        let p1 = EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u1", alpha: 0.2)])
        let p2 = EditProgram(ops: [.rank1Damp(module: Self.oProj, vectorKey: "u2", alpha: 0.3)])

        _ = try copy.apply(p1, tensors: ["u1": u1], verifying: original)
        let afterFirst = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern)
        let plannedSecond = try copy.planBudget(for: p2)
        _ = try copy.apply(p2, tensors: ["u2": u2], verifying: original)
        #expect(copy.accounting.totalExtraBytes == plannedSecond.totalExtraBytes)
        let afterSecond = try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self)

        let denseRef = try TinyModel.make(.qwen35)
        let origLinear = try #require(denseRef.namedModules().first(where: { $0.0 == Self.oProj })?.1 as? Linear)
        let w0 = origLinear.weight.asType(.float32)
        let w1 = w0 - 0.2 * matmul(u1.reshaped(64, 1), matmul(u1.reshaped(1, 64), w0))
        let w2 = w1 - 0.3 * matmul(u2.reshaped(64, 1), matmul(u2.reshaped(1, 64), w1))
        try MLXInferenceEngine.replace(Self.oProj, with: Linear(weight: w2.asType(origLinear.weight.dtype), bias: origLinear.bias), in: denseRef)
        let refLogits = try Self.logitsFloat(denseRef)
        #expect((zip(afterSecond, refLogits).map { abs($0 - $1) }.max() ?? .infinity) < 1e-4)

        try copy.undoLast(verifying: original)
        #expect(try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == afterFirst)
        try copy.undoLast(verifying: original)
        #expect(try copy.logits(for: Self.promptTokens).asType(.float32).asArray(Float.self).map(\.bitPattern) == (try Self.logitsBits(original)))
        try copy.release(verifying: original)
    }
}
