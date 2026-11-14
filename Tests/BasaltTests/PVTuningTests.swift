import ArgumentParser
import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("PV-Tuning Phase 0 unit tests: 2/3/4-bit affine and nf4 code round-trips, differentiable dequantization, KL descent on hybrid and MoE fixtures, and cancellation safety (#410)", .serialized)
struct PVTuningTests {
    static func teacherBatch(from teacher: any LanguageModel, tokens: [Int32], topK: Int = 16) -> PVTeacherBatch {
        let logits = teacher(LMInput.Text(tokens: MLXArray(tokens)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
        let logp = logits - logSumExp(logits, axis: -1, keepDims: true)
        let ids = argPartition(-logp, kth: topK - 1, axis: -1)[0..., 0 ..< topK]
        let vals = takeAlong(logp, ids, axis: -1)
        eval(ids, vals)
        return PVTeacherBatch(
            tokens: tokens, topIDs: ids.asArray(Int32.self), topLogprobs: vals.asArray(Float.self),
            seqLen: tokens.count, topK: topK)
    }

    @Test("2-bit, 3-bit, 4-bit affine and 4-bit nf4 pack/unpack round-trips are bit-identical on 2D and 3D expert tensors and match MLX dequantized within 1e-6")
    func codeRoundTripAndDequant() throws {
        MLXRandom.seed(410)
        for shape in [[12, 64], [4, 8, 64]] {
            let w = MLXRandom.normal(shape).asType(.float32)
            for (bits, mode) in [(2, QuantizationMode.affine), (3, .affine), (4, .affine), (4, .nf4)] {
                let (packed, scales, biases) = MLX.quantized(w, groupSize: 32, bits: bits, mode: mode)
                let codes = PVCodeGrid.unpackCodes(packed, bits: bits)
                let repacked = PVCodeGrid.packCodes(codes, bits: bits)
                eval(packed, repacked)
                #expect(packed.asArray(UInt32.self) == repacked.asArray(UInt32.self))

                let ref = dequantized(packed, scales: scales, biases: biases, groupSize: 32, bits: bits, mode: mode).asType(.float32)
                let diff = PVCodeGrid.dequantizeDifferentiable(
                    codes: codes, scales: scales, biases: biases, groupSize: 32, mode: mode, dtype: scales.dtype).asType(.float32)
                let maxErr = abs(ref - diff).max()
                eval(maxErr)
                #expect(maxErr.item(Float.self) < 1e-6)
            }
        }
    }

    @Test("PV-Tuning lowers global teacher KL across 2-bit affine, 3-bit affine and nf4 on TinyModel.qwen35")
    func klDescentOnHybridQwen35() throws {
        for (bits, mode) in [(2, QuantizationMode.affine), (3, .affine), (4, .nf4)] {
            let teacher = try TinyModel.make(.qwen35)
            let student = try TinyModel.make(.qwen35)
            quantize(model: student as Module, groupSize: 32, bits: bits, mode: mode) { path, _ in
                PVTuner.textBlockIndex(path) != nil
            }
            eval(student as Module)

            var trainBatches: [PVTeacherBatch] = []
            for b in 0 ..< 4 {
                let toks: [Int32] = (0 ..< 24).map { Int32((b * 29 + $0 * 7 + 3) % 240) }
                trainBatches.append(Self.teacherBatch(from: teacher, tokens: toks))
            }
            var valBatches: [PVTeacherBatch] = []
            for v in 0 ..< 2 {
                let valToks: [Int32] = (0 ..< 24).map { Int32((v * 31 + $0 * 11 + 17) % 240) }
                valBatches.append(Self.teacherBatch(from: teacher, tokens: valToks))
            }

            var cfg = PVTuneConfig()
            cfg.globalContinuousSteps = 6
            cfg.continuousStepsPerGroup = 3
            cfg.batchesPerDiscreteStep = 4
            cfg.continuousLR = 1e-3
            cfg.discreteFractions = [0.02, 0.005, 0.001, 0.0002]
            cfg.groupSizeBlocks = 2

            let summary = try PVTuner.tune(student, train: trainBatches, val: valBatches, config: cfg)
            #expect(summary.trainKLAfter < summary.trainKLBefore)
            #expect(summary.valKLAfter <= summary.valKLBefore)
            #expect(summary.totalUpdatedCodes > 0)
        }
    }

    @Test("PV-Tuning updates QuantizedSwitchLinear MoE experts and lowers teacher KL on TinyModel.gemma4Moe")
    func klDescentOnGemma4MoE() throws {
        let teacher = try TinyModel.make(.gemma4Moe)
        let student = try TinyModel.make(.gemma4Moe)
        quantize(model: student as Module, groupSize: 32, bits: 2, mode: .affine) { path, mod in
            guard PVTuner.textBlockIndex(path) != nil else { return false }
            if let lin = mod as? Linear { return lin.weight.dim(-1).isMultiple(of: 32) }
            return mod is SwitchLinear
        }
        eval(student as Module)

        var trainBatches: [PVTeacherBatch] = []
        for b in 0 ..< 4 {
            let toks: [Int32] = (0 ..< 20).map { Int32((b * 19 + $0 * 5 + 1) % 240) }
            trainBatches.append(Self.teacherBatch(from: teacher, tokens: toks))
        }
        let valToks: [Int32] = (0 ..< 20).map { Int32(($0 * 13 + 9) % 240) }
        let valBatches = [Self.teacherBatch(from: teacher, tokens: valToks)]

        var cfg = PVTuneConfig()
        cfg.globalContinuousSteps = 6
        cfg.continuousStepsPerGroup = 3
        cfg.batchesPerDiscreteStep = 4
        cfg.continuousLR = 1e-3
        cfg.discreteFractions = [0.01, 0.002, 0.0005]
        cfg.groupSizeBlocks = 2

        let summary = try PVTuner.tune(student, train: trainBatches, val: valBatches, config: cfg)
        #expect(summary.trainKLAfter < summary.trainKLBefore)
        #expect(summary.valKLAfter <= summary.valKLBefore)
        #expect(summary.totalUpdatedCodes > 0)
    }

    @Test("bfloat16 QuantizedLinear dtypes and monotone valKL are preserved under both normal and rollback steps")
    func bfloat16DTypeAndValKLInvariant() throws {
        let teacher = try TinyModel.make(.qwen35)
        let student = try TinyModel.make(.qwen35)
        let root = student as Module
        let bf16Params = root.parameters().flattened().map { k, v in
            (k, v.dtype.isFloatingPoint && !k.hasSuffix("A_log") ? v.asType(.bfloat16) : v)
        }
        root.update(parameters: ModuleParameters.unflattened(bf16Params))
        quantize(model: root, groupSize: 32, bits: 3, mode: .affine) { path, _ in
            PVTuner.textBlockIndex(path) != nil
        }
        eval(root)
        let beforeDTypes = Dictionary(uniqueKeysWithValues: root.parameters().flattened().map { ($0.0, $0.1.dtype) })

        var trainBatches: [PVTeacherBatch] = []
        for b in 0 ..< 4 {
            let toks: [Int32] = (0 ..< 20).map { Int32((b * 23 + $0 * 7 + 5) % 240) }
            trainBatches.append(Self.teacherBatch(from: teacher, tokens: toks))
        }
        let valToks: [Int32] = (0 ..< 20).map { Int32(($0 * 17 + 3) % 240) }
        let valBatches = [Self.teacherBatch(from: teacher, tokens: valToks)]

        var badCfg = PVTuneConfig()
        badCfg.globalContinuousSteps = 2
        badCfg.continuousStepsPerGroup = 2
        badCfg.continuousLR = 10.0
        badCfg.discreteFractions = [0.5]
        badCfg.groupSizeBlocks = 2

        let rollbackSummary = try PVTuner.tune(student, train: trainBatches, val: valBatches, config: badCfg)
        let afterDTypes = Dictionary(uniqueKeysWithValues: root.parameters().flattened().map { ($0.0, $0.1.dtype) })
        #expect(beforeDTypes == afterDTypes)
        #expect(rollbackSummary.valKLAfter <= rollbackSummary.valKLBefore + 1e-5)
    }

    @Test("Cancelling a PV-Tuning Task restores all quantized modules, freezes parameters and exits training mode")
    func cancellationLeavesQuantizedModelIntact() async throws {
        let teacher = try TinyModel.make(.qwen3)
        let toks: [Int32] = (0 ..< 16).map { Int32($0 + 2) }
        let batch = Self.teacherBatch(from: teacher, tokens: toks)

        let task = Task { () throws -> (Bool, Bool, Bool, Bool) in
            let student = try TinyModel.make(.qwen3)
            let root = student as Module
            quantize(model: root, groupSize: 32, bits: 3, mode: .affine) { path, _ in
                PVTuner.textBlockIndex(path) != nil
            }
            eval(root)
            withUnsafeCurrentTask { $0?.cancel() }
            var cancelled = false
            do {
                _ = try PVTuner.tune(student, train: [batch], val: [batch], config: PVTuneConfig())
            } catch is CancellationError {
                cancelled = true
            }
            let hasTempWrapper = root.namedModules().contains { $0.1 is PVScaleLinear }
            return (cancelled, root.training, root.trainableParameters().flattened().isEmpty, hasTempWrapper)
        }
        let (cancelled, isTraining, trainableEmpty, hasTempWrapper) = try await task.value
        #expect(cancelled)
        #expect(!isTraining)
        #expect(trainableEmpty)
        #expect(!hasTempWrapper)
    }
}

@Suite("PV-Tuning conversion, shard rewrite and CLI validation (#410)", .serialized)
struct PVTuningConvertTests {
    @Test("collectTeacherBatches, requantizeModel, writeTunedShards with language_model prefix, and ConvertSubcommand CLI validation")
    func shardRewriteRequantAndCLIValidation() async throws {
        let teacherContainer = try TinyModel.container(.qwen3Small)
        let batches = await PVTuner.collectTeacherBatches(
            teacherContainer, texts: PVTuner.defaultCalibrationCorpus, count: 4, seqLen: 16, topK: 16)
        #expect(batches.count == 4)
        #expect(batches[0].seqLen == 16)

        let moe = try TinyModel.make(.gemma4Moe) as Module
        quantize(model: moe, groupSize: 32, bits: 4, mode: .affine) { path, mod in
            guard PVTuner.textBlockIndex(path) != nil else { return false }
            if let lin = mod as? Linear { return lin.weight.dim(-1).isMultiple(of: 32) }
            return mod is SwitchLinear
        }
        try PVTuner.requantizeModel(moe, bits: 3, groupSize: 32, mode: .affine)
        let switchBits = (moe.namedModules().first { $0.1 is QuantizedSwitchLinear }?.1 as? QuantizedSwitchLinear)?.bits
        #expect(switchBits == 3)

        let dir = FileManager.default.temporaryDirectory.appending(component: "pv-shards-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let student = try TinyModel.make(.qwen3Small)
        let root = student as Module
        quantize(model: root, groupSize: 32, bits: 3, mode: .affine) { path, _ in PVTuner.textBlockIndex(path) != nil }
        eval(root)
        let flat = Dictionary(uniqueKeysWithValues: root.parameters().flattened())
        var prefixed: [String: MLXArray] = [:]
        for (k, v) in flat { prefixed["language_model.\(k)"] = v + MLXArray(0, dtype: v.dtype) }
        eval(Array(prefixed.values))
        let shardURL = dir.appending(component: "model.safetensors")
        try save(arrays: prefixed, metadata: ["format": "mlx"], url: shardURL)

        var cfg = PVTuneConfig()
        cfg.globalContinuousSteps = 4
        cfg.continuousStepsPerGroup = 2
        cfg.batchesPerDiscreteStep = 2
        cfg.continuousLR = 1e-3
        cfg.discreteFractions = [0.02, 0.005]
        cfg.groupSizeBlocks = 2
        let studentContainer = ModelContainer(context: ModelContext(
            configuration: ModelConfiguration(id: "tiny-student"),
            model: student,
            processor: ByteInputProcessor(),
            tokenizer: ByteTokenizer()))
        let summary = try await PVTuner.tuneLoadedStudent(studentContainer, into: dir, allBatches: batches, config: cfg)
        #expect(summary.rewrittenTensors > 0)
        #expect(summary.valKLAfter <= summary.valKLBefore + 1e-5)
        await #expect(throws: BasaltError.self) {
            _ = try await PVTuner.tuneLoadedStudent(studentContainer, into: dir, allBatches: [batches[0]], config: cfg)
        }

        let reloaded = try loadArrays(url: shardURL)
        #expect(reloaded.count == prefixed.count)
        #expect(try ConvertSubcommand.resolveModelDirectory(dir.path).standardizedFileURL == dir.standardizedFileURL)
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.resolveModelDirectory("/nonexistent/basalt-pv-\(UUID().uuidString)")
        }

        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--pv-teacher", "t"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--dequantize", "bf16", "--experimental-pv-tune"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--format", "hf", "--experimental-pv-tune"])
        }
        let valid = try ConvertSubcommand.parse(["src", "--out", "dst", "--quantize", "affine:3:64", "--experimental-pv-tune", "--pv-teacher", "t"])
        #expect(valid.experimentalPvTune)
        #expect(valid.pvTeacher == "t")
    }
}
