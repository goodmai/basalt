import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("MACaQ + PV-Tuning end-to-end discrete and continuous refinement (#543)", .serialized)
struct MACaQPVTuneTests {
    static func makeCalibrationAndTeacher(
        teacher: any LanguageModel, batchCount: Int = 4, valCount: Int = 2, seqLen: Int = 24
    ) -> (cal: [[Int32]], train: [PVTeacherBatch], val: [PVTeacherBatch]) {
        var cal: [[Int32]] = []
        var train: [PVTeacherBatch] = []
        for b in 0 ..< batchCount {
            let toks = (0 ..< seqLen).map { Int32((b * 29 + $0 * 7 + 3) % 200) }
            cal.append(toks)
            train.append(PVTuningTests.teacherBatch(from: teacher, tokens: toks))
        }
        var val: [PVTeacherBatch] = []
        for v in 0 ..< valCount {
            let toks = (0 ..< seqLen).map { Int32((v * 31 + $0 * 11 + 17) % 200) }
            val.append(PVTuningTests.teacherBatch(from: teacher, tokens: toks))
        }
        return (cal, train, val)
    }

    @Test("Differentiable Hadamard matches fused Metal kernel within 1e-5 and produces finite VJPs for x and pre_scale")
    func differentiableHadamardParityAndAutograd() {
        MLXRandom.seed(543)
        let x = MLXRandom.normal([6, 64]).asType(.float32)
        let pre = MLXRandom.uniform(low: 0.4, high: 1.8, [64]).asType(.float32)
        let metalOut = MACaQFusedKernels.blockHadamard(x, scaleVector: pre, blockSize: 32, isInverse: false)
        let diffOut = MACaQFusedKernels.differentiableBlockHadamard(x, scaleVector: pre, blockSize: 32)
        let maxErr = abs(metalOut - diffOut).max()
        eval(maxErr)
        #expect(maxErr.item(Float.self) < 1e-5)

        let vg = valueAndGrad({ arrays in
            let y = MACaQFusedKernels.differentiableBlockHadamard(
                arrays[0], scaleVector: arrays[1], blockSize: 32)
            return [y.square().sum()]
        }, argumentNumbers: [0, 1])
        let (_, grads) = vg([x, pre])
        let gxNorm = grads[0].square().sum()
        let gpNorm = grads[1].square().sum()
        eval(gxNorm, gpNorm)
        #expect(gxNorm.item(Float.self) > 0 && gxNorm.item(Float.self).isFinite)
        #expect(gpNorm.item(Float.self) > 0 && gpNorm.item(Float.self).isFinite)

        let xSw = MLXRandom.normal([3, 1, 1, 64]).asType(.float32)
        let preExp = MLXRandom.uniform(low: 0.5, high: 1.6, [4, 64]).asType(.float32)
        let idx = MLXArray([Int32(2), 1, 0, 3, 3, 2]).reshaped(3, 2)
        let vgSw = valueAndGrad({ arrays in
            let y = MACaQFusedKernels.differentiableSwitchHadamard(
                arrays[0], indices: idx, preScale: arrays[1], blockSize: 32)
            return [y.square().sum()]
        }, argumentNumbers: [0, 1])
        let (_, swGrads) = vgSw([xSw, preExp])
        let gpSwNorm = swGrads[1].square().sum()
        eval(gpSwNorm)
        #expect(gpSwNorm.item(Float.self) > 0 && gpSwNorm.item(Float.self).isFinite)
    }

    @Test("PVMACaQGrid MXFP4 2D and 3D pack/unpack round-trips bit-identically and discreteStep lowers linear proxy objective")
    func mxfp4GridRoundTripAndDiscreteStep() {
        MLXRandom.seed(5431)
        let spec = PVCodeGrid.Spec(groupSize: 32, bits: 4, mode: .mxfp4)
        for shape in [[8, 64], [3, 6, 64]] {
            let w = MLXRandom.normal(shape).asType(.bfloat16)
            let (wq, sc, _) = MLX.quantized(w, groupSize: 32, bits: 4, mode: .mxfp4)
            let codes = PVMACaQGrid.unpackCodes(wq, spec: spec)
            let repacked = PVMACaQGrid.packCodes(codes, spec: spec)
            let repackedTwice = PVMACaQGrid.packCodes(PVMACaQGrid.unpackCodes(repacked, spec: spec), spec: spec)
            let wBefore = MLX.dequantized(wq, scales: sc, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32)
            let wRepacked = MLX.dequantized(repacked, scales: sc, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32)
            let maxDeqDiff = abs(wBefore - wRepacked).max()
            eval(repacked, repackedTwice, maxDeqDiff)
            #expect(maxDeqDiff.item(Float.self) == 0)
            #expect(repacked.asArray(UInt32.self) == repackedTwice.asArray(UInt32.self))

            let grad = MLXRandom.normal(shape).asType(.float32)
            let step = PVMACaQGrid.discreteStep(codes: codes, scales: sc, grad: grad, spec: spec, fraction: 0.1)
            #expect(step.updated > 0)
            let wAfter = MLX.dequantized(
                PVMACaQGrid.packCodes(step.codes, spec: spec),
                scales: sc, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32)
            let linDelta = ((wAfter - wBefore) * grad).sum()
            eval(linDelta)
            #expect(linDelta.item(Float.self) < 0)
        }
    }

    @Test("PV-Tuning lowers KL on 2-bit/3-bit affine and 4-bit MXFP4 MACaQ Qwen3.5, preserves dtypes, and reloads bit-identically")
    func klDescentAndReloadOnMACaQQwen35() async throws {
        let modes: [MACaQTargetMode] = [
            .affine(bits: 2, groupSize: 32),
            .affine(bits: 3, groupSize: 32),
            .mxfp4,
        ]
        for targetMode in modes {
            let teacher = try TinyModel.make(.qwen35)
            let student = try TinyModel.make(.qwen35)
            let root = student as Module
            let teacherParams = Dictionary(uniqueKeysWithValues: (teacher as Module).parameters().flattened())
            root.update(parameters: ModuleParameters.unflattened(Array(teacherParams)))
            eval(root)

            let data = Self.makeCalibrationAndTeacher(teacher: teacher)
            let initCal: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 13 + 91) % 200) }]
            var macCfg = MACaQConfig()
            macCfg.hadamardBlock = 32
            macCfg.maxFlipIterations = 2
            let (_, manifest) = try MACaQRefiner.quantizeModel(
                student, targetMode: targetMode, calibrationTokenBatches: initCal, config: macCfg)
            let dtypesBefore = Dictionary(uniqueKeysWithValues: root.parameters().flattened().map { ($0.0, $0.1.dtype) })

            var pvCfg = PVTuneConfig()
            pvCfg.globalContinuousSteps = 6
            pvCfg.continuousStepsPerGroup = 3
            pvCfg.batchesPerDiscreteStep = 4
            pvCfg.continuousLR = 1e-3
            pvCfg.discreteFractions = [0.02, 0.005, 0.001, 0.0002]
            pvCfg.groupSizeBlocks = 2

            let summary = try PVTuner.tune(student, train: data.train, val: data.val, config: pvCfg)
            let dtypesAfter = Dictionary(uniqueKeysWithValues: root.parameters().flattened().map { ($0.0, $0.1.dtype) })
            #expect(dtypesBefore == dtypesAfter)
            #expect(summary.trainKLAfter < summary.trainKLBefore)
            #expect(summary.valKLAfter <= summary.valKLBefore)
            if targetMode.bits < 4 {
                #expect(summary.totalUpdatedCodes > 0)
            }
            try await Self.assertDirectoryReloadParity(student: student, manifest: manifest, alpha: macCfg.alpha)
        }
    }

    private static func assertDirectoryReloadParity(
        student: any LanguageModel, manifest: [String: MACaQLayerSpec], alpha: Float
    ) async throws {
        let root = student as Module
        let dir = FileManager.default.temporaryDirectory.appending(component: "macaq-pv-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let srcDir = dir.appending(component: "src")
        let outDir = dir.appending(component: "out")
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
        try Data(TinyModel.Family.qwen35.config.utf8).write(to: srcDir.appending(component: "config.json"))
        try Data("{}".utf8).write(to: srcDir.appending(component: "tokenizer_config.json"))
        _ = try MACaQRefiner.writeMACaQDirectory(
            model: root, manifest: manifest, alpha: alpha, source: srcDir, output: outDir)
        let rewritten = try PVTuner.writeTunedShards(from: root, into: outDir)
        #expect(rewritten > 0)

        let loaded = try #require(try await MACaQNativeLoader.loadContainer(
            from: outDir, tokenizerLoader: ByteTokenizerLoader()))
        let prompt: [Int32] = [5, 19, 42, 7]
        let direct = student(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
        eval(direct)
        let directArr = direct.asArray(Float.self)
        let diff = await loaded.perform { ctx in
            let rel = ctx.model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
            eval(rel)
            return zip(directArr, rel.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        }
        #expect(diff == 0)
    }

    @Test("PV-Tuning refines MACaQSwitchLinear MoE experts and CLI accepts --experimental-macaq with --experimental-pv-tune")
    func moeRefinementAndCLIValidation() throws {
        let parsed = try ConvertSubcommand.parse([
            "src", "--out", "dst", "--experimental-macaq", "affine:3:64",
            "--experimental-pv-tune", "--pv-teacher", "teacher-dir",
        ])
        #expect(parsed.experimentalMacaq == "affine:3:64")
        #expect(parsed.experimentalPvTune)
        #expect(parsed.pvTeacher == "teacher-dir")

        let teacher = try TinyModel.make(.gemma4Moe)
        let student = try TinyModel.make(.gemma4Moe)
        let root = student as Module
        let teacherParams = Dictionary(uniqueKeysWithValues: (teacher as Module).parameters().flattened())
        root.update(parameters: ModuleParameters.unflattened(Array(teacherParams)))
        eval(root)

        let data = Self.makeCalibrationAndTeacher(teacher: teacher, batchCount: 4, valCount: 1, seqLen: 20)
        var macCfg = MACaQConfig()
        macCfg.hadamardBlock = 32
        macCfg.maxFlipIterations = 2
        macCfg.minExpertTokens = 32
        _ = try MACaQRefiner.quantizeModel(
            student, targetMode: .affine(bits: 2, groupSize: 32), calibrationTokenBatches: data.cal, config: macCfg)
        #expect(root.namedModules().contains { $0.1 is MACaQSwitchLinear })

        var pvCfg = PVTuneConfig()
        pvCfg.globalContinuousSteps = 6
        pvCfg.continuousStepsPerGroup = 3
        pvCfg.batchesPerDiscreteStep = 4
        pvCfg.continuousLR = 1e-3
        pvCfg.discreteFractions = [0.01, 0.002, 0.0005]
        pvCfg.groupSizeBlocks = 2

        let summary = try PVTuner.tune(student, train: data.train, val: data.val, config: pvCfg)
        #expect(summary.trainKLAfter < summary.trainKLBefore)
        #expect(summary.valKLAfter <= summary.valKLBefore)
        #expect(summary.totalUpdatedCodes > 0)
    }
}
