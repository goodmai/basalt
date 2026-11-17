import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

enum MACaQCPUReference {
    static func scaledHadamard(
        x: [Float],
        scale: [Float],
        rows: Int,
        cols: Int,
        blockSize: Int,
        inverse: Bool
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows * cols)
        let norm = 1.0 / Float(blockSize).squareRoot()
        let blocks = cols / blockSize
        for r in 0 ..< rows {
            for b in 0 ..< blocks {
                let base = r * cols + b * blockSize
                let sBase = b * blockSize
                for i in 0 ..< blockSize {
                    var sum: Float = 0
                    for j in 0 ..< blockSize {
                        let pop = (i & j).nonzeroBitCount
                        let hSign: Float = (pop & 1) == 0 ? 1.0 : -1.0
                        let xv = inverse ? x[base + j] : (x[base + j] * scale[sBase + j])
                        sum += xv * hSign
                    }
                    var res = sum * norm
                    if inverse { res *= scale[sBase + i] }
                    out[base + i] = res
                }
            }
        }
        return out
    }

    static func assertLinearParity(
        wDense: MLXArray,
        pre: MLXArray,
        bias: MLXArray,
        spec: MACaQLayerSpec,
        tol: Float
    ) {
        let cols = wDense.dim(1)
        let qMode = spec.quantizationMode
        let (wq, scales, biases) = MLX.quantized(
            wDense.asType(.bfloat16), groupSize: spec.groupSize, bits: spec.bits, mode: qMode)
        let wDeq = MLX.dequantized(
            wq, scales: scales, biases: biases,
            groupSize: spec.groupSize, bits: spec.bits, mode: qMode, dtype: .float32)
        let layer = MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: wq, scales: scales, biases: biases),
            bias: bias, preScale: pre, spec: spec)
        for rows in [1, 4, 17] {
            let x = MLXRandom.normal([rows, cols], scale: 0.5, key: MLXRandom.key(UInt64(53530 + rows))).asType(.bfloat16)
            eval(x, wDeq)
            let xRotHost = scaledHadamard(
                x: x.asType(.float32).asArray(Float.self),
                scale: pre.asType(.float32).asArray(Float.self),
                rows: rows, cols: cols, blockSize: spec.hadamardBlock, inverse: false)
            let ref = matmul(MLXArray(xRotHost, [rows, cols]), wDeq.T) + bias.asType(.float32)
            let got = layer(x).asType(.float32)
            let err = abs(got - ref).max() / maximum(abs(ref).max(), Float(1e-4))
            eval(err)
            #expect(err.item(Float.self) <= tol)
        }
    }

    static func assertSwitchParity() {
        let (experts, outs, cols) = (4, 12, 128)
        let w3D = MLXRandom.normal([experts, outs, cols], scale: 0.25, key: MLXRandom.key(53540)).asType(.bfloat16)
        let pre2D = MLXRandom.uniform(low: 0.5, high: 1.6, [experts, cols], key: MLXRandom.key(53541)).asType(.bfloat16)
        var (wqList, scList, biList, deqList) = ([MLXArray](), [MLXArray](), [MLXArray](), [MLXArray]())
        for e in 0 ..< experts {
            let q = MLX.quantized(w3D[e], groupSize: 32, bits: 4, mode: .affine)
            wqList.append(q.wq)
            scList.append(q.scales)
            if let b = q.biases { biList.append(b) }
            deqList.append(MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 32, bits: 4, mode: .affine, dtype: .float32))
        }
        var spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .affine, hadamardBlock: 32)
        spec.perExpertPreScale = true
        let sw = MACaQSwitchLinear(
            geometry: MACaQSwitchGeometry(experts: experts, outputs: outs, inputs: cols),
            packed: MACaQPackedWeights(weight: stacked(wqList), scales: stacked(scList), biases: stacked(biList)),
            bias: nil, preScale: pre2D, spec: spec)
        let x2D = MLXRandom.normal([3, cols], scale: 0.5, key: MLXRandom.key(53542)).asType(.bfloat16)
        let gotUnsorted = sw(expandedDimensions(x2D, axes: [-2, -3]), MLXArray([Int32(2), 1, 0, 3, 3, 2]).reshaped(3, 2)).asType(.float32)
        let gotSorted = sw(expandedDimensions(x2D, axis: -2), MLXArray([Int32(0), 2, 3]), sortedIndices: true).asType(.float32)
        eval(gotUnsorted, gotSorted)
        let pairs = [[2, 1], [0, 3], [3, 2]]
        for token in 0 ..< 3 {
            let xHost = x2D[token].asType(.float32).asArray(Float.self)
            for slot in 0 ..< 2 {
                let expIdx = pairs[token][slot]
                let xRot = scaledHadamard(
                    x: xHost, scale: pre2D[expIdx].asType(.float32).asArray(Float.self),
                    rows: 1, cols: cols, blockSize: 32, inverse: false)
                let ref = matmul(MLXArray(xRot, [1, cols]), deqList[expIdx].T)
                let err = abs(gotUnsorted[token, slot] - ref).max() / maximum(abs(ref).max(), Float(1e-4))
                eval(err)
                #expect(err.item(Float.self) <= 6e-2)
            }
            let sExp = [0, 2, 3][token]
            let sRot = scaledHadamard(
                x: xHost, scale: pre2D[sExp].asType(.float32).asArray(Float.self),
                rows: 1, cols: cols, blockSize: 32, inverse: false)
            let sRef = matmul(MLXArray(sRot, [1, cols]), deqList[sExp].T)
            let sErr = abs(gotSorted[token] - sRef).max() / maximum(abs(sRef).max(), Float(1e-4))
            eval(sErr)
            #expect(sErr.item(Float.self) <= 6e-2)
        }
    }
}

@Suite("MACaQ kernel and TinyModel parity (#535)", .serialized)
struct MACaQKernelParityTests {
        @Test("Single-simdgroup barrier-free blockHadamard instantiates every (T, TS, N, IS_INV) template and matches CPU FP32 Sylvester reference")
        func singleSimdHadamardMatchesCPU() {
            for (bIdx, blockSize) in [1, 2, 4, 8, 16, 32, 64, 128].enumerated() {
                let cols = max(blockSize * 2, 64)
                let rows = 5
                let k1 = MLXRandom.key(UInt64(53500 + bIdx * 3))
                let k2 = MLXRandom.key(UInt64(53501 + bIdx * 3))
                let k3 = MLXRandom.key(UInt64(53502 + bIdx * 3))
                let xF32 = MLXRandom.normal([rows, cols], key: k1).asType(.float32)
                let pF32 = (MLXRandom.uniform(low: 0.25, high: 2.5, [cols], key: k2).asType(.float32)
                    * which(MLXRandom.uniform(low: 0.0, high: 1.0, [cols], key: k3) .< Float(0.5), MLXArray(Float(-1)), MLXArray(Float(1))))
                eval(xF32, pF32)
                for dtype in [DType.float32, .float16, .bfloat16] {
                    let xIn = xF32.asType(dtype)
                    eval(xIn)
                    let xQuantHost = xIn.asType(.float32).asArray(Float.self)
                    let scaleDTypes: [DType] = (blockSize == 64) ? [.float32, .float16, .bfloat16] : [dtype]
                    for sType in scaleDTypes {
                        let pIn = pF32.asType(sType)
                        eval(pIn)
                        let pQuantHost = pIn.asType(.float32).asArray(Float.self)
                        let tol: Float = (dtype == .float32 && sType == .float32)
                            ? 1e-5 : ((dtype == .bfloat16 || sType == .bfloat16) ? 4e-2 : 5e-3)
                        for isInv in [false, true] {
                            let ref = MACaQCPUReference.scaledHadamard(
                                x: xQuantHost, scale: pQuantHost,
                                rows: rows, cols: cols, blockSize: blockSize, inverse: isInv)
                            let got = MACaQFusedKernels.blockHadamard(
                                xIn, scaleVector: pIn, blockSize: blockSize, isInverse: isInv).asType(.float32)
                            eval(got)
                            let maxDiff = zip(ref, got.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 0
                            #expect(maxDiff <= tol)
                        }
                    }
                }
            }
        }

        @Test("MACaQQuantizedLinear and MACaQSwitchLinear match CPU FP32 rotated reference across affine 2/3/4-bit group {32,64,128} and mxfp4")
        func quantizedModulesMatchCPUReference() {
            let cols = 256
            let outs = 13
            let wDense = MLXRandom.normal([outs, cols], scale: 0.25, key: MLXRandom.key(53520)).asType(.float32)
            let pre = MLXRandom.uniform(low: 0.4, high: 1.8, [cols], key: MLXRandom.key(53521)).asType(.bfloat16)
            let bias = MLXRandom.normal([outs], scale: 0.05, key: MLXRandom.key(53522)).asType(.bfloat16)
            eval(wDense, pre, bias)
            for blockSize in [32, 64] {
                for bits in [2, 3, 4] {
                    for groupSize in [32, 64, 128] {
                        let spec = MACaQLayerSpec(bits: bits, groupSize: groupSize, mode: .affine, hadamardBlock: blockSize)
                        MACaQCPUReference.assertLinearParity(wDense: wDense, pre: pre, bias: bias, spec: spec, tol: 8e-2)
                    }
                }
                let mxSpec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .mxfp4, hadamardBlock: blockSize)
                MACaQCPUReference.assertLinearParity(wDense: wDense, pre: pre, bias: bias, spec: mxSpec, tol: 8e-2)
            }
            MACaQCPUReference.assertSwitchParity()
        }

        @Test("MACaQ M=1 decode matches prefill greedy token IDs across TinyModel qwen3, gemma4, qwen35, and qwen3Moe")
        func tinyModelGreedyTokenParity() throws {
            for family in [TinyModel.Family.qwen3, .gemma4, .qwen35, .qwen3Moe] {
                let model = try TinyModel.make(family)
                let batches: [[Int32]] = [
                    (0 ..< 16).map { Int32(($0 * 7 + 3) % 200) },
                    (0 ..< 16).map { Int32(($0 * 11 + 5) % 200) },
                ]
                var cfg = MACaQConfig()
                cfg.hadamardBlock = 32
                cfg.maxFlipIterations = 8
                _ = try MACaQRefiner.quantizeModel(
                    model, targetMode: .affine(bits: 4, groupSize: 32),
                    calibrationTokenBatches: batches, config: cfg)
                let prompt: [Int32] = [5, 19, 42, 7, 88, 13, 64, 21, 9, 31]
                let prefillOut = model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: nil, state: nil)
                let singleCache = try model.newCache(parameters: nil)
                var state: LMOutput.State?
                var stepLogits = MLXArray.zeros([1])
                for tok in prompt {
                    let out = model(LMInput.Text(tokens: MLXArray([tok])[.newAxis]), cache: singleCache, state: state)
                    state = out.state
                    stepLogits = out.logits[0, -1]
                }
                let prefillLast = prefillOut.logits[0, -1]
                let tokBatch = argMax(prefillLast, axis: -1)
                let tokStep = argMax(stepLogits, axis: -1)
                let maxLogitDiff = abs(prefillLast.asType(.float32) - stepLogits.asType(.float32)).max()
                eval(tokBatch, tokStep, maxLogitDiff)
                #expect(tokBatch.item(Int32.self) == tokStep.item(Int32.self))
                #expect(maxLogitDiff.item(Float.self) < 5e-2)
            }
        }
}

@Suite("MACaQ Hessian flip-refinement and MXFP4 screening (#535)", .serialized)
struct MACaQRefinementMathTests {
        @Test("MXFP4 pack/unpack round-trips bit-identically against MLX dequantized and H-weighted exponent screen reduces weighted error on outlier blocks")
        func mxfp4RoundTripAndScreen() {
            var wHost = (0 ..< (8 * 64)).map { Float(($0 % 17) - 8) * 0.18 }
            wHost[0] = 5.4
            wHost[64] = -5.6
            let w = MLXArray(wHost, [8, 64])
            var hdHost = [Float](repeating: 1.0, count: 64)
            hdHost[0] = 0.01
            let hd = MLXArray(hdHost)

            let base = MACaQMath.mxfp4RTNSignedIndex(weight: w, exponentOffset: 1)
            let screened = MACaQMath.screenMXFP4Exponents(weight: w, hessianDiag: hd, offsets: [1, 0, -1])
            let packed = MACaQMath.packMXFP4(sidx: screened.sidx, e8m0: screened.e8m0)
            let mlxDeq = MLX.dequantized(
                packed.weight, scales: packed.scales, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4, dtype: .float32)
            let unpacked = MACaQMath.unpackMXFP4(weight: packed.weight, scales: packed.scales)
            let levels = MLXArray(MACaQMath.mxfp4SignedLevels)
            let manualDeq = levels.take(unpacked.sidx) * unpacked.scaleElem
            let roundTripErr = abs(mlxDeq - manualDeq).max()
            let baseDeq = levels.take(base.sidx) * base.scaleElem
            let baseWeightedErr = ((w - baseDeq).square() * hd.reshaped(1, 64)).sum()
            let screenWeightedErr = ((w - manualDeq).square() * hd.reshaped(1, 64)).sum()
            eval(roundTripErr, baseWeightedErr, screenWeightedErr)
            #expect(roundTripErr.item(Float.self) == 0)
            #expect(screenWeightedErr.item(Float.self) <= baseWeightedErr.item(Float.self))
        }

        @Test("Active-set greedy flipRefine strictly lowers quadratic Hessian loss across 2/3/4-bit affine, nf4, and mxfp4")
        func flipRefineMonotonicAcrossFormats() {
            let w = MLXRandom.normal([24, 64], key: MLXRandom.key(53550)).asType(.float32)
            let x = MLXRandom.normal([96, 64], key: MLXRandom.key(53551)).asType(.float32)
                * MLXRandom.uniform(low: 0.3, high: 4.0, [1, 64], key: MLXRandom.key(53552))
            let rawH = matmul(x.T, x) * Float(1.0 / 96.0)
            let frame = MACaQMath.transformFrame(weight: w, hessian: rawH, preScale: nil, blockSize: nil, ridge: 0.25)

            for bits in [2, 3, 4] {
                let (wq, sc, bi) = MLX.quantized(frame.weight, groupSize: 32, bits: bits, mode: .affine)
                let spec = PVCodeGrid.Spec(groupSize: 32, bits: bits, mode: .affine)
                var state = MACaQMath.makeGridState(packedWeight: wq, scales: sc, biases: bi, spec: spec).state
                let res = MACaQMath.flipRefine(state: &state, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: 64)
                #expect(res.flips > 0)
                #expect(res.lossAfter < res.lossBefore)
            }
            let (nfCodes, nfScales) = NF4Metal.quantized(frame.weight, groupSize: 32)
            var nfState = MACaQMath.makeGridState(
                packedWeight: nfCodes, scales: nfScales, biases: nil, spec: PVCodeGrid.Spec(nf4GroupSize: 32)).state
            let nfRes = MACaQMath.flipRefine(state: &nfState, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: 64)
            #expect(nfRes.flips > 0)
            #expect(nfRes.lossAfter < nfRes.lossBefore)

            let scr = MACaQMath.screenMXFP4Exponents(weight: frame.weight, hessianDiag: diagonal(frame.hessian))
            var mxState = MACaQGridState(
                codes: scr.sidx, scaleElem: scr.scaleElem, biasElem: nil,
                levelTable: MLXArray(MACaQMath.mxfp4SignedLevels), maxLevel: 14)
            let mxRes = MACaQMath.flipRefine(state: &mxState, targetWeight: frame.weight, hessian: frame.hessian, maxIterations: 64)
            #expect(mxRes.lossAfter <= mxRes.lossBefore)
        }

        @Test("MACaQ-R in-place refinement lowers unseen held-out teacher KL over RTN on anisotropic TinyModel.qwen35 and MoE TinyModel.gemma4Moe with cold-expert SoftCoverage")
        func macaqRefineLowersHeldOutKL() throws {
            let normScales = MLXArray((0 ..< 64).map { idx -> Float in
                (idx % 8 == 0) ? 6.0 : ((idx % 4 == 0) ? 2.5 : 0.35)
            })
            for (fIdx, family) in [TinyModel.Family.qwen35, .gemma4Moe].enumerated() {
                let teacher = try TinyModel.make(family)
                var initUpdates: [(String, MLXArray)] = []
                for (pIdx, pair) in (teacher as Module).parameters().flattened().sorted(by: { $0.0 < $1.0 }).enumerated() {
                    let (key, val) = pair
                    if key.contains("layernorm.weight"), val.shape == [64] {
                        initUpdates.append((key, normScales.asType(val.dtype)))
                    } else if val.dtype.isFloatingPoint, val.ndim >= 2 {
                        let scale = Float(1.0) / Float(max(1, val.dim(-1))).squareRoot()
                        let det = MLXRandom.normal(
                            val.shape, dtype: val.dtype, scale: scale,
                            key: MLXRandom.key(UInt64(53600 + fIdx * 500 + pIdx)))
                        initUpdates.append((key, det))
                    }
                }
                (teacher as Module).update(parameters: ModuleParameters.unflattened(initUpdates))
                eval(teacher as Module)

                let student = try TinyModel.make(family)
                let teacherParams = Dictionary(uniqueKeysWithValues: (teacher as Module).parameters().flattened())
                (student as Module).update(parameters: ModuleParameters.unflattened(Array(teacherParams)))
                quantize(model: student as Module, groupSize: 32, bits: 2, mode: .affine) { path, mod in
                    guard PVTuner.textBlockIndex(path) != nil, !path.contains("router") else { return false }
                    if let lin = mod as? Linear { return lin.weight.dim(-1).isMultiple(of: 32) }
                    return mod is SwitchLinear
                }
                eval(student as Module)
                var calBatches: [[Int32]] = []
                for b in 0 ..< 8 {
                    var row: [Int32] = []
                    for idx in 0 ..< 24 { row.append(Int32((b * 23 + idx * 7 + 3) % 120)) }
                    calBatches.append(row)
                }
                var valBatches: [PVTeacherBatch] = []
                for v in 0 ..< 2 {
                    var row: [Int32] = []
                    for idx in 0 ..< 24 { row.append(Int32(120 + ((v * 31 + idx * 13 + 11) % 120))) }
                    valBatches.append(PVTuningTests.teacherBatch(from: teacher, tokens: row))
                }
                let klBefore = PVTuner.meanKL(student, valBatches)
                var cfg = MACaQConfig()
                cfg.minExpertTokens = 80
                cfg.maxFlipIterations = 64
                cfg.ridge = 0.25
                let summary = try MACaQRefiner.refineModel(
                    student, calibrationTokenBatches: calBatches, config: cfg,
                    teacherWeight: { teacherParams["\($0).weight"] })
                let klAfter = PVTuner.meanKL(student, valBatches)
                #expect(summary.refinedLayers > 0)
                #expect(summary.totalFlips > 0)
                #expect(summary.meanLossRatio < 1.0)
                #expect(klAfter < klBefore)
            }
        }
}

@Suite("MACaQ directory conversion, loader, and CLI validation (#535)", .serialized)
struct MACaQDirectoryCLITests {
        @Test("WeightConverter mxfp4 + MACaQ-R, source validation guards, and full MACaQ directory conversion round-trip through MACaQNativeLoader")
        func directoryRoundTripAndCLIValidation() async throws {
            #expect(throws: (any Error).self) {
                _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--macaq-block", "64"])
            }
            #expect(throws: (any Error).self) {
                _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-macaq", "affine:5:64"])
            }
            let parsed = try ConvertSubcommand.parse([
                "src", "--out", "dst", "--quantize", "mxfp4", "--experimental-macaq-refine",
            ])
            #expect(parsed.experimentalMacaqRefine)
            #expect(try WeightConverter.precision(quantize: "mxfp4", dequantize: nil, format: .mlx) == .mxfp4)

            let scratch = try WeightConverterTests.Scratch()
            let mxfp4Out = scratch.url("mxfp4")
            try WeightConverter(precision: .mxfp4).run(source: scratch.source, output: mxfp4Out)
            let mxfp4Weights = try WeightConverterTests.weights(mxfp4Out)
            #expect(mxfp4Weights[WeightConverterTests.attention + ".scales"]?.dtype == .uint8)
            let denseReader = try MACaQShardTeacherReader(directory: scratch.source)
            let mxfp4Reader = try MACaQShardTeacherReader(directory: mxfp4Out)
            let wSrc = try #require(try denseReader.denseWeight(for: "language_model." + WeightConverterTests.attention))
            let wMxfp4 = try #require(try mxfp4Reader.denseWeight(for: WeightConverterTests.attention))
            #expect(wSrc.shape == wMxfp4.shape && wSrc.dtype == .float32)

            let dir = FileManager.default.temporaryDirectory.appending(component: "basalt-macaq-test-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: dir) }
            let bertDir = dir.appending(component: "bert")
            try FileManager.default.createDirectory(at: bertDir, withIntermediateDirectories: true)
            try Data(#"{"model_type":"bert"}"#.utf8).write(to: bertDir.appending(component: "config.json"))
            await #expect(throws: BasaltError.self) {
                try await MACaQRefiner.validateSourceDirectory(bertDir)
            }

            let srcDir = dir.appending(component: "src")
            let outDir = dir.appending(component: "out")
            try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
            try Data(TinyModel.Family.qwen35.config.utf8).write(to: srcDir.appending(component: "config.json"))
            try Data("{}".utf8).write(to: srcDir.appending(component: "tokenizer_config.json"))

            let modelDirect = try TinyModel.make(.qwen35)
            let batches: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 11 + 5) % 180) }]
            var cfg = MACaQConfig()
            cfg.hadamardBlock = 32
            cfg.maxFlipIterations = 16
            let (summary, manifest) = try MACaQRefiner.quantizeModel(
                modelDirect, targetMode: .affine(bits: 3, groupSize: 32), calibrationTokenBatches: batches, config: cfg)
            #expect(summary.refinedLayers > 0 && manifest.count == summary.refinedLayers)
            let written = try MACaQRefiner.writeMACaQDirectory(
                model: modelDirect as Module, manifest: manifest, alpha: cfg.alpha, source: srcDir, output: outDir)
            #expect(written.shards == 1 && written.bytes > 0)
            await #expect(throws: BasaltError.self) {
                try await MACaQRefiner.validateSourceDirectory(outDir)
            }

            let report = await ModelInspector().inspect(modelIdOrPath: outDir.path)
            #expect(report.intact && report.quantizationMode == "macaq")

            let container = try #require(try await MACaQNativeLoader.loadContainer(
                from: outDir, tokenizerLoader: ByteTokenizerLoader()))
            let promptTokens: [Int32] = [4, 17, 33, 91]
            let directLogits = modelDirect(LMInput.Text(tokens: MLXArray(promptTokens)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
            eval(directLogits)
            let directValues = directLogits.asArray(Float.self)
            let reloadMaxDiff = await container.perform { ctx in
                let rLogits = ctx.model(LMInput.Text(tokens: MLXArray(promptTokens)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
                eval(rLogits)
                let rValues = rLogits.asArray(Float.self)
                return zip(directValues, rValues).map { abs($0 - $1) }.max() ?? 1
            }
            #expect(reloadMaxDiff == 0)
        }
}

@Suite("MACaQ pre-quantized hybrid and FluxKlein DiT loader round-trip (#538)", .serialized)
struct MACaQMixedAndDiTLoaderTests {
    @Test("Pre-quantized 4-bit Qwen3.5 converted to 3-bit MACaQ preserves untouched 4-bit layers and reloads bit-identically")
    func preQuantizedQwen35RoundTrip() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(component: "basalt-macaq-538-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let srcDir = dir.appending(component: "src")
        let outDir = dir.appending(component: "out")
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
        var rawCfg = try JSONSerialization.jsonObject(with: Data(TinyModel.Family.qwen35.config.utf8)) as? [String: Any] ?? [:]
        rawCfg["quantization"] = ["bits": 4, "group_size": 32, "mode": "affine"]
        try JSONSerialization.data(withJSONObject: rawCfg, options: [.sortedKeys]).write(to: srcDir.appending(component: "config.json"))
        try Data("{}".utf8).write(to: srcDir.appending(component: "tokenizer_config.json"))

        let model = try TinyModel.make(.qwen35)
        quantize(model: model as Module, groupSize: 32, bits: 4, mode: .affine)
        eval(model as Module)

        var cfg = MACaQConfig()
        cfg.hadamardBlock = 32
        cfg.maxFlipIterations = 8
        let batches: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 7 + 3) % 180) }]
        let (_, manifest) = try MACaQRefiner.quantizeModel(
            model, targetMode: .affine(bits: 3, groupSize: 32), calibrationTokenBatches: batches, config: cfg)
        _ = try MACaQRefiner.writeMACaQDirectory(
            model: model as Module, manifest: manifest, alpha: cfg.alpha, source: srcDir, output: outDir)

        let report = await ModelInspector().inspect(modelIdOrPath: outDir.path)
        #expect(report.intact && report.quantizationMode == "macaq")

        let container = try #require(try await MACaQNativeLoader.loadContainer(from: outDir, tokenizerLoader: ByteTokenizerLoader()))
        let prompt: [Int32] = [3, 19, 44, 81]
        let direct = model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
        eval(direct)
        let directArr = direct.asArray(Float.self)
        let diff = await container.perform { ctx in
            let rel = ctx.model(LMInput.Text(tokens: MLXArray(prompt)[.newAxis]), cache: nil, state: nil).logits[0].asType(.float32)
            eval(rel)
            return zip(directArr, rel.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        }
        #expect(diff == 0)
    }

    @Test("FluxKleinSynthesizer loads MACaQ MXFP4 DiT safetensors checkpoint and synthesizes valid PNG")
    func fluxKleinLoadsMACaQDiT() async throws {
        let dir = FileManager.default.temporaryDirectory.appending(component: "basalt-macaq-dit-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let ditCfg = ImageDiTConfig.tiny(modelID: "black-forest-labs/FLUX.2-klein-4B", hiddenSize: 64)
        let dit = FluxKleinDiTModule(config: ditCfg, usePerceiver: false)
        let spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .mxfp4, hadamardBlock: 32)
        let targetPaths = ["dual_blocks.0.img_qkv", "single_blocks.0.qkv"]
        var manifestModules: [String: MACaQLayerSpec] = [:]
        for path in targetPaths {
            manifestModules[path] = spec
        }
        try MACaQNativeLoader.install(manifestModules, in: dit)
        let arrays = Dictionary(uniqueKeysWithValues: dit.parameters().flattened())
        try save(arrays: arrays, metadata: ["format": "mlx"], url: dir.appending(component: "model.safetensors"))

        let manifestObj = MACaQNativeLoader.Manifest(version: 1, alpha: 0.75, modules: manifestModules)
        let cfgJSON: [String: Any] = [
            "model_id": "black-forest-labs/FLUX.2-klein-4B",
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "latent_channels": 16,
            "patch_size": 2,
            MACaQNativeLoader.configKey: try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifestObj)),
        ]
        try JSONSerialization.data(withJSONObject: cfgJSON, options: [.sortedKeys]).write(to: dir.appending(component: "config.json"))

        let synth = try await ImageSynthesizerLoader.load(identifier: "black-forest-labs/FLUX.2-klein-4B", directory: dir)
        let artifact = try await ImageSynthesizerLoader.probeImageSynthesizer(synth)
        #expect(!artifact.imageData.isEmpty)
        await (synth as? FluxKleinSynthesizer)?.unload()
    }
}
