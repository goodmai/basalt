import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

private final class ReferenceGraphQTIPLinear: Linear {
    let source: QTIPLinear
    let inSigns: MLXArray
    let outSigns: MLXArray
    let inQ: MLXArray?
    let outQ: MLXArray?

    init(_ source: QTIPLinear) {
        self.source = source
        let sIn = MLXArray(source.rhtIn.signsValues)
        let sOut = MLXArray(source.rhtOut.signsValues)
        let qIn = source.rhtIn.qValues.map { MLXArray($0, [source.rhtIn.outer, source.rhtIn.outer]) }
        let qOut = source.rhtOut.qValues.map { MLXArray($0, [source.rhtOut.outer, source.rhtOut.outer]) }
        self.inSigns = sIn
        self.outSigns = sOut
        self.inQ = qIn
        self.outQ = qOut
        super.init(weight: MLXArray.zeros([1, 1], dtype: .float16), bias: nil)
        if let qIn, let qOut { eval(sIn, sOut, qIn, qOut) }
        else if let qIn { eval(sIn, sOut, qIn) }
        else if let qOut { eval(sIn, sOut, qOut) }
        else { eval(sIn, sOut) }
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xRot = source.rhtIn.forwardFloat32(x, signs: inSigns, qMatrix: inQ).asType(x.dtype)
        let wRot = source.decodedRotatedWeight(dtype: .float32)
        let yRot = matmul(xRot.asType(.float32), wRot.T).asType(x.dtype)
        var y = source.rhtOut.inverseFloat32(yRot, signs: outSigns, qMatrix: outQ).asType(x.dtype)
        if let b = source.bias { y = y + b }
        return y
    }
}

private enum CPUKroneckerOracle {
    static func walshHadamard(_ block: inout [Float], scale: Float) {
        let n = block.count
        var h = 1
        while h < n {
            let step = h << 1
            for base in stride(from: 0, to: n, by: step) {
                for j in 0 ..< h {
                    let a = block[base + j]
                    let b = block[base + j + h]
                    block[base + j] = a + b
                    block[base + j + h] = a - b
                }
            }
            h = step
        }
        for i in 0 ..< n { block[i] *= scale }
    }

    static func forward(_ row: [Float], spec: FusedKroneckerRHT.Spec) -> [Float] {
        let (p, n) = (spec.outer, spec.hadamardWidth)
        var blocks = [[Float]](repeating: [Float](repeating: 0, count: n), count: p)
        for c in 0 ..< p {
            for j in 0 ..< n {
                let d = c * n + j
                blocks[c][j] = row[d] * spec.signsValues[d]
            }
            walshHadamard(&blocks[c], scale: spec.hadamardScale)
        }
        guard let q = spec.qValues, p > 1 else { return blocks[0] }
        var out = [Float](repeating: 0, count: spec.dimension)
        for r in 0 ..< p {
            for c in 0 ..< p {
                let qrc = q[r * p + c]
                for j in 0 ..< n {
                    out[r * n + j] += qrc * blocks[c][j]
                }
            }
        }
        return out
    }

    static func inverse(_ row: [Float], spec: FusedKroneckerRHT.Spec) -> [Float] {
        let (p, n) = (spec.outer, spec.hadamardWidth)
        var mixed = [[Float]](repeating: [Float](repeating: 0, count: n), count: p)
        if let q = spec.qValues, p > 1 {
            for r in 0 ..< p {
                for c in 0 ..< p {
                    let qcr = q[c * p + r]
                    for j in 0 ..< n {
                        mixed[r][j] += qcr * row[c * n + j]
                    }
                }
            }
        } else {
            mixed[0] = row
        }
        var out = [Float](repeating: 0, count: spec.dimension)
        for r in 0 ..< p {
            walshHadamard(&mixed[r], scale: spec.hadamardScale)
            for j in 0 ..< n {
                let d = r * n + j
                out[d] = mixed[r][j] * spec.signsValues[d]
            }
        }
        return out
    }
}

@Suite("FusedKroneckerRHT single-dispatch Metal kernels, RMSNorm fusion, and QTIP dedup (#491)")
struct FusedKroneckerRHTTests {
    private static func roundtripMaxTol(for dtype: DType) -> Float {
        switch dtype {
        case .float32: 1e-4
        case .float16: 2e-3
        case .bfloat16: 8e-3
        default: 1e-4
        }
    }

    private static func parityTol(for dtype: DType) -> Float {
        dtype == .float32 ? 1e-4 : 2e-3
    }

    @Test("CPU Sylvester Walsh-Hadamard + Q_p oracle matches FusedKroneckerRHT bit-packed Metal kernel",
          arguments: [2, 4, 8, 16, 32, 48, 64, 80, 128, 256, 512, 1024, 1536, 2048])
    func cpuOracleExactParity(dimension: Int) throws {
        let spec = try #require(FusedKroneckerRHT.Spec(dimension: dimension, seed: 491))
        let buffers = FusedKroneckerRHT.Buffers(spec: spec)
        #expect(buffers.usesMetalKernel)
        let x = withRandomState(MLXRandom.RandomState(seed: UInt64(dimension) + 17)) {
            MLXRandom.normal([2, dimension])
        }
        eval(x)
        let hostIn = x.asArray(Float.self)
        let cpuFwd = CPUKroneckerOracle.forward(Array(hostIn[0 ..< dimension]), spec: spec)
            + CPUKroneckerOracle.forward(Array(hostIn[dimension ..< (2 * dimension)]), spec: spec)
        let cpuInv = CPUKroneckerOracle.inverse(Array(cpuFwd[0 ..< dimension]), spec: spec)
            + CPUKroneckerOracle.inverse(Array(cpuFwd[dimension ..< (2 * dimension)]), spec: spec)

        let gpuFwd = FusedKroneckerRHT.forward(x, buffers: buffers)
        let gpuInv = FusedKroneckerRHT.inverse(gpuFwd, buffers: buffers)
        eval(gpuFwd, gpuInv)

        let fwdErr = zip(cpuFwd, gpuFwd.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        let invErr = zip(cpuInv, gpuInv.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        let roundErr = zip(hostIn, gpuInv.asArray(Float.self)).map { abs($0 - $1) }.max() ?? 1
        #expect(fwdErr < 1e-4)
        #expect(invErr < 1e-4)
        #expect(roundErr < 1e-4)
    }

    @Test("Power-of-two and composite Kronecker parity + roundtrip across dtypes and batch sizes",
          arguments: [4, 16, 48, 64, 80, 256, 1024, 1536, 2048, 2112, 2816, 4096, 5120])
    func forwardInverseParityAndRoundtrip(dimension: Int) throws {
        let spec = try #require(FusedKroneckerRHT.Spec(dimension: dimension, seed: 2026))
        let buffers = FusedKroneckerRHT.Buffers(spec: spec)
        #expect(buffers.usesMetalKernel)
        let rowsList = dimension <= 2048 ? [1, 2, 5, 17] : [1, 5]
        for rows in rowsList {
            let base = withRandomState(MLXRandom.RandomState(seed: UInt64(dimension * 31 + rows))) {
                MLXRandom.normal([rows, dimension]) * Float(0.2)
            }
            for dtype in [DType.float32, DType.float16, DType.bfloat16] {
                let x = base.asType(dtype)
                let fwdF32 = FusedKroneckerRHT.forward(x, buffers: buffers, outputDType: .float32)
                let refFwdF32 = FusedKroneckerRHT.referenceForward(x, spec: spec, outputDType: .float32)
                let fwd = FusedKroneckerRHT.forward(x, buffers: buffers)
                let refFwd = FusedKroneckerRHT.referenceForward(x, spec: spec)
                let inv = FusedKroneckerRHT.inverse(fwd, buffers: buffers)
                let refInv = FusedKroneckerRHT.referenceInverse(fwd, spec: spec)
                let f32Diff = abs(fwdF32 - refFwdF32).max()
                let fwdDiff = abs(fwd.asType(.float32) - refFwd.asType(.float32)).max()
                let invDiff = abs(inv.asType(.float32) - refInv.asType(.float32)).max()
                let rtDiff = abs(inv.asType(.float32) - x.asType(.float32)).max()
                eval(f32Diff, fwdDiff, invDiff, rtDiff)
                #expect(fwd.dtype == dtype && inv.dtype == dtype)
                #expect(f32Diff.item(Float.self) < 1e-4)
                #expect(fwdDiff.item(Float.self) <= (dtype == .bfloat16 ? 8e-3 : Self.parityTol(for: dtype)))
                #expect(invDiff.item(Float.self) <= (dtype == .bfloat16 ? 8e-3 : Self.parityTol(for: dtype)))
                #expect(rtDiff.item(Float.self) <= Self.roundtripMaxTol(for: dtype))
            }
        }
    }

    @Test("1D [D] and 3D [B, L, D] shapes, fused inverse bias, and fused RMSNorm + forward parity",
          arguments: [4, 48, 64, 256, 1536, 2816, 5120])
    func rmsNormAndBiasAnd3DShapesParity(dimension: Int) throws {
        let spec = try #require(FusedKroneckerRHT.Spec(dimension: dimension, seed: 777))
        let buffers = FusedKroneckerRHT.Buffers(spec: spec)
        let (baseX, baseW, baseB) = withRandomState(MLXRandom.RandomState(seed: UInt64(dimension + 99))) {
            (
                MLXRandom.normal([2, 3, dimension]) * Float(0.25),
                MLXRandom.uniform(low: 0.15, high: 0.35, [dimension]),
                MLXRandom.normal([dimension]) * Float(0.05)
            )
        }
        let vec1D = baseX[0, 0]
        let vecOut = FusedKroneckerRHT.inverse(FusedKroneckerRHT.forward(vec1D, buffers: buffers), buffers: buffers)
        eval(vecOut)
        #expect(vecOut.shape == [dimension] && abs(vecOut - vec1D).max().item(Float.self) < 1e-4)

        for dtype in [DType.float32, DType.float16, DType.bfloat16] {
            let x = baseX.asType(dtype)
            let w = baseW.asType(dtype)
            let b = baseB.asType(dtype)
            let fusedNormF32 = FusedKroneckerRHT.rmsNormForward(x, normWeight: w, eps: 1e-5, buffers: buffers, outputDType: .float32)
            let refNormF32 = FusedKroneckerRHT.referenceRMSNormForward(x, normWeight: w, eps: 1e-5, spec: spec, outputDType: .float32)
            let fusedNorm = FusedKroneckerRHT.rmsNormForward(x, normWeight: w, eps: 1e-5, buffers: buffers)
            let refNorm = FusedKroneckerRHT.referenceRMSNormForward(x, normWeight: w, eps: 1e-5, spec: spec)
            let fusedBias = FusedKroneckerRHT.inverse(fusedNorm, buffers: buffers, bias: b)
            let refBias = FusedKroneckerRHT.referenceInverse(fusedNorm, spec: spec, bias: b)
            let f32Diff = abs(fusedNormF32 - refNormF32).max()
            let normDiff = abs(fusedNorm.asType(.float32) - refNorm.asType(.float32)).max()
            let biasDiff = abs(fusedBias.asType(.float32) - refBias.asType(.float32)).max()
            eval(f32Diff, normDiff, biasDiff)
            #expect(fusedNorm.shape == [2, 3, dimension] && fusedNorm.dtype == dtype)
            #expect(fusedBias.shape == [2, 3, dimension] && fusedBias.dtype == dtype)
            #expect(f32Diff.item(Float.self) < 1e-4)
            #expect(normDiff.item(Float.self) <= (dtype == .bfloat16 ? 8e-3 : Self.parityTol(for: dtype)))
            #expect(biasDiff.item(Float.self) <= (dtype == .bfloat16 ? 8e-3 : Self.parityTol(for: dtype)))
        }
    }

    @Test("SignedBlockHadamard Metal kernel and FusedKroneckerRHT large-width fallback match reference")
    func blockHadamardAndFallbackParity() throws {
        for (blockSize, width) in [(1, 16), (4, 16), (16, 64), (64, 256), (128, 512), (1024, 2048)] {
            let rot = try #require(SignedBlockHadamard(blockSize: blockSize, width: width))
            let (xBase, rawSigns) = withRandomState(MLXRandom.RandomState(seed: UInt64(width + blockSize))) {
                (MLXRandom.normal([3, 4, width]) * Float(0.2), MLXRandom.bernoulli(0.5, [width]))
            }
            let signs = MLX.where(rawSigns, MLXArray(Float(1)), MLXArray(Float(-1)))
            let scale = 1.0 / Float(blockSize).squareRoot()
            for dtype in [DType.float32, DType.float16, DType.bfloat16] {
                let x = xBase.asType(dtype)
                let fwd = rot.forward(x, signs: signs)
                let inv = rot.inverse(fwd, signs: signs)
                let refFwd = hadamardTransform((x.asType(.float32) * signs).reshaped(-1, blockSize), scale: scale)
                    .reshaped(x.shape).asType(dtype)
                let diff = abs(fwd.asType(.float32) - refFwd.asType(.float32)).max()
                let rt = abs(inv.asType(.float32) - x.asType(.float32)).max()
                eval(diff, rt)
                #expect(diff.item(Float.self) <= (dtype == .bfloat16 ? 8e-3 : Self.parityTol(for: dtype)))
                #expect(rt.item(Float.self) <= Self.roundtripMaxTol(for: dtype))
            }
        }
        let metal8kSpec = try #require(FusedKroneckerRHT.Spec(dimension: 8192, seed: 55))
        #expect(FusedKroneckerRHT.Buffers(spec: metal8kSpec).usesMetalKernel)
        let fallbackSpec = try #require(FusedKroneckerRHT.Spec(dimension: 130, seed: 55))
        let fallbackBuf = FusedKroneckerRHT.Buffers(spec: fallbackSpec)
        #expect(!fallbackBuf.usesMetalKernel)
        let xf = withRandomState(MLXRandom.RandomState(seed: 55)) { MLXRandom.normal([2, 130]) }
        let yf = FusedKroneckerRHT.forward(xf, buffers: fallbackBuf)
        let zf = FusedKroneckerRHT.inverse(yf, buffers: fallbackBuf)
        let err = abs(zf - xf).max()
        eval(err)
        #expect(err.item(Float.self) < 1e-4)
    }
}

@Suite("FusedKroneckerRHT QTIP sibling dedup, MoE collapse, and TinyModel parity (#491)")
struct FusedKroneckerRHTIntegrationTests {
    private static func parityTol(for dtype: DType) -> Float {
        dtype == .float32 ? 1e-4 : 2e-3
    }

    @Test("Sibling input RHT dedup + RMSNorm fusion, 8-expert MoE collapse, and legacy parameter loading")
    func siblingDedupMoECollapseAndLegacyLoad() throws {
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: .threeBPW), [512, 2]).asType(.float16)
        eval(tlut)
        let rhtIn = try #require(KroneckerRHT(dimension: 64, seed: 101))
        let qkvLayers = try (0 ..< 3).map { idx -> QTIPLinear in
            let rhtOut = try #require(KroneckerRHT(dimension: idx == 0 ? 64 : 32, seed: UInt64(200 + idx)))
            let w = withRandomState(MLXRandom.RandomState(seed: UInt64(300 + idx))) {
                MLXRandom.normal([rhtOut.dimension, 64]) * Float(0.15)
            }
            return QTIPQuantizer.quantizeMatrix(
                weight: w, bias: nil, spec: QTIPLayerSpec(rate: .threeBPW, rhtIn: rhtIn, rhtOut: rhtOut),
                choleskyFactors: nil, tlut: tlut)
        }
        let (xRaw, normW) = withRandomState(MLXRandom.RandomState(seed: 404)) {
            (MLXRandom.normal([2, 5, 64]) * Float(0.25), MLXRandom.uniform(low: 0.2, high: 0.4, [64]))
        }
        for dtype in [DType.float32, DType.float16, DType.bfloat16] {
            let x = xRaw.asType(dtype)
            let nw = normW.asType(dtype)
            let dedupPlain = QTIPLinear.applySiblings(x, layers: qkvLayers)
            let indivPlain = qkvLayers.map { $0(x) }
            for (lhs, rhs) in zip(dedupPlain, indivPlain) {
                let d = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
                eval(d)
                #expect(d.item(Float.self) == 0)
            }
            let dedupNorm = QTIPLinear.applySiblings(x, layers: qkvLayers, norm: (nw, 1e-5))
            let xNormRef = (x.asType(.float32) * rsqrt((x.asType(.float32).square()).mean(axis: -1, keepDims: true) + 1e-5) * nw.asType(.float32)).asType(dtype)
            let indivNorm = qkvLayers.map { $0(xNormRef) }
            for (lhs, rhs) in zip(dedupNorm, indivNorm) {
                let d = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
                eval(d)
                #expect(d.item(Float.self) <= (dtype == .bfloat16 ? 8e-3 : Self.parityTol(for: dtype)))
            }
        }
        try verifyMoECollapseAndLegacyParameters(tlut: tlut)
    }

    private func verifyMoECollapseAndLegacyParameters(tlut: MLXArray) throws {
        let outKey0 = QTIPQuantizer.sharedOutputKey(path: "model.layers.0.mlp.experts.0.down_proj")
        let outKey7 = QTIPQuantizer.sharedOutputKey(path: "model.layers.0.mlp.experts.7.down_proj")
        #expect(outKey0 == outKey7 && outKey0 == "model.layers.0.mlp.experts.*.down_proj.out")
        let sharedOut = try #require(KroneckerRHT(dimension: 64, seed: QTIPQuantizer.hashString(outKey0, seed: 99)))
        let sharedIn = try #require(KroneckerRHT(dimension: 48, seed: 88))
        let experts = (0 ..< 8).map { eIdx -> QTIPLinear in
            let w = withRandomState(MLXRandom.RandomState(seed: UInt64(500 + eIdx))) {
                MLXRandom.normal([64, 48]) * Float(0.15)
            }
            return QTIPQuantizer.quantizeMatrix(
                weight: w, bias: nil, spec: QTIPLayerSpec(rate: .threeBPW, rhtIn: sharedIn, rhtOut: sharedOut),
                choleskyFactors: nil, tlut: tlut)
        }
        let inputs = (0 ..< 8).map { eIdx in
            withRandomState(MLXRandom.RandomState(seed: UInt64(600 + eIdx))) {
                MLXRandom.normal([3, 48]) * Float(0.25)
            }
        }
        let rawWeights = softmax(MLXArray([Float(1.2), 0.7, -0.3, 0.4, 0.9, -0.5, 0.1, 0.2]).reshaped(1, 8), axis: -1)
        let weights = broadcast(rawWeights, to: [3, 8])
        for dtype in [DType.float32, DType.float16, DType.bfloat16] {
            let typedInputs = inputs.map { $0.asType(dtype) }
            let collapsed = QTIPLinear.applyMoECollapse(typedInputs, experts: experts, weights: weights.asType(dtype))
            let separate = zip(typedInputs, experts).enumerated().reduce(MLXArray.zeros([3, 64], dtype: .float32)) { acc, pair in
                let (idx, (inp, exp)) = pair
                return acc + exp(inp).asType(.float32) * weights[0..., idx ..< (idx + 1)]
            }.asType(dtype)
            let diff = abs(collapsed.asType(.float32) - separate.asType(.float32)).max()
            eval(diff)
            #expect(diff.item(Float.self) <= (dtype == .bfloat16 ? 8e-3 : Self.parityTol(for: dtype)))
        }
        let sample = experts[0]
        let paramKeys = Set(sample.parameters().flattened().map(\.0))
        #expect(paramKeys.isDisjoint(with: ["inSigns", "outSigns", "inQ", "outQ"]))
        var legacyFlat = Dictionary(uniqueKeysWithValues: sample.parameters().flattened())
        legacyFlat["inSigns"] = MLXArray(sample.rhtIn.signsValues)
        legacyFlat["outSigns"] = MLXArray(sample.rhtOut.signsValues)
        legacyFlat["outQ"] = MLXArray.zeros([1, 1], dtype: .float32)
        try sample.update(parameters: ModuleParameters.unflattened(legacyFlat), verify: [.all])
    }

    @Test("TinyModel QTIP end-to-end forward logits match stock KroneckerRHT MLX graph (#228)",
          arguments: [TinyModel.Family.qwen3, TinyModel.Family.gemma4])
    func tinyModelFusedVsStockGraphParity(family: TinyModel.Family) throws {
        let calib: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 7 + 3) % 200) }]
        let prompt = MLXArray([Int32(5), Int32(19), Int32(42), Int32(88), Int32(120)])[.newAxis]
        for dtype in [DType.float32, DType.float16, DType.bfloat16] {
            let model = try TinyModel.make(family)
            let root = model as Module
            let scale: Float = dtype == .float32 ? 1.0 : 0.15
            let castParams = root.parameters().flattened().map { key, arr in
                let shouldScale = scale != 1.0 && arr.ndim >= 2
                let adjusted = shouldScale ? (arr * scale) : arr
                return (key, adjusted.dtype.isFloatingPoint ? adjusted.asType(dtype) : adjusted)
            }
            try root.update(parameters: ModuleParameters.unflattened(castParams), verify: [.noUnusedKeys])
            eval(root)

            var cfg = QTIPConfig(rate: .threeBPW)
            cfg.minDimension = 16
            cfg.blockGroupSize = 4
            let summary = try QTIPQuantizer.quantizeModel(model, calibrationTokenBatches: calib, config: cfg)
            #expect(summary.quantizedLayers > 0)

            let fusedLogits = model(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits[0].asType(.float32)
            eval(fusedLogits)

            for (path, mod) in root.namedModules() {
                guard let qLayer = mod as? QTIPLinear else { continue }
                try PVTuner.swapModule(path, with: ReferenceGraphQTIPLinear(qLayer), in: root)
            }
            MLXInferenceEngine.settle(root)
            let refLogits = model(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits[0].asType(.float32)
            let maxDiff = abs(fusedLogits - refLogits).max()
            let fusedIDs = argMax(fusedLogits, axis: -1)
            let refIDs = argMax(refLogits, axis: -1)
            eval(maxDiff, fusedIDs, refIDs)
            let diffVal = maxDiff.item(Float.self)
            let limit: Float = dtype == .float32 ? 1e-4 : (dtype == .float16 ? 2e-3 : 2.5e-2)
            #expect(diffVal <= limit)
            if dtype == .float32 || dtype == .float16 {
                #expect(fusedIDs.asArray(Int32.self) == refIDs.asArray(Int32.self))
            }
        }
    }
}
