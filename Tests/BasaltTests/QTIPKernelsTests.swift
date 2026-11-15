import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

enum QTIPKernelTestOracle {
    static func randomPackedWords(count: Int, seed: UInt64) -> [UInt32] {
        var state = seed ^ 0x9E37_79B9_7F4A_7C15
        return (0 ..< count).map { _ in
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            z ^= (z >> 31)
            return UInt32(truncatingIfNeeded: z)
        }
    }

    static func cpuDecodedMatrix(
        words: [UInt32],
        scales: [Float],
        tlut: [Float],
        rate: QTIPRate,
        tileRows: Int,
        tileCols: Int
    ) -> MLXArray {
        let outFeatures = tileRows * 16
        let inFeatures = tileCols * 16
        let wpt = rate.wordsPerTile
        var host = [Float](repeating: 0, count: outFeatures * inFeatures)
        for tr in 0 ..< tileRows {
            let scale = scales[tr]
            for tc in 0 ..< tileCols {
                let tIdx = tr * tileCols + tc
                let slice = Array(words[(tIdx * wpt) ..< ((tIdx + 1) * wpt)])
                let tile = QTIPReference.decodeTile(words: slice, tlut: tlut, rate: rate, scale: scale)
                for r in 0 ..< 16 {
                    let dstBase = (tr * 16 + r) * inFeatures + tc * 16
                    let srcBase = r * 16
                    for c in 0 ..< 16 {
                        host[dstBase + c] = tile[srcBase + c]
                    }
                }
            }
        }
        return MLXArray(host, [outFeatures, inFeatures])
    }

    static func cpuUnrotatedWeight(_ layer: QTIPLinear, dtype: DType) -> MLXArray {
        let tileRows = layer.outFeatures / 16
        let tileCols = layer.inFeatures / 16
        let packed = layer.weight.reshaped(tileRows, tileCols * layer.rate.wordsPerTile)
        let scales = layer.rowScales.asType(.float32)
        let tlut = layer.tlut.asType(.float16).asType(.float32)
        eval(packed, scales, tlut)
        let wRot = cpuDecodedMatrix(
            words: packed.asArray(UInt32.self),
            scales: scales.asArray(Float.self),
            tlut: tlut.asArray(Float.self),
            rate: layer.rate,
            tileRows: tileRows,
            tileCols: tileCols
        )
        let outSigns = MLXArray(layer.rhtOut.signsValues)
        let inSigns = MLXArray(layer.rhtIn.signsValues)
        let outQ = layer.rhtOut.qValues.map { MLXArray($0, [layer.rhtOut.outer, layer.rhtOut.outer]) }
        let inQ = layer.rhtIn.qValues.map { MLXArray($0, [layer.rhtIn.outer, layer.rhtIn.outer]) }
        let wUnrotOut = layer.rhtOut.inverseFloat32(wRot.T, signs: outSigns, qMatrix: outQ).T
        let wFull = layer.rhtIn.inverseFloat32(wUnrotOut, signs: inSigns, qMatrix: inQ).asType(dtype)
        eval(wFull)
        return wFull
    }

    static func relativeError(_ actual: MLXArray, _ reference: MLXArray) -> Float {
        let num = abs(actual.asType(.float32) - reference.asType(.float32)).max()
        let den = maximum(Float(1e-5), abs(reference.asType(.float32)).max())
        let ratio = num / den
        eval(ratio)
        return ratio.item(Float.self)
    }

    static func tolerance(for dtype: DType) -> Float {
        switch dtype {
        case .float32: 1e-5
        case .float16: 2e-3
        default: 1e-2
        }
    }
}

@Suite("QTIPKernels fused qmv, smallM, and 64MB-tiled qmmTiled parity (#409 Phase 1)", .serialized)
struct QTIPDenseKernelsTests {
    @Test("qmv, smallM, and qmmTiled match CPU QTIPReference and QTIPMetal.decode across rates, dtypes, and constant/device scales",
          arguments: QTIPRate.allCases, [DType.float32, .float16, .bfloat16])
    func denseParityAcrossShapesRatesAndDTypes(rate: QTIPRate, dtype: DType) {
        let tlutF16 = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        eval(tlutF16)
        let tlutHost = tlutF16.asType(.float32).asArray(Float.self)
        let tol = QTIPKernelTestOracle.tolerance(for: dtype)

        for (outFeatures, inFeatures) in [(16, 16), (48, 80), (64, 128), (144, 96)] {
            let (tileRows, tileCols) = (outFeatures / 16, inFeatures / 16)
            let words = QTIPKernelTestOracle.randomPackedWords(
                count: tileRows * tileCols * rate.wordsPerTile,
                seed: UInt64(rate.rawValue * 1000 + outFeatures * 7 + inFeatures)
            )
            let packed = MLXArray(words, [tileRows, tileCols * rate.wordsPerTile])
            let scalesF16 = MLXArray((0 ..< tileRows).map { Float(0.25 + 0.15 * Double($0)) }).asType(.float16)
            eval(packed, scalesF16)
            let scalesHost = scalesF16.asType(.float32).asArray(Float.self)
            let cpuW = QTIPKernelTestOracle.cpuDecodedMatrix(
                words: words, scales: scalesHost, tlut: tlutHost,
                rate: rate, tileRows: tileRows, tileCols: tileCols)
            let gpuW = QTIPMetal.decode(
                packed: packed, rowScales: scalesF16, tlut: tlutF16,
                rate: rate, tileCols: tileCols, dtype: dtype)
            eval(cpuW, gpuW)

            for rows in [1, 2, 4, 8, 9, 17] {
                let x = MLXRandom.normal([rows, inFeatures], key: MLXRandom.key(UInt64(rows + outFeatures))).asType(dtype)
                let cpuRef = matmul(x.asType(.float32), cpuW.T)
                let gpuRef = matmul(x, gpuW.T)
                let actual = QTIPKernels.linear(
                    x, packed: packed, rowScales: scalesF16, tlut: tlutF16, rate: rate, inFeatures: inFeatures)
                #expect(actual.shape == [rows, outFeatures] && actual.dtype == dtype)
                #expect(QTIPKernelTestOracle.relativeError(actual, cpuRef) < tol)
                #expect(QTIPKernelTestOracle.relativeError(actual, gpuRef) < tol)
            }
        }
    }

    @Test("All smallM batch sizes M=1..8, 1D/3D shapes, float32 TLUT/scales, and multi-chunk qmmTiled match CPU reference",
          arguments: QTIPRate.allCases)
    func allBatchTemplatesAndShapesMatchCPU(rate: QTIPRate) {
        let (outFeatures, inFeatures) = (80, 48)
        let (tileRows, tileCols) = (outFeatures / 16, inFeatures / 16)
        let tlutF32 = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2])
        let tlutHost = tlutF32.asType(.float16).asType(.float32).asArray(Float.self)
        let words = QTIPKernelTestOracle.randomPackedWords(
            count: tileRows * tileCols * rate.wordsPerTile, seed: UInt64(900 + rate.rawValue))
        let packed3D = MLXArray(words, [tileRows, tileCols, rate.wordsPerTile])
        let scalesF32 = MLXArray((0 ..< tileRows).map { Float(0.4 + 0.1 * Double($0)) })
        eval(tlutF32, packed3D, scalesF32)
        let cpuW = QTIPKernelTestOracle.cpuDecodedMatrix(
            words: words, scales: scalesF32.asArray(Float.self), tlut: tlutHost,
            rate: rate, tileRows: tileRows, tileCols: tileCols)

        let x1D = MLXRandom.normal([inFeatures], key: MLXRandom.key(71))
        let y1D = QTIPKernels.qmv(x1D, packed: packed3D, rowScales: scalesF32, tlut: tlutF32, rate: rate)
        #expect(y1D.shape == [outFeatures])
        #expect(QTIPKernelTestOracle.relativeError(y1D, matmul(x1D, cpuW.T)) < 1e-5)

        for rows in 1 ... 8 {
            let x = MLXRandom.normal([rows, inFeatures], key: MLXRandom.key(UInt64(100 + rows)))
            let y = QTIPKernels.smallM(x, packed: packed3D, rowScales: scalesF32, tlut: tlutF32, rate: rate)
            #expect(y.shape == [rows, outFeatures])
            #expect(QTIPKernelTestOracle.relativeError(y, matmul(x, cpuW.T)) < 1e-5)
        }

        let x3DSmall = MLXRandom.normal([2, 3, inFeatures], key: MLXRandom.key(201)).asType(.bfloat16)
        let y3DSmall = QTIPKernels.linear(x3DSmall, packed: packed3D, rowScales: scalesF32, tlut: tlutF32, rate: rate)
        #expect(y3DSmall.shape == [2, 3, outFeatures] && y3DSmall.dtype == .bfloat16)
        #expect(QTIPKernelTestOracle.relativeError(y3DSmall, matmul(x3DSmall.asType(.float32), cpuW.T)) < 1e-2)

        let x3DTiled = MLXRandom.normal([2, 5, inFeatures], key: MLXRandom.key(202))
        let forcedSmallTileBytes = 32 * inFeatures * MemoryLayout<Float>.size
        let y3DTiled = QTIPKernels.qmmTiled(
            x3DTiled, packed: packed3D, rowScales: scalesF32, tlut: tlutF32,
            rate: rate, tileBytes: forcedSmallTileBytes)
        #expect(y3DTiled.shape == [2, 5, outFeatures])
        #expect(QTIPKernelTestOracle.relativeError(y3DTiled, matmul(x3DTiled, cpuW.T)) < 1e-5)

        let empty = MLXArray.zeros([0, inFeatures], dtype: .float16)
        let emptyOut = QTIPKernels.linear(empty, packed: packed3D, rowScales: scalesF32, tlut: tlutF32, rate: rate)
        #expect(emptyOut.shape == [0, outFeatures] && emptyOut.dtype == .float16)
    }
}

@Suite("QTIPKernels MoE gatherQmv, QTIPSwitchLinear, and TinyModel admission parity (#409 Phase 1)", .serialized)
struct QTIPGatherAndModelParityTests {
    @Test("gatherQmv and QTIPSwitchLinear match CPU expert decode and per-expert QTIPLinear on top-2 and top-4 routing",
          arguments: QTIPRate.allCases, [DType.float32, .float16, .bfloat16])
    func gatherQmvAndSwitchLinearParity(rate: QTIPRate, dtype: DType) throws {
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        eval(tlut)
        let tol = QTIPKernelTestOracle.tolerance(for: dtype)
        try verifyConstantIndicesAndScalesGather(rate: rate, dtype: dtype, tlut: tlut, tol: tol)
        try verifySwitchLinearAgainstPerExpertQTIPLinear(rate: rate, dtype: dtype, tlut: tlut, tol: tol)
    }

    private func verifyConstantIndicesAndScalesGather(
        rate: QTIPRate, dtype: DType, tlut: MLXArray, tol: Float
    ) throws {
        let (numExperts, tileRows, tileCols) = (3, 2, 3)
        let (outFeatures, inFeatures) = (tileRows * 16, tileCols * 16)
        let tlutHost = tlut.asType(.float32).asArray(Float.self)
        let wpt = rate.wordsPerTile
        let words = QTIPKernelTestOracle.randomPackedWords(
            count: numExperts * tileRows * tileCols * wpt, seed: UInt64(3100 + rate.rawValue))
        let packed = MLXArray(words, [numExperts, tileRows, tileCols * wpt])
        let scales = MLXArray((0 ..< (numExperts * tileRows)).map { Float(0.3 + 0.1 * Double($0)) }, [numExperts, tileRows])
            .asType(.float16)
        eval(packed, scales)
        let scalesHost = scales.asType(.float32).asArray(Float.self)
        let cpuExperts = (0 ..< numExperts).map { e in
            let wOff = e * tileRows * tileCols * wpt
            let sOff = e * tileRows
            return QTIPKernelTestOracle.cpuDecodedMatrix(
                words: Array(words[wOff ..< (wOff + tileRows * tileCols * wpt)]),
                scales: Array(scalesHost[sOff ..< (sOff + tileRows)]),
                tlut: tlutHost, rate: rate, tileRows: tileRows, tileCols: tileCols)
        }

        let x1 = MLXRandom.normal([1, inFeatures], key: MLXRandom.key(17)).asType(dtype)
        let idxTop2 = MLXArray([Int32(2), Int32(0)], [1, 2])
        let yTop2 = QTIPKernels.gatherQmv(
            x1, packedExperts: packed, rowScales: scales, tlut: tlut, indices: idxTop2, rate: rate)
        let refTop2 = stacked([2, 0].map { matmul(x1.asType(.float32), cpuExperts[$0].T)[0] }, axis: 0)
            .reshaped(1, 2, outFeatures)
        #expect(yTop2.shape == [1, 2, outFeatures] && yTop2.dtype == dtype)
        #expect(QTIPKernelTestOracle.relativeError(yTop2, refTop2) < tol)

        let x5DUp = MLXRandom.normal([2, 2, 1, 1, inFeatures], key: MLXRandom.key(19)).asType(dtype)
        let idx3D = MLXArray([UInt32(1), 2, 0, 1, 2, 0, 1, 0], [2, 2, 2])
        let y5DUp = QTIPKernels.gatherMatmul(
            x: x5DUp, packedExperts: packed, rowScales: scales, tlut: tlut, indices: idx3D, rate: rate)
        let denseStack = stacked(cpuExperts, axis: 0)
        let ref5DUp = MLX.gatherMM(x5DUp.asType(.float32), denseStack.swappedAxes(-1, -2), rhsIndices: idx3D)
        #expect(y5DUp.shape == [2, 2, 2, 1, outFeatures] && y5DUp.dtype == dtype)
        #expect(QTIPKernelTestOracle.relativeError(y5DUp, ref5DUp) < tol)

        let x5DDown = MLXRandom.normal([2, 2, 2, 1, inFeatures], key: MLXRandom.key(23)).asType(dtype)
        let y5DDown = QTIPKernels.gatherMatmul(
            x: x5DDown, packedExperts: packed.reshaped(numExperts, tileRows, tileCols, wpt),
            rowScales: scales, tlut: tlut, indices: idx3D, rate: rate)
        let ref5DDown = MLX.gatherMM(x5DDown.asType(.float32), denseStack.swappedAxes(-1, -2), rhsIndices: idx3D)
        #expect(y5DDown.shape == [2, 2, 2, 1, outFeatures] && y5DDown.dtype == dtype)
        #expect(QTIPKernelTestOracle.relativeError(y5DDown, ref5DDown) < tol)
        verifyPreRotatedAndGeneralShapes(
            packed: packed, scales: scales, tlut: tlut, denseStack: denseStack, rate: rate, dtype: dtype)
    }

    private func verifyPreRotatedAndGeneralShapes(
        packed: MLXArray, scales: MLXArray, tlut: MLXArray, denseStack: MLXArray, rate: QTIPRate, dtype: DType
    ) {
        let (numExperts, outFeatures, inFeatures) = (denseStack.dim(0), denseStack.dim(1), denseStack.dim(2))
        let tol = QTIPKernelTestOracle.tolerance(for: dtype)
        let xmInner = MLXRandom.normal([2, 3, inFeatures], key: MLXRandom.key(29)).asType(dtype)
        let idx1D = MLXArray([UInt32(2), 0])
        let ymInner = QTIPKernels.gatherMatmul(
            x: xmInner, packedExperts: packed, rowScales: scales, tlut: tlut, indices: idx1D, rate: rate)
        let refmInner = MLX.gatherMM(xmInner.asType(.float32), denseStack.swappedAxes(-1, -2), rhsIndices: idx1D)
        #expect(ymInner.shape == [2, 3, outFeatures] && QTIPKernelTestOracle.relativeError(ymInner, refmInner) < tol)

        let x5DUp = MLXRandom.normal([2, 2, 1, 1, inFeatures], key: MLXRandom.key(31)).asType(dtype)
        let idx3D = MLXArray([UInt32(1), 2, 0, 1, 2, 0, 1, 0], [2, 2, 2])
        let ref5DUp = MLX.gatherMM(x5DUp.asType(.float32), denseStack.swappedAxes(-1, -2), rhsIndices: idx3D)
        let preRot = QTIPSwitchLinear(
            packedExperts: packed, rowScales: scales, tlut: tlut,
            bias: MLXArray.ones([numExperts, outFeatures], dtype: .float32) * Float(0.25), rate: rate)
        #expect(preRot.groupSize == 16 && preRot.bits == max(2, rate.stepBits / 2) && preRot.mode == .affine)
        let yPre5D = preRot(x5DUp, idx3D, sortedIndices: true)
        #expect(QTIPKernelTestOracle.relativeError(yPre5D, ref5DUp + Float(0.25)) < tol)
        let x1 = MLXRandom.normal([1, inFeatures], key: MLXRandom.key(37)).asType(dtype)
        let emptyIdx = MLXArray.zeros([0, 2], dtype: .int32)
        #expect(preRot(x1, emptyIdx).shape == [0, 2, outFeatures])
    }

    private func verifySwitchLinearAgainstPerExpertQTIPLinear(
        rate: QTIPRate, dtype: DType, tlut: MLXArray, tol: Float
    ) throws {
        let (numExperts, outFeatures, inFeatures) = (6, 48, 80)
        let rhtIn = try #require(KroneckerRHT(dimension: inFeatures, seed: 4091))
        let rhtOut = try #require(KroneckerRHT(dimension: outFeatures, seed: 4092))
        let spec = QTIPLayerSpec(rate: rate, rhtIn: rhtIn, rhtOut: rhtOut)

        var perExpert: [QTIPLinear] = []
        for e in 0 ..< numExperts {
            let (w, b) = withRandomState(MLXRandom.RandomState(seed: UInt64(500 + e))) {
                (MLXRandom.normal([outFeatures, inFeatures]), MLXRandom.normal([outFeatures]) * Float(0.1))
            }
            eval(w, b)
            perExpert.append(QTIPQuantizer.quantizeMatrix(
                weight: w, bias: b.asType(dtype), spec: spec, choleskyFactors: nil, tlut: tlut))
        }
        let packedExperts = stacked(perExpert.map(\.weight), axis: 0)
        let expertScales = stacked(perExpert.map(\.rowScales), axis: 0)
        let expertBiases = stacked(try perExpert.map { try #require($0.bias) }, axis: 0)
        let switchLayer = QTIPSwitchLinear(
            packedExperts: packedExperts, rowScales: expertScales, tlut: tlut,
            bias: expertBiases, rate: rate, spec: spec)
        eval(switchLayer)
        #expect(switchLayer.expertCount == numExperts && switchLayer.inFeatures == inFeatures && switchLayer.outFeatures == outFeatures)

        let tokens = 3
        let topK = 4
        let x = MLXRandom.normal([tokens, inFeatures], key: MLXRandom.key(77)).asType(dtype)
        let hostIdx: [[Int]] = [[5, 1, 3, 0], [2, 4, 5, 1], [0, 3, 2, 4]]
        let indices = MLXArray(hostIdx.flatMap { $0.map(Int32.init) }, [tokens, topK])
        let actual = switchLayer(x, indices)
        #expect(actual.shape == [tokens, topK, outFeatures] && actual.dtype == dtype)

        var expectedRows: [MLXArray] = []
        for t in 0 ..< tokens {
            let xt = x[t ..< (t + 1)]
            for slot in 0 ..< topK {
                expectedRows.append(perExpert[hostIdx[t][slot]](xt)[0])
            }
        }
        let expected = stacked(expectedRows, axis: 0).reshaped(tokens, topK, outFeatures)
        eval(actual, expected)
        let switchTol: Float = dtype == .bfloat16 ? 2e-2 : max(tol, 3e-3)
        #expect(QTIPKernelTestOracle.relativeError(actual, expected) < switchTol)
    }
}

@Suite("QTIPKernels TinyModel #228 admission parity (#409 Phase 1)", .serialized)
struct QTIPTinyModelParityTests {
    @Test("TinyModel greedy decode and prefill logits with QTIPKernels match CPU QTIPReference dense oracle (#228)",
          arguments: [TinyModel.Family.qwen3, .qwen35, .gemma4])
    func tinyModelEndToEndAdmissionParity(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let calibBatches: [[Int32]] = [
            (0 ..< 24).map { Int32(($0 * 7 + 3) % 200) },
            (0 ..< 24).map { Int32(($0 * 13 + 11) % 200) }
        ]
        var cfg = QTIPConfig(rate: .threeBPW)
        cfg.minDimension = 16
        cfg.blockGroupSize = 4
        let summary = try QTIPQuantizer.quantizeModel(model, calibrationTokenBatches: calibBatches, config: cfg)
        #expect(summary.quantizedLayers > 0)

        let shortPrompt: [Int] = [5, 19, 42, 88, 120]
        let longPrompt: [Int] = [5, 19, 42, 88, 120, 14, 33, 77, 105, 144]
        let shortQTIPLogits = model(
            LMInput.Text(tokens: MLXArray(shortPrompt.map(Int32.init))[.newAxis]), cache: nil, state: nil
        ).logits[0].asType(.float32)
        let longQTIPLogits = model(
            LMInput.Text(tokens: MLXArray(longPrompt.map(Int32.init))[.newAxis]), cache: nil, state: nil
        ).logits[0].asType(.float32)
        eval(shortQTIPLogits, longQTIPLogits)
        let qtipGreedy = try CacheBlendQualityTests.greedy(longPrompt, count: 6, model: model)

        let root = model as Module
        let embedDType = root.parameters().flattened()
            .first(where: { $0.1.dtype.isFloatingPoint && $0.0.contains("embed") })?.1.dtype ?? .float32
        for (path, mod) in root.namedModules() {
            guard let qLayer = mod as? QTIPLinear else { continue }
            let cpuWeight = QTIPKernelTestOracle.cpuUnrotatedWeight(qLayer, dtype: embedDType)
            let explicit = Linear(weight: cpuWeight, bias: qLayer.bias)
            eval(explicit)
            try PVTuner.swapModule(path, with: explicit, in: root)
        }
        MLXInferenceEngine.settle(root)

        let shortCPULogits = model(
            LMInput.Text(tokens: MLXArray(shortPrompt.map(Int32.init))[.newAxis]), cache: nil, state: nil
        ).logits[0].asType(.float32)
        let longCPULogits = model(
            LMInput.Text(tokens: MLXArray(longPrompt.map(Int32.init))[.newAxis]), cache: nil, state: nil
        ).logits[0].asType(.float32)
        let shortDiff = abs(shortQTIPLogits - shortCPULogits).max()
        let longDiff = abs(longQTIPLogits - longCPULogits).max()
        eval(shortDiff, longDiff)
        let cpuGreedy = try CacheBlendQualityTests.greedy(longPrompt, count: 6, model: model)

        #expect(shortDiff.item(Float.self) < 5e-2)
        #expect(longDiff.item(Float.self) < 5e-2)
        #expect(qtipGreedy == cpuGreedy)
    }
}

@Suite("GGUFEmbedding.asLinear matvec vs matmul and CPU parity (#409 Phase 1)", .serialized)
struct GGUFEmbeddingLinearTests {
    @Test("GGUFEmbedding.asLinear dispatches matvec for M=1..8 and matches M=16 matmul and CPU GGUFDequant",
          arguments: [GGMLType.q8_0, .q4K, .iq4XS, .iq3XXS], [DType.float32, .float16, .bfloat16])
    func embeddingAsLinearMatvecMatchesMatmulAndCPU(type: GGMLType, dtype: DType) throws {
        let (vocab, dimensions) = (37, 256)
        let bytes = GGUFKernelTests.blocks(
            type, count: vocab * dimensions / type.block.elements, seed: UInt64(type.rawValue) + 99)
        let embedding = GGUFEmbedding(type: type, count: vocab, dimensions: dimensions, dtype: dtype)
        try embedding.update(
            parameters: ModuleParameters.unflattened(["weight": GGUFKernelTests.packed(bytes, rows: vocab)]),
            verify: .all)
        let cpuWeight = GGUFKernelTests.dense(bytes, type, rows: vocab, columns: dimensions)
        let tol = QTIPKernelTestOracle.tolerance(for: dtype)

        let x16 = MLXRandom.normal([16, dimensions], key: MLXRandom.key(31)).asType(dtype)
        let y16 = embedding.asLinear(x16)
        let y1 = embedding.asLinear(x16[0 ..< 1])
        let y4 = embedding.asLinear(x16[0 ..< 4])
        let y8 = embedding.asLinear(x16[0 ..< 8])
        let x3D = x16[0 ..< 6].reshaped(2, 3, dimensions)
        let y3D = embedding.asLinear(x3D)

        #expect(y1.shape == [1, vocab] && y1.dtype == dtype)
        #expect(y4.shape == [4, vocab] && y4.dtype == dtype)
        #expect(y8.shape == [8, vocab] && y8.dtype == dtype)
        #expect(y3D.shape == [2, 3, vocab] && y3D.dtype == dtype)
        #expect(y16.shape == [16, vocab] && y16.dtype == dtype)

        #expect(QTIPKernelTestOracle.relativeError(y1, matmul(x16[0 ..< 1].asType(.float32), cpuWeight.T)) < tol)
        #expect(QTIPKernelTestOracle.relativeError(y8, matmul(x16[0 ..< 8].asType(.float32), cpuWeight.T)) < tol)
        #expect(QTIPKernelTestOracle.relativeError(y3D, matmul(x3D.asType(.float32), cpuWeight.T)) < tol)
        #expect(QTIPKernelTestOracle.relativeError(y16, matmul(x16.asType(.float32), cpuWeight.T)) < tol)
        #expect(QTIPKernelTestOracle.relativeError(y1, y16[0 ..< 1]) < tol)
        #expect(QTIPKernelTestOracle.relativeError(y4, y16[0 ..< 4]) < tol)
    }
}
