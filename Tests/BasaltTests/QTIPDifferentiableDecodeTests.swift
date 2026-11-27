import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltInference

@Suite("QTIPDifferentiableDecode parity and valueAndGrad tests (#876)", .serialized)
struct QTIPDifferentiableDecodeTests {
    private static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = abs(a.asType(.float32) - b.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    private static func makeQTIPLinear(
        outFeatures: Int = 32,
        inFeatures: Int = 32,
        rate: QTIPRate = .twoBPW,
        withBias: Bool = true,
        float32TLUT: Bool = false,
        seed: UInt64 = 8760
    ) throws -> QTIPLinear {
        let rhtIn = try #require(KroneckerRHT(dimension: inFeatures, seed: seed))
        let rhtOut = try #require(KroneckerRHT(dimension: outFeatures, seed: seed + 1))
        let spec = QTIPLayerSpec(rate: rate, rhtIn: rhtIn, rhtOut: rhtOut)
        let tileRows = outFeatures / 16
        let tileCols = inFeatures / 16
        let rawTLUT = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2])
        let tlut = float32TLUT ? rawTLUT.asType(.float32) : rawTLUT.asType(.float16)
        let tiles = MLXRandom.normal([tileRows * tileCols, 16, 16], scale: 0.3, key: MLXRandom.key(seed + 2))
        let encoded = QTIPMetal.encodeTiles(tiles, tlut: tlut, rate: rate).packed
        let packed = encoded.reshaped(tileRows, tileCols * rate.wordsPerTile)
        let scales = MLXRandom.uniform(low: 0.5, high: 1.5, [tileRows], key: MLXRandom.key(seed + 3))
            .asType(float32TLUT ? .float32 : .float16)
        let bias = withBias ? MLXRandom.normal([outFeatures], scale: 0.1, key: MLXRandom.key(seed + 4)) : nil
        eval(packed, scales, tlut)
        if let bias { eval(bias) }
        return QTIPLinear(packedWeight: packed, rowScales: scales, tlut: tlut, bias: bias, spec: spec)
    }

    @Test("QTIPLinear training decode matches Metal inference and independent CPU reference across all QTIPRates and dtypes (#228)", arguments: QTIPRate.allCases)
    func qtipLinearTrainingMatchesMetalAndCPUReference(rate: QTIPRate) throws {
        let layer = try Self.makeQTIPLinear(outFeatures: 48, inFeatures: 48, rate: rate, withBias: true, seed: UInt64(8760 + rate.rawValue))
        let denseRef = layer.decodedWeight(dtype: .float32)
        eval(denseRef)
        for dtype in [DType.float32, .float16, .bfloat16] {
            let x = MLXRandom.normal([2, 48], scale: 0.4, key: MLXRandom.key(8770)).asType(dtype)
            layer.train(false)
            let inferBefore = layer(x)
            layer.train(true)
            let trainOut = layer(x)
            layer.train(false)
            let inferAfter = layer(x)
            var cpuRef = matmul(x.asType(.float32), denseRef.T)
            if let bias = layer.bias { cpuRef = cpuRef + bias.asType(.float32) }
            eval(inferBefore, trainOut, inferAfter, cpuRef)
            #expect(inferBefore.shape == [2, 48] && trainOut.shape == [2, 48])
            #expect(trainOut.dtype == dtype)
            #expect(arrayEqual(inferBefore, inferAfter).item(Bool.self))
            let tol: Float = dtype == .bfloat16 ? 2e-2 : (dtype == .float16 ? 5e-3 : 2e-3)
            #expect(Self.maxAbsDiff(inferBefore, trainOut) < tol)
            #expect(Self.maxAbsDiff(cpuRef, trainOut) < tol)
        }
        layer.train(true)
        let emptyOut = layer(MLXArray.zeros([0, 48], dtype: .float32))
        eval(emptyOut)
        #expect(emptyOut.shape == [0, 48])
    }

    @Test("QTIPLinear training decode handles float32 tlut/scales, float64 CPU fallback, and rotateNormalizedInput")
    func qtipLinearEdgeCasesAndRotateNormalizedInput() throws {
        let layer = try Self.makeQTIPLinear(outFeatures: 32, inFeatures: 32, rate: .twoHalfBPW, withBias: true, float32TLUT: true, seed: 8790)
        let x64 = Device.withDefaultDevice(.cpu) {
            let arr = MLXRandom.normal([2, 32], scale: 0.3, key: MLXRandom.key(8791)).asType(.float64)
            eval(arr)
            return arr
        }
        layer.train(false)
        let infer64 = layer(x64)
        layer.train(true)
        let train64 = layer(x64)
        eval(infer64, train64)
        #expect(train64.dtype == .float64)
        let diff64 = Device.withDefaultDevice(.cpu) { Self.maxAbsDiff(infer64, train64) }
        #expect(diff64 < 2e-3)

        let x32 = MLXRandom.normal([2, 32], scale: 0.3, key: MLXRandom.key(8792))
        let normW = MLXRandom.uniform(low: 0.8, high: 1.2, [32], key: MLXRandom.key(8793))
        layer.train(false)
        let normInfer = layer.callPreRotated(layer.rotateNormalizedInput(x32, normWeight: normW, eps: 1e-6))
        layer.train(true)
        let normTrain = layer.callPreRotated(layer.rotateNormalizedInput(x32, normWeight: normW, eps: 1e-6))
        eval(normInfer, normTrain)
        #expect(Self.maxAbsDiff(normInfer, normTrain) < 2e-3)
    }

    @Test("valueAndGrad matches finite differences for input x, unfrozen row_scales, and bias on Kronecker outer > 1 QTIPLinear")
    func valueAndGradMatchesFiniteDifferences() throws {
        let layer = try Self.makeQTIPLinear(outFeatures: 48, inFeatures: 48, rate: .threeBPW, withBias: true, float32TLUT: true, seed: 8800)
        layer.unfreeze(keys: ["row_scales", "bias"])
        layer.train(true)
        let x = MLXRandom.normal([2, 48], scale: 0.3, key: MLXRandom.key(8801))
        let lossW = MLXRandom.normal([2, 48], scale: 0.5, key: MLXRandom.key(8802))

        let xVG = valueAndGrad { args in [sum(layer(args[0]) * lossW)] }
        let (xVal, xGrad) = xVG([x])
        eval(xVal[0], xGrad[0])
        #expect(xVal[0].item(Float.self).isFinite)
        let xGradVals = xGrad[0].asArray(Float.self)
        #expect(xGradVals.allSatisfy { $0.isFinite })
        let eps: Float = 1e-3
        for idx in [0, 17, 47] {
            var xp = x.asArray(Float.self)
            var xm = xp
            xp[idx] += eps
            xm[idx] -= eps
            let lp = sum(layer(MLXArray(xp, [2, 48])) * lossW)
            let lm = sum(layer(MLXArray(xm, [2, 48])) * lossW)
            eval(lp, lm)
            let fd = (lp.item(Float.self) - lm.item(Float.self)) / (2 * eps)
            #expect(abs(xGradVals[idx] - fd) < 5e-3)
        }

        let paramVG = valueAndGrad(model: layer) { (model: QTIPLinear, input: MLXArray, weights: MLXArray) -> MLXArray in
            sum(model(input) * weights)
        }
        let (pVal, pGrads) = paramVG(layer, x, lossW)
        eval(pVal, pGrads)
        let flat = Dictionary(uniqueKeysWithValues: pGrads.flattened())
        let scaleGrad = try #require(flat["row_scales"])
        let biasGrad = try #require(flat["bias"])
        eval(scaleGrad, biasGrad)
        let scaleGradVals = scaleGrad.asArray(Float.self)
        let biasGradVals = biasGrad.asArray(Float.self)
        #expect(scaleGradVals.allSatisfy { $0.isFinite })
        #expect(biasGradVals.allSatisfy { $0.isFinite })

        let expectedBiasGrad = lossW.sum(axis: 0)
        eval(expectedBiasGrad)
        #expect(Self.maxAbsDiff(biasGrad, expectedBiasGrad) < 1e-4)

        let origScales = layer.rowScales.asArray(Float.self)
        for sIdx in 0 ..< origScales.count {
            var sp = origScales
            var sm = origScales
            sp[sIdx] += eps
            sm[sIdx] -= eps
            try layer.update(parameters: ModuleParameters.unflattened(["row_scales": MLXArray(sp)]), verify: .none)
            let lp = sum(layer(x) * lossW)
            try layer.update(parameters: ModuleParameters.unflattened(["row_scales": MLXArray(sm)]), verify: .none)
            let lm = sum(layer(x) * lossW)
            eval(lp, lm)
            let fd = (lp.item(Float.self) - lm.item(Float.self)) / (2 * eps)
            #expect(abs(scaleGradVals[sIdx] - fd) < 5e-3)
        }
        try layer.update(parameters: ModuleParameters.unflattened(["row_scales": MLXArray(origScales)]), verify: .none)
    }
}
