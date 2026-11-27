import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private enum Shard4AuditHelpers {
    static func quantizedLayer(
        inputs: Int, outputs: Int, groupSize: Int, bits: Int, dtype: DType, seed: UInt64
    ) -> QuantizedLinear {
        let key = MLXRandom.key(seed)
        let dense = MLXRandom.normal([outputs, inputs], scale: 0.08, key: key).asType(dtype)
        let offset = MLXRandom.normal([outputs], scale: 0.05, key: key).asType(dtype)
        let layer = QuantizedLinear(weight: dense, bias: offset, groupSize: groupSize, bits: bits)
        eval(layer)
        return layer
    }

    static func applyGivensLayerToCov(
        _ cov: [Float], dim: Int, schedule: LIRGivensSchedule, layer: Int
    ) -> [Float] {
        var g = [Float](repeating: 0, count: dim * dim)
        for r in 0 ..< dim { g[r * dim + r] = 1 }
        for p in 0 ..< schedule.pairsPerLayer {
            let off = (layer * schedule.pairsPerLayer + p) * 2
            let i = Int(schedule.flatPairs[off])
            let j = Int(schedule.flatPairs[off + 1])
            guard i >= 0, j >= 0, i < dim, j < dim, i != j else { continue }
            let c = schedule.flatAngles[off]
            let s = schedule.flatAngles[off + 1]
            g[i * dim + i] = c
            g[i * dim + j] = -s
            g[j * dim + i] = s
            g[j * dim + j] = c
        }
        var tmp = [Float](repeating: 0, count: dim * dim)
        for r in 0 ..< dim {
            for col in 0 ..< dim {
                var acc: Float = 0
                for k in 0 ..< dim { acc += cov[r * dim + k] * g[k * dim + col] }
                tmp[r * dim + col] = acc
            }
        }
        var out = [Float](repeating: 0, count: dim * dim)
        for r in 0 ..< dim {
            for col in 0 ..< dim {
                var acc: Float = 0
                for k in 0 ..< dim { acc += g[k * dim + r] * tmp[k * dim + col] }
                out[r * dim + col] = acc
            }
        }
        return out
    }

    static func offDiagonalEnergy(_ cov: [Float], dim: Int) -> Float {
        var energy: Float = 0
        for r in 0 ..< dim {
            for c in 0 ..< dim where r != c {
                let v = cov[r * dim + c]
                energy += v * v
            }
        }
        return energy
    }

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }
}

@Suite("Metal Audit Shard 4: FlatQuant & LIRQuant regression tests", .serialized)
struct FlatQuantAndLIRQuantAuditRegressionTests {
    @Test("FlatQuant Kronecker handles x.size < 8, invalid shapes, empty activations, and CPU default device")
    func flatQuantSmallBuffersGuardsAndCPUFallback() {
        let rng = MLXRandom.RandomState(seed: 4101)
        for (d1, d2) in [(2, 2), (1, 4), (1, 1)] {
            let k = d1 * d2
            let x = MLXRandom.normal([1, k], dtype: .float32, key: rng)
            let p1 = MLXRandom.normal([d1, d1], dtype: .float32, key: rng) * Float(0.5) + eye(d1, dtype: .float32)
            let p2 = MLXRandom.normal([d2, d2], dtype: .float32, key: rng) * Float(0.5) + eye(d2, dtype: .float32)
            let bounds = MLXArray([Float(-1.5), Float(1.5)])
            for dtype in [DType.float32, .float16, .bfloat16] {
                let xT = x.asType(dtype)
                let p1T = p1.asType(dtype)
                let p2T = p2.asType(dtype)
                let gpuOut = FlatQuantKernels.transform(xT, p1: p1T, p2: p2T, clipBounds: bounds)
                let refOut = FlatQuantKernels.differentiableTransform(xT, p1: p1T, p2: p2T, clipBounds: bounds)
                let cpuOut = Device.withDefaultDevice(.cpu) {
                    FlatQuantKernels.transform(xT, p1: p1T, p2: p2T, clipBounds: bounds)
                }
                let tol: Float = dtype == .bfloat16 ? 2e-2 : (dtype == .float16 ? 4e-3 : 1e-4)
                #expect(Shard4AuditHelpers.maxAbsDiff(gpuOut, refOut) <= tol)
                #expect(Shard4AuditHelpers.maxAbsDiff(gpuOut, cpuOut) <= tol)
            }
        }

        let x4 = MLXArray([Float(1), 2, 3, 4], [1, 4])
        let rectP1 = MLXArray.ones([2, 3], dtype: .float32)
        let sqP2 = eye(2, dtype: .float32)
        let vecP1 = MLXArray.ones([2], dtype: .float32)
        #expect(Shard4AuditHelpers.maxAbsDiff(FlatQuantKernels.transform(x4, p1: rectP1, p2: sqP2), x4) == 0)
        #expect(Shard4AuditHelpers.maxAbsDiff(FlatQuantKernels.transform(x4, p1: sqP2, p2: rectP1), x4) == 0)
        #expect(Shard4AuditHelpers.maxAbsDiff(FlatQuantKernels.transform(x4, p1: vecP1, p2: sqP2), x4) == 0)
        #expect(Shard4AuditHelpers.maxAbsDiff(FlatQuantKernels.transform(x4, p1: sqP2, p2: sqP2, clipBounds: MLXArray([Float(1)])), x4) == 0)
        #expect(Shard4AuditHelpers.maxAbsDiff(FlatQuantKernels.transform(x4, p1: sqP2, p2: sqP2, clipBounds: MLXArray([Float(1), 2, 3])), x4) == 0)

        let weight = MLXRandom.normal([16, 32], dtype: .float32, key: rng)
        let emptyActs = MLXArray.zeros([0, 32], dtype: .float32)
        let factors = FlatQuantCalibrator.calibrateFactors(weight: weight, activations: emptyActs, clipQuantile: 0.95)
        #expect(factors.clipBounds == nil)
        #expect(factors.p1.asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(factors.p2.asArray(Float.self).allSatisfy { $0.isFinite })

        let lin = withRandomState(rng) { Linear(256, 32, bias: true) }
        let x256 = MLXRandom.normal([2, 256], dtype: .float16, key: rng)
        let fqLayer = FlatQuantCalibrator.quantizeLayer(linear: lin, activations: x256.asType(.float32), bits: 8, groupSize: 32)
        fqLayer.train(false)
        let gpuLayerOut = fqLayer(x256)
        let cpuLayerOut = Device.withDefaultDevice(.cpu) { fqLayer(x256) }
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuLayerOut, cpuLayerOut) < 5e-3)
    }

    @Test("LIRQuant Givens calibration strictly zeroes off-diagonal covariance across multi-layer schedule")
    func lirQuantGivensSignZeroesOffDiagonalCovariance() {
        let cov2: [Float] = [3.0, 1.0, 1.0, 1.0]
        let sched2 = LIRQuantCalibrator.buildSchedule(subspaceCov: cov2, subspaceWidth: 2, layers: 1)
        #expect(sched2.numLayers == 1 && sched2.pairsPerLayer == 1)
        let rot2 = Shard4AuditHelpers.applyGivensLayerToCov(cov2, dim: 2, schedule: sched2, layer: 0)
        #expect(abs(rot2[1]) < 1e-5 && abs(rot2[2]) < 1e-5)
        #expect(abs(rot2[0] - (2.0 + Float(2.0).squareRoot())) < 1e-4)
        #expect(abs(rot2[3] - (2.0 - Float(2.0).squareRoot())) < 1e-4)

        let cov4: [Float] = [
            3.0, 1.0, 0.6, -0.4,
            1.0, 1.2, -0.3, 0.5,
            0.6, -0.3, 4.0, 1.1,
            -0.4, 0.5, 1.1, 2.0,
        ]
        let sched4 = LIRQuantCalibrator.buildSchedule(subspaceCov: cov4, subspaceWidth: 4, layers: 2)
        let e0 = Shard4AuditHelpers.offDiagonalEnergy(cov4, dim: 4)
        let afterL0 = Shard4AuditHelpers.applyGivensLayerToCov(cov4, dim: 4, schedule: sched4, layer: 0)
        let e1 = Shard4AuditHelpers.offDiagonalEnergy(afterL0, dim: 4)
        #expect(abs(afterL0[0 * 4 + 1]) < 1e-5 && abs(afterL0[2 * 4 + 3]) < 1e-5)
        #expect(e1 < e0)

        let afterL1 = Shard4AuditHelpers.applyGivensLayerToCov(afterL0, dim: 4, schedule: sched4, layer: 1)
        let e2 = Shard4AuditHelpers.offDiagonalEnergy(afterL1, dim: 4)
        #expect(abs(afterL1[0 * 4 + 2]) < 1e-5 && abs(afterL1[1 * 4 + 3]) < 1e-5)
        #expect(e2 < e1)

        let badSched = LIRQuantCalibrator.buildSchedule(subspaceCov: [1.0, 0.0], subspaceWidth: 4, layers: 2)
        #expect(badSched.numLayers == 0 && badSched.pairsPerLayer == 0)
    }

    @Test("LIRQuant guards OOB pairs, OOB perm, < 8 pair buffers, mismatched sizes, k > 4096, and CPU fallback")
    func lirQuantBoundsGuardsSmallBuffersAndCPUFallback() {
        let rng = MLXRandom.RandomState(seed: 4102)
        let k = 32
        let x = MLXRandom.normal([2, k], dtype: .float32, key: rng)
        let perm = MLXArray(Array(0 ..< Int32(k)))
        let scale = MLXRandom.uniform(low: 0.7, high: 1.3, [k], dtype: .float32, key: rng)
        let oobPairs = MLXArray([Int32(0), 5000, 4096, 4097, -3, 2, 32, 33, 1, 5])
        let oobAngles = MLXArray([Float(0.6), 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.8, 0.6])
        let oobBasis = LIRBasisParameters(
            perm: perm, givensPairs: oobPairs, givensAngles: oobAngles,
            diagScale: scale, numLayers: 1, pairsPerLayer: 5)
        let gpuOOB = LIRQuantKernels.transform(x, basis: oobBasis)
        let diffOOB = LIRQuantKernels.differentiableTransform(x, basis: oobBasis)
        let cpuOOB = Device.withDefaultDevice(.cpu) { LIRQuantKernels.transform(x, basis: oobBasis) }
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuOOB, diffOOB) < 1e-4)
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuOOB, cpuOOB) < 1e-4)

        let tinyPairs2D = MLXArray([Int32(2), 7], [1, 1, 2])
        let tinyAngles2D = MLXArray([Float(0.8), 0.6], [1, 1, 2])
        let tinyBasis = LIRBasisParameters(
            perm: perm, givensPairs: tinyPairs2D, givensAngles: tinyAngles2D,
            diagScale: scale, numLayers: 1, pairsPerLayer: 1)
        for dtype in [DType.float32, .float16, .bfloat16] {
            let xT = x.asType(dtype)
            let gpuTiny = LIRQuantKernels.transform(xT, basis: tinyBasis)
            let refTiny = LIRQuantKernels.differentiableTransform(xT, basis: tinyBasis)
            let tol: Float = dtype == .bfloat16 ? 2e-2 : (dtype == .float16 ? 4e-3 : 1e-4)
            #expect(Shard4AuditHelpers.maxAbsDiff(gpuTiny, refTiny) <= tol)
        }

        var badPermHost = Array(0 ..< Int32(k))
        badPermHost[0] = -1
        badPermHost[1] = Int32(k + 9)
        let badPermBasis = LIRBasisParameters(
            perm: MLXArray(badPermHost), givensPairs: tinyPairs2D, givensAngles: tinyAngles2D,
            diagScale: scale, numLayers: 1, pairsPerLayer: 1)
        #expect(Shard4AuditHelpers.maxAbsDiff(
            LIRQuantKernels.transform(x, basis: badPermBasis),
            LIRQuantKernels.differentiableTransform(x, basis: badPermBasis)) < 1e-4)

        let shortPairsBasis = LIRBasisParameters(
            perm: perm, givensPairs: MLXArray([Int32(0), 1]), givensAngles: MLXArray([Float(1), 0]),
            diagScale: scale, numLayers: 2, pairsPerLayer: 4)
        #expect(Shard4AuditHelpers.maxAbsDiff(LIRQuantKernels.transform(x, basis: shortPairsBasis), x) == 0)

        let badScaleBasis = LIRBasisParameters(
            perm: perm, givensPairs: tinyPairs2D, givensAngles: tinyAngles2D,
            diagScale: MLXArray.ones([k - 1], dtype: .float32), numLayers: 1, pairsPerLayer: 1)
        #expect(Shard4AuditHelpers.maxAbsDiff(LIRQuantKernels.transform(x, basis: badScaleBasis), x) == 0)

        let kWide = 4128
        let xWide = MLXRandom.normal([1, kWide], dtype: .float32, key: rng)
        let wideBasis = LIRBasisParameters(
            perm: MLXArray(Array(0 ..< Int32(kWide))),
            givensPairs: MLXArray([Int32(0), 1, 2, 3]),
            givensAngles: MLXArray([Float(0.8), 0.6, 0.6, -0.8]),
            diagScale: MLXArray.ones([kWide], dtype: .float32),
            numLayers: 1, pairsPerLayer: 2)
        let gpuWide = LIRQuantKernels.transform(xWide, basis: wideBasis)
        let cpuWide = Device.withDefaultDevice(.cpu) { LIRQuantKernels.transform(xWide, basis: wideBasis) }
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuWide, cpuWide) < 1e-4)
    }

    @Test("LIRQuant differentiableTransform supports MLX valueAndGrad wrt x, givensAngles, and diagScale")
    func lirQuantDifferentiableTransformValueAndGrad() {
        let rng = MLXRandom.RandomState(seed: 4103)
        let k = 32
        let x = MLXRandom.normal([2, k], dtype: .float32, key: rng)
        let perm = MLXArray(Array(0 ..< Int32(k)))
        let pairs = MLXArray([Int32(0), 1, 2, 3, -1, -1, 0, 99, 0, 2, 1, 3, -1, -1, 4, 5])
        let angles = MLXArray([
            Float(0.8), 0.6, 0.9, -0.43589, 1.0, 0.0, 0.7, 0.7,
            0.7071, 0.7071, 0.6, 0.8, 1.0, 0.0, 0.95, -0.31225,
        ])
        let diag = MLXRandom.uniform(low: 0.7, high: 1.3, [k], dtype: .float32, key: rng)
        let lossW = MLXRandom.normal([2, k], dtype: .float32, key: rng)

        let vg = valueAndGrad({ (args: [MLXArray]) -> [MLXArray] in
            let basis = LIRBasisParameters(
                perm: perm, givensPairs: pairs, givensAngles: args[1],
                diagScale: args[2], numLayers: 2, pairsPerLayer: 4)
            let y = LIRQuantKernels.differentiableTransform(args[0], basis: basis)
            return [sum(y * lossW)]
        }, argumentNumbers: [0, 1, 2])
        let (vals, grads) = vg([x, angles, diag])
        eval(vals[0], grads[0], grads[1], grads[2])

        let dAngles = grads[1].asArray(Float.self)
        #expect(vals[0].item(Float.self).isFinite)
        #expect(grads[0].asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(dAngles.allSatisfy { $0.isFinite })
        #expect(grads[2].asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(abs(grads[0]).max().item(Float.self) > 1e-4)
        #expect(abs(grads[2]).max().item(Float.self) > 1e-4)
        #expect(dAngles[4] == 0 && dAngles[5] == 0 && dAngles[6] == 0 && dAngles[7] == 0)

        let eps: Float = 1e-3
        var plusArr = angles.asArray(Float.self)
        var minusArr = plusArr
        plusArr[0] += eps
        minusArr[0] -= eps
        let lossPlus = sum(LIRQuantKernels.differentiableTransform(x, basis: LIRBasisParameters(
            perm: perm, givensPairs: pairs, givensAngles: MLXArray(plusArr),
            diagScale: diag, numLayers: 2, pairsPerLayer: 4)) * lossW)
        let lossMinus = sum(LIRQuantKernels.differentiableTransform(x, basis: LIRBasisParameters(
            perm: perm, givensPairs: pairs, givensAngles: MLXArray(minusArr),
            diagScale: diag, numLayers: 2, pairsPerLayer: 4)) * lossW)
        eval(lossPlus, lossMinus)
        let fd = (lossPlus.item(Float.self) - lossMinus.item(Float.self)) / (2 * eps)
        #expect(abs(dAngles[0] - fd) < 5e-3)
    }
}

@Suite("Metal Audit Shard 4: FusedTransformerKernels & FusedKroneckerRHT regression tests", .serialized)
struct FusedTransformerAndRHTAuditRegressionTests {
    @Test("FusedTransformerKernels guards CPU fallback, outputs == 0/1, empty rows, and bad weight shapes")
    func fusedTransformerGuardsCPUAndSmallOutputs() {
        #expect(FusedTransformerKernels.supportedDTypes == [.float16, .bfloat16, .float32])
        let layer1 = Shard4AuditHelpers.quantizedLayer(
            inputs: 32, outputs: 1, groupSize: 32, bits: 4, dtype: .float16, seed: 4201)
        let norm32 = RMSNorm(dimensions: 32, eps: 1e-5)
        let x1 = MLXRandom.normal([1, 2, 32], key: MLXRandom.key(4202)).asType(.float16)
        let u1 = MLXRandom.normal([1, 2, 32], key: MLXRandom.key(4203)).asType(.float16)
        let gpuNorm1 = FusedTransformerKernels.rmsNormQuantizedMatmul(x1, norm: norm32, linear: layer1)
        let cpuNorm1 = Device.withDefaultDevice(.cpu) {
            FusedTransformerKernels.rmsNormQuantizedMatmul(x1, norm: norm32, linear: layer1)
        }
        let gpuGate1 = FusedTransformerKernels.gatedQuantizedMatmul(gate: x1, up: u1, activation: .silu, down: layer1)
        let cpuGate1 = Device.withDefaultDevice(.cpu) {
            FusedTransformerKernels.gatedQuantizedMatmul(gate: x1, up: u1, activation: .silu, down: layer1)
        }
        #expect(gpuNorm1.shape == [1, 2, 1] && cpuNorm1.shape == [1, 2, 1])
        #expect(gpuNorm1.dtype == cpuNorm1.dtype)
        #expect(gpuGate1.dtype == cpuGate1.dtype)
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuNorm1, cpuNorm1) < 2e-3)
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuGate1, cpuGate1) < 2e-3)

        let zeroWeights = AffineQuantizedWeights(
            weight: MLXArray.zeros([0, 8], dtype: .uint32),
            scales: MLXArray.zeros([0, 1], dtype: .float16),
            biases: MLXArray.zeros([0, 1], dtype: .float16),
            groupSize: 64, bits: 4)
        let norm64 = RMSNorm(dimensions: 64, eps: 1e-5)
        let x64 = MLXRandom.normal([1, 2, 64], key: MLXRandom.key(4204)).asType(.float16)
        let zeroNormOut = FusedTransformerKernels.rmsNormQuantizedMatmul(
            x64, normWeight: norm64.weight, eps: 1e-5, weights: zeroWeights)
        let zeroGateOut = FusedTransformerKernels.gatedQuantizedMatmul(
            gate: x64, up: x64, activation: .geluApprox, weights: zeroWeights)
        eval(zeroNormOut, zeroGateOut)
        #expect(zeroNormOut.shape == [1, 2, 0] && zeroGateOut.shape == [1, 2, 0])

        let layer64 = Shard4AuditHelpers.quantizedLayer(
            inputs: 64, outputs: 16, groupSize: 64, bits: 4, dtype: .float16, seed: 4205)
        let emptyX = MLXArray.zeros([1, 0, 64], dtype: .float16)
        let emptyOut = FusedTransformerKernels.rmsNormQuantizedMatmul(emptyX, norm: norm64, linear: layer64)
        eval(emptyOut)
        #expect(emptyOut.shape == [1, 0, 16])

        let badWeights = AffineQuantizedWeights(
            weight: MLXArray.zeros([16, 4], dtype: .uint32),
            scales: MLXArray.zeros([16, 1], dtype: .float16),
            biases: MLXArray.zeros([16, 1], dtype: .float16),
            groupSize: 64, bits: 4)
        #expect(throws: MLXError.self) { try withError { _ = FusedTransformerKernels.rmsNormQuantizedMatmul(x64, normWeight: norm64.weight, eps: 1e-5, weights: badWeights) } }
        #expect(throws: MLXError.self) { try withError { _ = FusedTransformerKernels.gatedQuantizedMatmul(gate: x64, up: x64, activation: .silu, weights: badWeights) } }
    }

    @Test("FusedRMSNormQuantizedLinear, FusedGatedQuantizedMLP, FlatQuantLinear, and LIRQuantLinear support valueAndGrad when training")
    func fusedModulesBypassMetalKernelDuringTraining() {
        let linear = Shard4AuditHelpers.quantizedLayer(
            inputs: 64, outputs: 16, groupSize: 64, bits: 4, dtype: .float32, seed: 4301)
        let norm = RMSNorm(dimensions: 64, eps: 1e-5)
        let fusedNorm = FusedRMSNormQuantizedLinear(replacing: linear, normWeight: norm.weight, eps: norm.eps)
        #expect(!fusedNorm.training)
        fusedNorm.train(true)

        let gate = Shard4AuditHelpers.quantizedLayer(
            inputs: 64, outputs: 64, groupSize: 64, bits: 4, dtype: .float32, seed: 4302)
        let up = Shard4AuditHelpers.quantizedLayer(
            inputs: 64, outputs: 64, groupSize: 64, bits: 4, dtype: .float32, seed: 4303)
        let down = Shard4AuditHelpers.quantizedLayer(
            inputs: 64, outputs: 16, groupSize: 64, bits: 4, dtype: .float32, seed: 4304)
        let fusedMLP = FusedGatedQuantizedMLP(gateProj: gate, upProj: up, downProj: down, activation: .silu)
        #expect(!fusedMLP.training)
        fusedMLP.train(true)

        let x = MLXRandom.normal([1, 2, 64], key: MLXRandom.key(4305))
        let (_, normGrads) = valueAndGrad({ (args: [MLXArray]) -> [MLXArray] in [sum(fusedNorm(args[0]))] })([x])
        let (_, stockNormGrads) = valueAndGrad({ (args: [MLXArray]) -> [MLXArray] in [sum(linear(norm(args[0])))] })([x])
        let (_, mlpGrads) = valueAndGrad({ (args: [MLXArray]) -> [MLXArray] in [sum(fusedMLP(args[0]))] })([x])
        let (_, stockMLPGrads) = valueAndGrad({ (args: [MLXArray]) -> [MLXArray] in
            [sum(down(silu(gate(args[0])) * up(args[0])))]
        })([x])
        eval(normGrads[0], stockNormGrads[0], mlpGrads[0], stockMLPGrads[0])
        #expect(Shard4AuditHelpers.maxAbsDiff(normGrads[0], stockNormGrads[0]) < 1e-4)
        #expect(Shard4AuditHelpers.maxAbsDiff(mlpGrads[0], stockMLPGrads[0]) < 1e-4)

        let rng = MLXRandom.RandomState(seed: 4306)
        let dense = withRandomState(rng) { Linear(64, 16, bias: false) }
        let fq = FlatQuantCalibrator.quantizeLayer(linear: dense, activations: nil, bits: 4, groupSize: 32)
        fq.train(true)
        let lir = LIRQuantCalibrator.quantizeLayer(
            linear: dense, activations: nil, options: .init(subspaceBlocks: 1, givensLayers: 2, bits: 4, groupSize: 32))
        lir.train(true)
        let (_, fqGrads) = valueAndGrad({ (args: [MLXArray]) -> [MLXArray] in [sum(fq(args[0]))] })([x])
        let (_, lirGrads) = valueAndGrad({ (args: [MLXArray]) -> [MLXArray] in [sum(lir(args[0]))] })([x])
        eval(fqGrads[0], lirGrads[0])
        #expect(fqGrads[0].asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(lirGrads[0].asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test("FusedKroneckerRHT handles CPU default device, broadcastable signs, and dynamic num_blocks in MSL")
    func fusedKroneckerRHTCPUAndBroadcastSignsParity() throws {
        let spec = try #require(FusedKroneckerRHT.Spec(dimension: 48, seed: 4401))
        let buffers = FusedKroneckerRHT.Buffers(spec: spec)
        let rng = MLXRandom.RandomState(seed: 4402)
        let x = MLXRandom.normal([2, 48], dtype: .float32, key: rng)
        let normW = MLXRandom.uniform(low: 0.5, high: 1.5, [1, 48], dtype: .float32, key: rng)
        let bias = MLXRandom.normal([1, 48], dtype: .float32, key: rng) * Float(0.1)

        let gpuFwd = FusedKroneckerRHT.forward(x, buffers: buffers)
        let cpuFwd = Device.withDefaultDevice(.cpu) { FusedKroneckerRHT.forward(x, buffers: buffers) }
        let gpuNorm = FusedKroneckerRHT.rmsNormForward(x, normWeight: normW, eps: 1e-5, buffers: buffers)
        let cpuNorm = Device.withDefaultDevice(.cpu) {
            FusedKroneckerRHT.rmsNormForward(x, normWeight: normW, eps: 1e-5, buffers: buffers)
        }
        let gpuInv = FusedKroneckerRHT.inverse(gpuFwd, buffers: buffers, bias: bias)
        let cpuInv = Device.withDefaultDevice(.cpu) {
            FusedKroneckerRHT.inverse(gpuFwd, buffers: buffers, bias: bias)
        }
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuFwd, cpuFwd) < 1e-4)
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuNorm, cpuNorm) < 1e-4)
        #expect(Shard4AuditHelpers.maxAbsDiff(gpuInv, cpuInv) < 1e-4)

        let bx = MLXRandom.normal([3, 64], dtype: .float32, key: rng)
        let broadcastShapes: [MLXArray] = [
            MLXArray(Float(-1.0)),
            MLXRandom.uniform(low: 0.5, high: 1.5, [3, 1], dtype: .float32, key: rng),
            MLXRandom.uniform(low: 0.5, high: 1.5, [3, 64], dtype: .float32, key: rng),
        ]
        for signs in broadcastShapes {
            for blockSize in [32, 64] {
                let scale = 1.0 / Float(blockSize).squareRoot()
                let gotFwd = FusedKroneckerRHT.blockHadamardFallback(bx, signs: signs, blockSize: blockSize, isInverse: false)
                let refFwd = hadamardTransform((bx * signs).reshaped(-1, blockSize), scale: scale).reshaped(bx.shape)
                let gotInv = FusedKroneckerRHT.blockHadamardFallback(bx, signs: signs, blockSize: blockSize, isInverse: true)
                let refInv = hadamardTransform(bx.reshaped(-1, blockSize), scale: scale).reshaped(bx.shape) * signs
                #expect(Shard4AuditHelpers.maxAbsDiff(gotFwd, refFwd) < 1e-4)
                #expect(Shard4AuditHelpers.maxAbsDiff(gotInv, refInv) < 1e-4)
            }
        }

        for width in [256, 512, 768] {
            let mx = MLXRandom.normal([2, width], dtype: .float16, key: rng)
            let mSigns = MLXRandom.uniform(low: 0.5, high: 1.5, [width], dtype: .float16, key: rng)
            let gpuBlk = FusedKroneckerRHT.blockHadamardForward(mx, signs: mSigns, blockSize: 256)
            let cpuBlk = Device.withDefaultDevice(.cpu) {
                FusedKroneckerRHT.blockHadamardForward(mx, signs: mSigns, blockSize: 256)
            }
            let gpuInvBlk = FusedKroneckerRHT.blockHadamardInverse(gpuBlk, signs: mSigns, blockSize: 256)
            let cpuInvBlk = Device.withDefaultDevice(.cpu) {
                FusedKroneckerRHT.blockHadamardInverse(gpuBlk, signs: mSigns, blockSize: 256)
            }
            #expect(Shard4AuditHelpers.maxAbsDiff(gpuBlk, cpuBlk) < 3e-3)
            #expect(Shard4AuditHelpers.maxAbsDiff(gpuInvBlk, cpuInvBlk) < 3e-3)
        }
    }

    @Test("FusedRMSNormQuantizedLinear skips input_layernorm on DFlashAttention cross-attention")
    func dFlashDraftModelSkipsInputLayerNormFusion() {
        let cfg = DFlashConfig(
            hiddenSize: 64, vocabularySize: 128, numHiddenLayers: 1,
            numAttentionHeads: 4, blockSize: 4)
        let model = DFlashDraftModel(cfg)
        quantize(model: model, groupSize: 64, bits: 4)
        let anchor = MLXArray([Int32(3)]).reshaped(1, 1)
        let targetHidden = MLXRandom.normal([1, 4, 64], dtype: .float32, key: MLXRandom.key(4299))
        let before = model(anchorTokens: anchor, targetHidden: targetHidden)
        eval(before)

        let fused = FusedRMSNormQuantizedLinear.install(in: model)
        #expect(fused == 2)
        let after = model(anchorTokens: anchor, targetHidden: targetHidden)
        eval(after)
        #expect(Shard4AuditHelpers.maxAbsDiff(before, after) < 5e-3)
    }

    @Test("FusedRMSNormQuantizedLinear skips input_layernorm on RicursiveAttention with value residual")
    func ricursiveValresModelSkipsInputLayerNormFusion() throws {
        let json = """
        {"model_type":"ricursive","vocab_size":128,"hidden_size":64,"intermediate_size":128,
         "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":4,"head_dim":16,
         "use_gated_attention":false,"use_valres":true}
        """
        let cfg = try JSONDecoder().decode(RicursiveConfiguration.self, from: Data(json.utf8))
        let model = RicursiveModel(cfg)
        quantize(model: model, groupSize: 64, bits: 4)
        let tokens = MLXArray([Int32(1), 5, 9]).reshaped(1, 3)
        let before = model(tokens, cache: nil)
        eval(before)

        let fused = FusedRMSNormQuantizedLinear.install(in: model)
        #expect(fused == 0)
        let after = model(tokens, cache: nil)
        eval(after)
        #expect(Shard4AuditHelpers.maxAbsDiff(before, after) == 0)
    }
}
