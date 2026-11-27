import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Metal Audit Shard 3 regressions: CPU fallback, non-finite parity, k<=0/0-D/<8 shapes, KDA dtype/template cleanup, and training autograd bypass", .serialized)
struct MetalAuditShard3RegressionTests {
    struct KDASpec {
        let batch: Int
        let steps: Int
        let hk: Int
        let hv: Int
        let dk: Int
        let dv: Int
        let dtype: DType
    }

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let delta = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(delta)
        return delta.item(Float.self)
    }

    static func allFinite(_ array: MLXArray) -> Bool {
        let flat = array.asType(.float32).reshaped(-1)
        eval(flat)
        return flat.asArray(Float.self).allSatisfy(\.isFinite)
    }

    static func routerTolerance(_ dtype: DType) -> Float {
        switch dtype {
        case .bfloat16: 2e-2
        case .float16: 2e-3
        default: 1e-5
        }
    }

    static func makeKDAInputs(_ spec: KDASpec) -> (inputs: BailingHybridKDAInputs, state: MLXArray, mask: MLXArray) {
        let qCount = spec.batch * spec.steps * spec.hk * spec.dk
        let vCount = spec.batch * spec.steps * spec.hv * spec.dv
        let gCount = spec.batch * spec.steps * spec.hv * spec.dk
        let bCount = spec.batch * spec.steps * spec.hv
        let sCount = spec.batch * spec.hv * spec.dv * spec.dk
        let q = MLXArray((0 ..< qCount).map { Float(($0 * 17) % 31 - 15) * 0.03 }, [spec.batch, spec.steps, spec.hk, spec.dk]).asType(spec.dtype)
        let k = MLXArray((0 ..< qCount).map { Float(($0 * 13) % 29 - 14) * 0.03 }, [spec.batch, spec.steps, spec.hk, spec.dk]).asType(spec.dtype)
        let v = MLXArray((0 ..< vCount).map { Float(($0 * 11) % 23 - 11) * 0.04 }, [spec.batch, spec.steps, spec.hv, spec.dv]).asType(spec.dtype)
        let decay = MLXArray((0 ..< gCount).map { exp(Float(-0.2 - Double($0 % 7) * 0.2)) }, [spec.batch, spec.steps, spec.hv, spec.dk])
        let beta = MLXArray((0 ..< bCount).map { Float(0.2 + Double($0 % 5) * 0.12) }, [spec.batch, spec.steps, spec.hv])
        let state = MLXArray((0 ..< sCount).map { Float(($0 * 7) % 19 - 9) * 0.02 }, [spec.batch, spec.hv, spec.dv, spec.dk])
        let mask = MLXArray((0 ..< spec.batch * spec.steps).map { ($0 % 4) != 2 }, [spec.batch, spec.steps])
        return (BailingHybridKDAInputs(q: q, k: k, v: v, decay: decay, beta: beta), state, mask)
    }

    @Test("FusedMoERouterKernel, FlintRouterKernel, and BailingHybridKDAKernel fall back cleanly on CPU default device and match GPU outputs (#228)")
    func cpuDefaultDeviceFallbackForAllThreeKernels() {
        let logits = MLXArray((0 ..< 32).map { Float(($0 * 19) % 23 - 11) * 0.25 }, [4, 8])
        let bias = MLXArray([Float(0.3), -0.2, 0.1, 0.0, -0.4, 0.25, 0.05, -0.1], [8])
        let kda = Self.makeKDAInputs(KDASpec(batch: 1, steps: 6, hk: 2, hv: 4, dk: 32, dv: 8, dtype: .float32))
        eval(logits, bias, kda.inputs.q, kda.inputs.k, kda.inputs.v, kda.inputs.decay, kda.inputs.beta, kda.state, kda.mask)

        #expect(FusedMoERouterKernel.supports(logits, k: 2))
        #expect(FlintRouterKernel.supports(logits, expertBias: bias, k: 2))
        #expect(BailingHybridKDAKernel.canUseKernel(kda.inputs, state: kda.state, mask: kda.mask))

        let gpuMoe = FusedMoERouterKernel.route(logits, k: 2, normTopkProb: true, softmaxBeforeTopK: true)
        let gpuMoePost = FusedMoERouterKernel.route(logits, k: 2, normTopkProb: true, softmaxBeforeTopK: false)
        let gpuFlintSig = FlintRouterKernel.route(logits, expertBias: bias, k: 2, scoreFunction: "sigmoid")
        let gpuFlintSm = FlintRouterKernel.route(logits, expertBias: nil, k: 2, scoreFunction: "softmax")
        let (gpuKdaOut, gpuKdaState) = BailingHybridKDAKernel.run(kda.inputs, state: kda.state, mask: kda.mask)
        eval(
            gpuMoe.indices, gpuMoe.weights, gpuMoePost.indices, gpuMoePost.weights,
            gpuFlintSig.indices, gpuFlintSig.weights, gpuFlintSm.indices, gpuFlintSm.weights,
            gpuKdaOut, gpuKdaState
        )

        Device.withDefaultDevice(.cpu) {
            #expect(!FusedMoERouterKernel.supports(logits, k: 2))
            #expect(!FlintRouterKernel.supports(logits, expertBias: bias, k: 2))
            #expect(!BailingHybridKDAKernel.canUseKernel(kda.inputs, state: kda.state, mask: kda.mask))

            let cpuMoe = FusedMoERouterKernel.route(logits, k: 2, normTopkProb: true, softmaxBeforeTopK: true)
            let cpuMoePost = FusedMoERouterKernel.route(logits, k: 2, normTopkProb: true, softmaxBeforeTopK: false)
            let cpuFlintSig = FlintRouterKernel.route(logits, expertBias: bias, k: 2, scoreFunction: "sigmoid")
            let cpuFlintSm = FlintRouterKernel.route(logits, expertBias: nil, k: 2, scoreFunction: "softmax")
            let (cpuOpsOut, cpuOpsState) = BailingHybridKDAOps.recurrentUpdate(kda.inputs, state: kda.state, mask: kda.mask)
            let (cpuRunOut, cpuRunState) = BailingHybridKDAKernel.run(kda.inputs, state: kda.state, mask: kda.mask)
            eval(
                cpuMoe.indices, cpuMoe.weights, cpuMoePost.indices, cpuMoePost.weights,
                cpuFlintSig.indices, cpuFlintSig.weights, cpuFlintSm.indices, cpuFlintSm.weights,
                cpuOpsOut, cpuOpsState, cpuRunOut, cpuRunState
            )

            #expect(cpuMoe.indices.asArray(Int32.self) == gpuMoe.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(cpuMoe.weights, gpuMoe.weights) < 1e-5)
            #expect(cpuMoePost.indices.asArray(Int32.self) == gpuMoePost.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(cpuMoePost.weights, gpuMoePost.weights) < 1e-5)
            #expect(cpuFlintSig.indices.asArray(Int32.self) == gpuFlintSig.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(cpuFlintSig.weights, gpuFlintSig.weights) < 1e-5)
            #expect(cpuFlintSm.indices.asArray(Int32.self) == gpuFlintSm.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(cpuFlintSm.weights, gpuFlintSm.weights) < 1e-5)
            #expect(Self.maxAbsDiff(cpuOpsOut, gpuKdaOut) < 1e-4 && Self.maxAbsDiff(cpuOpsState, gpuKdaState) < 1e-4)
            #expect(Self.maxAbsDiff(cpuRunOut, gpuKdaOut) < 1e-4 && Self.maxAbsDiff(cpuRunState, gpuKdaState) < 1e-4)
        }
    }
}

extension MetalAuditShard3RegressionTests {
    @Test("FusedMoERouterKernel matches reference and CPU fallback on all--Inf, all-NaN, +Inf, and mixed non-finite rows without NaN weights (#228)",
          arguments: [DType.float32, .float16, .bfloat16])
    func fusedMoERouterNonFiniteGPUvsCPUParity(dtype: DType) {
        let inf = Float.infinity
        let negInf = -Float.infinity
        let nan = Float.nan
        let rows: [[Float]] = [
            [negInf, negInf, negInf, negInf, negInf, negInf, negInf, negInf],
            [nan, nan, nan, nan, nan, nan, nan, nan],
            [negInf, nan, negInf, nan, negInf, nan, negInf, nan],
            [inf, 1.0, 2.0, negInf, nan, 0.5, -1.0, 0.0],
            [inf, 1.0, inf, 2.0, inf, negInf, nan, 0.5],
            [inf, inf, inf, inf, inf, inf, inf, inf],
            [1.0, 2.5, 2.5, negInf, nan, 2.5, 0.5, -2.0],
        ]
        let logits = MLXArray(rows.flatMap { $0 }, [rows.count, 8]).asType(dtype)
        eval(logits)
        let tol = Self.routerTolerance(dtype)

        for k in [2, 4] {
            for (softmaxFirst, normTopk) in [(true, true), (true, false), (false, true)] {
                let gpu = FusedMoERouterKernel.route(
                    logits, k: k, normTopkProb: normTopk, softmaxBeforeTopK: softmaxFirst)
                let ref = FusedMoERouterKernel.reference(
                    logits, k: k, normTopkProb: normTopk, softmaxBeforeTopK: softmaxFirst)
                let cpu = Device.withDefaultDevice(.cpu) {
                    FusedMoERouterKernel.route(
                        logits, k: k, normTopkProb: normTopk, softmaxBeforeTopK: softmaxFirst)
                }
                eval(gpu.indices, gpu.weights, ref.indices, ref.weights, cpu.indices, cpu.weights)

                #expect(Self.allFinite(gpu.weights) && Self.allFinite(ref.weights) && Self.allFinite(cpu.weights))
                #expect(gpu.indices.asArray(Int32.self) == ref.indices.asArray(Int32.self))
                #expect(gpu.indices.asArray(Int32.self) == cpu.indices.asArray(Int32.self))
                #expect(Self.maxAbsDiff(gpu.weights, ref.weights) < tol)
                #expect(Self.maxAbsDiff(gpu.weights, cpu.weights) < tol)
            }
        }
    }

    @Test("FlintRouterKernel sanitizes non-finite expertBias and logits identically across GPU, referenceSigmoid, softmax, and CPU fallback (#228)",
          arguments: [DType.float32, .float16, .bfloat16])
    func flintRouterNonFiniteBiasAndSoftmaxGPUvsCPUParity(dtype: DType) {
        let inf = Float.infinity
        let negInf = -Float.infinity
        let nan = Float.nan
        let rows: [[Float]] = [
            [negInf, negInf, negInf, negInf, negInf, negInf, negInf, negInf],
            [nan, nan, nan, nan, nan, nan, nan, nan],
            [inf, 1.0, inf, -1.0, nan, negInf, 0.25, 0.5],
            [0.5, 1.5, -0.5, 2.0, 0.0, -1.5, 1.0, 0.25],
        ]
        let logits = MLXArray(rows.flatMap { $0 }, [rows.count, 8]).asType(dtype)
        let dirtyBias = MLXArray([nan, inf, negInf, 0.4, -0.2, nan, inf, 0.1], [8]).asType(.float32)
        eval(logits, dirtyBias)
        let tol = Self.routerTolerance(dtype)

        for normTopk in [true, false] {
            let gpuSig = FlintRouterKernel.route(logits, expertBias: dirtyBias, k: 3, normTopkProb: normTopk, scoreFunction: "sigmoid")
            let refSig = FlintRouterKernel.referenceSigmoid(logits, expertBias: dirtyBias, k: 3, normTopkProb: normTopk)
            let cpuSig = Device.withDefaultDevice(.cpu) {
                FlintRouterKernel.route(logits, expertBias: dirtyBias, k: 3, normTopkProb: normTopk, scoreFunction: "sigmoid")
            }
            eval(gpuSig.indices, gpuSig.weights, refSig.indices, refSig.weights, cpuSig.indices, cpuSig.weights)
            #expect(Self.allFinite(gpuSig.weights) && Self.allFinite(refSig.weights))
            #expect(gpuSig.indices.asArray(Int32.self) == refSig.indices.asArray(Int32.self))
            #expect(gpuSig.indices.asArray(Int32.self) == cpuSig.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(gpuSig.weights, refSig.weights) < tol)
            #expect(Self.maxAbsDiff(gpuSig.weights, cpuSig.weights) < tol)

            let gpuSmBias = FlintRouterKernel.route(logits, expertBias: dirtyBias, k: 3, normTopkProb: normTopk, scoreFunction: "softmax")
            let refSmBias = FlintRouterKernel.reference(logits, expertBias: dirtyBias, k: 3, normTopkProb: normTopk, scoreFunction: "softmax")
            let gpuSmNil = FlintRouterKernel.route(logits, expertBias: nil, k: 3, normTopkProb: normTopk, scoreFunction: "softmax")
            let refSmNil = FlintRouterKernel.reference(logits, expertBias: nil, k: 3, normTopkProb: normTopk, scoreFunction: "softmax")
            eval(
                gpuSmBias.indices, gpuSmBias.weights, refSmBias.indices, refSmBias.weights,
                gpuSmNil.indices, gpuSmNil.weights, refSmNil.indices, refSmNil.weights
            )
            #expect(Self.allFinite(gpuSmBias.weights) && Self.allFinite(gpuSmNil.weights))
            #expect(gpuSmBias.indices.asArray(Int32.self) == refSmBias.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(gpuSmBias.weights, refSmBias.weights) < tol)
            #expect(gpuSmNil.indices.asArray(Int32.self) == refSmNil.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(gpuSmNil.weights, refSmNil.weights) < tol)
        }
    }
}

extension MetalAuditShard3RegressionTests {
    @Test("Routers preserve leading shape and weightDType on k<=0, empty, and 0-D logits without crashing")
    func routerEmptyZeroKAndZeroDContracts() {
        let logits3D = MLXArray.ones([2, 3, 8], dtype: .float16)
        let bias8 = MLXArray.zeros([8], dtype: .float32)
        for k in [0, -2] {
            let moeFast = FusedMoERouterKernel.route(logits3D, k: k, weightDType: .bfloat16)
            let moeRef = FusedMoERouterKernel.reference(logits3D, k: k, weightDType: .bfloat16)
            #expect(moeFast.indices.shape == [2, 3, 0] && moeFast.indices.dtype == .int32)
            #expect(moeFast.weights.shape == [2, 3, 0] && moeFast.weights.dtype == .bfloat16)
            #expect(moeFast.indices.shape == moeRef.indices.shape && moeFast.weights.dtype == moeRef.weights.dtype)

            for (fn, bias): (String, MLXArray?) in [("sigmoid", bias8), ("softmax", bias8), ("softmax", nil)] {
                let flFast = FlintRouterKernel.route(logits3D, expertBias: bias, k: k, scoreFunction: fn, weightDType: .bfloat16)
                let flRef = FlintRouterKernel.reference(logits3D, expertBias: bias, k: k, scoreFunction: fn, weightDType: .bfloat16)
                #expect(flFast.indices.shape == [2, 3, 0] && flFast.indices.dtype == .int32)
                #expect(flFast.weights.shape == [2, 3, 0] && flFast.weights.dtype == .bfloat16)
                #expect(flFast.indices.shape == flRef.indices.shape && flFast.weights.dtype == flRef.weights.dtype)
            }
        }

        let scalar = MLXArray(Float(1.5))
        let emptyRows = MLXArray.zeros([0, 8], dtype: .float16)
        let emptyExperts = MLXArray.zeros([2, 0], dtype: .float16)
        for (input, expectedShape): (MLXArray, [Int]) in [(scalar, [0]), (emptyRows, [0, 2]), (emptyExperts, [2, 0])] {
            let moe = FusedMoERouterKernel.route(input, k: 2, weightDType: .float16)
            #expect(moe.indices.shape == expectedShape && moe.weights.shape == expectedShape)
            #expect(moe.indices.dtype == .int32 && moe.weights.dtype == .float16)
            for (fn, bias): (String, MLXArray?) in [("sigmoid", bias8), ("softmax", bias8), ("softmax", nil)] {
                let fl = FlintRouterKernel.route(input, expertBias: bias, k: 2, scoreFunction: fn, weightDType: .float16)
                #expect(fl.indices.shape == expectedShape && fl.weights.shape == expectedShape)
                #expect(fl.indices.dtype == .int32 && fl.weights.dtype == .float16)
            }
        }
    }

    @Test("Routers handle sub-8 constant-address-space inputs and high-width V=1024/256 K=16 without register spill or dynamic index corruption (#228)")
    func routerSub8ConstantInputsAndHighWidthParity() {
        for (shape, k) in [([1, 1], 1), ([1, 4], 2), ([2, 3], 2), ([4], 2)] {
            let count = shape.reduce(1, *)
            let experts = shape.last ?? 1
            let logits = MLXArray((0 ..< count).map { Float(($0 * 11) % 13 - 6) * 0.4 }, shape)
            let bias = MLXArray((0 ..< experts).map { Float($0) * 0.15 - 0.2 }, [experts])
            eval(logits, bias)
            #expect(FusedMoERouterKernel.supports(logits, k: k))
            #expect(FlintRouterKernel.supports(logits, expertBias: bias, k: k))

            let moeFast = FusedMoERouterKernel.route(logits, k: k)
            let moeRef = FusedMoERouterKernel.reference(logits, k: k)
            let flFast = FlintRouterKernel.route(logits, expertBias: bias, k: k)
            let flRef = FlintRouterKernel.referenceSigmoid(logits, expertBias: bias, k: k)
            eval(moeFast.indices, moeFast.weights, moeRef.indices, moeRef.weights, flFast.indices, flFast.weights, flRef.indices, flRef.weights)
            #expect(moeFast.indices.asArray(Int32.self) == moeRef.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(moeFast.weights, moeRef.weights) < 1e-5)
            #expect(flFast.indices.asArray(Int32.self) == flRef.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(flFast.weights, flRef.weights) < 1e-5)
        }

        let wideVals: [Float] = (0 ..< (1024 * 5)).map { Float((($0 * 37) % 1021) - 510) * 0.01 }
        let wideLogits = MLXArray(wideVals, [1024, 5]).transposed(1, 0)
        for softmaxFirst in [true, false] {
            let fast = FusedMoERouterKernel.route(wideLogits, k: 16, normTopkProb: true, softmaxBeforeTopK: softmaxFirst)
            let ref = FusedMoERouterKernel.reference(wideLogits, k: 16, normTopkProb: true, softmaxBeforeTopK: softmaxFirst)
            eval(fast.indices, fast.weights, ref.indices, ref.weights)
            #expect(fast.indices.asArray(Int32.self) == ref.indices.asArray(Int32.self))
            #expect(Self.maxAbsDiff(fast.weights, ref.weights) < 1e-5)
        }

        let flintVals: [Float] = (0 ..< (256 * 5)).map { Float((($0 * 29) % 251) - 125) * 0.02 }
        let flintBiasVals: [Float] = (0 ..< 256).map { Float(($0 * 17) % 41 - 20) * 0.02 }
        let flintLogits = MLXArray(flintVals, [256, 5]).transposed(1, 0)
        let flintBias = MLXArray(flintBiasVals, [256])
        let flWide = FlintRouterKernel.route(flintLogits, expertBias: flintBias, k: 16, normTopkProb: true)
        let flWideRef = FlintRouterKernel.referenceSigmoid(flintLogits, expertBias: flintBias, k: 16, normTopkProb: true)
        eval(flWide.indices, flWide.weights, flWideRef.indices, flWideRef.weights)
        #expect(flWide.indices.asArray(Int32.self) == flWideRef.indices.asArray(Int32.self))
        #expect(Self.maxAbsDiff(flWide.weights, flWideRef.weights) < 1e-5)
    }
}

extension MetalAuditShard3RegressionTests {
    @Test("BailingHybridKDAKernel rejects float64 and complex64 instead of crashing MSL compilation")
    func bailingKDARejectsFloat64AndComplex64() {
        #expect(BailingHybridKDAKernel.supportedDTypes == [.float32, .float16, .bfloat16])

        let base = Self.makeKDAInputs(KDASpec(batch: 1, steps: 2, hk: 1, hv: 1, dk: 32, dv: 4, dtype: .float32))
        let f64Inputs = BailingHybridKDAInputs(
            q: base.inputs.q.asType(.float64, stream: .cpu),
            k: base.inputs.k.asType(.float64, stream: .cpu),
            v: base.inputs.v.asType(.float64, stream: .cpu),
            decay: base.inputs.decay.asType(.float64, stream: .cpu),
            beta: base.inputs.beta.asType(.float64, stream: .cpu)
        )
        #expect(f64Inputs.q.dtype == .float64)
        #expect(!BailingHybridKDAKernel.canUseKernel(f64Inputs))
        let (gpuFallbackOut, gpuFallbackState) = BailingHybridKDAKernel.run(f64Inputs, state: nil)
        let f64Mask = MLXArray([Float(1.0), Float(0.0)], [1, 2]).asType(.float64, stream: .cpu)
        #expect(!BailingHybridKDAKernel.canUseKernel(base.inputs, mask: f64Mask))
        let (maskedOut, maskedState) = BailingHybridKDAKernel.run(base.inputs, state: nil, mask: f64Mask)
        eval(gpuFallbackOut, gpuFallbackState, maskedOut, maskedState)
        #expect(gpuFallbackOut.shape == [1, 2, 1, 4] && gpuFallbackOut.dtype == .float64)
        #expect(maskedOut.shape == [1, 2, 1, 4])
        Device.withDefaultDevice(.cpu) {
            let (outOps, stateOps) = BailingHybridKDAOps.recurrentUpdate(f64Inputs, state: nil)
            let (outRun, stateRun) = BailingHybridKDAKernel.run(f64Inputs, state: nil)
            eval(outOps, stateOps, outRun, stateRun)
            #expect(outOps.shape == [1, 2, 1, 4] && outOps.dtype == .float64)
            #expect(outRun.shape == [1, 2, 1, 4] && outRun.dtype == .float64)
            #expect(Self.maxAbsDiff(outOps, outRun) < 1e-5 && Self.maxAbsDiff(stateOps, stateRun) < 1e-5)
            #expect(Self.maxAbsDiff(outOps, gpuFallbackOut) < 1e-5 && Self.maxAbsDiff(stateOps, gpuFallbackState) < 1e-5)
        }

        let c64Inputs = BailingHybridKDAInputs(
            q: base.inputs.q.asType(.complex64),
            k: base.inputs.k.asType(.complex64),
            v: base.inputs.v.asType(.complex64),
            decay: base.inputs.decay,
            beta: base.inputs.beta
        )
        #expect(!BailingHybridKDAKernel.canUseKernel(c64Inputs))
        let (c64Out, c64State) = BailingHybridKDAOps.recurrentUpdate(c64Inputs, state: nil)
        eval(c64Out, c64State)
        #expect(c64Out.shape == [1, 2, 1, 4] && c64Out.dtype == .complex64)
    }

    @Test("BailingHybridKDAKernel matches ops reference across runtime Dv/Hk/Hv shapes, sub-8 v/beta/mask, and genuinely non-contiguous slices (#228)")
    func bailingKDARuntimeShapesAndNonContiguousParity() {
        let shapeCases: [(KDASpec, Bool)] = [
            (KDASpec(batch: 1, steps: 1, hk: 1, hv: 1, dk: 32, dv: 1, dtype: .float32), true),
            (KDASpec(batch: 1, steps: 5, hk: 1, hv: 3, dk: 32, dv: 3, dtype: .float32), false),
            (KDASpec(batch: 2, steps: 6, hk: 2, hv: 6, dk: 32, dv: 7, dtype: .float32), true),
        ]
        for (spec, masked) in shapeCases {
            let sample = Self.makeKDAInputs(spec)
            let mask: MLXArray? = masked ? sample.mask : nil
            #expect(BailingHybridKDAKernel.canUseKernel(sample.inputs, state: sample.state, mask: mask))
            let (gpuOut, gpuState) = BailingHybridKDAKernel.run(sample.inputs, state: sample.state, mask: mask)
            let (refOut, refState) = BailingHybridKDAOps.recurrentUpdateOps(sample.inputs, state: sample.state, mask: mask)
            eval(gpuOut, gpuState, refOut, refState)
            #expect(Self.maxAbsDiff(gpuOut, refOut) < 1e-4)
            #expect(Self.maxAbsDiff(gpuState, refState) < 1e-4)
        }

        let qVals: [Float] = (0 ..< 1280).map { (i: Int) -> Float in Float((i * 17) % 31 - 15) * 0.03 }
        let kVals: [Float] = (0 ..< 1280).map { (i: Int) -> Float in Float((i * 13) % 29 - 14) * 0.03 }
        let vVals: [Float] = (0 ..< 960).map { (i: Int) -> Float in Float((i * 11) % 23 - 11) * 0.04 }
        let gVals: [Float] = (0 ..< 2560).map { (i: Int) -> Float in exp(Float(-0.2 - Double(i % 7) * 0.2)) }
        let bVals: [Float] = (0 ..< 80).map { (i: Int) -> Float in Float(0.2 + Double(i % 5) * 0.12) }
        let sVals: [Float] = (0 ..< 3072).map { (i: Int) -> Float in Float((i * 7) % 19 - 9) * 0.02 }
        let rawQ = MLXArray(qVals, [2, 2, 10, 32])
        let rawK = MLXArray(kVals, [2, 2, 10, 32])
        let rawV = MLXArray(vVals, [2, 4, 10, 12])
        let rawG = MLXArray(gVals, [2, 4, 10, 32])
        let rawBeta = MLXArray(bVals, [2, 10, 4])
        let rawS = MLXArray(sVals, [2, 12, 4, 32])
        let rawMask = MLXArray((0 ..< 20).map { ($0 % 3) != 1 }, [2, 10])
        eval(rawQ, rawK, rawV, rawG, rawBeta, rawS, rawMask)

        let strided = BailingHybridKDAInputs(
            q: rawQ.transposed(0, 2, 1, 3)[0..., .stride(by: 2), 0..., 0...],
            k: rawK.transposed(0, 2, 1, 3)[0..., .stride(by: 2), 0..., 0...],
            v: rawV.transposed(0, 2, 1, 3)[0..., .stride(by: 2), 0..., 0...],
            decay: rawG.transposed(0, 2, 1, 3)[0..., .stride(by: 2), 0..., 0...],
            beta: rawBeta[0..., .stride(by: 2), 0...]
        )
        let ncState = rawS.transposed(0, 2, 1, 3)
        let ncMask = rawMask[0..., .stride(by: 2)]
        let (ncGpuOut, ncGpuState) = BailingHybridKDAKernel.run(strided, state: ncState, mask: ncMask)
        let (ncRefOut, ncRefState) = BailingHybridKDAOps.recurrentUpdateOps(strided, state: ncState, mask: ncMask)
        eval(ncGpuOut, ncGpuState, ncRefOut, ncRefState)
        #expect(Self.maxAbsDiff(ncGpuOut, ncRefOut) < 1e-4)
        #expect(Self.maxAbsDiff(ncGpuState, ncRefState) < 1e-4)
    }
}

extension MetalAuditShard3RegressionTests {
    @Test("Training mode bypasses FlintRouterKernel so valueAndGrad succeeds on FlintSparseMoEBlock and NemotronMoEGateModule")
    func autogradValueAndGradBypassOnFlintAndNemotronMoE() throws {
        let flintModel = try #require(try TinyModel.make(.flint) as? FlintModel)
        let flintMoE = try #require(flintModel.model.layers[2].mlp as? FlintSparseMoEBlock)
        let xFlint = MLXRandom.normal([1, 4, flintModel.configuration.hiddenSize], scale: 0.5, key: MLXRandom.key(301))
        eval(xFlint)
        flintMoE.train(false)
        let flintInfer = flintMoE(xFlint)
        flintMoE.train(true)
        let flintTrain = flintMoE(xFlint)
        eval(flintInfer, flintTrain)
        #expect(Self.maxAbsDiff(flintInfer, flintTrain) < 1e-4)
        let flintVG = valueAndGrad(model: flintMoE) { block, arrays in [block(arrays[0]).square().mean()] }
        let (flintLosses, flintGrads) = flintVG(flintMoE, [xFlint])
        let flintGradMap = Dictionary(uniqueKeysWithValues: flintGrads.flattened())
        eval(flintLosses + Array(flintGradMap.values))
        let flintGateGrad = try #require(flintGradMap["gate.weight"])
        #expect(Self.allFinite(flintLosses[0]) && Self.allFinite(flintGateGrad) && abs(flintGateGrad).max().item(Float.self) > 0)

        var softmaxCfg = try JSONDecoder.json5().decode(
            FlintConfiguration.self, from: Data(TinyModel.Family.flint.config.utf8)).validated()
        softmaxCfg.routerScoreFunction = "softmax"
        let softmaxMoE = FlintSparseMoEBlock(softmaxCfg, layerIdx: 2)
        eval(softmaxMoE)
        softmaxMoE.train(true)
        let smVG = valueAndGrad(model: softmaxMoE) { block, arrays in [block(arrays[0]).square().mean()] }
        let (smLosses, smGrads) = smVG(softmaxMoE, [xFlint])
        let smGradMap = Dictionary(uniqueKeysWithValues: smGrads.flattened())
        eval(smLosses + Array(smGradMap.values))
        let smGateGrad = try #require(smGradMap["gate.weight"])
        #expect(Self.allFinite(smLosses[0]) && Self.allFinite(smGateGrad) && abs(smGateGrad).max().item(Float.self) > 0)

        let nemJSON = TinyModel.Family.sigmoidMoe.config.replacingOccurrences(of: "\"n_group\": 2", with: "\"n_group\": 1")
        let nemCfg = try JSONDecoder.json5().decode(NemotronHConfiguration.self, from: Data(nemJSON.utf8))
        let nemGate = NemotronMoEGateModule(nemCfg)
        nemGate.update(parameters: ModuleParameters.unflattened([
            ("weight", MLXRandom.normal([nemCfg.nRoutedExperts, nemCfg.hiddenSize], scale: 0.2, key: MLXRandom.key(302))),
            ("e_score_correction_bias", MLXRandom.normal([nemCfg.nRoutedExperts], scale: 0.1, key: MLXRandom.key(303))),
        ]))
        let xNem = MLXRandom.normal([1, 4, nemCfg.hiddenSize], scale: 0.5, key: MLXRandom.key(304))
        eval(nemGate, xNem)
        nemGate.train(false)
        let nemInfer = nemGate(xNem)
        nemGate.train(true)
        let nemTrain = nemGate(xNem)
        eval(nemInfer.indices, nemInfer.weights, nemTrain.indices, nemTrain.weights)
        #expect(nemInfer.indices.asArray(Int32.self) == nemTrain.indices.asArray(Int32.self))
        #expect(Self.maxAbsDiff(nemInfer.weights, nemTrain.weights) < 1e-5)
        let nemVG = valueAndGrad(model: nemGate) { gate, arrays in [gate(arrays[0]).weights.square().mean()] }
        let (nemLosses, nemGrads) = nemVG(nemGate, [xNem])
        let nemGradMap = Dictionary(uniqueKeysWithValues: nemGrads.flattened())
        eval(nemLosses + Array(nemGradMap.values))
        let nemWeightGrad = try #require(nemGradMap["weight"])
        #expect(Self.allFinite(nemLosses[0]) && Self.allFinite(nemWeightGrad) && abs(nemWeightGrad).max().item(Float.self) > 0)
    }

    @Test("Training mode bypasses BailingHybridKDAKernel so valueAndGrad succeeds on BailingHybridLinearAttention")
    func autogradValueAndGradBypassOnBailingKDA() throws {
        let bailingModel = try #require(try TinyModel.make(.bailingHybrid) as? BailingHybridModel)
        let kda = try #require(bailingModel.model.layers[0].attention as? BailingHybridLinearAttention)
        let xKDA = MLXRandom.normal([1, 6, bailingModel.configuration.hiddenSize], scale: 0.4, key: MLXRandom.key(305))
        eval(xKDA)
        kda.train(false)
        let kdaInfer = kda(xKDA, cache: nil)
        kda.train(true)
        let kdaTrain = kda(xKDA, cache: nil)
        eval(kdaInfer, kdaTrain)
        #expect(Self.maxAbsDiff(kdaInfer, kdaTrain) < 1e-4)
        let kdaVG = valueAndGrad(model: kda) { module, arrays in [module(arrays[0], cache: nil).square().mean()] }
        let (kdaLosses, kdaGrads) = kdaVG(kda, [xKDA])
        let kdaGradMap = Dictionary(uniqueKeysWithValues: kdaGrads.flattened())
        eval(kdaLosses + Array(kdaGradMap.values))
        let qGrad = try #require(kdaGradMap["q_proj.weight"])
        let vGrad = try #require(kdaGradMap["v_proj.weight"])
        let oGrad = try #require(kdaGradMap["o_proj.weight"])
        #expect(Self.allFinite(kdaLosses[0]) && Self.allFinite(qGrad) && Self.allFinite(vGrad) && Self.allFinite(oGrad))
        #expect(abs(qGrad).max().item(Float.self) > 0 && abs(vGrad).max().item(Float.self) > 0 && abs(oGrad).max().item(Float.self) > 0)
    }
}
