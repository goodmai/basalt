import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("MACaQ-R discrete GGUF Flip-Refinement, QuIP# E8P 2-bit lattice Metal kernel & TurboBoA Babai-LLL (#549)")
struct GGUFMacAqE8PAndTurboBoATests {
    @Test("GGUFMacAqRefiner unpacks and repacks Q4_K, Q5_K, and Q6_K bit-identically and reduces Hessian loss")
    func ggufMacAqRoundTripAndHessianRefinement() throws {
        MLXRandom.seed(5490)
        let rows = 4
        let cols = 256
        let target = MLXRandom.normal([rows, cols], dtype: .float32) * Float(1.5)
        let xCal = MLXRandom.normal([64, cols], dtype: .float32)
        let hessian = (matmul(xCal.T, xCal) / Float(64.0)) + eye(cols, dtype: .float32) * Float(0.05)
        eval(target, hessian)
        let targetHost = target.asArray(Float.self)

        var q4Bytes = [UInt8](repeating: 0, count: rows * 144)
        for r in 0 ..< rows {
            GSQRCOBlockEncoder.encodeQ4K(targetHost, v0: r * 256, into: &q4Bytes, b0: r * 144)
        }
        let q4State = try GGUFMacAqRefiner.unpackKQuant(bytes: q4Bytes, type: .q4K, rows: rows, cols: cols)
        let q4Deq = GGUFDequant.floats(Data(q4Bytes), type: .q4K, count: rows * cols)
        let q4GridVals = q4State.dequantizedValues().asArray(Float.self)
        let q4MaxDiff = zip(q4Deq, q4GridVals).map { abs($0 - $1) }.max() ?? 0
        #expect(q4MaxDiff < 1e-5)
        let q4Repack = GGUFMacAqRefiner.repackKQuant(
            originalBytes: q4Bytes, codes: q4State.codes.asArray(Int32.self), type: .q4K)
        #expect(q4Repack == q4Bytes)

        let perturbedTarget = target + MLXRandom.normal([rows, cols], dtype: .float32) * Float(0.25)
        let (refinedQ4, flipRes) = try GGUFMacAqRefiner.refineTensorBytes(
            rawBytes: q4Bytes, type: .q4K, targetWeight: perturbedTarget, hessian: hessian, maxIterations: 32)
        #expect(refinedQ4.count == q4Bytes.count)
        #expect(flipRes.lossAfter <= flipRes.lossBefore)
        let refinedVals = MLXArray(GGUFDequant.floats(Data(refinedQ4), type: .q4K, count: rows * cols), [rows, cols])
        let recomputedLoss = MACaQMath.hessianLoss(values: refinedVals, target: perturbedTarget, hessian: hessian)
        #expect(abs(recomputedLoss - flipRes.lossAfter) < 1e-2)

        for kType in [GGMLType.q5K, .q6K] {
            let blkBytes = kType.block.bytes
            var raw = [UInt8](repeating: 0, count: rows * blkBytes)
            for i in 0 ..< raw.count { raw[i] = UInt8((i * 37 + 13) & 0xff) }
            if kType == .q5K {
                for r in 0 ..< rows {
                    raw[r * blkBytes] = 0x00
                    raw[r * blkBytes + 1] = 0x38
                    raw[r * blkBytes + 2] = 0x00
                    raw[r * blkBytes + 3] = 0x34
                }
            } else {
                for r in 0 ..< rows {
                    raw[r * blkBytes + 208] = 0x00
                    raw[r * blkBytes + 209] = 0x38
                }
            }
            let st = try GGUFMacAqRefiner.unpackKQuant(bytes: raw, type: kType, rows: rows, cols: cols)
            let deqRef = GGUFDequant.floats(Data(raw), type: kType, count: rows * cols)
            let gridVals = st.dequantizedValues().asArray(Float.self)
            let diff = zip(deqRef, gridVals).map { abs($0 - $1) }.max() ?? 0
            #expect(diff < 1e-5)
            let repacked = GGUFMacAqRefiner.repackKQuant(
                originalBytes: raw, codes: st.codes.asArray(Int32.self), type: kType)
            #expect(repacked == raw)
        }
    }

    @Test("QuIP# E8P 2-bit Gosset lattice codebook satisfies exact E8 parity and fused Metal kernel matches CPU float32 and TinyModel (#228)")
    func e8pLatticeAndFusedMetalParity() throws {
        MLXRandom.seed(2402)
        for trial in 0 ..< 32 {
            let vec = (0 ..< 8).map { j in sinf(Float(trial * 8 + j) * 0.73) * 2.1 }
            let code = E8PLattice.nearestCode(vec)
            let pt = E8PLattice.cpuDecodeCode(code)
            #expect(E8PLattice.isValidE8Point(pt))
        }

        let (nOut, k) = (16, 64)
        let w = MLXRandom.normal([nOut, k], dtype: .float32)
        let bias = MLXRandom.normal([nOut], dtype: .float32) * Float(0.05)
        let pack = E8PMetalKernels.quantizeWeight(w, seed: 77)
        eval(pack.packedCodes, pack.rowScale, pack.rhtSigns, bias)
        let wordsHost = pack.packedCodes.asArray(UInt32.self)
        let signsHost = pack.rhtSigns.asArray(Float.self)
        let rsHost = pack.rowScale.asArray(Float.self)
        let biasHost = bias.asArray(Float.self)

        for dtype in [DType.float32, .float16, .bfloat16] {
            for rows in [1, 4, 8] {
                let xIn = MLXRandom.normal([rows, k], dtype: .float32).asType(dtype)
                let xHost = xIn.asType(.float32).asArray(Float.self)
                let cpuPack = E8PMetalKernels.QuantizedPack(
                    packedCodes: pack.packedCodes,
                    rowScale: pack.rowScale.asType(dtype),
                    rhtSigns: pack.rhtSigns.asType(dtype))
                let gpuOut = E8PMetalKernels.matmul(xIn, pack: cpuPack, bias: bias.asType(dtype)).asType(.float32)
                eval(gpuOut)
                let cPack = E8PMetalKernels.CPUPack(
                    packedWords: wordsHost, nOut: nOut,
                    rhtSigns: cpuPack.rhtSigns.asType(.float32).asArray(Float.self),
                    rowScale: cpuPack.rowScale.asType(.float32).asArray(Float.self),
                    bias: bias.asType(dtype).asType(.float32).asArray(Float.self))
                let cpuRef = MLXArray(E8PMetalKernels.cpuMatmul(xHost, rows: rows, pack: cPack)).asType(dtype).asType(.float32).asArray(Float.self)
                let tol: Float = dtype == .bfloat16 ? 1.6e-2 : (dtype == .float16 ? 2e-3 : 5e-4)
                for (g, r) in zip(gpuOut.asArray(Float.self), cpuRef) {
                    let ulpAllowance = abs(r) * (dtype == .bfloat16 ? Float(8e-3) : (dtype == .float16 ? Float(1e-3) : Float(1e-5)))
                    #expect(abs(g - r) <= max(tol, ulpAllowance))
                }
                _ = signsHost
                _ = rsHost
                _ = biasHost
            }
        }

        let model = try TinyModel.make(.qwen3Small)
        var swaps: [(String, Module)] = []
        for (path, leaf) in model.leafModules().flattened() {
            if let lin = leaf as? Linear, !(lin is QuantizedLinear),
               path != "lm_head", lin.shape.1.isMultiple(of: 32) {
                let p = E8PMetalKernels.quantizeWeight(lin.weight)
                swaps.append((path, E8PLinear(pack: p, bias: lin.bias)))
            }
        }
        model.update(modules: ModuleChildren.unflattened(swaps))
        let logits = model(LMInput.Text(tokens: MLXArray([Int32(1), 2, 3, 4], [1, 4])), cache: nil, state: nil).logits
        eval(logits)
        #expect(logits.shape == [1, 4, 256])
        let finite = logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite }
        #expect(finite)
    }

    @Test("BabaiLLLCompensator LLL basis reduction lowers orthogonality defect and TurboBoA 8-step beats RTN Hessian loss")
    func lllReductionAndTurboBoACompensation() {
        let skewed: [Float] = [
            1.0, 3.0, 2.0, -4.0,
            0.0, 1.0, -2.0, 3.0,
            0.0, 0.0, 1.0, 2.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        let red = BabaiLLLCompensator.lllReduce(basis: skewed, dim: 4)
        #expect(red.defectAfter < red.defectBefore)
        let uMat = MLXArray(red.unimodularTransform.map(Float.init), [4, 4])
        let bOrig = MLXArray(skewed, [4, 4])
        let bRed = MLXArray(red.reducedBasis, [4, 4])
        let recon = matmul(bOrig, uMat)
        eval(recon)
        #expect(abs(recon - bRed).max().item(Float.self) < 1e-4)

        MLXRandom.seed(2602)
        let k = 128
        let w = MLXRandom.normal([32, k], dtype: .float32)
        let latent = MLXRandom.normal([96, 16], dtype: .float32)
        let mix = MLXRandom.normal([16, k], dtype: .float32)
        let xCorr = matmul(latent, mix) + MLXRandom.normal([96, k], dtype: .float32) * Float(0.15)
        let hCorr = matmul(xCorr.T, xCorr) / Float(96.0)

        let affineRes = BabaiLLLCompensator.turboBoAQuantize(
            weight: w, hessian: hCorr,
            spec: .init(bits: 3, groupSize: 32, mode: .affine, numSteps: 8, reorderChannels: true))
        #expect(affineRes.lossBoA < affineRes.lossRTN)

        let mxRes = BabaiLLLCompensator.turboBoAQuantize(
            weight: w, hessian: hCorr,
            spec: .init(bits: 4, groupSize: 32, mode: .mxfp4, numSteps: 8, reorderChannels: true))
        #expect(mxRes.lossBoA <= mxRes.lossRTN)
    }
}
