import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Fused Metal substrate, MACaQ W4A4/W4A8, FlatQuant Kronecker (P1⊗P2), and LIRQuant/ParoQuant Givens kernels (#547)")
struct MetalSubstrateAndBasisTransformTests {
    struct LinearRefFixture {
        var wDeqHost: [Float]
        var rsHost: [Float]?
        var biasHost: [Float]?
        var block: Int
        var nOut: Int
        var k: Int
        var rows: Int
        var actMode: MACaQActivationMode
        var dtype: DType
        var tol: Float
    }

    private func assertLinearCPUParity(
        gotHost: [Float], xHost: [Float], preHost: [Float], fix: LinearRefFixture
    ) {
        var expected = [Float](repeating: 0, count: fix.rows * fix.nOut)
        for r in 0 ..< fix.rows {
            let rowSlice = Array(xHost[(r * fix.k) ..< ((r + 1) * fix.k)])
            var xRot = BasaltMetalSubstrate.cpuHadamardBlock(rowSlice, scale: preHost, blockSize: fix.block)
            if fix.actMode == .w4a8 {
                xRot = stride(from: 0, to: fix.k, by: 32).flatMap { b0 in
                    BasaltMetalSubstrate.cpuFP8E4M3QuantDequantBlock32(Array(xRot[b0 ..< (b0 + 32)]))
                }
            } else if fix.actMode == .w4a4 {
                xRot = stride(from: 0, to: fix.k, by: 32).flatMap { b0 in
                    BasaltMetalSubstrate.cpuMXFP4QuantDequantBlock32(Array(xRot[b0 ..< (b0 + 32)]))
                }
            }
            for n in 0 ..< fix.nOut {
                var acc: Float = 0
                for c in 0 ..< fix.k { acc += xRot[c] * fix.wDeqHost[n * fix.k + c] }
                if let rs = fix.rsHost { acc *= rs[n] }
                if let b = fix.biasHost { acc += b[n] }
                expected[r * fix.nOut + n] = acc
            }
        }
        let roundedRef = MLXArray(expected).asType(fix.dtype).asType(.float32).asArray(Float.self)
        for idx in 0 ..< expected.count {
            let ulpAllowance = abs(expected[idx]) * (fix.dtype == .bfloat16 ? Float(8e-3) : (fix.dtype == .float16 ? Float(1e-3) : Float(1e-5)))
            #expect(abs(gotHost[idx] - roundedRef[idx]) <= max(fix.tol, ulpAllowance))
        }
    }

    private func castWeightsForDType(
        wq: MLXArray, ws: MLXArray, wb: MLXArray?, spec: MACaQLayerSpec, dtype: DType
    ) -> (MACaQPackedWeights, [Float]) {
        let qMode = spec.quantizationMode
        let wsCast = qMode == .mxfp4 ? ws : ws.asType(dtype)
        let wbCast = wb?.asType(dtype)
        let deqScales = qMode == .mxfp4 ? wsCast : wsCast.asType(.float32)
        let wDeq = MLX.dequantized(
            wq, scales: deqScales, biases: wbCast?.asType(.float32),
            groupSize: spec.groupSize, bits: spec.bits, mode: qMode, dtype: .float32
        ).asArray(Float.self)
        return (MACaQPackedWeights(weight: wq, scales: wsCast, biases: wbCast), wDeq)
    }

    @Test("MACaQFusedMatmul linear matches pure CPU Hadamard + dequantized reference across dtypes, blocks, modes, row_scale, and W4A4/W4A8 (#228)")
    func macaqFusedLinearMatchesCPUReference() {
        MLXRandom.seed(547)
        let (k, nOut) = (64, 16)
        let wDense = MLXRandom.normal([nOut, k], dtype: .float32)
        let pre = MLXRandom.uniform(low: 0.6, high: 1.4, [k], dtype: .float32)
        let rowSc = MLXRandom.uniform(low: 0.75, high: 1.25, [nOut], dtype: .float32)
        let linBias = MLXRandom.normal([nOut], dtype: .float32) * Float(0.1)

        for block in [32, 64] {
            let wRot = MACaQMath.rotateBlocks(wDense / pre.reshaped(1, k), blockSize: block)
            for (bits, group, qMode) in [(4, 32, QuantizationMode.mxfp4), (4, 32, .affine), (3, 32, .affine), (2, 64, .affine), (8, 32, .affine)] {
                let (wq, ws, wb) = MLX.quantized(wRot, groupSize: group, bits: bits, mode: qMode)
                let spec = MACaQLayerSpec(bits: bits, groupSize: group, mode: qMode, hadamardBlock: block)
                for dtype in [DType.float32, .float16, .bfloat16] {
                    let (packedCast, wDeqCast) = castWeightsForDType(wq: wq, ws: ws, wb: wb, spec: spec, dtype: dtype)
                    let preCast = pre.asType(dtype)
                    let preCastHost = preCast.asType(.float32).asArray(Float.self)
                    let rsCast = rowSc.asType(dtype)
                    let biasCast = linBias.asType(dtype)
                    let rsCastHost = rsCast.asType(.float32).asArray(Float.self)
                    let biasCastHost = biasCast.asType(.float32).asArray(Float.self)
                    let tol: Float = dtype == .bfloat16 ? 2.5e-2 : (dtype == .float16 ? 6e-3 : 5e-4)
                    for rows in [1, 2, 4, 8] {
                        let xIn = MLXRandom.normal([rows, k], dtype: .float32).asType(dtype)
                        let xHost = xIn.asType(.float32).asArray(Float.self)
                        for (hasRS, hasB, actMode) in [(false, false, MACaQActivationMode.fp16), (true, true, .fp16), (false, true, .w4a8), (true, false, .w4a4)] where dtype == .float32 || actMode == .fp16 {
                            let inputs = MACaQFusedLinearInputs(
                                packed: packedCast, preScale: preCast, spec: spec,
                                rowScale: hasRS ? rsCast : nil,
                                bias: hasB ? biasCast : nil,
                                activationMode: actMode)
                            let got = MACaQFusedMatmul.linear(xIn, inputs: inputs).asType(.float32)
                            eval(got)
                            let fix = LinearRefFixture(
                                wDeqHost: wDeqCast, rsHost: hasRS ? rsCastHost : nil, biasHost: hasB ? biasCastHost : nil,
                                block: block, nOut: nOut, k: k, rows: rows, actMode: actMode, dtype: dtype, tol: tol)
                            assertLinearCPUParity(
                                gotHost: got.asArray(Float.self), xHost: xHost, preHost: preCastHost, fix: fix)
                        }
                    }
                }
            }
        }
    }

    @Test("MACaQFusedMatmul gather MoE matches unfused reference with 1D/2D preScale, broadcast x, and negative expert indices (#228)")
    func macaqFusedGatherMoEParity() {
        MLXRandom.seed(5471)
        let (experts, nOut, k) = (4, 16, 64)
        for perExpert in [false, true] {
            for qMode in [QuantizationMode.affine, .mxfp4] {
                let sw = SwitchLinear(inputDims: k, outputDims: nOut, numExperts: experts, bias: true)
                var spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: qMode, hadamardBlock: 32)
                spec.perExpertPreScale = perExpert
                spec.hasBias = true
                let layer = MACaQSwitchLinear(replacing: sw, spec: spec)

                var wList: [MLXArray] = []
                var sList: [MLXArray] = []
                var bList: [MLXArray] = []
                for _ in 0 ..< experts {
                    let raw = MLXRandom.normal([nOut, k], dtype: .float16)
                    let (wq, ws, wb) = MLX.quantized(raw, groupSize: 32, bits: 4, mode: qMode)
                    wList.append(wq)
                    sList.append(ws)
                    if let wb { bList.append(wb) }
                }
                let preShape = perExpert ? [experts, k] : [k]
                let pre = MLXRandom.uniform(low: 0.7, high: 1.3, preShape, dtype: .float16)
                let biasArr = MLXRandom.normal([experts, nOut], dtype: .float16) * Float(0.05)
                var updates: [(String, MLXArray)] = [
                    ("weight", stacked(wList, axis: 0)),
                    ("scales", stacked(sList, axis: 0)),
                    ("pre_scale", pre),
                    ("bias", biasArr),
                ]
                if !bList.isEmpty { updates.append(("biases", stacked(bList, axis: 0))) }
                layer.update(parameters: ModuleParameters.unflattened(updates))

                let xBroadcast = MLXRandom.normal([1, 2, 1, 1, k], dtype: .float16)
                let indices = MLXArray([Int32(0), 2, 1, 3], [1, 2, 2])
                let got = layer(xBroadcast, indices).asType(.float32)

                layer.train(true)
                let ref = layer(xBroadcast, indices).asType(.float32)
                layer.train(false)
                eval(got, ref)
                #expect(abs(got - ref).max().item(Float.self) < 4e-3)

                let negIdx = MLXArray([Int32(-1), 1], [1, 1, 2])
                let xSingle = MLXRandom.normal([1, 1, 1, 1, k], dtype: .float16)
                let negOut = layer(xSingle, negIdx).asType(.float32)
                eval(negOut)
                let zeroSlice = negOut[0, 0, 0, 0].asArray(Float.self)
                let allZero = zeroSlice.allSatisfy { $0 == 0 }
                #expect(allZero)
            }
        }
    }
}

@Suite("FlatQuant Kronecker (P1⊗P2), LIRQuant/ParoQuant Givens, and TinyModel greedy parity (#547)")
struct FlatQuantAndLIRQuantTests {
    @Test("FlatQuant Kronecker (P1⊗P2) and LIRQuant/ParoQuant Givens Metal kernels match CPU float32 and preserve exact dense matmul (#228)")
    func flatQuantAndLIRQuantParity() {
        MLXRandom.seed(2410)
        let (rows, nOut, k) = (6, 24, 64)
        let x = MLXRandom.normal([rows, k], dtype: .float32)
        let lin = Linear(k, nOut, bias: true)

        let fqFactors = FlatQuantCalibrator.calibrateFactors(weight: lin.weight, activations: x, clipQuantile: 0.98)
        let wConj = FlatQuantCalibrator.conjugateWeight(lin.weight, factors: fqFactors)
        let xUnclipped = FlatQuantKernels.transform(x, p1: fqFactors.p1, p2: fqFactors.p2)
        let exactRef = matmul(x, lin.weight.asType(.float32).T)
        let conjOut = matmul(xUnclipped, wConj.T)
        eval(exactRef, conjOut)
        #expect(abs(exactRef - conjOut).max().item(Float.self) < 1e-4)

        let clipVec = fqFactors.clipBounds!
        let clipArr = clipVec.asArray(Float.self)
        let cpuFactors = FlatQuantKernels.CPUFactors(
            d1: fqFactors.p1.dim(0), d2: fqFactors.p2.dim(0),
            p1: fqFactors.p1.asArray(Float.self), p2: fqFactors.p2.asArray(Float.self),
            clipBounds: (clipArr[0], clipArr[1]))
        let cpuFQ = FlatQuantKernels.cpuTransform(x.asArray(Float.self), rows: rows, factors: cpuFactors)
        for dtype in [DType.float32, .float16, .bfloat16] {
            let gpuFQ = FlatQuantKernels.transform(
                x.asType(dtype), p1: fqFactors.p1.asType(dtype), p2: fqFactors.p2.asType(dtype), clipBounds: clipVec
            ).asType(.float32)
            eval(gpuFQ)
            let tol: Float = dtype == .bfloat16 ? 2e-2 : (dtype == .float16 ? 4e-3 : 1e-4)
            let maxDiff = zip(gpuFQ.asArray(Float.self), cpuFQ).map { abs($0 - $1) }.max() ?? 0
            #expect(maxDiff <= tol)
        }

        let fqLayer = FlatQuantCalibrator.quantizeLayer(linear: lin, activations: x, bits: 8, groupSize: 32)
        let fqOut = fqLayer(x)
        let fqUnrot = fqLayer.unrotatedDenseWeight()
        eval(fqOut, fqUnrot)
        #expect(abs(fqUnrot - lin.weight.asType(.float32)).max().item(Float.self) < 2e-2)

        let lirLayer = LIRQuantCalibrator.quantizeLayer(
            linear: lin, activations: x, options: .init(subspaceBlocks: 1, givensLayers: 2, bits: 8, groupSize: 32))
        let schedule = LIRGivensSchedule(
            numLayers: lirLayer.numLayers, pairsPerLayer: lirLayer.pairsPerLayer,
            flatPairs: lirLayer.givensPairs.asArray(Int32.self),
            flatAngles: lirLayer.givensAngles.asArray(Float.self))
        let cpuLIR = LIRQuantKernels.cpuTransform(
            x.asArray(Float.self), rows: rows,
            perm: lirLayer.perm.asArray(Int32.self),
            schedule: schedule,
            diagScale: lirLayer.diagScale.asType(.float32).asArray(Float.self))
        let basis = LIRBasisParameters(
            perm: lirLayer.perm, givensPairs: lirLayer.givensPairs, givensAngles: lirLayer.givensAngles,
            diagScale: lirLayer.diagScale, numLayers: lirLayer.numLayers, pairsPerLayer: lirLayer.pairsPerLayer)
        let gpuLIR = LIRQuantKernels.transform(x, basis: basis).asType(.float32)
        let lirUnrot = lirLayer.unrotatedDenseWeight()
        eval(gpuLIR, lirUnrot)
        let lirDiff = zip(gpuLIR.asArray(Float.self), cpuLIR).map { abs($0 - $1) }.max() ?? 0
        #expect(lirDiff < 2e-4)
        #expect(abs(lirUnrot - lin.weight.asType(.float32)).max().item(Float.self) < 2e-2)
    }

    @Test("TinyModel dense and MoE greedy token parity between fused MACaQ Metal kernels and stock MLX graph (#228)")
    func tinyModelGreedyTokenParity() throws {
        for family in [TinyModel.Family.qwen3Small, .qwen3Moe] {
            let model = try TinyModel.make(family)
            var specs: [String: MACaQLayerSpec] = [:]
            for (path, leaf) in model.leafModules().flattened() {
                if let lin = leaf as? Linear, !(lin is QuantizedLinear),
                   path != "lm_head", lin.shape.1.isMultiple(of: 32) {
                    specs[path] = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .mxfp4, hadamardBlock: 32)
                } else if let sw = leaf as? SwitchLinear, !(sw is QuantizedSwitchLinear) {
                    let inDim = sw.parameters().flattened().first(where: { $0.0 == "weight" })?.1.dim(-1) ?? 0
                    if inDim >= 32, inDim.isMultiple(of: 32) {
                        specs[path] = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .mxfp4, hadamardBlock: 32)
                    }
                }
            }
            let denseSnapshot = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            try MACaQNativeLoader.install(specs, in: model)
            var packedUpdates: [(String, MLXArray)] = []
            for (path, spec) in specs {
                guard let wOrig = denseSnapshot["\(path).weight"] else { continue }
                if wOrig.ndim == 2 {
                    let wRot = MACaQMath.rotateBlocks(wOrig.asType(.float32), blockSize: spec.hadamardBlock).asType(.float16)
                    let (wq, ws, _) = MLX.quantized(wRot, groupSize: 32, bits: 4, mode: .mxfp4)
                    packedUpdates.append(("\(path).weight", wq))
                    packedUpdates.append(("\(path).scales", ws))
                } else if wOrig.ndim == 3 {
                    var wqE: [MLXArray] = []
                    var wsE: [MLXArray] = []
                    for e in 0 ..< wOrig.dim(0) {
                        let wRot = MACaQMath.rotateBlocks(wOrig[e].asType(.float32), blockSize: spec.hadamardBlock).asType(.float16)
                        let (wq, ws, _) = MLX.quantized(wRot, groupSize: 32, bits: 4, mode: .mxfp4)
                        wqE.append(wq)
                        wsE.append(ws)
                    }
                    packedUpdates.append(("\(path).weight", stacked(wqE, axis: 0)))
                    packedUpdates.append(("\(path).scales", stacked(wsE, axis: 0)))
                }
            }
            try model.update(parameters: ModuleParameters.unflattened(packedUpdates), verify: .none)

            let prompt = MLXArray([Int32(1), 5, 9, 13], [1, 4])
            model.train(false)
            let fusedLogits = model(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits.asType(.float32)
            model.train(true)
            let stockLogits = model(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits.asType(.float32)
            model.train(false)
            eval(fusedLogits, stockLogits)
            #expect(abs(fusedLogits - stockLogits).max().item(Float.self) < 2e-3)
            let fusedGreedy = argMax(fusedLogits, axis: -1).asArray(Int32.self)
            let stockGreedy = argMax(stockLogits, axis: -1).asArray(Int32.self)
            #expect(fusedGreedy == stockGreedy)
        }
    }
}
