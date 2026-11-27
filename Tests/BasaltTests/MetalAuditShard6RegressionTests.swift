import Foundation
import MLX
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = abs(a.asType(.float32) - b.asType(.float32)).max()
    eval(diff)
    return diff.item(Float.self)
}

@Suite("Metal audit shard 6: ThreeBitQMV & SmallMQuantizedLinear regressions", .serialized)
struct MetalAuditShard6ThreeBitAndSmallMTests {
    @Test("ThreeBitQMV compiles and matches MLX with <8-element constant buffers, odd N, all dtypes, and strict fits/handles guards")
    func threeBitQMVConstantBuffersAndGuards() throws {
        #expect(!ThreeBitQMV.fits(bits: 3, mode: .affine, groupSize: 32, inputs: 48, hasBiases: true))
        #expect(!ThreeBitQMV.fits(bits: 3, mode: .affine, groupSize: 64, inputs: 0, hasBiases: true))
        #expect(!ThreeBitQMV.fits(bits: 3, mode: .affine, groupSize: 64, inputs: 64, outputs: 0, hasBiases: true))
        #expect(ThreeBitQMV.fits(bits: 3, mode: .affine, groupSize: 32, inputs: 32, outputs: 1, hasBiases: true))

        let badScales = MLXArray.ones([4, 2], dtype: .float16)
        #expect(!ThreeBitQMV.handlesSingleRow(MLXArray.ones([1, 96], dtype: .float16), scales: badScales, biases: badScales))
        #expect(!ThreeBitQMV.handlesSingleRow(MLXArray.ones([1, 64], dtype: .float16), scales: badScales, biases: badScales, groupSize: 64))

        for dtype in [DType.float16, .bfloat16, .float32] {
            let tol: Float = dtype == .float32 ? 1e-4 : 3e-2
            for outputs in [1, 5] {
                let wDense = MLXRandom.normal([1, outputs, 32], scale: 0.1, key: MLXRandom.key(UInt64(outputs + 71)))
                let (wq3, s3, bOpt) = MLX.quantized(wDense, groupSize: 32, bits: 3, mode: .affine)
                let b3 = try #require(bOpt)
                let (w, s, b) = (wq3[0], s3[0].asType(dtype), b3[0].asType(dtype))
                let x = MLXRandom.normal([1, 1, 32], key: MLXRandom.key(72)).asType(dtype)
                #expect(ThreeBitQMV.handlesSingleRow(x, scales: s, biases: b, groupSize: 32))
                let got = ThreeBitQMV.matmul(x, indices: nil, weight: w, scales: s, biases: b, groupSize: 32)
                let ref = MLX.quantizedMM(x, w, scales: s, biases: b, transpose: true, groupSize: 32, bits: 3, mode: .affine)
                eval(got, ref)
                #expect(got.shape == ref.shape && got.dtype == dtype)
                #expect(maxAbsDiff(got, ref) < tol)

                let idx = MLXArray([Int32(0)], [1])
                let xG = MLXRandom.normal([1, 1, 32], key: MLXRandom.key(73)).asType(dtype)
                let sG = s3.asType(dtype)
                let bG = b3.asType(dtype)
                #expect(ThreeBitQMV.handlesGather(xG, indices: idx, scales: sG, biases: bG, groupSize: 32))
                let gotG = ThreeBitQMV.matmul(xG, indices: idx, weight: wq3, scales: sG, biases: bG, groupSize: 32)
                let refG = MLX.gatherQuantizedMM(xG, wq3, scales: sG, biases: bG, rhsIndices: idx, transpose: true, groupSize: 32, bits: 3, mode: .affine)
                eval(gotG, refG)
                #expect(gotG.shape == refG.shape)
                #expect(maxAbsDiff(gotG, refG) < tol)
            }
        }
    }

    @Test("ThreeBitQMV falls back safely on CPU default device for dense, gather, and OOB expert indices")
    func threeBitQMVCPUFallbackParity() throws {
        let wDense = MLXRandom.normal([2, 8, 64], scale: 0.1, key: MLXRandom.key(74))
        let (wq, sRaw, bOpt) = MLX.quantized(wDense, groupSize: 64, bits: 3, mode: .affine)
        let s = sRaw.asType(.float32)
        let b = try #require(bOpt).asType(.float32)
        let xDense = MLXRandom.normal([1, 1, 64], key: MLXRandom.key(75))
        let xGather = MLXRandom.normal([2, 1, 64], key: MLXRandom.key(76))
        let idx = MLXArray([Int32(1), Int32(-1)], [2])
        eval(wq, s, b, xDense, xGather, idx)

        let gpuDense = ThreeBitQMV.matmul(xDense, indices: nil, weight: wq[0], scales: s[0], biases: b[0], groupSize: 64)
        let gpuGather = ThreeBitQMV.matmul(xGather, indices: idx, weight: wq, scales: s, biases: b, groupSize: 64)
        eval(gpuDense, gpuGather)

        let (cpuHandlesDense, cpuHandlesGather, cpuDense, cpuGather) = Device.withDefaultDevice(.cpu) {
            let hd = ThreeBitQMV.handlesSingleRow(xDense, scales: s[0], biases: b[0], groupSize: 64)
            let hg = ThreeBitQMV.handlesGather(xGather, indices: idx, scales: s, biases: b, groupSize: 64)
            let d = ThreeBitQMV.matmul(xDense, indices: nil, weight: wq[0], scales: s[0], biases: b[0], groupSize: 64)
            let g = ThreeBitQMV.matmul(xGather, indices: idx, weight: wq, scales: s, biases: b, groupSize: 64)
            eval(d, g)
            return (hd, hg, d, g)
        }
        #expect(!cpuHandlesDense && !cpuHandlesGather)
        #expect(maxAbsDiff(gpuDense, cpuDense) < 1e-4)
        #expect(maxAbsDiff(gpuGather, cpuGather) < 1e-4)
        #expect(abs(cpuGather[1]).max().item(Float.self) == 0)
    }

    @Test("SmallMQuantizedLinear rejects 0-output, mismatched K, K % groupSize != 0, supports training autograd, and matches at K < 256")
    func smallMGuardsAndTrainingParity() {
        let zeroOut = QuantizedLinear(
            weight: MLXArray.zeros([0, 8], dtype: .uint32), bias: nil,
            scales: MLXArray.zeros([0, 1], dtype: .float16), biases: MLXArray.zeros([0, 1], dtype: .float16),
            groupSize: 64, bits: 4, mode: .affine)
        let zeroIn = QuantizedLinear(
            weight: MLXArray.zeros([32, 0], dtype: .uint32), bias: nil,
            scales: MLXArray.zeros([32, 0], dtype: .float16), biases: MLXArray.zeros([32, 0], dtype: .float16),
            groupSize: 64, bits: 4, mode: .affine)
        let badGroup = QuantizedLinear(
            weight: MLXArray.zeros([32, 8], dtype: .uint32), bias: nil,
            scales: MLXArray.zeros([32, 1], dtype: .float16), biases: MLXArray.zeros([32, 1], dtype: .float16),
            groupSize: 128, bits: 4, mode: .affine)
        #expect(!SmallMQuantizedMatmul.fits(zeroOut))
        #expect(!SmallMQuantizedMatmul.fits(zeroIn))
        #expect(!SmallMQuantizedMatmul.fits(badGroup))

        for k in [64, 128] {
            let dense = MLXRandom.normal([32, k], scale: 0.05, key: MLXRandom.key(UInt64(k + 80))).asType(.float32)
            let stock = QuantizedLinear(weight: dense, bias: nil, groupSize: 64, bits: 4)
            let layer = SmallMQuantizedLinear(replacing: stock)
            let x = MLXRandom.normal([1, 4, k], key: MLXRandom.key(81)).asType(.float32)
            #expect(SmallMQuantizedMatmul.handles(x, layer: layer))
            #expect(!SmallMQuantizedMatmul.handles(MLXArray(Float(1)), layer: layer))
            #expect(!SmallMQuantizedMatmul.handles(MLXRandom.normal([1, 4, k + 64], key: MLXRandom.key(82)), layer: layer))
            let got = layer(x)
            let ref = stock(x)
            eval(got, ref)
            #expect(maxAbsDiff(got, ref) < 1e-4)

            layer.train(true)
            #expect(!SmallMQuantizedMatmul.handles(x, layer: layer))
            let vg = valueAndGrad({ arrays in [layer(arrays[0]).square().sum()] }, argumentNumbers: [0])
            let (_, grads) = vg([x])
            let gradNorm = grads[0].square().sum()
            eval(gradNorm)
            #expect(gradNorm.item(Float.self).isFinite && gradNorm.item(Float.self) > 0)
            layer.train(false)
        }

        let w = MLXRandom.normal([32, 128], scale: 0.05, key: MLXRandom.key(83)).asType(.bfloat16)
        let (qw, sc, bi) = MLX.quantized(w, groupSize: 64, bits: 4, mode: .affine)
        let spec = MACaQLayerSpec(bits: 4, groupSize: 64, mode: .affine, hadamardBlock: 64)
        let macaq = MACaQQuantizedLinear(
            packed: MACaQPackedWeights(weight: qw, scales: sc, biases: bi),
            bias: nil, preScale: MLXArray.ones([128], dtype: .bfloat16), spec: spec)
        macaq.train(false)
        let xM = MLXRandom.normal([1, 4, 128], key: MLXRandom.key(84)).asType(.bfloat16)
        let baseM = macaq(xM)
        #expect(SmallMQuantizedLinear.install(in: macaq) == 1)
        let smallM = macaq(xM)
        eval(baseM, smallM)
        #expect(maxAbsDiff(baseM, smallM) < 3e-2)
    }

    @Test("SmallMQuantizedLinear and SmallMQuantizedMatmul fall back safely on CPU default device")
    func smallMCPUFallbackParity() throws {
        let dense = MLXRandom.normal([32, 128], scale: 0.05, key: MLXRandom.key(85)).asType(.float32)
        let stock = QuantizedLinear(weight: dense, bias: nil, groupSize: 64, bits: 4)
        let layer = SmallMQuantizedLinear(replacing: stock)
        let biases = try #require(layer.biases)
        let x = MLXRandom.normal([1, 4, 128], key: MLXRandom.key(86)).asType(.float32)
        eval(layer, x)

        let gpuOut = layer(x)
        eval(gpuOut)

        let (cpuHandles, cpuLayer, cpuDirect) = Device.withDefaultDevice(.cpu) {
            let h = SmallMQuantizedMatmul.handles(x, layer: layer)
            let a = layer(x)
            let b = SmallMQuantizedMatmul.matmul(x, weight: layer.weight, scales: layer.scales, biases: biases, groupSize: 64)
            eval(a, b)
            return (h, a, b)
        }
        #expect(!cpuHandles)
        #expect(maxAbsDiff(gpuOut, cpuLayer) < 1e-4)
        #expect(maxAbsDiff(gpuOut, cpuDirect) < 1e-4)
    }
}

@Suite("Metal audit shard 6: MACaQFusedMatmul & MACaQFusedKernels regressions", .serialized)
struct MetalAuditShard6MACaQTests {
    @Test("MACaQFusedMatmul rejects affine with nil biases and handles nOut == 0 / k == 0 safely")
    func macaqFusedMatmulAffineNilBiasesAndEmptyGuards() {
        let wDense = MLXRandom.normal([8, 64], scale: 0.1, key: MLXRandom.key(90))
        let (wq, ws, _) = MLX.quantized(wDense, groupSize: 32, bits: 4, mode: .affine)
        let pre = MLXArray.ones([64], dtype: .float32)
        let spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .affine, hadamardBlock: 32)
        let nilBiasInputs = MACaQFusedLinearInputs(
            packed: MACaQPackedWeights(weight: wq, scales: ws, biases: nil),
            preScale: pre, spec: spec)
        let x = MLXRandom.normal([2, 64], key: MLXRandom.key(91))
        #expect(!MACaQFusedMatmul.supportsLinear(x, inputs: nilBiasInputs))
        let gotNil = MACaQFusedMatmul.linear(x, inputs: nilBiasInputs)
        let zeroBiases = MLXArray.zeros(ws.shape, dtype: ws.dtype)
        let refInputs = MACaQFusedLinearInputs(
            packed: MACaQPackedWeights(weight: wq, scales: ws, biases: zeroBiases),
            preScale: pre, spec: spec)
        let ref = MACaQFusedMatmul.linear(x, inputs: refInputs)
        eval(gotNil, ref)
        #expect(gotNil.shape == [2, 8])
        #expect(maxAbsDiff(gotNil, ref) < 1e-4)

        let emptyOutInputs = MACaQFusedLinearInputs(
            packed: MACaQPackedWeights(
                weight: MLXArray.zeros([0, 8], dtype: .uint32),
                scales: MLXArray.zeros([0, 2], dtype: .float32),
                biases: MLXArray.zeros([0, 2], dtype: .float32)),
            preScale: pre, spec: spec)
        #expect(!MACaQFusedMatmul.supportsLinear(x, inputs: emptyOutInputs))
        let emptyOut = MACaQFusedMatmul.linear(x, inputs: emptyOutInputs)
        eval(emptyOut)
        #expect(emptyOut.shape == [2, 0])

        let emptyK = MLXArray.zeros([2, 0], dtype: .float32)
        let emptyKOut = MACaQFusedMatmul.linear(emptyK, inputs: refInputs)
        eval(emptyKOut)
        #expect(emptyKOut.shape == [2, 8])
        #expect(abs(emptyKOut).max().item(Float.self) == 0)

        let emptyKBias = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8])
        let emptyKBiasInputs = MACaQFusedLinearInputs(
            packed: refInputs.packed, preScale: pre, spec: spec, bias: emptyKBias)
        let emptyKBiasOut = MACaQFusedMatmul.linear(emptyK, inputs: emptyKBiasInputs)
        eval(emptyKBiasOut)
        #expect(maxAbsDiff(emptyKBiasOut, broadcast(emptyKBias, to: [2, 8])) == 0)
    }

    @Test("MACaQFusedMatmul compiles and matches CPU reference for <8-element weights, odd N, and partial MTILE")
    func macaqFusedMatmulTinyBuffersPartialTileAndOddN() {
        for qMode in [QuantizationMode.affine, .mxfp4] {
            for nOut in [1, 5] {
                let wDense = MLXRandom.normal([nOut, 32], scale: 0.1, key: MLXRandom.key(UInt64(nOut + 100)))
                let (wq, ws, wb) = MLX.quantized(wDense, groupSize: 32, bits: 4, mode: qMode)
                let pre = MLXRandom.uniform(low: 0.6, high: 1.4, [32], key: MLXRandom.key(101))
                let spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: qMode, hadamardBlock: 32)
                let inputs = MACaQFusedLinearInputs(
                    packed: MACaQPackedWeights(weight: wq, scales: ws, biases: wb),
                    preScale: pre, spec: spec)
                for rows in [1, 3, 5] {
                    let x = MLXRandom.normal([rows, 32], key: MLXRandom.key(UInt64(rows + 110)))
                    eval(x, wq, ws, pre)
                    if let wb { eval(wb) }
                    #expect(MACaQFusedMatmul.supportsLinear(x, inputs: inputs))
                    let gpu = MACaQFusedMatmul.linear(x, inputs: inputs)
                    eval(gpu)
                    let cpu = Device.withDefaultDevice(.cpu) {
                        let out = MACaQFusedMatmul.linear(x, inputs: inputs)
                        eval(out)
                        return out
                    }
                    #expect(gpu.shape == [rows, nOut] && cpu.shape == [rows, nOut])
                    #expect(maxAbsDiff(gpu, cpu) < 2e-4)
                }
            }
        }
    }

    @Test("MACaQFusedMatmul linear and gather fall back cleanly on CPU default device")
    func macaqFusedMatmulCPUFallbackParity() {
        let wDense = MLXRandom.normal([8, 64], scale: 0.1, key: MLXRandom.key(120))
        let (wq, ws, wb) = MLX.quantized(wDense, groupSize: 32, bits: 4, mode: .affine)
        let pre = MLXRandom.uniform(low: 0.7, high: 1.3, [64], key: MLXRandom.key(121))
        let rowScale = MLXRandom.uniform(low: 0.8, high: 1.2, [8], key: MLXRandom.key(122))
        let bias = MLXRandom.normal([8], scale: 0.1, key: MLXRandom.key(123))
        let spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .affine, hadamardBlock: 32)
        let x = MLXRandom.normal([2, 64], key: MLXRandom.key(124))
        eval(wq, ws, pre, rowScale, bias, x)
        if let wb { eval(wb) }

        for actMode in MACaQActivationMode.allCases {
            let inputs = MACaQFusedLinearInputs(
                packed: MACaQPackedWeights(weight: wq, scales: ws, biases: wb),
                preScale: pre, spec: spec, rowScale: rowScale, bias: bias, activationMode: actMode)
            let gpu = MACaQFusedMatmul.linear(x, inputs: inputs)
            eval(gpu)
            let (cpuSupports, cpu) = Device.withDefaultDevice(.cpu) {
                let s = MACaQFusedMatmul.supportsLinear(x, inputs: inputs)
                let out = MACaQFusedMatmul.linear(x, inputs: inputs)
                eval(out)
                return (s, out)
            }
            #expect(!cpuSupports)
            let tol: Float = actMode == .fp16 ? 5e-4 : 5e-3
            #expect(maxAbsDiff(gpu, cpu) < tol)
        }

        let w3 = stacked([wq, wq], axis: 0)
        let s3 = stacked([ws, ws], axis: 0)
        let b3 = wb.map { stacked([$0, $0], axis: 0) }
        let sw = MACaQSwitchLinear(
            geometry: MACaQSwitchGeometry(experts: 2, outputs: 8, inputs: 64),
            packed: MACaQPackedWeights(weight: w3, scales: s3, biases: b3),
            bias: nil, preScale: pre, spec: spec)
        let xG = MLXRandom.normal([1, 1, 1, 64], key: MLXRandom.key(125))
        let idx = MLXArray([Int32(0), Int32(1)], [1, 2])
        eval(w3, s3, sw, xG, idx)
        let gpuG = MACaQFusedMatmul.gather(xG, indices: idx, weight: w3, switchLayer: sw, bias: nil)
        eval(gpuG)
        let (cpuSupG, cpuG) = Device.withDefaultDevice(.cpu) {
            let s = MACaQFusedMatmul.supportsGather(xG, indices: idx, switchLayer: sw)
            let out = MACaQFusedMatmul.gather(xG, indices: idx, weight: w3, switchLayer: sw, bias: nil)
            eval(out)
            return (s, out)
        }
        #expect(!cpuSupG)
        #expect(maxAbsDiff(gpuG, cpuG) < 2e-4)
    }

    @Test("MACaQFusedKernels blockHadamard and switchBlockHadamard match GPU on CPU with broadcast 1D preScale, unclipped signs, and OOB indices")
    func macaqFusedKernelsSwitchHadamardAndCPUFallbackParity() {
        for width in [64, 128] {
            let x = MLXRandom.normal([2, width], key: MLXRandom.key(UInt64(width + 130)))
            let scaleVals: [Float] = (0 ..< width).map { i in
                if i == 0 { return -2.5 }
                if i == 1 { return 1e-6 }
                if i == 2 { return 2e4 }
                return (i % 2 == 0) ? 1.25 : -0.75
            }
            let pre1D = MLXArray(scaleVals, [width])
            eval(x, pre1D)
            for inv in [false, true] {
                let gpu = MACaQFusedKernels.blockHadamard(x, scaleVector: pre1D, blockSize: 32, isInverse: inv)
                eval(gpu)
                let cpu = Device.withDefaultDevice(.cpu) {
                    let out = MACaQFusedKernels.blockHadamard(x, scaleVector: pre1D, blockSize: 32, isInverse: inv)
                    eval(out)
                    return out
                }
                #expect(gpu.shape == cpu.shape)
                let rel = maxAbsDiff(gpu, cpu) / max(1, abs(gpu).max().item(Float.self))
                #expect(rel < 1e-4)
            }
        }

        let xBroadcast = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
        let idx = MLXArray([Int32(0), Int32(-1), Int32(3)], [1, 3])
        let pre1D = MLXArray((0 ..< 64).map { $0 == 2 ? Float(2e4) : ($0 % 3 == 0 ? Float(-1.5) : ($0 == 1 ? Float(1e-6) : Float(0.8))) }, [64])
        let pre2D = stacked([pre1D, -pre1D, pre1D * Float(0.5)], axis: 0)
        eval(xBroadcast, idx, pre1D, pre2D)

        for pre in [pre1D, pre2D] {
            let gpuSw = MACaQFusedKernels.switchBlockHadamard(xBroadcast, indices: idx, preScale: pre, blockSize: 32, numExperts: 3)
            eval(gpuSw)
            let cpuSw = Device.withDefaultDevice(.cpu) {
                let out = MACaQFusedKernels.switchBlockHadamard(xBroadcast, indices: idx, preScale: pre, blockSize: 32, numExperts: 3)
                eval(out)
                return out
            }
            #expect(gpuSw.shape == [1, 3, 1, 64] && cpuSw.shape == [1, 3, 1, 64])
            let rel = maxAbsDiff(gpuSw, cpuSw) / max(1, abs(gpuSw).max().item(Float.self))
            #expect(rel < 1e-4)
            #expect(abs(gpuSw[0, 0]).max().item(Float.self) > 2500)
            #expect(abs(cpuSw[0, 0]).max().item(Float.self) > 2500)
            #expect(abs(gpuSw[0, 1]).max().item(Float.self) == 0)
            #expect(abs(cpuSw[0, 1]).max().item(Float.self) == 0)
            #expect(abs(gpuSw[0, 2]).max().item(Float.self) == 0)
            #expect(abs(cpuSw[0, 2]).max().item(Float.self) == 0)
        }
    }
}

@Suite("Metal audit shard 6: GGUF & NF4Metal regressions", .serialized)
struct MetalAuditShard6GGUFAndNF4Tests {
    @Test("GGUFLinear and GGUFEmbedding handle empty tensors, FP16 across K widths, and IQ3 legacyMatvec large activations without overflow")
    func ggufEmptyTensorsAndFP16AndLegacyIQ3Range() throws {
        let emptyEmb = GGUFEmbedding(type: .q4K, count: 8, dimensions: 256, dtype: .float16)
        let outEmptyTok = emptyEmb(MLXArray.zeros([0], dtype: .int32))
        let outEmpty2D = emptyEmb(MLXArray.zeros([2, 0], dtype: .int32))
        eval(outEmptyTok, outEmpty2D)
        #expect(outEmptyTok.shape == [0, 256])
        #expect(outEmpty2D.shape == [2, 0, 256])

        let zeroVocabEmb = GGUFEmbedding(type: .q4K, count: 0, dimensions: 256, dtype: .float16)
        let zeroEmbLin = zeroVocabEmb.asLinear(MLXArray.ones([2, 256], dtype: .float16))
        eval(zeroEmbLin)
        #expect(zeroEmbLin.shape == [2, 0])

        let zeroOutLin = GGUFLinear(type: .q4K, outputs: 0, inputs: 256)
        let zeroInLin = GGUFLinear(type: .q4K, outputs: 8, inputs: 0)
        let outZeroN = zeroOutLin(MLXArray.ones([2, 256], dtype: .float16))
        let outZeroK = zeroInLin(MLXArray.zeros([2, 0], dtype: .float16))
        let outEmptyX = GGUFLinear(type: .q4K, outputs: 8, inputs: 256)(MLXArray.zeros([0, 256], dtype: .float16))
        eval(outZeroN, outZeroK, outEmptyX)
        #expect(outZeroN.shape == [2, 0] && outZeroK.shape == [2, 8] && outEmptyX.shape == [0, 8])

        for type in [GGMLType.iq3XXS, .iq3S] {
            let bytes = GGUFKernelTests.blocks(type, count: 8 * 512 / type.block.elements, seed: 141, scale: 0.05)
            let packed = GGUFKernelTests.packed(bytes, rows: 8)
            let dense = GGUFKernelTests.dense(bytes, type, rows: 8, columns: 512)
            let xBig = (MLXRandom.normal([1, 512], key: MLXRandom.key(142)) * Float(3000)).asType(.bfloat16)
            let got = GGUFKernels.legacyMatvec(xBig, packed, type: type)
            let ref = matmul(xBig.asType(.float32), dense.T)
            eval(got, ref)
            #expect(isFinite(got).all().item(Bool.self))
            #expect(GGUFKernelTests.relativeError(got, ref) < 1e-2)
        }

        for k in [256, 512] {
            let bytes = GGUFKernelTests.blocks(.q8_0, count: 8 * k / 32, seed: UInt64(k + 150), scale: 0.01)
            let packed = GGUFKernelTests.packed(bytes, rows: 8)
            let dense = GGUFKernelTests.dense(bytes, .q8_0, rows: 8, columns: k)
            let x = MLXRandom.normal([2, k], key: MLXRandom.key(151)).asType(.float16)
            let got = GGUFKernels.legacyMatvec(x, packed, type: .q8_0)
            let ref = matmul(x.asType(.float32), dense.T)
            eval(got, ref)
            #expect(GGUFKernelTests.relativeError(got, ref) < 5e-3)
        }
    }

    @Test("GGUFKernels, GGUFLinear, and GGUFEmbedding fall back safely to CPU dequantizer on CPU default device")
    func ggufCPUFallbackParity() throws {
        let bytes = GGUFKernelTests.blocks(.q4K, count: 8, seed: 160, scale: 0.01)
        let packed = GGUFKernelTests.packed(bytes, rows: 8)
        let layer = GGUFLinear(type: .q4K, outputs: 8, inputs: 256)
        try layer.update(parameters: ModuleParameters.unflattened(["weight": packed]), verify: .all)
        let emb = GGUFEmbedding(type: .q4K, count: 8, dimensions: 256, dtype: .float32)
        try emb.update(parameters: ModuleParameters.unflattened(["weight": packed]), verify: .all)
        let x = MLXRandom.normal([2, 256], key: MLXRandom.key(161))
        let tok = MLXArray([Int32(0), Int32(3), Int32(7)], [3])
        eval(packed, layer, emb, x, tok)

        let gpuDeq = GGUFKernels.dequantize(packed, type: .q4K, columns: 256, dtype: .float32)
        let gpuLin = layer(x)
        let gpuLegacy = GGUFKernels.legacyMatvec(x, packed, type: .q4K)
        let gpuEmb = emb(tok)
        let gpuEmbLin = emb.asLinear(x)
        eval(gpuDeq, gpuLin, gpuLegacy, gpuEmb, gpuEmbLin)

        let (cpuDeq, cpuLin, cpuLegacy, cpuEmb, cpuEmbLin) = Device.withDefaultDevice(.cpu) {
            let d = GGUFKernels.dequantize(packed, type: .q4K, columns: 256, dtype: .float32)
            let l = layer(x)
            let lg = GGUFKernels.legacyMatvec(x, packed, type: .q4K)
            let e = emb(tok)
            let el = emb.asLinear(x)
            eval(d, l, lg, e, el)
            return (d, l, lg, e, el)
        }
        #expect(maxAbsDiff(gpuDeq, cpuDeq) < 1e-5)
        #expect(maxAbsDiff(gpuLin, cpuLin) < 1e-4)
        #expect(maxAbsDiff(gpuLegacy, cpuLegacy) < 1e-4)
        #expect(maxAbsDiff(gpuEmb, cpuEmb) < 1e-5)
        #expect(maxAbsDiff(gpuEmbLin, cpuEmbLin) < 1e-4)
    }

    @Test("NF4Metal compiles for <8-element codes, matches CPU across M=1..16 and odd N, and writes exact OOB zeros in gatherMatmul")
    func nf4TinyBuffersRegisterTileM16AndGather() {
        for dtype in [DType.float16, .bfloat16, .float32] {
            let tol: Float = dtype == .float32 ? 2e-4 : 3e-2
            let wTiny = MLXRandom.normal([1, 32], scale: 0.1, key: MLXRandom.key(170))
            let (cTiny, sTiny) = NF4Metal.quantized(wTiny, groupSize: 32)
            let xTiny = MLXRandom.normal([2, 32], key: MLXRandom.key(171)).asType(dtype)
            eval(cTiny, sTiny, xTiny)
            let gpuTiny = NF4Metal.matmul(xTiny, codes: cTiny, scales: sTiny, groupSize: 32)
            eval(gpuTiny)
            let cpuTiny = Device.withDefaultDevice(.cpu) {
                let out = NF4Metal.matmul(xTiny, codes: cTiny, scales: sTiny, groupSize: 32)
                eval(out)
                return out
            }
            #expect(maxAbsDiff(gpuTiny, cpuTiny) < tol)

            let wOdd = MLXRandom.normal([7, 128], scale: 0.1, key: MLXRandom.key(172))
            let (cOdd, sOdd) = NF4Metal.quantized(wOdd, groupSize: 32)
            eval(cOdd, sOdd)
            for m in [1, 5, 8, 9, 13, 16] {
                let x = MLXRandom.normal([m, 128], key: MLXRandom.key(UInt64(m + 180))).asType(dtype)
                eval(x)
                let gpu = NF4Metal.matmul(x, codes: cOdd, scales: sOdd, groupSize: 32)
                eval(gpu)
                let cpu = Device.withDefaultDevice(.cpu) {
                    let out = NF4Metal.matmul(x, codes: cOdd, scales: sOdd, groupSize: 32)
                    eval(out)
                    return out
                }
                #expect(gpu.shape == [m, 7] && cpu.shape == [m, 7])
                #expect(maxAbsDiff(gpu, cpu) < tol)
            }
        }

        let wG = MLXRandom.normal([1, 1, 32], scale: 0.1, key: MLXRandom.key(190))
        let (cG, sG) = NF4Metal.quantized(wG, groupSize: 32)
        let idx = MLXArray([Int32(0), Int32(-1), Int32(1)], [3])
        for m in [3, 16] {
            let xG = MLXRandom.normal([3, m, 32], key: MLXRandom.key(UInt64(m + 195))).asType(.float32)
            eval(cG, sG, idx, xG)
            let gpuG = NF4Metal.gatherMatmul(xG, codes: cG, scales: sG, rhsIndices: idx, groupSize: 32)
            eval(gpuG)
            let cpuG = Device.withDefaultDevice(.cpu) {
                let out = NF4Metal.gatherMatmul(xG, codes: cG, scales: sG, rhsIndices: idx, groupSize: 32)
                eval(out)
                return out
            }
            #expect(gpuG.shape == [3, m, 1] && cpuG.shape == [3, m, 1])
            #expect(maxAbsDiff(gpuG, cpuG) < 2e-4)
            #expect(abs(gpuG[1]).max().item(Float.self) == 0)
            #expect(abs(gpuG[2]).max().item(Float.self) == 0)
        }
    }

    @Test("NF4Metal training mode supports valueAndGrad without CustomKernel vjp crash")
    func nf4TrainingAutograd() {
        let w = MLXRandom.normal([8, 64], scale: 0.1, key: MLXRandom.key(200))
        let (codes, scales) = NF4Metal.quantized(w, groupSize: 32)
        let x = MLXRandom.normal([4, 64], key: MLXRandom.key(201))
        let vg = valueAndGrad({ arrays in
            [NF4Metal.matmul(arrays[0], codes: codes, scales: scales, groupSize: 32, training: true).square().sum()]
        }, argumentNumbers: [0])
        let (_, grads) = vg([x])
        let norm = grads[0].square().sum()
        eval(norm)
        #expect(norm.item(Float.self).isFinite && norm.item(Float.self) > 0)

        let w3 = MLXRandom.normal([2, 8, 64], scale: 0.1, key: MLXRandom.key(202))
        let (c3, s3) = NF4Metal.quantized(w3, groupSize: 32)
        let x3 = MLXRandom.normal([2, 2, 64], key: MLXRandom.key(203))
        let idx = MLXArray([Int32(0), Int32(1)], [2])
        let vgGather = valueAndGrad({ arrays in
            [NF4Metal.gatherMatmul(arrays[0], codes: c3, scales: s3, rhsIndices: idx, groupSize: 32, training: true).square().sum()]
        }, argumentNumbers: [0])
        let (_, gGather) = vgGather([x3])
        let gNorm = gGather[0].square().sum()
        eval(gNorm)
        #expect(gNorm.item(Float.self).isFinite && gNorm.item(Float.self) > 0)
    }
}
