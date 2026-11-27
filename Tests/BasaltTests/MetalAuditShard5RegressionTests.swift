import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCore
@testable import BasaltInference

@Suite("Metal Audit Shard 5: E8P small-buffer, geometry, CPU fallback & #228 TinyModel parity", .serialized)
struct MetalAuditShard5E8PTests {
    @Test("E8PMetalKernels handles packed_codes.size < 8 across all dtypes and matches CPU reference")
    func smallPackedCodesAndDTypeParity() {
        let rng = MLXRandom.RandomState(seed: 5501)
        for nOut in [1, 3] {
            let k = 32
            let w = MLXRandom.normal([nOut, k], dtype: .float32, key: rng)
            let bias = MLXRandom.normal([nOut], dtype: .float32, key: rng) * Float(0.1)
            let pack = E8PMetalKernels.quantizeWeight(w, seed: UInt64(100 + nOut))
            eval(pack.packedCodes, pack.rowScale, pack.rhtSigns, bias)
            #expect(pack.packedCodes.size < 8)
            let wordsHost = pack.packedCodes.asArray(UInt32.self)
            for dtype in [DType.float32, .float16, .bfloat16] {
                for rows in [1, 2, 5] {
                    let x = MLXRandom.normal([rows, k], dtype: .float32, key: rng).asType(dtype)
                    let dPack = E8PMetalKernels.QuantizedPack(
                        packedCodes: pack.packedCodes,
                        rowScale: pack.rowScale.asType(dtype),
                        rhtSigns: pack.rhtSigns.asType(dtype))
                    let dBias = bias.asType(dtype)
                    let gpuWithBias = E8PMetalKernels.matmul(x, pack: dPack, bias: dBias)
                    let gpuNoBias = E8PMetalKernels.matmul(x, pack: dPack, bias: nil)
                    eval(gpuWithBias, gpuNoBias)
                    #expect(gpuWithBias.shape == [rows, nOut] && gpuWithBias.dtype == dtype)
                    #expect(gpuNoBias.shape == [rows, nOut] && gpuNoBias.dtype == dtype)
                    let cpuPack = E8PMetalKernels.CPUPack(
                        packedWords: wordsHost,
                        nOut: nOut,
                        rhtSigns: dPack.rhtSigns.asType(.float32).asArray(Float.self),
                        rowScale: dPack.rowScale.asType(.float32).asArray(Float.self),
                        bias: dBias.asType(.float32).asArray(Float.self))
                    let xHost = x.asType(.float32).asArray(Float.self)
                    let ref = E8PMetalKernels.cpuMatmul(xHost, rows: rows, pack: cpuPack)
                    var cpuPackNoBias = cpuPack
                    cpuPackNoBias.bias = nil
                    let refNoBias = E8PMetalKernels.cpuMatmul(xHost, rows: rows, pack: cpuPackNoBias)
                    let tol: Float = dtype == .bfloat16 ? 2e-2 : (dtype == .float16 ? 3e-3 : 5e-4)
                    for (g, r) in zip(gpuWithBias.asType(.float32).asArray(Float.self), ref) {
                        #expect(abs(g - r) <= max(tol, abs(r) * tol))
                    }
                    for (g, r) in zip(gpuNoBias.asType(.float32).asArray(Float.self), refNoBias) {
                        #expect(abs(g - r) <= max(tol, abs(r) * tol))
                    }
                }
            }
        }
    }

    @Test("E8PMetalKernels handles empty tensors, empty quantizeWeight, unsupported dtype, and CPU fallback")
    func emptyUnsupportedDTypeAndCPUFallback() {
        let rng = MLXRandom.RandomState(seed: 5502)
        let emptyPack = E8PMetalKernels.quantizeWeight(MLXArray.zeros([0, 32], dtype: .float32), seed: 11)
        #expect(emptyPack.packedCodes.shape == [0, 2] && emptyPack.rowScale.shape == [0])

        let w = MLXRandom.normal([6, 64], dtype: .float32, key: rng)
        let bias = MLXRandom.normal([6], dtype: .float32, key: rng) * Float(0.2)
        let pack = E8PMetalKernels.quantizeWeight(w, seed: 42)
        let xEmptyRows = MLXArray.zeros([0, 64], dtype: .float16)
        let yEmptyRows = E8PMetalKernels.matmul(xEmptyRows, pack: pack, bias: bias)
        #expect(yEmptyRows.shape == [0, 6] && yEmptyRows.dtype == .float16)

        let yEmptyOut = E8PMetalKernels.matmul(MLXArray.zeros([2, 32], dtype: .float32), pack: emptyPack)
        #expect(yEmptyOut.shape == [2, 0] && yEmptyOut.dtype == .float32)

        let zeroKPack = E8PMetalKernels.QuantizedPack(
            packedCodes: MLXArray.zeros([4, 0], dtype: .uint32),
            rowScale: MLXArray.ones([4], dtype: .float32),
            rhtSigns: MLXArray.zeros([0], dtype: .float32))
        let zeroKBias = MLXArray([Float(1), 2, 3, 4])
        let yZeroK = E8PMetalKernels.matmul(MLXArray.zeros([2, 0], dtype: .float32), pack: zeroKPack, bias: zeroKBias)
        eval(yZeroK)
        #expect(yZeroK.shape == [2, 4])
        #expect(yZeroK.asArray(Float.self) == [1, 2, 3, 4, 1, 2, 3, 4])

        let xInt = MLXArray(Array(repeating: Int32(1), count: 64), [1, 64])
        let yInt = E8PMetalKernels.matmul(xInt, pack: pack, bias: nil)
        eval(yInt)
        #expect(yInt.shape == [1, 6] && yInt.dtype == .int32)

        let x = MLXRandom.normal([3, 64], dtype: .float32, key: rng)
        let gpuOut = E8PMetalKernels.matmul(x, pack: pack, bias: bias)
        eval(gpuOut, pack.rowScale)
        let (cpuRowScales, cpuPackedShape, cpuOutHost) = Device.withDefaultDevice(.cpu) {
            let cp = E8PMetalKernels.quantizeWeight(w, seed: 42)
            let co = E8PMetalKernels.matmul(x, pack: pack, bias: bias)
            eval(cp.packedCodes, cp.rowScale, co)
            return (cp.rowScale.asArray(Float.self), cp.packedCodes.shape, co.asArray(Float.self))
        }
        #expect(cpuPackedShape == [6, 4])
        let gpuRowScales = pack.rowScale.asArray(Float.self)
        #expect(gpuRowScales.count == cpuRowScales.count && !gpuRowScales.isEmpty)
        #expect(zip(gpuRowScales, cpuRowScales).allSatisfy { abs($0 - $1) < 1e-5 })
        let gpuOutHost = gpuOut.asArray(Float.self)
        #expect(gpuOutHost.count == cpuOutHost.count && !gpuOutHost.isEmpty)
        #expect(zip(gpuOutHost, cpuOutHost).allSatisfy { abs($0 - $1) < 5e-4 })
    }

    @Test("E8PLinear TinyModel.qwen3Small matches dequantized stock Linear logits and greedy token IDs (#228)")
    func tinyModelAdmissionParity() throws {
        let model = try TinyModel.make(.qwen3Small)
        let root = model as Module
        let embedDType = root.parameters().flattened()
            .first(where: { $0.1.dtype.isFloatingPoint && $0.0.contains("embed") })?.1.dtype ?? .float32
        var e8pSwaps: [(String, Module)] = []
        for (path, leaf) in root.leafModules().flattened() {
            if let lin = leaf as? Linear, !(lin is QuantizedLinear),
               path != "lm_head", lin.shape.1.isMultiple(of: 32) {
                let pack = E8PMetalKernels.quantizeWeight(lin.weight, seed: 549)
                e8pSwaps.append((path, E8PLinear(pack: pack, bias: lin.bias)))
            }
        }
        #expect(!e8pSwaps.isEmpty)
        root.update(modules: ModuleChildren.unflattened(e8pSwaps))
        eval(root)

        let prompt: [Int] = [5, 19, 42, 88, 120]
        let promptArr = MLXArray(prompt.map(Int32.init))[.newAxis]
        let e8pLogits = model(LMInput.Text(tokens: promptArr), cache: nil, state: nil).logits[0].asType(.float32)
        eval(e8pLogits)
        let e8pGreedy = try CacheBlendQualityTests.greedy(prompt, count: 6, model: model)

        var denseSwaps: [(String, Module)] = []
        for (path, leaf) in root.leafModules().flattened() {
            guard let e8p = leaf as? E8PLinear else { continue }
            let wDense = Self.dequantizedE8PWeight(e8p, dtype: embedDType)
            denseSwaps.append((path, Linear(weight: wDense, bias: e8p.bias)))
        }
        #expect(denseSwaps.count == e8pSwaps.count)
        root.update(modules: ModuleChildren.unflattened(denseSwaps))
        eval(root)

        let refLogits = model(LMInput.Text(tokens: promptArr), cache: nil, state: nil).logits[0].asType(.float32)
        let maxDiff = abs(e8pLogits - refLogits).max()
        eval(refLogits, maxDiff)
        let refGreedy = try CacheBlendQualityTests.greedy(prompt, count: 6, model: model)
        #expect(maxDiff.item(Float.self) < 1e-2)
        #expect(e8pGreedy == refGreedy)
    }

    private static func dequantizedE8PWeight(_ layer: E8PLinear, dtype: DType) -> MLXArray {
        let codes = layer.weight.asType(.uint32)
        let signs = layer.rhtSigns.asType(.float32)
        let scales = layer.rowScale.asType(.float32)
        eval(codes, signs, scales)
        let nOut = codes.dim(0)
        let wordsPerRow = codes.dim(1)
        let k = wordsPerRow * 16
        let words = codes.asArray(UInt32.self)
        var wDeq = [Float](repeating: 0, count: nOut * k)
        for n in 0 ..< nOut {
            for pair in 0 ..< wordsPerRow {
                let word = words[n * wordsPerRow + pair]
                let d0 = E8PLattice.cpuDecodeCode(UInt16(truncatingIfNeeded: word))
                let d1 = E8PLattice.cpuDecodeCode(UInt16(truncatingIfNeeded: word >> 16))
                let base = n * k + pair * 16
                for j in 0 ..< 8 {
                    wDeq[base + j] = d0[j]
                    wDeq[base + 8 + j] = d1[j]
                }
            }
        }
        let wRot = MLXArray(wDeq, [nOut, k])
        let wUnrot = hadamardTransform(wRot.reshaped(-1, 32), scale: 1.0 / Float(32).squareRoot()).reshaped(nOut, k) * signs
        return (wUnrot * scales.reshaped(nOut, 1)).asType(dtype)
    }
}

@Suite("Metal Audit Shard 5: QTIPMetal FP16/BF16, empty, <8 buffers, TCOLS-free & CPU fallback", .serialized)
struct MetalAuditShard5QTIPMetalTests {
    @Test("QTIPMetal.encodeTiles handles FP16/BF16 tiles, BF16 tlut, empty tiles, and CPU fallback parity",
          arguments: QTIPRate.allCases)
    func encodeTilesDTypesEmptyAndCPU(rate: QTIPRate) {
        let tlutF16 = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        let tlutBF16 = tlutF16.asType(.bfloat16)
        let tilesF32 = MLXRandom.normal([4, 256], key: MLXRandom.key(UInt64(600 + rate.rawValue)))
        let tilesF16 = tilesF32.asType(.float16)
        let tilesBF16 = tilesF32.asType(.bfloat16)
        eval(tlutF16, tlutBF16, tilesF32, tilesF16, tilesBF16)

        let (p16, h16) = QTIPMetal.encodeTiles(tilesF16, tlut: tlutF16, rate: rate)
        let (p16Ref, h16Ref) = QTIPMetal.encodeTiles(tilesF16.asType(.float32), tlut: tlutF16, rate: rate)
        let (pBF, hBF) = QTIPMetal.encodeTiles(tilesBF16, tlut: tlutBF16, rate: rate)
        let (pBFRef, hBFRef) = QTIPMetal.encodeTiles(tilesBF16.asType(.float32), tlut: tlutBF16.asType(.float16), rate: rate)
        eval(p16, h16, p16Ref, h16Ref, pBF, hBF, pBFRef, hBFRef)
        #expect(p16.asArray(UInt32.self) == p16Ref.asArray(UInt32.self))
        #expect(h16.asArray(Float.self) == h16Ref.asArray(Float.self))
        #expect(pBF.asArray(UInt32.self) == pBFRef.asArray(UInt32.self))
        #expect(hBF.asArray(Float.self) == hBFRef.asArray(Float.self))

        for dtype in [DType.float32, .float16, .bfloat16] {
            let (ep, eh) = QTIPMetal.encodeTiles(MLXArray.zeros([0, 256], dtype: dtype), tlut: tlutF16, rate: rate)
            #expect(ep.shape == [0, rate.wordsPerTile] && ep.dtype == .uint32)
            #expect(eh.shape == [0, 256] && eh.dtype == .float32)
        }

        let (gpuP, gpuH) = QTIPMetal.encodeTiles(tilesF32[0 ..< 2], tlut: tlutF16, rate: rate)
        eval(gpuP, gpuH)
        let (cpuP, cpuH) = Device.withDefaultDevice(.cpu) {
            let (cp, ch) = QTIPMetal.encodeTiles(tilesF32[0 ..< 2], tlut: tlutF16, rate: rate)
            eval(cp, ch)
            return (cp.asArray(UInt32.self), ch.asArray(Float.self))
        }
        #expect(cpuP.count == 2 * rate.wordsPerTile && cpuH.count == 512)
        let refHost = tilesF32[0 ..< 2].asArray(Float.self)
        let gpuMSE = zip(refHost, gpuH.asArray(Float.self)).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }
        let cpuMSE = zip(refHost, cpuH).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }
        #expect(abs(gpuMSE - cpuMSE) / max(1e-9, cpuMSE) < 1e-4)
    }

    @Test("QTIPMetal.decode handles empty shapes, <8 rowScales, varied tileCols, non-uint32, and CPU fallback",
          arguments: QTIPRate.allCases)
    func decodeEdgeCasesAndCPU(rate: QTIPRate) {
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        eval(tlut)
        let tlutHost = tlut.asType(.float32).asArray(Float.self)

        let emptyRows = QTIPMetal.decode(
            packed: MLXArray.zeros([0, 2 * rate.wordsPerTile], dtype: .uint32),
            rowScales: MLXArray.zeros([0], dtype: .float16),
            tlut: tlut, rate: rate, tileCols: 2, dtype: .float16)
        #expect(emptyRows.shape == [0, 32] && emptyRows.dtype == .float16)

        let emptyCols = QTIPMetal.decode(
            packed: MLXArray.zeros([2, 0], dtype: .uint32),
            rowScales: MLXArray.ones([2], dtype: .float16),
            tlut: tlut, rate: rate, tileCols: 0, dtype: .bfloat16)
        #expect(emptyCols.shape == [32, 0] && emptyCols.dtype == .bfloat16)

        for (tileRows, tileCols) in [(1, 1), (3, 2), (1, 5)] {
            let words = QTIPKernelTestOracle.randomPackedWords(
                count: tileRows * tileCols * rate.wordsPerTile,
                seed: UInt64(700 + rate.rawValue * 31 + tileRows * 7 + tileCols)
            ).map { $0 & 0x7fff_ffff }
            let packed = MLXArray(words, [tileRows, tileCols * rate.wordsPerTile])
            let scales = MLXArray((0 ..< tileRows).map { Float(0.3 + 0.2 * Double($0)) }).asType(.float16)
            eval(packed, scales)
            #expect(scales.size < 8)
            let expected = QTIPKernelTestOracle.cpuDecodedMatrix(
                words: words, scales: scales.asType(.float32).asArray(Float.self),
                tlut: tlutHost, rate: rate, tileRows: tileRows, tileCols: tileCols)
            let gpuF32 = QTIPMetal.decode(packed: packed, rowScales: scales, tlut: tlut, rate: rate, tileCols: tileCols, dtype: .float32)
            let nonU32 = QTIPMetal.decode(packed: packed.asType(.int32), rowScales: scales, tlut: tlut, rate: rate, tileCols: tileCols, dtype: .float32)
            eval(gpuF32, nonU32)
            #expect(QTIPKernelTestOracle.relativeError(gpuF32, expected) < 1e-5)
            #expect(QTIPKernelTestOracle.relativeError(nonU32, expected) < 1e-5)

            let cpuDecoded = Device.withDefaultDevice(.cpu) {
                let c = QTIPMetal.decode(packed: packed, rowScales: scales, tlut: tlut, rate: rate, tileCols: tileCols, dtype: .float16)
                eval(c)
                return c.asType(.float32).asArray(Float.self)
            }
            let gpuF16 = QTIPMetal.decode(packed: packed, rowScales: scales, tlut: tlut, rate: rate, tileCols: tileCols, dtype: .float16)
            eval(gpuF16)
            let gpuF16Host = gpuF16.asType(.float32).asArray(Float.self)
            #expect(gpuF16Host.count == cpuDecoded.count && !gpuF16Host.isEmpty)
            #expect(zip(gpuF16Host, cpuDecoded).allSatisfy { abs($0 - $1) < 2e-3 })
        }
    }
}

@Suite("Metal Audit Shard 5: QTIPKernels empty 3D packed, gatherOutputShape & CPU fallback", .serialized)
struct MetalAuditShard5QTIPKernelsTests {
    @Test("QTIPKernels handles empty 3D packed and zero inFeatures without reshaped(0, -1) crash")
    func empty3DPackedAndZeroWidth() {
        let rate = QTIPRate.twoBPW
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        let empty3D = MLXArray.zeros([0, 3, rate.wordsPerTile], dtype: .uint32)
        let emptyScales = MLXArray.zeros([0], dtype: .float16)

        let y1 = QTIPKernels.qmv(MLXArray.zeros([1, 48], dtype: .float32), packed: empty3D, rowScales: emptyScales, tlut: tlut, rate: rate)
        let y4 = QTIPKernels.smallM(MLXArray.zeros([4, 48], dtype: .float16), packed: empty3D, rowScales: emptyScales, tlut: tlut, rate: rate)
        let y10 = QTIPKernels.qmmTiled(MLXArray.zeros([10, 48], dtype: .bfloat16), packed: empty3D, rowScales: emptyScales, tlut: tlut, rate: rate)
        let y0 = QTIPKernels.linear(MLXArray.zeros([0, 48], dtype: .float32), packed: empty3D, rowScales: emptyScales, tlut: tlut, rate: rate)
        #expect(y1.shape == [1, 0] && y1.dtype == .float32)
        #expect(y4.shape == [4, 0] && y4.dtype == .float16)
        #expect(y10.shape == [10, 0] && y10.dtype == .bfloat16)
        #expect(y0.shape == [0, 0] && y0.dtype == .float32)

        let zeroWidthPacked = MLXArray.zeros([2, 0, rate.wordsPerTile], dtype: .uint32)
        let scales2 = MLXArray.ones([2], dtype: .float16)
        let yZeroK = QTIPKernels.linear(MLXArray.zeros([2, 0], dtype: .float32), packed: zeroWidthPacked, rowScales: scales2, tlut: tlut, rate: rate)
        eval(yZeroK)
        #expect(yZeroK.shape == [2, 32] && yZeroK.asArray(Float.self).allSatisfy { $0 == 0 })
    }

    @Test("QTIPKernels.gatherMatmul and QTIPSwitchLinear return 0 on empty broadcast dimensions")
    func gatherOutputShapeEmptyBroadcast() {
        let rate = QTIPRate.twoHalfBPW
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        let packed = MLXArray.zeros([3, 2, 3 * rate.wordsPerTile], dtype: .uint32)
        let scales = MLXArray.ones([3, 2], dtype: .float16)

        let yA = QTIPKernels.gatherMatmul(
            x: MLXArray.zeros([2, 1, 1, 48], dtype: .float16),
            packedExperts: packed, rowScales: scales, tlut: tlut,
            indices: MLXArray.zeros([2, 0], dtype: .int32), rate: rate)
        #expect(yA.shape == [2, 0, 1, 32] && yA.dtype == .float16)

        let yB = QTIPKernels.gatherMatmul(
            x: MLXArray.zeros([0, 1, 1, 48], dtype: .float32),
            packedExperts: packed, rowScales: scales, tlut: tlut,
            indices: MLXArray.zeros([1, 2], dtype: .int32), rate: rate)
        #expect(yB.shape == [0, 2, 1, 32] && yB.dtype == .float32)

        let yC = QTIPKernels.gatherMatmul(
            x: MLXArray.zeros([2, 2, 0, 48], dtype: .bfloat16),
            packedExperts: packed, rowScales: scales, tlut: tlut,
            indices: MLXArray.zeros([2, 2], dtype: .int32), rate: rate)
        #expect(yC.shape == [2, 2, 0, 32] && yC.dtype == .bfloat16)

        let sw = QTIPSwitchLinear(packedExperts: packed, rowScales: scales, tlut: tlut, rate: rate)
        let ySw = sw(MLXArray.zeros([2, 1, 1, 48], dtype: .float16), MLXArray.zeros([2, 0], dtype: .int32))
        #expect(ySw.shape == [2, 0, 1, 32] && ySw.dtype == .float16)
    }

    @Test("QTIPKernels linear, qmv, smallM, qmmTiled, and gatherMatmul match GPU under CPU default device",
          arguments: QTIPRate.allCases)
    func cpuFallbackParity(rate: QTIPRate) {
        let (numExperts, tileRows, tileCols) = (3, 2, 3)
        let inFeatures = tileCols * 16
        let wpt = rate.wordsPerTile
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        let words = QTIPKernelTestOracle.randomPackedWords(
            count: numExperts * tileRows * tileCols * wpt, seed: UInt64(820 + rate.rawValue))
        let packedExperts = MLXArray(words, [numExperts, tileRows, tileCols * wpt])
        let scalesExperts = MLXArray((0 ..< (numExperts * tileRows)).map { Float(0.25 + 0.1 * Double($0)) }, [numExperts, tileRows]).asType(.float16)
        let packed0 = packedExperts[0]
        let scales0 = scalesExperts[0]
        let x1 = MLXRandom.normal([1, inFeatures], key: MLXRandom.key(81))
        let x4 = MLXRandom.normal([4, inFeatures], key: MLXRandom.key(82))
        let x10 = MLXRandom.normal([10, inFeatures], key: MLXRandom.key(83))
        let idx = MLXArray([Int32(2), Int32(0)], [1, 2])
        eval(tlut, packedExperts, scalesExperts, x1, x4, x10, idx)

        let gpu1 = QTIPKernels.linear(x1, packed: packed0, rowScales: scales0, tlut: tlut, rate: rate)
        let gpu4 = QTIPKernels.linear(x4, packed: packed0, rowScales: scales0, tlut: tlut, rate: rate)
        let gpu10 = QTIPKernels.linear(x10, packed: packed0, rowScales: scales0, tlut: tlut, rate: rate)
        let gpuG = QTIPKernels.gatherQmv(x1, packedExperts: packedExperts, rowScales: scalesExperts, tlut: tlut, indices: idx, rate: rate)
        eval(gpu1, gpu4, gpu10, gpuG)

        let cpuResults = Device.withDefaultDevice(.cpu) {
            let c1 = QTIPKernels.linear(x1, packed: packed0, rowScales: scales0, tlut: tlut, rate: rate)
            let c4 = QTIPKernels.linear(x4, packed: packed0, rowScales: scales0, tlut: tlut, rate: rate)
            let c10 = QTIPKernels.linear(x10, packed: packed0, rowScales: scales0, tlut: tlut, rate: rate)
            let cG = QTIPKernels.gatherQmv(x1, packedExperts: packedExperts, rowScales: scalesExperts, tlut: tlut, indices: idx, rate: rate)
            eval(c1, c4, c10, cG)
            return (c1, c4, c10, cG)
        }
        #expect(QTIPKernelTestOracle.relativeError(cpuResults.0, gpu1) < 1e-5)
        #expect(QTIPKernelTestOracle.relativeError(cpuResults.1, gpu4) < 1e-5)
        #expect(QTIPKernelTestOracle.relativeError(cpuResults.2, gpu10) < 1e-5)
        #expect(QTIPKernelTestOracle.relativeError(cpuResults.3, gpuG) < 1e-5)
    }
}

@Suite("Metal Audit Shard 5: IQMetalKernels zero-columns, non-uint8, M=2..8, varied K & CPU fallback", .serialized)
struct MetalAuditShard5IQTests {
    @Test("IQMetalKernels guards columns == 0, rows == 0, outputs == 0, and non-uint8 packed without div0 or MSL crash")
    func emptyColumnsAndNonUInt8Guards() {
        for type in [GGMLType.iq4NL, .iq4XS, .iq3XXS, .q4K] {
            let emptyColsW = MLXArray.zeros([8, 0], dtype: .uint8)
            let mv1DZeroK = IQMetalKernels.matvec(MLXArray.zeros([0], dtype: .float32), emptyColsW, type: type)
            let mv2DZeroK = IQMetalKernels.matvec(MLXArray.zeros([1, 0], dtype: .float16), emptyColsW, type: type)
            let smZeroK = IQMetalKernels.smallM(MLXArray.zeros([4, 0], dtype: .bfloat16), emptyColsW, type: type)
            eval(mv1DZeroK, mv2DZeroK, smZeroK)
            #expect(mv1DZeroK.shape == [8] && mv1DZeroK.asArray(Float.self).allSatisfy { $0 == 0 })
            #expect(mv2DZeroK.shape == [1, 8] && mv2DZeroK.dtype == .float16)
            #expect(smZeroK.shape == [4, 8] && smZeroK.dtype == .bfloat16)

            let cols = 256
            let rowBytes = (cols / type.block.elements) * type.block.bytes
            let emptyRowsW = MLXArray.zeros([0, rowBytes], dtype: .uint8)
            #expect(IQMetalKernels.matvec(MLXArray.zeros([cols], dtype: .float32), emptyRowsW, type: type).shape == [0])
            #expect(IQMetalKernels.matvec(MLXArray.zeros([1, cols], dtype: .float32), emptyRowsW, type: type).shape == [1, 0])
            #expect(IQMetalKernels.smallM(MLXArray.zeros([4, cols], dtype: .float32), emptyRowsW, type: type).shape == [4, 0])

            let outputs = 6
            let bytes = GGUFKernelTests.blocks(type, count: outputs * cols / type.block.elements, seed: UInt64(900 + type.rawValue), scale: 0.005)
            let packed = GGUFKernelTests.packed(bytes, rows: outputs)
            #expect(IQMetalKernels.smallM(MLXArray.zeros([0, cols], dtype: .float32), packed, type: type).shape == [0, outputs])

            let packedInt = packed.asType(.int32)
            let x1 = MLXRandom.normal([1, cols], key: MLXRandom.key(91)) * Float(0.25)
            let x3 = MLXRandom.normal([3, cols], key: MLXRandom.key(92)) * Float(0.25)
            let gpu1 = IQMetalKernels.matvec(x1, packed, type: type)
            let nonU8MV = IQMetalKernels.matvec(x1, packedInt, type: type)
            let gpu3 = IQMetalKernels.smallM(x3, packed, type: type)
            let nonU8SM = IQMetalKernels.smallM(x3, packedInt, type: type)
            eval(gpu1, nonU8MV, gpu3, nonU8SM)
            #expect(GGUFKernelTests.relativeError(nonU8MV, gpu1) < 1e-5)
            #expect(GGUFKernelTests.relativeError(nonU8SM, gpu3) < 1e-5)
        }
    }

    @Test("IQMetalKernels covers M=2,3,5,6,7,8, 1D matvec, and varied K without K template arg",
          arguments: [GGMLType.iq4NL, .iq4XS, .iq3XXS, .iq2XS, .iq1S, .q4K])
    func allSmallMRowsAndVariedKParity(type: GGMLType) {
        let outputs = 13
        for cols in [256, 512, 768] {
            let bytes = GGUFKernelTests.blocks(type, count: outputs * cols / type.block.elements, seed: UInt64(cols + Int(type.rawValue)), scale: 0.004)
            let packed = GGUFKernelTests.packed(bytes, rows: outputs)
            let denseW = GGUFKernelTests.dense(bytes, type, rows: outputs, columns: cols)

            let x1D = MLXRandom.normal([cols], key: MLXRandom.key(UInt64(cols + 1))) * Float(0.2)
            let y1D = IQMetalKernels.matvec(x1D, packed, type: type)
            let ref1D = matmul(x1D.reshaped(1, cols), denseW.T).reshaped(outputs)
            eval(y1D, ref1D)
            #expect(y1D.shape == [outputs] && GGUFKernelTests.relativeError(y1D, ref1D) < 1e-5)

            for m in [2, 3, 5, 6, 7, 8] {
                let xm = MLXRandom.normal([m, cols], key: MLXRandom.key(UInt64(cols + m * 17))) * Float(0.2)
                let ym = IQMetalKernels.smallM(xm, packed, type: type)
                let refM = matmul(xm, denseW.T)
                eval(ym, refM)
                #expect(ym.shape == [m, outputs] && GGUFKernelTests.relativeError(ym, refM) < 1e-5)
            }
        }
    }

    @Test("IQMetalKernels matvec and smallM fall back cleanly under CPU default device and match GPU",
          arguments: [GGMLType.iq4NL, .iq4XS, .iq3XXS, .iq2XS, .iq1S, .q2K, .q3K, .q4K])
    func cpuFallbackParity(type: GGMLType) {
        let (outputs, cols) = (11, 256)
        let bytes = GGUFKernelTests.blocks(type, count: outputs * cols / type.block.elements, seed: UInt64(1100 + type.rawValue), scale: 0.004)
        let packed = GGUFKernelTests.packed(bytes, rows: outputs)
        let x1D = MLXRandom.normal([cols], key: MLXRandom.key(301)) * Float(0.2)
        let x3 = (MLXRandom.normal([3, cols], key: MLXRandom.key(302)) * Float(0.2)).asType(.float16)
        eval(packed, x1D, x3)

        let gpu1D = IQMetalKernels.matvec(x1D, packed, type: type)
        let gpu3 = IQMetalKernels.smallM(x3, packed, type: type)
        eval(gpu1D, gpu3)

        let (cpu1D, cpu3) = Device.withDefaultDevice(.cpu) {
            let c1 = IQMetalKernels.matvec(x1D, packed, type: type)
            let c3 = IQMetalKernels.smallM(x3, packed, type: type)
            eval(c1, c3)
            return (c1, c3)
        }
        #expect(cpu1D.shape == [outputs] && cpu1D.dtype == .float32)
        #expect(cpu3.shape == [3, outputs] && cpu3.dtype == .float16)
        #expect(GGUFKernelTests.relativeError(cpu1D, gpu1D) < 1e-5)
        #expect(GGUFKernelTests.relativeError(cpu3, gpu3.asType(.float32)) < 2e-3)
    }
}
