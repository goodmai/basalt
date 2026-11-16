import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("QTIP bitshift trellis, Kronecker RHT, BlockLDLQ, and TinyModel parity (#409)")
struct QTIPQuantizerTests {
    @Test("HYB Q=9 16-bit permutation is bijective and hits all 2048 signed centroids uniformly")
    func hybPermutationAndLUT() {
        let perms = (0 ..< 65536).map { QTIPCodebook.perm16(UInt16($0)) }
        #expect(Set(perms).count == 65536)
        var counts = [Int](repeating: 0, count: 2048)
        for edge in 0 ..< 65536 {
            counts[Int(QTIPCodebook.edgeCodeIndex(UInt16(edge)))] += 1
        }
        #expect(counts.allSatisfy { $0 == 32 })

        for rate in QTIPRate.allCases {
            #expect(QTIPRate(bpw: rate.bpw) == rate)
            let lut = QTIPCodebook.baseLUTValues(rate: rate)
            #expect(lut.count == 1024)
            #expect(lut.allSatisfy { $0.isFinite && $0 >= 0 })
        }
        #expect(QTIPRate(bpw: 4.0) == nil)
    }

    @Test("KroneckerRHT preserves inner products and inverts across native and composite widths",
          arguments: [32, 64, 128, 2112, 2816, 5120, 6144, 10240, 12288, 17408])
    func kroneckerRHTOrthogonality(width: Int) throws {
        let rht = try #require(KroneckerRHT(dimension: width, seed: 409))
        let signs = MLXArray(rht.signsValues)
        let qMat = rht.qValues.map { MLXArray($0, [rht.outer, rht.outer]) }
        let (u, v) = withRandomState(MLXRandom.RandomState(seed: UInt64(width))) {
            (MLXRandom.normal([4, width]), MLXRandom.normal([4, width]))
        }
        let uRot = rht.forwardFloat32(u, signs: signs, qMatrix: qMat)
        let vRot = rht.forwardFloat32(v, signs: signs, qMatrix: qMat)
        let uBack = rht.inverseFloat32(uRot, signs: signs, qMatrix: qMat)
        let maxInvErr = abs(uBack - u).max()
        let dotOrig = (u * v).sum(axis: -1)
        let dotRot = (uRot * vRot).sum(axis: -1)
        let relDotErr = (abs(dotOrig - dotRot) / maximum(Float(1.0), abs(dotOrig))).max()
        eval(maxInvErr, relDotErr)
        #expect(maxInvErr.item(Float.self) < 1e-4)
        #expect(relDotErr.item(Float.self) < 1e-4)
    }

    @Test("Metal bitshift Viterbi encoder and parallel decoder match CPU reference and chunked launches",
          arguments: QTIPRate.allCases)
    func metalVsCPUReferenceAndDistortion(rate: QTIPRate) {
        let tlutVals = QTIPCodebook.baseLUTValues(rate: rate)
        let edgeVals = QTIPCodebook.edgeCodeTableValues()
        let tlut = MLXArray(tlutVals, [512, 2]).asType(.float16)
        eval(tlut)
        let tlutF16Vals = tlut.asType(.float32).asArray(Float.self)

        let numTiles = 32
        let tilesArr = withRandomState(MLXRandom.RandomState(seed: 1234 + UInt64(rate.rawValue))) {
            MLXRandom.normal([numTiles, 256])
        }
        eval(tilesArr)
        let flatHost = tilesArr.asArray(Float.self)

        let (gpuPacked, gpuDecoded) = QTIPMetal.encodeTiles(tilesArr, tlut: tlut, rate: rate)
        let (chunkPacked, chunkDecoded) = QTIPMetal.encodeTiles(
            tilesArr, tlut: tlut, rate: rate, maxTilesPerLaunch: 12)
        eval(gpuPacked, gpuDecoded, chunkPacked, chunkDecoded)
        let gpuPackedHost = gpuPacked.asArray(UInt32.self)
        let gpuDecodedHost = gpuDecoded.asArray(Float.self)
        #expect(gpuPackedHost == chunkPacked.asArray(UInt32.self))
        #expect(gpuDecodedHost == chunkDecoded.asArray(Float.self))

        var (cpuMSE, gpuMSE) = (0.0, 0.0)
        for t in 0 ..< 4 {
            let slice = Array(flatHost[(t * 256) ..< ((t + 1) * 256)])
            let (cpuWords, cpuDec) = QTIPReference.encodeTile(
                tile: slice, tlut: tlutF16Vals, edgeCodes: edgeVals, rate: rate)
            let gpuWords = Array(gpuPackedHost[(t * rate.wordsPerTile) ..< ((t + 1) * rate.wordsPerTile)])
            let cpuFromGPUWords = QTIPReference.decodeTile(words: gpuWords, tlut: tlutF16Vals, rate: rate)
            let gpuDecSlice = Array(gpuDecodedHost[(t * 256) ..< ((t + 1) * 256)])
            #expect((zip(cpuFromGPUWords, gpuDecSlice).map { abs($0 - $1) }.max() ?? 1) < 1e-6)
            cpuMSE += zip(slice, cpuDec).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }
            gpuMSE += zip(slice, gpuDecSlice).reduce(0.0) { $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1)) }
            #expect(cpuWords.count == gpuWords.count)
        }
        #expect(abs(gpuMSE - cpuMSE) / max(1e-9, cpuMSE) < 1e-4)

        let fullGPUMSE = zip(flatHost, gpuDecodedHost).reduce(0.0) {
            $0 + Double(($1.0 - $1.1) * ($1.0 - $1.1))
        } / Double(flatHost.count)
        switch rate {
        case .twoBPW: #expect(fullGPUMSE < 0.088)
        case .twoHalfBPW: #expect(fullGPUMSE < 0.046)
        case .threeBPW: #expect(fullGPUMSE < 0.025)
        }

        let scales = MLXArray((0 ..< 8).map { Float(0.5 + 0.2 * Double($0)) }).asType(.float16)
        let reshapedPacked = gpuPacked.reshaped(8, 4 * rate.wordsPerTile)
        for dt in [DType.float32, DType.float16, DType.bfloat16] {
            let decMat = QTIPMetal.decode(
                packed: reshapedPacked, rowScales: scales, tlut: tlut, rate: rate, tileCols: 4, dtype: dt)
            eval(decMat)
            #expect(decMat.shape == [128, 64] && decMat.dtype == dt)
        }
    }

    @Test("BlockLDLQ with correlated Hessian beats identity and scalar RTN, and panel blocking matches strip-by-strip",
          arguments: QTIPRate.allCases)
    func blockLDLQBeatsIdentityAndScalarRTN(rate: QTIPRate) throws {
        let (m, n, samples) = (64, 512, 512)
        let (w, xRaw) = withRandomState(MLXRandom.RandomState(seed: 2026)) {
            let wBase = MLXRandom.normal([m, n])
            let latent = MLXRandom.normal([samples, 16])
            let mixer = MLXRandom.normal([16, n])
            let noise = MLXRandom.normal([samples, n]) * Float(0.08)
            let scales = exp(MLXArray((0 ..< n).map { Float(Double($0) / Double(n) * 2.5) })).reshaped(1, n)
            return (wBase, (matmul(latent, mixer) + noise) * scales)
        }
        eval(w, xRaw)
        let yRef = matmul(xRaw, w.T)
        eval(yRef)

        let rhtIn = try #require(KroneckerRHT(dimension: n, seed: 11))
        let rhtOut = try #require(KroneckerRHT(dimension: m, seed: 22))
        let spec = QTIPLayerSpec(rate: rate, rhtIn: rhtIn, rhtOut: rhtOut)
        let tlut = MLXArray(QTIPCodebook.baseLUTValues(rate: rate), [512, 2]).asType(.float16)
        eval(tlut)

        let hRot = try #require(QTIPQuantizer.rotatedHessian(samples: [xRaw], rhtIn: rhtIn))
        let chol = QTIPQuantizer.upperCholesky(hRot, damping: 0.01)

        let (wNorm, _) = QTIPQuantizer.prepareNormalizedWeight(weight: w, rhtIn: rhtIn, rhtOut: rhtOut)
        let pStrip = QTIPQuantizer.encodeBlockLDLQ(wNorm: wNorm, factors: chol, tlut: tlut, rate: rate, panelStrips: 1)
        let pPanel = QTIPQuantizer.encodeBlockLDLQ(wNorm: wNorm, factors: chol, tlut: tlut, rate: rate, panelStrips: 16)
        eval(pStrip, pPanel)
        #expect(pStrip.asArray(UInt32.self) == pPanel.asArray(UInt32.self))

        let qId = QTIPQuantizer.quantizeMatrix(weight: w, bias: nil, spec: spec, choleskyFactors: nil, tlut: tlut)
        let qLDL = QTIPQuantizer.quantizeMatrix(weight: w, bias: nil, spec: spec, choleskyFactors: chol, tlut: tlut)
        #expect(qLDL.shape == (m, n))

        let errId = ((matmul(xRaw, qId.decodedWeight().T) - yRef).square()).mean()
        let errLDL = ((matmul(xRaw, qLDL.decodedWeight().T) - yRef).square()).mean()
        eval(errId, errLDL)
        #expect(errLDL.item(Float.self) < errId.item(Float.self) * 0.85)

        if rate == .twoBPW || rate == .threeBPW {
            let scalarBits = rate == .twoBPW ? 2 : 3
            let rtnLayer = QuantizedLinear(weight: w, bias: nil, groupSize: 64, bits: scalarBits, mode: .affine)
            let wRTN = dequantized(
                rtnLayer.weight, scales: rtnLayer.scales, biases: rtnLayer.biases,
                groupSize: 64, bits: scalarBits, mode: .affine)
            let errRTN = ((matmul(xRaw, wRTN.T) - yRef).square()).mean()
            eval(errRTN)
            #expect(errLDL.item(Float.self) < errRTN.item(Float.self) * 0.70)
        }
    }

    @Test("TinyModel forward pass with QTIPLinear matches explicit dequantized Linear across families",
          arguments: [TinyModel.Family.qwen3, TinyModel.Family.qwen35, TinyModel.Family.gemma4])
    func tinyModelEndToEndParity(family: TinyModel.Family) throws {
        let model = try TinyModel.make(family)
        let calibBatches: [[Int32]] = [
            (0 ..< 32).map { Int32(($0 * 7 + 3) % 200) },
            (0 ..< 32).map { Int32(($0 * 13 + 11) % 200) }
        ]
        var cfg = QTIPConfig(rate: .threeBPW)
        cfg.minDimension = 16
        cfg.blockGroupSize = 2
        let summary = try QTIPQuantizer.quantizeModel(model, calibrationTokenBatches: calibBatches, config: cfg)
        #expect(summary.quantizedLayers > 0 && summary.quantizedWeights > 0)

        let prompt = MLXArray([Int32(5), Int32(19), Int32(42), Int32(88), Int32(120)])[.newAxis]
        let qtipLogits = model(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits[0].asType(.float32)
        eval(qtipLogits)

        let root = model as Module
        let actualDType = model.parameters().flattened()
            .first(where: { $0.1.dtype.isFloatingPoint && $0.0.contains("embed") })?.1.dtype ?? .float32
        for (path, mod) in root.namedModules() {
            guard let qLayer = mod as? QTIPLinear else { continue }
            let explicit = Linear(weight: qLayer.decodedWeight(dtype: actualDType), bias: qLayer.bias)
            eval(explicit)
            try PVTuner.swapModule(path, with: explicit, in: root)
        }
        MLXInferenceEngine.settle(root)
        let denseLogits = model(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits[0].asType(.float32)
        let maxDiff = abs(qtipLogits - denseLogits).max()
        let qGreedy = argMax(qtipLogits, axis: -1)
        let dGreedy = argMax(denseLogits, axis: -1)
        eval(maxDiff, qGreedy, dGreedy)
        #expect(maxDiff.item(Float.self) < 5e-2)
        #expect(qGreedy.asArray(Int32.self) == dGreedy.asArray(Int32.self))
    }
}

@Suite("QTIP multi-rate staging, native loader round-trip, and CLI validation (#409)")
struct QTIPNativeLoaderAndCLITests {
    @Test("quantizeModelMultiRate + applyStagedRate and QTIPNativeLoader match standalone quantizeModel bit-identically")
    func multiRateStagingAndNativeLoaderRoundTrip() throws {
        let calibBatches: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 11 + 5) % 180) }]
        let dir = FileManager.default.temporaryDirectory.appending(component: "qtip-stage-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let modelMulti = try TinyModel.make(.qwen35)
        let modelDirect = try TinyModel.make(.qwen35)
        let directWeights = Dictionary(uniqueKeysWithValues: (modelMulti as Module).parameters().flattened())
        try (modelDirect as Module).update(parameters: ModuleParameters.unflattened(directWeights), verify: [.noUnusedKeys])
        eval(modelMulti, modelDirect)

        var cfg = QTIPConfig(rate: .twoHalfBPW)
        cfg.minDimension = 16
        cfg.blockGroupSize = 8
        let stageDir = dir.appending(component: "stage")
        let (pSummary, plan) = try QTIPQuantizer.quantizeModelMultiRate(
            modelMulti, calibrationTokenBatches: calibBatches, rates: [.twoHalfBPW, .threeBPW],
            baseConfig: cfg, stageDirectory: stageDir)
        #expect(pSummary.quantizedLayers > 0)
        let applied = try QTIPQuantizer.applyStagedRate(modelMulti as Module, plan: plan, rate: .threeBPW)
        #expect(applied == pSummary.quantizedLayers)

        var cfg3 = cfg
        cfg3.rate = .threeBPW
        _ = try QTIPQuantizer.quantizeModel(modelDirect, calibrationTokenBatches: calibBatches, config: cfg3)

        let prompt = MLXArray([Int32(4), Int32(17), Int32(33), Int32(91)])[.newAxis]
        let multiLogits = modelMulti(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits[0].asType(.float32)
        let directLogits = modelDirect(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits[0].asType(.float32)
        let stageDiff = abs(multiLogits - directLogits).max()
        eval(stageDiff)
        #expect(stageDiff.item(Float.self) == 0)

        let multiGroupModel = try TinyModel.make(.qwen3)
        var mgCfg = cfg
        mgCfg.blockGroupSize = 1
        let mgStage = dir.appending(component: "stage-mg")
        let (mgSummary, mgPlan) = try QTIPQuantizer.quantizeModelMultiRate(
            multiGroupModel, calibrationTokenBatches: calibBatches, rates: [.twoBPW, .threeBPW],
            baseConfig: mgCfg, stageDirectory: mgStage)
        #expect(mgPlan.groups.count >= 2)
        #expect(try QTIPQuantizer.applyStagedRate(multiGroupModel as Module, plan: mgPlan, rate: .threeBPW) == mgSummary.quantizedLayers)
    }

    @Test("QTIPNativeLoader writes, inspects, and reloads ModelContainer bit-identically across families",
          arguments: [TinyModel.Family.prism, TinyModel.Family.gemma4Moe, TinyModel.Family.lfm2])
    func nativeLoaderShardRoundTrip(family: TinyModel.Family) async throws {
        let calibBatches: [[Int32]] = [(0 ..< 24).map { Int32(($0 * 11 + 5) % 180) }]
        let dir = FileManager.default.temporaryDirectory.appending(component: "qtip-native-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let modelDirect = try TinyModel.make(family)
        var cfg = QTIPConfig(rate: .twoHalfBPW)
        cfg.minDimension = 16
        cfg.blockGroupSize = 8
        let summary = try QTIPQuantizer.quantizeModel(modelDirect, calibrationTokenBatches: calibBatches, config: cfg)

        let srcDir = dir.appending(component: "src")
        let outDir = dir.appending(component: "out")
        try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
        try Data(family.config.utf8).write(to: srcDir.appending(component: "config.json"))
        try Data("{}".utf8).write(to: srcDir.appending(component: "tokenizer_config.json"))
        let written = try QTIPNativeLoader.writeDirectory(
            model: modelDirect as Module, rate: .twoHalfBPW, seed: cfg.seed, source: srcDir, output: outDir)
        #expect(written.shards == 1 && written.bytes > 0)

        let report = await ModelInspector().inspect(modelIdOrPath: outDir.path)
        #expect(report.intact && report.quantizationMode == "qtip")

        let manifest = try #require(QTIPNativeLoader.manifest(in: outDir))
        #expect(manifest.bpw == 2.5 && manifest.layers.count == summary.quantizedLayers)
        let container = try #require(try await QTIPNativeLoader.loadContainer(
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

    @Test("ConvertSubcommand validates --experimental-qtip rates and incompatible flags")
    func convertSubcommandValidation() throws {
        for bpw in ["2.0", "2.5", "3.0"] {
            let parsed = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-qtip", bpw])
            #expect(parsed.experimentalQtip == Double(bpw))
        }
        let withCalib = try ConvertSubcommand.parse([
            "src", "--out", "dst", "--experimental-qtip", "2.5", "--pv-calibration", "cal.txt"
        ])
        #expect(withCalib.pvCalibration == "cal.txt")
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-qtip", "4.0"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-qtip", "2.0", "--quantize", "affine:4:64"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-qtip", "2.0", "--dequantize", "bf16"])
        }
        let withPV = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-qtip", "2.0", "--experimental-pv-tune"])
        #expect(withPV.experimentalQtip == 2.0 && withPV.experimentalPvTune)
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--experimental-qtip", "2.0", "--adapter", "lora"])
        }
    }
}
