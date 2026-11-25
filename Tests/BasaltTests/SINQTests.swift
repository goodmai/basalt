import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltCLI
@testable import BasaltCore
@testable import BasaltInference

@Suite("SINQ, No-overhead SINQ, A-SINQ & EAQuant dual-axis Sinkhorn-Knopp normalization (#548)")
struct SINQTests {
    @Test("No-overhead SINQ RMSNorm absorption and EAQuant router compensation match exact float32 matmul")
    func algebraicEquivalenceStandardAndGemmaAndRouter() {
        let rng = MLXRandom.RandomState(seed: 548)
        let x = MLXRandom.normal([4, 64], dtype: .float32, key: rng)
        let gamma = MLXRandom.uniform(low: 0.5, high: 1.5, [64], dtype: .float32, key: rng)
        let wProj = MLXRandom.normal([32, 64], dtype: .float32, key: rng) * MLXArray((0 ..< 64).map { Float(1.0 + Double($0 % 8) * 1.5) }).reshaped(1, 64)
        let wRouter = MLXRandom.normal([8, 64], dtype: .float32, key: rng)

        let factors = SINQNormalizer.solve(wProj)
        #expect(factors.imbalanceAfter <= factors.imbalanceBefore + 1e-4)

        let hStd = MLXFast.rmsNorm(x, weight: gamma, eps: 1e-5)
        let refProj = matmul(hStd, wProj.T)
        let refRouter = matmul(hStd, wRouter.T)

        let gammaAbsorbed = SINQNormalizer.scaleNormWeight(gamma, colScale: factors.colScale, addsUnitOffset: false)
        let wNorm = wProj / (factors.rowScale.reshaped(32, 1) * factors.colScale.reshaped(1, 64))
        let wReconstructed = wNorm * factors.rowScale.reshaped(32, 1)
        let wRouterComp = SINQNormalizer.compensateDenseConsumer(wRouter, colScale: factors.colScale)

        let hAbsorbed = MLXFast.rmsNorm(x, weight: gammaAbsorbed, eps: 1e-5)
        let gotProj = matmul(hAbsorbed, wReconstructed.T)
        let gotRouter = matmul(hAbsorbed, wRouterComp.T)
        eval(refProj, gotProj, refRouter, gotRouter)

        #expect(abs(refProj - gotProj).max().item(Float.self) < 1e-4)
        #expect(abs(refRouter - gotRouter).max().item(Float.self) < 1e-4)

        let gemmaOffset = gamma - MLXArray(Float(1.0))
        let hGemmaRef = MLXFast.rmsNorm(x, weight: gemmaOffset + MLXArray(Float(1.0)), eps: 1e-6)
        let gemmaAbsorbed = SINQNormalizer.scaleNormWeight(gemmaOffset, colScale: factors.colScale, addsUnitOffset: true)
        let hGemmaGot = MLXFast.rmsNorm(x, weight: gemmaAbsorbed + MLXArray(Float(1.0)), eps: 1e-6)
        let gotGemmaProj = matmul(hGemmaGot, wReconstructed.T)
        eval(hGemmaRef, gotGemmaProj)
        #expect(abs(matmul(hGemmaRef, wProj.T) - gotGemmaProj).max().item(Float.self) < 1e-4)
    }

    @Test("SINQ reduces matrix imbalance and quantization reconstruction MSE over RTN across INT3, INT4, NF4, and MXFP4")
    func sinqReducesQuantizationErrorAcrossFormats() {
        let rng = MLXRandom.RandomState(seed: 2026)
        let base = MLXRandom.normal([64, 128], dtype: .float32, key: rng)
        let colSpikes = MLXArray((0 ..< 128).map { $0 % 16 == 0 ? Float(12.0) : Float(1.0) }).reshaped(1, 128)
        let rowSpikes = MLXArray((0 ..< 64).map { $0 % 8 == 0 ? Float(6.0) : Float(1.0) }).reshaped(64, 1)
        let w = base * rowSpikes * colSpikes
        let x = MLXRandom.normal([16, 128], dtype: .float32, key: rng)
        let targetOut = matmul(x, w.T)

        let factors = SINQNormalizer.solve(w)
        #expect(factors.imbalanceAfter < factors.imbalanceBefore)

        for bits in [3, 4] {
            let (rtnQ, rtnS, rtnB) = MLX.quantized(w.asType(.float16), groupSize: 32, bits: bits, mode: .affine)
            let rtnW = MLX.dequantized(rtnQ, scales: rtnS, biases: rtnB, groupSize: 32, bits: bits, mode: .affine).asType(.float32)
            let rtnMSE = mean(square(matmul(x, rtnW.T) - targetOut)).item(Float.self)

            let sinqPacked = SINQNormalizer.quantizeNormalized(
                base: "l", weight: w.asType(.float16),
                rowScale: factors.rowScale, colScale: factors.colScale,
                format: .affine(bits: bits, groupSize: 32))
            let sinqDeq = MLX.dequantized(
                sinqPacked["l.weight"]!, scales: sinqPacked["l.scales"]!, biases: sinqPacked["l.biases"],
                groupSize: 32, bits: bits, mode: .affine).asType(.float32)
            let sinqOut = matmul(x * factors.colScale.reshaped(1, 128), sinqDeq.T)
            let sinqMSE = mean(square(sinqOut - targetOut)).item(Float.self)
            #expect(sinqMSE < rtnMSE)
        }

        let (nf4CodesRTN, nf4ScalesRTN) = NF4Metal.quantized(w, groupSize: 32)
        let nf4RTNOut = NF4Metal.matmul(x.asType(.float16), codes: nf4CodesRTN, scales: nf4ScalesRTN, groupSize: 32).asType(.float32)
        let nf4RTNMSE = mean(square(nf4RTNOut - targetOut)).item(Float.self)

        let nf4SINQ = SINQNormalizer.quantizeNormalized(
            base: "l", weight: w.asType(.float16),
            rowScale: factors.rowScale, colScale: factors.colScale,
            format: .nf4(groupSize: 32))
        let nf4SINQOut = NF4Metal.matmul(
            (x * factors.colScale.reshaped(1, 128)).asType(.float16),
            codes: nf4SINQ["l.weight"]!, scales: nf4SINQ["l.scales"]!, groupSize: 32).asType(.float32)
        let nf4SINQMSE = mean(square(nf4SINQOut - targetOut)).item(Float.self)
        #expect(nf4SINQMSE < nf4RTNMSE)

        let (mxQRTN, mxSRTN, _) = MLX.quantized(w.asType(.float16), groupSize: 32, bits: 4, mode: .mxfp4)
        let mxWRTN = MLX.dequantized(mxQRTN, scales: mxSRTN, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4).asType(.float32)
        let mxRTNMSE = mean(square(matmul(x, mxWRTN.T) - targetOut)).item(Float.self)

        let mxSINQ = SINQNormalizer.quantizeNormalized(
            base: "l", weight: w.asType(.float16),
            rowScale: factors.rowScale, colScale: factors.colScale,
            format: .mxfp4)
        let mxWSINQ = MLX.dequantized(
            mxSINQ["l.weight"]!, scales: mxSINQ["l.scales"]!, biases: nil,
            groupSize: 32, bits: 4, mode: .mxfp4).asType(.float32)
        let mxSINQMSE = mean(square(matmul(x * factors.colScale.reshaped(1, 128), mxWSINQ.T) - targetOut)).item(Float.self)
        #expect(mxSINQMSE < mxRTNMSE)
    }

    @Test("WeightConverter with SINQ and EAQuant converts dense and MoE TinyModel checkpoints into stock MLX models")
    func endToEndTinyModelSINQAndEAQuant() throws {
        for family in [TinyModel.Family.qwen3Small, .qwen3Moe, .gemma3] {
            let rootDir = FileManager.default.temporaryDirectory.appending(path: "sinq-\(family.rawValue)-\(UUID().uuidString)")
            let srcDir = rootDir.appending(component: "src")
            let outDir = rootDir.appending(component: "out")
            try FileManager.default.createDirectory(at: srcDir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: rootDir) }

            let model = try TinyModel.make(family)
            let flat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            try save(arrays: flat, url: srcDir.appending(component: "model.safetensors"))
            var cfg = try JSONSerialization.jsonObject(with: Data(family.config.utf8)) as! [String: Any]
            if cfg["model_type"] == nil { cfg["model_type"] = "qwen3" }
            try JSONSerialization.data(withJSONObject: cfg).write(to: srcDir.appending(component: "config.json"))

            let converter = WeightConverter(precision: .affine(bits: 4, groupSize: 32), recipe: .sinq)
            let summary = try converter.run(source: srcDir, output: outDir)
            let sinq = try #require(summary.sinq)
            #expect(sinq.absorbedGroups > 0)
            #expect(sinq.normalizedTensors > 0)
            if family == .qwen3Moe {
                #expect(sinq.routerProjectionsAdjusted > 0)
            }

            let outData = try Data(contentsOf: outDir.appending(component: "config.json"))
            let reloaded = try family.model(outData)
            let outArrays = try loadArrays(url: outDir.appending(component: "model.safetensors"))
            quantize(model: reloaded) { path, _ in
                outArrays["\(path).scales"] != nil ? (32, 4, .affine) : nil
            }
            try reloaded.update(parameters: ModuleParameters.unflattened(outArrays), verify: .noUnusedKeys)
            let prompt = MLXArray([Int32(1), 2, 3, 4], [1, 4])
            let logits = reloaded(LMInput.Text(tokens: prompt), cache: nil, state: nil).logits
            eval(logits)
            #expect(logits.shape == [1, 4, 256])
            let vals = logits.asType(.float32).asArray(Float.self)
            let allFinite = vals.allSatisfy { $0.isFinite }
            #expect(allFinite)
        }
    }

    @Test("SINQLinearTap captures exact RMS and CLI validates --sinq / --a-sinq / --quant-recipe")
    func activationTapAndCLIRecipeValidation() throws {
        let lin = Linear(weight: MLXArray.ones([8, 16], dtype: .float32), bias: nil)
        let box = SINQActivationBox(inFeatures: 16)
        let tap = SINQLinearTap(wrapping: lin, box: box)
        let x = MLXArray.ones([4, 16], dtype: .float32) * Float(3.0)
        let y = tap(x)
        eval(y)
        #expect(y.shape == [4, 8])
        let rms = try #require(box.rms())
        #expect(abs(rms - Float(3.0)).max().item(Float.self) < 1e-4)

        let parsed = try ConvertSubcommand.parse(["src", "--out", "dst", "--quantize", "affine:4:64", "--sinq"])
        #expect(try parsed.resolvedSINQRecipe() == .sinq)
        let parsedA = try ConvertSubcommand.parse(["src", "--out", "dst", "--quantize", "mxfp4", "--a-sinq", "--pv-calibration", "cal.txt"])
        #expect(try parsedA.resolvedSINQRecipe() == .asinq)
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--quantize", "affine:4:64", "--sinq", "--a-sinq"])
        }
        #expect(throws: (any Error).self) {
            _ = try ConvertSubcommand.parse(["src", "--out", "dst", "--dequantize", "bf16", "--sinq"])
        }
    }
}
