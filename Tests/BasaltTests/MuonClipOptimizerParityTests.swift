import ArgumentParser
import Foundation
import Synchronization
import Testing
import MLX
import MLXNN
import MLXLMCommon
import MLXOptimizers
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

private final class DenseAttentionTestBlock: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let qHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    init(
        hiddenDim: Int,
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        bias: Bool = true,
        withQKNorm: Bool = false
    ) {
        self.qHeads = qHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.scale = 1.0 / Float(headDim).squareRoot()
        _qProj.wrappedValue = Linear(hiddenDim, qHeads * headDim, bias: bias)
        _kProj.wrappedValue = Linear(hiddenDim, kvHeads * headDim, bias: bias)
        if withQKNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim)
        }
        super.init()
    }

    func qkLayer(index: Int = 0) -> QKAttentionLayer {
        let components = QKAttentionLayer.Components(
            inputNorm: nil,
            qProj: qProj,
            kProj: kProj,
            qNorm: qNorm,
            kNorm: kNorm,
            rope: nil
        )
        let geometry = QKAttentionLayer.Geometry(
            qHeads: qHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            scale: scale,
            isGatedQuery: false
        )
        return QKAttentionLayer(layerIndex: index, components: components, geometry: geometry)
    }

    func fullCausalHeadMaxLogits(_ hidden: MLXArray) -> [Float] {
        let (b, l) = (hidden.dim(0), hidden.dim(1))
        var q = qProj(hidden).reshaped([b, l, qHeads, headDim])
        var k = kProj(hidden).reshaped([b, l, kvHeads, headDim])
        if let qNorm { q = qNorm(q) }
        if let kNorm { k = kNorm(k) }
        q = q.transposed(0, 2, 1, 3).asType(.float32) * scale
        let groups = qHeads / kvHeads
        k = k.transposed(0, 2, 1, 3).asType(.float32)
        if groups > 1 { k = repeated(k, count: groups, axis: 1) }
        var scores = matmul(q, k.transposed(0, 1, 3, 2))
        let allowed = MLXArray.arange(l).reshaped([1, 1, 1, l]) .<= MLXArray.arange(l).reshaped([1, 1, l, 1])
        scores = MLX.where(allowed, scores, MLXArray(-Float.infinity))
        let maxPerHead = scores.max(axes: [0, 2, 3])
        eval(maxPerHead)
        return maxPerHead.asArray(Float.self)
    }
}

private final class SingleMatrixModule: Module {
    var weight: MLXArray

    init(weight: MLXArray) {
        self.weight = weight
        super.init()
    }
}

private enum CPUMuonReference {
    struct StepConfig {
        let rows: Int
        let cols: Int
        let lr: Double
        let momentum: Double
        let weightDecay: Double
    }

    static func matMul(_ a: [Double], _ rA: Int, _ cA: Int, _ b: [Double], _ cB: Int) -> [Double] {
        var out = Array(repeating: 0.0, count: rA * cB)
        for i in 0 ..< rA {
            for k in 0 ..< cA {
                let aik = a[i * cA + k]
                for j in 0 ..< cB { out[i * cB + j] += aik * b[k * cB + j] }
            }
        }
        return out
    }

    static func transpose(_ a: [Double], rows: Int, cols: Int) -> [Double] {
        var out = Array(repeating: 0.0, count: rows * cols)
        for i in 0 ..< rows {
            for j in 0 ..< cols { out[j * rows + i] = a[i * cols + j] }
        }
        return out
    }

    static func newtonSchulz5(_ input: [Double], rows: Int, cols: Int, steps: Int = 5) -> [Double] {
        let transposed = rows > cols
        var x = transposed ? transpose(input, rows: rows, cols: cols) : input
        let r = transposed ? cols : rows
        let c = transposed ? rows : cols
        let frob = sqrt(x.reduce(0.0) { $0 + $1 * $1 }) + 1e-7
        x = x.map { $0 / frob }
        let (ca, cb, cc) = (3.4445, -4.7750, 2.0315)
        for _ in 0 ..< steps {
            let xT = transpose(x, rows: r, cols: c)
            let aMat = matMul(x, r, c, xT, r)
            let a2 = matMul(aMat, r, r, aMat, r)
            let bMat = zip(aMat, a2).map { cb * $0 + cc * $1 }
            let bx = matMul(bMat, r, r, x, c)
            x = zip(x, bx).map { ca * $0 + $1 }
        }
        return transposed ? transpose(x, rows: r, cols: c) : x
    }

    static func step(w: inout [Double], v: inout [Double], grad: [Float], config: StepConfig) {
        let g = grad.map(Double.init)
        for i in 0 ..< v.count {
            v[i] = config.momentum * v[i] + (1.0 - config.momentum) * g[i]
        }
        let nesterov = zip(g, v).map { (1.0 - config.momentum) * $0 + config.momentum * $1 }
        let ortho = newtonSchulz5(nesterov, rows: config.rows, cols: config.cols)
        let scale = 0.2 * sqrt(Double(max(config.rows, config.cols)))
        for i in 0 ..< w.count {
            let decayed = w[i] * (1.0 - config.lr * config.weightDecay)
            w[i] = decayed - config.lr * scale * ortho[i]
        }
    }
}

private enum MuonClipTestSupport {
    static func values(count: Int, seed: UInt64, scale: Double = 1.0) -> [Float] {
        var rng = SplitMix64(seed: seed)
        return (0 ..< count).map { idx in
            let u = Double(rng.next() >> 11) / Double(1 << 53)
            return Float(((u - 0.5) * 2.0 + 0.05 * sin(Double(idx))) * scale)
        }
    }

    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = MLX.abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    static func rms(_ array: MLXArray) -> Float {
        let value = MLX.sqrt(square(array.asType(.float32)).mean())
        eval(value)
        return value.item(Float.self)
    }
}

@Suite("MuonClip and Matched-RMS Muon: CPU parity, GQA QK-Clip, 4-bit QLoRA safety, CLI/MCP parsing, and resume (#846)", .serialized, .timeLimit(.minutes(2)))
struct MuonClipOptimizerParityTests {
    @Test("MatchedRMSMuonOptimizer matches CPU Newton-Schulz-5 reference to < 1e-4 across tall, wide, and square shapes with symmetric RMS ~ 0.2 * lr")
    func matchedRMSParityAcrossAspectRatios() {
        let lr: Float = 0.02
        let wd: Float = 0.05
        var rmsByShape: [Float] = []

        for (rows, cols, seed) in [(64, 8, UInt64(11)), (8, 64, UInt64(17)), (16, 16, UInt64(23))] {
            let initVals = MuonClipTestSupport.values(count: rows * cols, seed: seed)
            let before = MLXArray(initVals, [rows, cols])
            let model = SingleMatrixModule(weight: MLXArray(initVals, [rows, cols]))
            let opt = MatchedRMSMuonOptimizer(learningRate: lr, momentum: 0.95, weightDecay: wd)
            var cpuW = initVals.map(Double.init)
            var cpuV = Array(repeating: 0.0, count: rows * cols)

            for step in 0 ..< 4 {
                let gradVals = MuonClipTestSupport.values(count: rows * cols, seed: seed + UInt64(step + 100))
                opt.update(model: model, gradients: ModuleParameters.unflattened([("weight", MLXArray(gradVals, [rows, cols]))]))
                eval(model, opt)
                let stepCfg = CPUMuonReference.StepConfig(rows: rows, cols: cols, lr: Double(lr), momentum: 0.95, weightDecay: Double(wd))
                CPUMuonReference.step(w: &cpuW, v: &cpuV, grad: gradVals, config: stepCfg)
                if step == 0 {
                    let pureModel = SingleMatrixModule(weight: MLXArray(initVals, [rows, cols]))
                    let pureOpt = MatchedRMSMuonOptimizer(learningRate: lr, momentum: 0.95, weightDecay: 0.0)
                    pureOpt.update(model: pureModel, gradients: ModuleParameters.unflattened([("weight", MLXArray(gradVals, [rows, cols]))]))
                    eval(pureModel, pureOpt)
                    rmsByShape.append(MuonClipTestSupport.rms(before - pureModel.weight))
                }
            }

            let actual = model.weight.asType(.float32).asArray(Float.self)
            let diff = zip(actual, cpuW).map { Swift.abs($0 - Float($1)) }.max() ?? 0
            #expect(diff < 1e-4, "shape [\(rows), \(cols)] maxAbsDiff=\(diff)")
        }

        for observed in rmsByShape {
            let target = 0.2 * lr
            #expect(observed > target * 0.5 && observed < target * 1.5, "rms=\(observed) target=\(target)")
        }
        #expect(Swift.abs(rmsByShape[0] - rmsByShape[1]) < 0.2 * lr * 0.35)
    }

    @Test("MatrixOptimizerMath.gqaSafeQKClipScales bounds spiking heads cleanly with zero distortion to non-spiking sibling heads")
    func gqaSafeScalingFormulas() {
        let tau: Float = 100.0
        let singleSpike = MatrixOptimizerMath.gqaSafeQKClipScales(
            headMaxLogits: [400.0, 50.0, 80.0, 60.0],
            tau: tau,
            alpha: 0.5,
            kvHeads: 2
        )
        #expect(Swift.abs(singleSpike.kScales[0] - 1.0) < 1e-6)
        #expect(Swift.abs(singleSpike.qScales[0] - 0.25) < 1e-6)
        #expect(Swift.abs(singleSpike.qScales[1] - 1.0) < 1e-6)
        #expect(Swift.abs(singleSpike.kScales[1] - 1.0) < 1e-6)

        let dualSpike = MatrixOptimizerMath.gqaSafeQKClipScales(
            headMaxLogits: [400.0, 200.0, 40.0, 30.0],
            tau: tau,
            alpha: 0.5,
            kvHeads: 2
        )
        #expect(dualSpike.kScales[0] < 1.0 && dualSpike.qScales[0] < 1.0 && dualSpike.qScales[1] < 1.0)
        #expect(Swift.abs(dualSpike.qScales[0] * dualSpike.kScales[0] - 0.25) < 1e-5)
        #expect(Swift.abs(dualSpike.qScales[1] * dualSpike.kScales[0] - 0.50) < 1e-5)
        #expect(Swift.abs(dualSpike.qScales[2] - 1.0) < 1e-6 && Swift.abs(dualSpike.kScales[1] - 1.0) < 1e-6)

        let mha = MatrixOptimizerMath.gqaSafeQKClipScales(
            headMaxLogits: [250.0, 50.0],
            tau: tau,
            alpha: 0.5,
            kvHeads: 2
        )
        #expect(Swift.abs(mha.qScales[0] - Float(0.4).squareRoot()) < 1e-5)
        #expect(Swift.abs(mha.kScales[0] - Float(0.4).squareRoot()) < 1e-5)
        #expect(Swift.abs(mha.qScales[1] - 1.0) < 1e-6 && Swift.abs(mha.kScales[1] - 1.0) < 1e-6)
    }

    @Test("Dense MHA and GQA QK-Clip matches full causal logits when tiled, bounds spiking heads <= tau, preserves non-spiking sibling heads, and handles trainable QK-Norm")
    func denseMHAAndGQAQKClipParity() throws {
        let hidden = MLXArray(MuonClipTestSupport.values(count: 2 * 10 * 32, seed: 501, scale: 0.8), [2, 10, 32])
        let gqa = DenseAttentionTestBlock(hiddenDim: 32, qHeads: 4, kvHeads: 2, headDim: 8, bias: true, withQKNorm: false)
        var qW = MuonClipTestSupport.values(count: 32 * 32, seed: 502, scale: 0.35)
        for row in 0 ..< 8 {
            for col in 0 ..< 32 { qW[row * 32 + col] *= 25.0 }
        }
        let kW = MuonClipTestSupport.values(count: 16 * 32, seed: 503, scale: 0.6)
        gqa.qProj.update(parameters: ModuleParameters.unflattened([("weight", MLXArray(qW, [32, 32])), ("bias", MLXArray.zeros([32]))]))
        gqa.kProj.update(parameters: ModuleParameters.unflattened([("weight", MLXArray(kW, [16, 32])), ("bias", MLXArray.zeros([16]))]))
        eval(gqa)

        let layer = gqa.qkLayer()
        let tiled = QKClipController.layerMaxLogits(layer, hidden: hidden, isCausal: true, tileSize: 3)
        eval(tiled)
        let tiledValues = tiled.asArray(Float.self)
        let naiveBefore = gqa.fullCausalHeadMaxLogits(hidden)
        for (a, b) in zip(tiledValues, naiveBefore) {
            #expect(Swift.abs(a - b) < 1e-4)
        }
        let tau: Float = max(naiveBefore[1] * 1.5, 5.0)
        #expect(naiveBefore[0] > tau && naiveBefore[1] < tau)

        let clipped = QKClipController.applyClip(to: layer, headMaxLogits: tiledValues, tau: tau, alpha: 0.5)
        #expect(clipped)
        let naiveAfter = gqa.fullCausalHeadMaxLogits(hidden)
        #expect(naiveAfter[0] <= tau + 1e-3)
        #expect(Swift.abs(naiveAfter[1] - naiveBefore[1]) < 1e-4)
        #expect(Swift.abs(naiveAfter[2] - naiveBefore[2]) < 1e-4)
        #expect(Swift.abs(naiveAfter[3] - naiveBefore[3]) < 1e-4)

        let mhaNorm = DenseAttentionTestBlock(hiddenDim: 32, qHeads: 2, kvHeads: 2, headDim: 8, bias: false, withQKNorm: true)
        mhaNorm.qNorm?.update(parameters: ModuleParameters.unflattened([("weight", MLXArray.ones([8]) * Float(8.0))]))
        mhaNorm.kNorm?.update(parameters: ModuleParameters.unflattened([("weight", MLXArray.ones([8]) * Float(8.0))]))
        eval(mhaNorm)
        let normBefore = mhaNorm.fullCausalHeadMaxLogits(hidden)
        let peakNorm = try #require(normBefore.max())
        let normTau: Float = max(peakNorm * 0.4, 1.0)
        #expect(QKClipController.applyClip(to: mhaNorm.qkLayer(), headMaxLogits: normBefore, tau: normTau, alpha: 0.5))
        let normAfter = mhaNorm.fullCausalHeadMaxLogits(hidden)
        #expect(normAfter.allSatisfy { $0 <= normTau + 1e-3 })
    }
}

extension MuonClipOptimizerParityTests {
    @Test("4-bit QLoRA refuses muonclip cleanly while unfrozen QK-Norm model re-measures S_max <= tau after clipping")
    func qLoRARefusalAndFullFineTuneReMeasuredCap() throws {
        let qLoraModel = try TinyModel.make(.qwen3)
        MLXNN.quantize(model: qLoraModel, groupSize: 32, bits: 4, filter: { _, module in module is Linear })
        eval(qLoraModel)
        _ = try MLXInferenceEngine.attachLoRA(to: qLoraModel, rank: 4)
        defer { MLXInferenceEngine.detachLoRA(from: qLoraModel) }

        let qLoraLayers = QKClipController.attentionLayers(of: qLoraModel)
        #expect(!qLoraLayers.isEmpty && qLoraLayers.allSatisfy { !QKClipController.canBoundLogits($0) })
        #expect(throws: BasaltError.self) { try QKClipController.validateTrainableSupport(for: qLoraModel, prefix: "train") }

        let fullModel = try TinyModel.make(.qwen3)
        fullModel.unfreeze(recursive: true)
        try QKClipController.validateTrainableSupport(for: fullModel, prefix: "train")

        let inputs = MLXArray([Int32(1), 5, 9, 13, 21, 34, 55, 89]).reshaped([1, 8])
        let layers = QKClipController.attentionLayers(of: fullModel)
        let probeBefore = QKResidualProbe()
        _ = QKClipController.forward(fullModel, inputs: inputs, probe: probeBefore)
        let beforeArrays = QKClipController.layerMaxLogits(model: fullModel, layers: layers, inputs: inputs, probe: probeBefore, isCausal: true)
        probeBefore.reset()
        eval(beforeArrays)
        let layer0Before = beforeArrays[0].asArray(Float.self).max() ?? 0
        #expect(layer0Before > 0)

        var options = TrainingOptimizerOptions(kind: .muonclip)
        let tau = layer0Before * 0.5
        options.qkClipTau = tau
        let qProjBefore = layers[0].components.qProj.parameters().flattened().first(where: { $0.0 == "weight" })?.1.asType(.float32).asArray(Float.self)
        let stats = QKClipController.measureAndClip(model: fullModel, inputs: inputs, isCausal: true, options: options)
        #expect(stats.clippedLayers >= 1 && stats.clippedHeads >= 1)
        let qProjAfter = layers[0].components.qProj.parameters().flattened().first(where: { $0.0 == "weight" })?.1.asType(.float32).asArray(Float.self)
        #expect(qProjBefore != nil && qProjBefore == qProjAfter)

        let probeAfter = QKResidualProbe()
        _ = QKClipController.forward(fullModel, inputs: inputs, probe: probeAfter)
        let afterArrays = QKClipController.layerMaxLogits(model: fullModel, layers: layers, inputs: inputs, probe: probeAfter, isCausal: true)
        probeAfter.reset()
        eval(afterArrays)
        let layer0After = afterArrays[0].asArray(Float.self).max() ?? 0
        #expect(layer0After <= tau + 1e-3, "layer 0 re-measured S_max=\(layer0After) exceeded tau=\(tau)")
        let allAfter = afterArrays.flatMap { $0.asArray(Float.self) }
        #expect(allAfter.allSatisfy { $0.isFinite })
    }

    @Test("CLI subcommands, MCP schemas, and adapter validators accept soap and refuse muonclip early before work starts")
    func cliAndMCPParsingForMuonClipAndSOAP() throws {
        var fullClip = TrainingOptimizerOptions(kind: .muonclip)
        fullClip.qkClipTau = 75.0
        fullClip.qkClipAlpha = 0.4
        try fullClip.validate(prefix: "full")
        #expect(throws: BasaltError.self) { try fullClip.validateAdapterCompatible(prefix: "adapter") }

        let trainSoap = try TrainSubcommand.parse([
            "org/model", "--dataset", "local.jsonl", "--optimizer", "soap",
            "--soap-precond-freq", "5", "--soap-max-precond-dim", "512", "--soap-shampoo-beta", "0.9",
        ])
        #expect(trainSoap.optimizer == .soap)
        let soapOpts = trainSoap.optimizerOptions.buildOptions()
        #expect(soapOpts.soapPrecondFreq == 5 && soapOpts.soapMaxPrecondDim == 512 && soapOpts.soapShampooBeta == 0.9)

        let distillBase = ["org/model", "--prompts", "p.jsonl", "--teacher-url", "http://127.0.0.1:8080/v1", "--teacher-model", "t"]
        let distillSoap = try DistillSubcommand.parse(distillBase + ["--optimizer", "soap", "--soap-precond-freq", "4"])
        #expect(distillSoap.optimizer == .soap && distillSoap.optimizerOptions.buildOptions().soapPrecondFreq == 4)

        #expect(throws: (any Error).self) { _ = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "muonclip", "--qk-clip-tau", "75.0"]) }
        #expect(throws: (any Error).self) { _ = try DistillSubcommand.parse(distillBase + ["--optimizer", "muonclip"]) }
        #expect(throws: (any Error).self) { _ = try TrainJepaSubcommand.parse(["org/model", "--dataset", "s.jsonl", "--optimizer", "muonclip"]) }
        #expect(throws: (any Error).self) { _ = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "soap", "--soap-precond-freq", "0"]) }
        #expect(throws: (any Error).self) { _ = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "soap", "--soap-shampoo-beta", "1.0"]) }
        #expect(throws: (any Error).self) { _ = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "adamw", "--qk-clip-tau", "50"]) }
        #expect(throws: (any Error).self) { _ = try TrainSubcommand.parse(["org/model", "--dataset", "local.jsonl", "--optimizer", "muon", "--soap-precond-freq", "5"]) }

        let mcpJepa = try LeJEPATrainRequest(mcp: ["datasetPath": "s.jsonl", "optimizer": "soap", "soapMaxPrecondDim": 256])
        #expect(mcpJepa.optimizer == .soap && mcpJepa.effectiveOptimizerOptions.soapMaxPrecondDim == 256)
        let mcpTrain = try TrainRequest(mcp: ["heldOut": ["Hi"], "optimizer": "soap", "soapPrecondFreq": 8])
        #expect(mcpTrain.optimizer == .soap && mcpTrain.effectiveOptimizerOptions.soapPrecondFreq == 8)

        #expect(ResearchTool.trainProperties["qkClipTau"] != nil && ResearchTool.trainProperties["soapMaxPrecondDim"] != nil)
        #expect(ResearchTool.trainProperties["soapBeta1"] != nil && ResearchTool.trainProperties["soapBeta2"] != nil && ResearchTool.trainProperties["soapEps"] != nil)
        #expect(ResearchTool.distillProperties["qkClipTau"] != nil && ResearchTool.distillProperties["soapPrecondFreq"] != nil)
        #expect(ResearchTool.trainJepaProperties["soapShampooBeta"] != nil)
        #expect(throws: MCPFailure.self) { _ = try DistillRequest(mcp: ["prompts": "p.jsonl", "teacherUrl": "http://127.0.0.1:8080/v1", "teacherModel": "t", "heldOut": ["Hi"], "optimizer": "muonclip"]) }
        #expect(throws: MCPFailure.self) { _ = try TrainRequest(mcp: ["heldOut": ["Hi"], "optimizer": "muonclip"]) }
        #expect(throws: MCPFailure.self) { _ = try LeJEPATrainRequest(mcp: ["datasetPath": "s.jsonl", "optimizer": "soap", "soapShampooBeta": 1.5]) }
        #expect(throws: MCPFailure.self) { _ = try TrainRequest(mcp: ["heldOut": ["Hi"], "optimizer": "adamw", "qkClipTau": 50.0]) }
        #expect(throws: MCPFailure.self) { _ = try TrainRequest(mcp: ["heldOut": ["Hi"], "optimizer": "muon", "soapPrecondFreq": 5]) }

        var dpoClip = OfflineDPOConfig(pairsPath: "pairs.jsonl")
        dpoClip.optimizer = .muonclip
        var grpoClip = GRPOConfig()
        grpoClip.optimizer = .muonclip
        #expect(throws: BasaltError.self) { try dpoClip.validate() }
        #expect(throws: BasaltError.self) { try grpoClip.validate() }
    }

    @Test("Distillation with Matched-RMS Muon and SOAP lowers held-out NLL and resumes identically, while LoRA MuonClip refuses cleanly")
    func distillMuonSOAPAndLoRAMuonClipRefusal() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        let scratch = FileManager.default.temporaryDirectory.appending(path: "r5-distill-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        var refusedRequest = PreemptibleTrainingTests.request(output: scratch.appending(component: "refused-muonclip"), iterations: 2)
        refusedRequest.optimizer = .muonclip
        refusedRequest.optimizerOptions = TrainingOptimizerOptions(kind: .muonclip)
        await #expect(throws: BasaltError.self) { _ = try await engine.distill(refusedRequest, lease: nil) }
        #expect(!FileManager.default.fileExists(atPath: scratch.appending(component: "refused-muonclip").path))

        for kind in [TrainingOptimizerKind.muon, TrainingOptimizerKind.soap] {
            let (full, copy) = (scratch.appending(component: "full-\(kind.rawValue)"), scratch.appending(component: "resumed-\(kind.rawValue)"))
            let copied = Mutex(false)
            let snapshot = StepLease(enter: {
                let source = AdapterCheckpoint.directory(in: full)
                if PreemptibleTrainingTests.savedStep(full) == 2, !copied.withLock({ $0 }) {
                    try? FileManager.default.createDirectory(at: copy, withIntermediateDirectories: true)
                    try? FileManager.default.copyItem(at: source, to: AdapterCheckpoint.directory(in: copy))
                    copied.withLock { $0 = true }
                }
                return 0
            }, leave: { _ in })

            var options = TrainingOptimizerOptions(kind: kind)
            options.soapPrecondFreq = 3
            var request = PreemptibleTrainingTests.request(output: full, iterations: 6)
            request.learningRate = 5e-3
            request.optimizer = kind
            request.optimizerOptions = options
            let reference = try await PreemptibleTrainingTests.run(engine, request, lease: snapshot)
            #expect(reference.losses.count == 6 && reference.losses.allSatisfy(\.isFinite))
            #expect(reference.nllAfter < reference.nllBefore, "\(kind.rawValue): \(reference.nllBefore) -> \(reference.nllAfter)")

            var resume = PreemptibleTrainingTests.request(output: copy, iterations: 6)
            resume.learningRate = 5e-3
            resume.optimizer = kind
            resume.optimizerOptions = options
            resume.resume = true
            let resumed = try await PreemptibleTrainingTests.run(engine, resume)
            #expect(resumed.resumedAt == 2)
            #expect(zip(resumed.losses, reference.losses).allSatisfy { Swift.abs($0 - $1) < 1e-4 })
            #expect(Swift.abs(resumed.nllAfter - reference.nllAfter) < 1e-4)
            #expect(await DistillTests.baseWeights(container) == before)
        }
    }
}
