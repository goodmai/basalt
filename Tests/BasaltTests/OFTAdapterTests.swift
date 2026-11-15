import Foundation
import Testing
import MLX
import MLXNN
import MLXOptimizers
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference
@testable import BasaltServer
@testable import BasaltCLI

private enum OFTTestSupport {
    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    static func singularValues(_ matrix: MLXArray) -> MLXArray {
        let s: MLXArray = MLXLinalg.svd(matrix.asType(.float32), stream: .cpu)
        eval(s)
        return s
    }

    static func trainLayerMSE(
        layer: OFTLinear,
        inputs: MLXArray,
        targets: MLXArray,
        optimizer: any Optimizer,
        steps: Int
    ) -> [Float] {
        let module = layer as Module
        let gradFn = valueAndGrad(model: module) { _, arrays in
            let diff = layer(arrays[0]).asType(.float32) - arrays[1]
            return [(diff * diff).mean()]
        }
        var losses: [Float] = []
        for _ in 0 ..< steps {
            let (vals, grads) = gradFn(module, [inputs, targets])
            optimizer.update(model: module, gradients: grads)
            eval(module, optimizer, vals)
            losses.append(vals[0].item(Float.self))
        }
        let endDiff = layer(inputs).asType(.float32) - targets
        let endLoss = (endDiff * endDiff).mean()
        eval(endLoss)
        losses.append(endLoss.item(Float.self))
        return losses
    }

    static func trainLanguageModelSteps(
        model: any LanguageModel,
        inputs: MLXArray,
        targets: MLXArray,
        steps: Int,
        learningRate: Float
    ) -> [Float] {
        let optimizer = Adam(learningRate: learningRate)
        let module = model as Module
        let gradFn = valueAndGrad(model: module) { _, arrays in
            let logits = model(arrays[0], cache: nil).asType(.float32)
            return [crossEntropy(logits: logits, targets: arrays[1], reduction: .mean)]
        }
        var losses: [Float] = []
        for _ in 0 ..< steps {
            let (vals, grads) = gradFn(module, [inputs, targets])
            optimizer.update(model: module, gradients: grads)
            eval(module, optimizer, vals)
            losses.append(vals[0].item(Float.self))
        }
        let finalLoss = crossEntropy(
            logits: model(inputs, cache: nil).asType(.float32),
            targets: targets,
            reduction: .mean
        )
        eval(finalLoss)
        losses.append(finalLoss.item(Float.self))
        return losses
    }

    static func maxModelDefect(in model: Module) -> Float {
        model.namedModules()
            .compactMap { ($0.1 as? any OFTAdapterLayer)?.maxOrthogonalityDefect() }
            .max() ?? 0.0
    }

    static func measureStepLatency(
        spec: AdapterSpec,
        warmup: Int = 2,
        iterations: Int = 5
    ) throws -> (params: Int, forwardMs: Double, stepMs: Double, defect: Float) {
        let model = try TinyModel.make(.qwen3)
        _ = try PHMAdapterApplicator.apply(to: model, spec: spec)
        let params = PHMAdapterApplicator.parameterCount(in: model)
        let inputs = MLXArray([Int32(2), 5, 9, 14, 21, 27, 33, 42], [1, 8])
        let targets = MLXArray([Int32(5), 9, 14, 21, 27, 33, 42, 51], [1, 8])
        for _ in 0 ..< warmup { eval(model(inputs, cache: nil)) }
        let clock = ContinuousClock()
        let fwdStart = clock.now
        for _ in 0 ..< iterations { eval(model(inputs, cache: nil)) }
        let fwdElapsed = fwdStart.duration(to: clock.now)
        let optimizer = Adam(learningRate: 5e-3)
        let module = model as Module
        let gradFn = valueAndGrad(model: module) { _, arrays in
            [crossEntropy(logits: model(arrays[0], cache: nil).asType(.float32), targets: arrays[1], reduction: .mean)]
        }
        for _ in 0 ..< warmup {
            let (vals, grads) = gradFn(module, [inputs, targets])
            optimizer.update(model: module, gradients: grads)
            eval(module, optimizer, vals)
        }
        let stepStart = clock.now
        for _ in 0 ..< iterations {
            let (vals, grads) = gradFn(module, [inputs, targets])
            optimizer.update(model: module, gradients: grads)
            eval(module, optimizer, vals)
        }
        let stepElapsed = stepStart.duration(to: clock.now)
        let fwdMs = Double(fwdElapsed.components.seconds) * 1000.0 + Double(fwdElapsed.components.attoseconds) / 1e15
        let stepMs = Double(stepElapsed.components.seconds) * 1000.0 + Double(stepElapsed.components.attoseconds) / 1e15
        return (params, fwdMs / Double(iterations), stepMs / Double(iterations), maxModelDefect(in: model))
    }
}

@Suite("OFT orthogonal adapters: step-0 bit identity and 20-step GPU orthogonality invariants (#363)", .serialized)
struct OFTInitializationAndOrthogonalityTests {
    @Test(
        "Step-0 initialization (Q=0 -> R=I) matches dense and quantized Qwen3 and Gemma4 within 1e-6",
        arguments: [TinyModel.Family.qwen3, TinyModel.Family.gemma4]
    )
    func stepZeroIdentityDenseAndQuantized(family: TinyModel.Family) throws {
        let inputs = MLXArray([Int32(3), 7, 11, 19, 23, 31], [1, 6])
        for param in OrthogonalParameterization.allCases {
            for shared in [false, true] {
                let dense = try TinyModel.make(family)
                let baseDense = dense(inputs, cache: nil).asType(.float32)
                eval(baseDense)
                let spec = AdapterSpec(method: .oft(blockSize: 8, parameterization: param, blockShared: shared))
                let replaced = try PHMAdapterApplicator.apply(to: dense, spec: spec)
                #expect(!replaced.isEmpty)
                #expect(dense.namedModules().contains { $0.1 is OFTLinear })
                #expect(OFTTestSupport.maxModelDefect(in: dense) == 0.0)
                let adaptedDense = dense(inputs, cache: nil).asType(.float32)
                eval(adaptedDense)
                #expect(OFTTestSupport.maxAbsDiff(adaptedDense, baseDense) < 1e-6)
            }
            let quantizedModel = try TinyModel.make(family)
            quantize(model: quantizedModel, groupSize: 32, bits: 4) { _, mod in
                (mod as? Linear).map { $0.weight.dim(-1).isMultiple(of: 32) } ?? false
            }
            eval(quantizedModel)
            let baseQuant = quantizedModel(inputs, cache: nil).asType(.float32)
            eval(baseQuant)
            let qSpec = AdapterSpec(method: .oft(blockSize: 8, parameterization: param, blockShared: false))
            _ = try PHMAdapterApplicator.apply(to: quantizedModel, spec: qSpec)
            #expect(quantizedModel.namedModules().contains { $0.1 is QuantizedOFTLinear })
            let adaptedQuant = quantizedModel(inputs, cache: nil).asType(.float32)
            eval(adaptedQuant)
            #expect(OFTTestSupport.maxAbsDiff(adaptedQuant, baseQuant) < 1e-6)
        }
    }

    @Test(
        "After 20 Adam and SGD steps across blockSize in [16, 32], ||R^T R - I||_F < 1e-5 for matrixExp and cayley",
        arguments: OrthogonalParameterization.allCases, [16, 32]
    )
    func standaloneTwentyStepOrthogonality(param: OrthogonalParameterization, blockSize: Int) throws {
        try withRandomState(MLXRandom.RandomState(seed: UInt64(3630 + blockSize + (param == .cayley ? 0 : 100)))) {
            let inputDim = 64
            let outputDim = 48
            let base = Linear(inputDim, outputDim, bias: true)
            let x = MLXRandom.normal([8, inputDim])
            let targetBasis = OFTSkewBasis(blockSize: blockSize)
            let targetSkew = MLXRandom.normal([inputDim / blockSize, targetBasis.freePerBlock]) * Float(0.35)
            let targetR = OFTMath.orthogonalBlocks(params: targetSkew, basis: targetBasis, parameterization: param)
            let targetWeight = OFTMath.rotateWeight(base.weight, blocks: targetR).asType(base.weight.dtype)
            let teacher = Linear(weight: targetWeight, bias: base.bias)
            let y = teacher(x).asType(.float32)
            eval(base, x, y)

            let adamLayer = try OFTLinear(linear: base, blockSize: blockSize, parameterization: param, blockShared: false)
            let adamLosses = OFTTestSupport.trainLayerMSE(
                layer: adamLayer,
                inputs: x,
                targets: y,
                optimizer: Adam(learningRate: 2e-2),
                steps: 20
            )
            #expect(adamLosses[20] < adamLosses[0])
            #expect(adamLayer.maxOrthogonalityDefect() < 1e-5)

            let sgdLayer = try OFTLinear(linear: base, blockSize: blockSize, parameterization: param, blockShared: true)
            let sgdLosses = OFTTestSupport.trainLayerMSE(
                layer: sgdLayer,
                inputs: x,
                targets: y,
                optimizer: SGD(learningRate: 5e-2),
                steps: 20
            )
            #expect(sgdLosses[20] < sgdLosses[0])
            #expect(sgdLayer.maxOrthogonalityDefect() < 1e-5)
        }
    }

    @Test(
        "TinyModel Qwen3 20-step Adam training maintains ||R^T R - I||_F < 1e-5 across blockSize in [4, 8]",
        arguments: OrthogonalParameterization.allCases, [4, 8]
    )
    func tinyModelTwentyStepOrthogonality(param: OrthogonalParameterization, blockSize: Int) throws {
        let model = try TinyModel.make(.qwen3)
        let spec = AdapterSpec(method: .oft(blockSize: blockSize, parameterization: param, blockShared: false))
        _ = try PHMAdapterApplicator.apply(to: model, spec: spec)
        let inputs = MLXArray([Int32(4), 9, 15, 22, 28, 35], [1, 6])
        let targets = MLXArray([Int32(9), 15, 22, 28, 35, 44], [1, 6])
        let losses = OFTTestSupport.trainLanguageModelSteps(
            model: model,
            inputs: inputs,
            targets: targets,
            steps: 20,
            learningRate: 1e-2
        )
        #expect(losses.count == 21)
        #expect(losses[20] < losses[0])
        #expect(OFTTestSupport.maxModelDefect(in: model) < 1e-5)
    }
}

@Suite("OFT orthogonal adapters: fused() parity, singular-value preservation, budgets, and checkpoints (#363)", .serialized)
struct OFTFusionBudgetAndCheckpointTests {
    @Test(
        "fused() matches OFTLinear(x) within 1e-5 and preserves exact singular values of W_0 within 1e-4",
        arguments: OrthogonalParameterization.allCases, [false, true]
    )
    func fusedParityAndSingularValuePreservation(param: OrthogonalParameterization, shared: Bool) throws {
        try withRandomState(MLXRandom.RandomState(seed: UInt64(3635 + (shared ? 10 : 0) + (param == .cayley ? 1 : 2)))) {
            let inputDim = 64
            let outputDim = 48
            let blockSize = 16
            let base = Linear(inputDim, outputDim, bias: true)
            let oft = try OFTLinear(linear: base, blockSize: blockSize, parameterization: param, blockShared: shared)
            let paramBlocks = shared ? 1 : inputDim / blockSize
            let freePerBlock = blockSize * (blockSize - 1) / 2
            let randomSkew = MLXRandom.normal([paramBlocks, freePerBlock]) * Float(0.6)
            oft.update(parameters: .unflattened(["oft_r": randomSkew]))
            eval(base, oft)

            #expect(oft.maxOrthogonalityDefect() < 1e-5)
            let fusedLinear = try #require(oft.fused() as? Linear)
            for shape in [[1, 1, inputDim], [3, 5, inputDim]] {
                let x = MLXRandom.normal(shape)
                let liveOut = oft(x)
                let fusedOut = fusedLinear(x)
                eval(liveOut, fusedOut)
                #expect(OFTTestSupport.maxAbsDiff(liveOut, fusedOut) < 1e-5)
            }

            let svBase = OFTTestSupport.singularValues(base.weight)
            let svFused = OFTTestSupport.singularValues(fusedLinear.weight)
            #expect(svBase.shape == svFused.shape)
            #expect(OFTTestSupport.maxAbsDiff(svBase, svFused) < 1e-4)

            let qBase = QuantizedLinear(Linear(64, 64, bias: false), groupSize: 32, bits: 4)
            let qOft = try QuantizedOFTLinear(linear: qBase, blockSize: 16, parameterization: param, blockShared: shared)
            qOft.update(parameters: .unflattened(["oft_r": randomSkew * Float(0.1)]))
            let probe = MLXRandom.normal([2, 64])
            let qLive = qOft(probe)
            let qFusedLayer = try #require(qOft.fused() as? QuantizedLinear)
            let qFused = qFusedLayer(probe)
            eval(qLive, qFused)
            #expect(OFTTestSupport.maxAbsDiff(qLive, qFused) < 0.5)
        }
    }

    @Test("Exact parameter counting and matchingBudget align .oft, .lora, and .phm to true trainable element counts")
    func parameterCountingAndMatchingBudget() throws {
        let methods: [AdapterMethod] = [
            .oft(blockSize: 8, parameterization: .cayley, blockShared: false),
            .oft(blockSize: 8, parameterization: .matrixExp, blockShared: true),
            .oft(blockSize: 16, parameterization: .cayley, blockShared: false),
            .oft(blockSize: 32, parameterization: .matrixExp, blockShared: true),
        ]
        for method in methods {
            let model = try TinyModel.make(.qwen3)
            let spec = AdapterSpec(method: method)
            let dims = try PHMAdapterApplicator.projectionDims(in: model, spec: spec)
            _ = try PHMAdapterApplicator.apply(to: model, spec: spec)
            #expect(PHMAdapterApplicator.parameterCount(in: model) == method.parameterCount(projections: dims))

            let gateOnly = try TinyModel.make(.qwen3)
            let gateSpec = AdapterSpec(method: method, targetSuffixes: ["gate_proj"])
            _ = try PHMAdapterApplicator.apply(to: gateOnly, spec: gateSpec)
            #expect(PHMAdapterApplicator.parameterCount(in: gateOnly) == method.parameterCount(inputDim: 64, outputDim: 128, layerCount: 4))
        }

        let probeModel = try TinyModel.make(.qwen3)
        let loraSpec = AdapterSpec(method: .lora(rank: 2, scale: 16.0))
        let dims = try PHMAdapterApplicator.projectionDims(in: probeModel, spec: loraSpec)
        let matchedOFT = try loraSpec.matchingBudget(to: .oft(parameterization: .cayley, blockShared: false), projections: dims)
        #expect(matchedOFT.method == .oft(blockSize: 8, parameterization: .cayley, blockShared: false))
        #expect(matchedOFT.method.parameterCount(projections: dims) == 7168)
        let roundTripLoRA = try matchedOFT.matchingBudget(to: .lora(scale: 16.0), projections: dims)
        #expect(roundTripLoRA.method == .lora(rank: 2, scale: 16.0))
        let matchedPHM = try matchedOFT.matchingBudget(to: .phm(hypercomplexDim: 8, sharedBasis: true, scale: 16.0), projections: dims)
        #expect(matchedPHM.method == .phm(hypercomplexDim: 8, factorRank: 2, sharedBasis: true, scale: 16.0))
    }

    @Test("OFTAdapterContainer safetensors round-trip, fusion, bit-exact revert, and invalid shape refusal")
    func checkpointRoundTripAndValidation() throws {
        let model = try TinyModel.make(.qwen3)
        let inputs = MLXArray([Int32(5), 11, 17, 23, 29, 37], [1, 6])
        let targets = MLXArray([Int32(11), 17, 23, 29, 37, 43], [1, 6])
        let baseLogits = model(inputs, cache: nil).asType(.float32)
        eval(baseLogits)
        let spec = AdapterSpec(method: .oft(blockSize: 8, parameterization: .matrixExp, blockShared: false))
        _ = try PHMAdapterApplicator.apply(to: model, spec: spec)
        _ = OFTTestSupport.trainLanguageModelSteps(model: model, inputs: inputs, targets: targets, steps: 4, learningRate: 1e-2)
        let adaptedLogits = model(inputs, cache: nil).asType(.float32)
        eval(adaptedLogits)

        let scratch = FileManager.default.temporaryDirectory.appending(path: "oft-ckpt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let config = OFTAdapterConfiguration(
            numLayers: 4,
            oftParameters: .init(blockSize: 8, parameterization: .matrixExp, blockShared: false)
        )
        try OFTAdapterContainer.save(configuration: config, model: model, to: scratch)
        let loaded = try OFTAdapterContainer.from(directory: scratch)
        let fresh = try TinyModel.make(.qwen3)
        try fresh.load(adapter: loaded)
        let reloadedLogits = fresh(inputs, cache: nil).asType(.float32)
        eval(reloadedLogits)
        #expect(OFTTestSupport.maxAbsDiff(reloadedLogits, adaptedLogits) < 1e-6)

        _ = PHMAdapterApplicator.fuse(in: fresh)
        #expect(!fresh.namedModules().contains { $0.1 is LoRALayer })
        let fusedLogits = fresh(inputs, cache: nil).asType(.float32)
        eval(fusedLogits)
        #expect(OFTTestSupport.maxAbsDiff(fusedLogits, adaptedLogits) < 1e-4)

        _ = PHMAdapterApplicator.revert(in: model)
        let revertedLogits = model(inputs, cache: nil).asType(.float32)
        eval(revertedLogits)
        #expect(OFTTestSupport.maxAbsDiff(revertedLogits, baseLogits) == 0.0)

        #expect(throws: BasaltError.self) { _ = try OFTLinear(linear: Linear(60, 64, bias: false), blockSize: 8) }
        #expect(throws: BasaltError.self) { try AdapterMethod.oft(blockSize: 1).validate() }
    }
}

@Suite("OFT orthogonal adapters: LabTrain, OnPolicyDistill, X-ray orthogonalityDefect, CLI, MCP, and bench (#363)", .serialized)
struct OFTTrainingPipelineAndSurfaceTests {
    @Test("LabTrain with .oft(blockSize: 8, .cayley) lowers loss, saves 0700 OFT safetensors, applies and undoes cleanly")
    func labTrainEndToEndWithOFT() async throws {
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let scratch = try LabTrainTests.scratch()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let directory = scratch.appending(path: "oft-lab")
        var request = LabTrainTests.request(output: directory)
        request.layers = [1, 3]
        request.method = .oft(blockSize: 8, parameterization: .cayley, blockShared: false)
        request.iterations = 30
        request.learningRate = 1e-2
        let pairs = TrainingTask.arithmetic(count: 4, seed: 1)
        let before = await DistillTests.baseWeights(container)
        let run = try await container.performEvaluating(nonSendable: ()) { [request] context, _ in
            try MLXInferenceEngine.trainAdapter(pairs, heldOut: Array(pairs.prefix(1)), request: request, directory: directory, context: context)
        }
        let first = try #require(run.losses.first)
        let last = try #require(run.losses.last)
        #expect(last < first, "\(run.losses)")
        #expect(try FileManager.default.attributesOfItem(atPath: directory.path)[.posixPermissions] as? Int == 0o700)
        let saved = try OFTAdapterContainer.from(directory: directory)
        #expect(saved.configuration.fineTuneType == "oft" && saved.configuration.oftParameters.blockSize == 8)

        let verdict = try await engine.applyAdapter(.directory(directory.path), gate: nil)
        let overlay = try #require(verdict.overlay)
        var tensorReq = TensorRequest(name: "model.layers.1.self_attn.q_proj.weight", maxValues: 0)
        tensorReq.spectrum = true
        let report = try await engine.tensor(tensorReq)
        let defect = try #require(report.spectrum?.orthogonalityDefect)
        #expect(defect < 1e-5)
        try await engine.undo(overlay.id)
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("OnPolicyDistill with .oft(blockSize: 8, .cayley) lowers reverse KL to teacher and leaves base weights bit-identical")
    func onPolicyDistillWithOFT() async throws {
        let directory = try DistillTests.scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let host = try OnPolicyDistillTests.teacherHost()
        let engine = try await TinyModel.engine(.qwen3)
        let container = try #require(await engine.container)
        let before = await DistillTests.baseWeights(container)
        var request = try OnPolicyDistillTests.onPolicy(host, prompts: 8, in: directory)
        request.method = .oft(blockSize: 8, parameterization: .cayley, blockShared: false)
        request.iterations = 40
        request.learningRate = 5e-3
        let report = try await engine.distill(
            request,
            session: URLSession(configuration: StubURLProtocol.configuration),
            home: directory,
            environment: [:]
        )
        let klBefore = try #require(report.reverseKLBefore)
        let klAfter = try #require(report.reverseKLAfter)
        #expect(klAfter < klBefore, "\(klBefore) -> \(klAfter)")
        let saved = try OFTAdapterContainer.from(directory: directory.appending(path: "adapter"))
        #expect(saved.configuration.oftParameters.parameterization == .cayley)
        if let overlay = report.verdict?.overlay { try await engine.undo(overlay.id) }
        #expect(await DistillTests.baseWeights(container) == before)
    }

    @Test("X-ray WeightSpectrum.measure reports orthogonalityDefect < 1e-5 on orthogonal matrices and > 0.1 on non-orthogonal")
    func xraySpectrumOrthogonalityDefect() throws {
        let basis = OFTSkewBasis(blockSize: 16)
        let skew = MLXArray((0 ..< basis.freePerBlock).map { Float($0 + 1) * 0.02 }, [1, basis.freePerBlock])
        let rBlock = OFTMath.orthogonalBlocks(params: skew, basis: basis, parameterization: .cayley)[0]
        eval(rBlock)
        let orthoSpectrum = try WeightSpectrum.measure(rows: 16, columns: 16) { rBlock[$0] }
        let orthoDefect = try #require(orthoSpectrum.orthogonalityDefect)
        #expect(orthoDefect < 1e-5)

        let scaled = rBlock * Float(1.5)
        eval(scaled)
        let nonOrthoSpectrum = try WeightSpectrum.measure(rows: 16, columns: 16) { scaled[$0] }
        let nonOrthoDefect = try #require(nonOrthoSpectrum.orthogonalityDefect)
        #expect(nonOrthoDefect > 1.0)

        let rect = basis.identity[0][0 ..< 16, 0 ..< 8]
        eval(rect)
        let rectSpectrum = try WeightSpectrum.measure(rows: 16, columns: 8) { rect[$0] }
        #expect(rectSpectrum.orthogonalityDefect == nil)
    }

    @Test("CLI DistillSubcommand and MCP TrainRequest/DistillRequest parse OFT flags and reject invalid values")
    func cliAndMCPArgumentParsing() throws {
        let cliDefault = try DistillSubcommand.parse([
            "tiny-model", "--prompts", "p.jsonl", "--teacher-url", "http://127.0.0.1:8080/v1", "--teacher-model", "t",
        ])
        #expect(try cliDefault.adapter.resolveMethod(fallbackRank: cliDefault.rank) == nil)

        let cliOFT = try DistillSubcommand.parse([
            "tiny-model", "--prompts", "p.jsonl", "--teacher-url", "http://127.0.0.1:8080/v1", "--teacher-model", "t",
            "--adapter-method", "oft", "--oft-block-size", "32", "--oft-parameterization", "matrixExp", "--oft-block-shared",
        ])
        #expect(try cliOFT.adapter.resolveMethod(fallbackRank: 16) == .oft(blockSize: 32, parameterization: .matrixExp, blockShared: true))

        let cliBad = try DistillSubcommand.parse([
            "tiny-model", "--prompts", "p.jsonl", "--teacher-url", "http://127.0.0.1:8080/v1", "--teacher-model", "t",
            "--adapter-method", "oft", "--oft-parameterization", "euler",
        ])
        #expect(throws: (any Error).self) { _ = try cliBad.adapter.resolveMethod(fallbackRank: 16) }

        let mcpTrain = try TrainRequest(mcp: [
            "heldOut": ["What is 2+2?"],
            "adapterMethod": "oft",
            "oftBlockSize": 8,
            "oftParameterization": "cayley",
            "oftBlockShared": true,
        ])
        #expect(mcpTrain.method == .oft(blockSize: 8, parameterization: .cayley, blockShared: true))

        let mcpDistill = try DistillRequest(mcp: [
            "prompts": "p.jsonl",
            "teacherUrl": "http://127.0.0.1:8080/v1",
            "teacherModel": "teacher",
            "heldOut": ["Explain gravity."],
            "oftBlockSize": 16,
            "oftParameterization": "matrixExp",
        ])
        #expect(mcpDistill.method == .oft(blockSize: 16, parameterization: .matrixExp, blockShared: false))
        #expect(ResearchTool.trainProperties["adapterMethod"] != nil)
        #expect(ResearchTool.distillProperties["oftParameterization"] != nil)
        #expect(throws: MCPFailure.self) {
            _ = try TrainRequest(mcp: ["heldOut": ["Hi"], "adapterMethod": "oft", "oftBlockSize": 1])
        }
    }

    @Test(
        "Equal-budget micro-benchmark: LoRA(r=2) vs OFT(b=8, cayley) vs OFT(b=8, matrixExp)",
        .enabled(if: ProcessInfo.processInfo.environment["BASALT_BENCH"] == "1")
    )
    func equalBudgetLoRAvsOFTBenchmark() throws {
        let lora = try OFTTestSupport.measureStepLatency(spec: AdapterSpec(method: .lora(rank: 2, scale: 16.0)))
        let cayley = try OFTTestSupport.measureStepLatency(
            spec: AdapterSpec(method: .oft(blockSize: 8, parameterization: .cayley, blockShared: false))
        )
        let matExp = try OFTTestSupport.measureStepLatency(
            spec: AdapterSpec(method: .oft(blockSize: 8, parameterization: .matrixExp, blockShared: false))
        )
        #expect(lora.params == 8192 && cayley.params == 7168 && matExp.params == 7168)
        #expect(lora.stepMs > 0 && cayley.stepMs > 0 && matExp.stepMs > 0)
        #expect(cayley.defect < 1e-5 && matExp.defect < 1e-5)
    }
}
