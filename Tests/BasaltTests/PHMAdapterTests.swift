import Foundation
import Testing
import MLX
import MLXNN
import MLXOptimizers
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

private enum PHMTestSupport {
    static func maxAbsDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let diff = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        eval(diff)
        return diff.item(Float.self)
    }

    static func maxRelDiff(_ lhs: MLXArray, _ rhs: MLXArray) -> Float {
        let num = abs(lhs.asType(.float32) - rhs.asType(.float32)).max()
        let den = maximum(abs(rhs.asType(.float32)).max(), MLXArray(Float(1.0)))
        let rel = num / den
        eval(rel)
        return rel.item(Float.self)
    }

    static func explicitKronDelta(basis: MLXArray, factorS: MLXArray, factorT: MLXArray) -> MLXArray {
        let n = basis.dim(0)
        let outDims = n * factorS.dim(1)
        let inDims = n * factorT.dim(1)
        var acc = MLXArray.zeros([outDims, inDims])
        for index in 0 ..< n {
            let bSlice = matmul(factorS[index], factorT[index].T)
            acc = acc + kron(basis[index], bSlice)
        }
        eval(acc)
        return acc
    }

    static func runAdamWSteps(
        model: any LanguageModel,
        inputs: MLXArray,
        targets: MLXArray,
        steps: Int,
        learningRate: Float = 1e-2
    ) -> [Float] {
        let optimizer = AdamW(learningRate: learningRate, weightDecay: 0.01)
        let module = model as Module
        let gradFn = valueAndGrad(model: module) { _, arrays in
            let logits = model(arrays[0], cache: nil).asType(.float32)
            return [crossEntropy(logits: logits, targets: arrays[1], reduction: .mean)]
        }
        var losses: [Float] = []
        for _ in 0 ..< steps {
            let (values, grads) = gradFn(module, [inputs, targets])
            optimizer.update(model: module, gradients: grads)
            eval(module, optimizer, values)
            losses.append(values[0].item(Float.self))
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

    static func measureTrainingLatency(
        spec: AdapterSpec,
        warmup: Int = 2,
        iterations: Int = 5
    ) throws -> (params: Int, forwardMs: Double, stepMs: Double) {
        let model = try TinyModel.make(.qwen3)
        _ = try PHMAdapterApplicator.apply(to: model, spec: spec)
        let params = PHMAdapterApplicator.parameterCount(in: model)
        let inputs = MLXArray([Int32(2), 5, 9, 14, 21, 27, 33, 42], [1, 8])
        let targets = MLXArray([Int32(5), 9, 14, 21, 27, 33, 42, 51], [1, 8])
        for _ in 0 ..< warmup {
            let out = model(inputs, cache: nil)
            eval(out)
        }
        let clock = ContinuousClock()
        let fwdStart = clock.now
        for _ in 0 ..< iterations {
            let out = model(inputs, cache: nil)
            eval(out)
        }
        let fwdElapsed = fwdStart.duration(to: clock.now)
        let optimizer = AdamW(learningRate: 5e-3)
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
        let fwdMs = Double(fwdElapsed.components.seconds) * 1000.0
            + Double(fwdElapsed.components.attoseconds) / 1e15
        let stepMs = Double(stepElapsed.components.seconds) * 1000.0
            + Double(stepElapsed.components.attoseconds) / 1e15
        return (params, fwdMs / Double(iterations), stepMs / Double(iterations))
    }

    static func verifySaveReloadAndFuse(
        model: Module,
        adaptedLogits: MLXArray,
        inputs: MLXArray
    ) throws {
        let scratch = FileManager.default.temporaryDirectory.appending(path: "phm-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: scratch) }
        let config = PHMAdapterConfiguration(
            numLayers: 4,
            phmParameters: .init(hypercomplexDim: 8, factorRank: 2, sharedBasis: true, scale: 16.0)
        )
        try PHMAdapterContainer.save(configuration: config, model: model, to: scratch)
        let container = try PHMAdapterContainer.from(directory: scratch)
        let reloaded = try TinyModel.make(.qwen3)
        try reloaded.load(adapter: container)
        let reloadedLogits = reloaded(inputs, cache: nil).asType(.float32)
        eval(reloadedLogits)
        #expect(maxAbsDiff(reloadedLogits, adaptedLogits) < 1e-5)

        _ = PHMAdapterApplicator.fuse(in: reloaded)
        #expect(!reloaded.namedModules().contains { $0.1 is LoRALayer })
        let fusedLogits = reloaded(inputs, cache: nil).asType(.float32)
        eval(fusedLogits)
        #expect(maxAbsDiff(fusedLogits, adaptedLogits) < 1e-4)
    }
}

@Suite("PHM and Compacter hypercomplex adapters: Kronecker contraction identity, parameter budgets, TinyModel training, fusion, and revert (#451)", .serialized)
struct PHMAdapterTests {
    @Test("Reshaped Kronecker forward pass matches explicit kron sum within 1e-5 across n in {4, 8, 16}", arguments: [4, 8, 16])
    func kroneckerAlgebraicIdentity(hypercomplexDim: Int) throws {
        try withRandomState(MLXRandom.RandomState(seed: UInt64(451 + hypercomplexDim))) {
            let inputDim = 64
            let outputDim = 48
            let factorRank = 2
            let scale: Float = 3.5
            let base = Linear(inputDim, outputDim, bias: true)
            let phm = try PHMLinear(
                linear: base,
                hypercomplexDim: hypercomplexDim,
                factorRank: factorRank,
                scale: scale
            )
            let a = MLXRandom.normal([hypercomplexDim, hypercomplexDim, hypercomplexDim])
            let s = MLXRandom.normal([hypercomplexDim, outputDim / hypercomplexDim, factorRank])
            let t = MLXRandom.normal([hypercomplexDim, inputDim / hypercomplexDim, factorRank])
            phm.update(parameters: .unflattened(["phm_a": a, "phm_s": s, "phm_t": t]))
            eval(base, phm)
            let explicitDelta = PHMTestSupport.explicitKronDelta(basis: a, factorS: s, factorT: t)
            let materialized = phm.materializeDeltaWeight()
            eval(materialized)
            #expect(materialized.shape == [outputDim, inputDim])
            #expect(PHMTestSupport.maxAbsDiff(materialized, explicitDelta) < 1e-5)
            for shape in [[1, 1, inputDim], [2, 5, inputDim]] {
                let x = MLXRandom.normal(shape)
                let actual = phm(x)
                let expected = base(x) + scale * matmul(x, explicitDelta.T)
                eval(actual, expected)
                #expect(actual.shape == Array(shape.dropLast()) + [outputDim])
                #expect(PHMTestSupport.maxRelDiff(actual, expected) < 1e-5)
                #expect(PHMTestSupport.maxAbsDiff(actual, expected) < 5e-4)
                let bf16Out = phm(x.asType(.bfloat16))
                eval(bf16Out)
                #expect(bf16Out.shape == actual.shape)
            }
        }
    }

    @Test("AdapterMethod.parameterCount matches trainableParameters element count to the exact integer")
    func exactParameterCountFormula() throws {
        let methods: [AdapterMethod] = [
            .lora(rank: 2, scale: 16.0),
            .phm(hypercomplexDim: 8, factorRank: 2, sharedBasis: false, scale: 16.0),
            .phm(hypercomplexDim: 8, factorRank: 2, sharedBasis: true, scale: 16.0),
            .phm(hypercomplexDim: 16, factorRank: 1, sharedBasis: true, scale: 16.0),
        ]
        for method in methods {
            let fullModel = try TinyModel.make(.qwen3)
            let fullSpec = AdapterSpec(method: method)
            let dims = try PHMAdapterApplicator.projectionDims(in: fullModel, spec: fullSpec)
            _ = try PHMAdapterApplicator.apply(to: fullModel, spec: fullSpec)
            let actualFull = PHMAdapterApplicator.parameterCount(in: fullModel)
            #expect(actualFull == method.parameterCount(projections: dims))

            let uniformModel = try TinyModel.make(.qwen3)
            let uniformSpec = AdapterSpec(method: method, targetSuffixes: ["gate_proj"])
            _ = try PHMAdapterApplicator.apply(to: uniformModel, spec: uniformSpec)
            let actualUniform = PHMAdapterApplicator.parameterCount(in: uniformModel)
            let analyticalUniform = method.parameterCount(inputDim: 64, outputDim: 128, layerCount: 4)
            #expect(actualUniform == analyticalUniform)
        }
        let lora2 = AdapterMethod.lora(rank: 2, scale: 20.0)
        let matched = try lora2.matchingPHM(hypercomplexDim: 8, sharedBasis: true, inputDim: 64, outputDim: 64, layerCount: 28)
        #expect(matched == .phm(hypercomplexDim: 8, factorRank: 2, sharedBasis: true, scale: 20.0))
    }

    @Test("TinyModel Qwen3 end-to-end: step-0 bit identity, 3-step AdamW monotonic descent, fusion, safetensors reload, and bit-identical revert")
    func tinyQwen3EndToEndLifecycle() throws {
        let model = try TinyModel.make(.qwen3)
        let inputs = MLXArray([Int32(3), 7, 11, 19, 23, 31], [1, 6])
        let targets = MLXArray([Int32(7), 11, 19, 23, 31, 41], [1, 6])
        let baseLogits = model(inputs, cache: nil).asType(.float32)
        eval(baseLogits)
        let spec = AdapterSpec(method: .phm(hypercomplexDim: 8, factorRank: 2, sharedBasis: true, scale: 16.0))
        let appliedPaths = try withRandomState(MLXRandom.RandomState(seed: 4510)) {
            try PHMAdapterApplicator.apply(to: model, spec: spec)
        }
        #expect(appliedPaths.count == 28)
        let step0Logits = model(inputs, cache: nil).asType(.float32)
        eval(step0Logits)
        #expect(PHMTestSupport.maxAbsDiff(step0Logits, baseLogits) == 0.0)
        let initialBasis = try #require(model.trainableParameters().flattened().first { $0.0.hasSuffix(".phm_a") }).1 + 0
        eval(initialBasis)

        let losses = PHMTestSupport.runAdamWSteps(model: model, inputs: inputs, targets: targets, steps: 3, learningRate: 1e-2)
        #expect(losses.count == 4)
        #expect(losses[1] < losses[0])
        #expect(losses[2] < losses[1])
        #expect(losses[3] < losses[2])
        let updatedBasis = try #require(model.trainableParameters().flattened().first { $0.0.hasSuffix(".phm_a") }).1
        #expect(PHMTestSupport.maxAbsDiff(updatedBasis, initialBasis) > 1e-6)
        let adaptedLogits = model(inputs, cache: nil).asType(.float32)
        eval(adaptedLogits)
        try PHMTestSupport.verifySaveReloadAndFuse(model: model, adaptedLogits: adaptedLogits, inputs: inputs)

        _ = PHMAdapterApplicator.revert(in: model)
        #expect(!model.namedModules().contains { $0.1 is LoRALayer })
        let revertedLogits = model(inputs, cache: nil).asType(.float32)
        eval(revertedLogits)
        #expect(PHMTestSupport.maxAbsDiff(revertedLogits, baseLogits) == 0.0)

        let sparseSpec = AdapterSpec(method: .phm(hypercomplexDim: 8, factorRank: 1, sharedBasis: true, scale: 8.0), layers: [1, 3])
        let sparsePaths = try PHMAdapterApplicator.apply(to: model, spec: sparseSpec)
        #expect(sparsePaths.count == 14)
        #expect(PHMAdapterApplicator.revert(in: model).count == 14)
        let sparseReverted = model(inputs, cache: nil).asType(.float32)
        eval(sparseReverted)
        #expect(PHMTestSupport.maxAbsDiff(sparseReverted, baseLogits) == 0.0)
    }

    @Test("Quantized TinyModel Qwen3: QuantizedPHMLinear step-0 identity, training, requantized fusion, and lossless revert")
    func quantizedTinyQwen3Lifecycle() throws {
        let model = try TinyModel.make(.qwen3Small)
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        let inputs = MLXArray([Int32(4), 8, 15, 16, 23, 42], [1, 6])
        let targets = MLXArray([Int32(8), 15, 16, 23, 42, 7], [1, 6])
        let baseLogits = model(inputs, cache: nil).asType(.float32)
        eval(baseLogits)
        let spec = AdapterSpec(method: .phm(hypercomplexDim: 8, factorRank: 2, sharedBasis: true, scale: 8.0))
        _ = try withRandomState(MLXRandom.RandomState(seed: 4511)) {
            try PHMAdapterApplicator.apply(to: model, spec: spec)
        }
        #expect(model.namedModules().contains { $0.1 is QuantizedPHMLinear })
        let step0Logits = model(inputs, cache: nil).asType(.float32)
        eval(step0Logits)
        #expect(PHMTestSupport.maxAbsDiff(step0Logits, baseLogits) == 0.0)

        let losses = PHMTestSupport.runAdamWSteps(model: model, inputs: inputs, targets: targets, steps: 3, learningRate: 1e-2)
        #expect(losses[3] < losses[0])
        let adaptedLogits = model(inputs, cache: nil).asType(.float32)
        eval(adaptedLogits)

        let singleBase = QuantizedLinear(Linear(64, 64, bias: false), groupSize: 32, bits: 4)
        let singlePHM = try QuantizedPHMLinear(linear: singleBase, hypercomplexDim: 8, factorRank: 2, scale: 2.0)
        singlePHM.update(parameters: .unflattened(["phm_s": MLXArray.ones([8, 8, 2]) * 0.01]))
        let probe = MLXRandom.normal([2, 64])
        let unfusedOut = singlePHM(probe)
        let fusedLayer = try #require(singlePHM.fused() as? QuantizedLinear)
        let fusedOut = fusedLayer(probe)
        eval(unfusedOut, fusedOut)
        #expect(PHMTestSupport.maxAbsDiff(unfusedOut, fusedOut) < 0.5)

        _ = PHMAdapterApplicator.revert(in: model)
        let revertedLogits = model(inputs, cache: nil).asType(.float32)
        eval(revertedLogits)
        #expect(PHMTestSupport.maxAbsDiff(revertedLogits, baseLogits) == 0.0)
    }

    @Test("Divisibility and hyperparameter validation throws typed BasaltError without mutating target model")
    func validationRejectsInvalidShapesCleanly() throws {
        #expect(throws: BasaltError.self) {
            _ = try PHMLinear(linear: Linear(60, 64, bias: false), hypercomplexDim: 8, factorRank: 1)
        }
        #expect(throws: BasaltError.self) {
            _ = try PHMLinear(linear: Linear(64, 50, bias: false), hypercomplexDim: 8, factorRank: 1)
        }
        #expect(throws: BasaltError.self) {
            try AdapterMethod.phm(hypercomplexDim: 0, factorRank: 1).validate()
        }
        #expect(throws: BasaltError.self) {
            try AdapterMethod.phm(hypercomplexDim: 8, factorRank: 0).validate()
        }
        let model = try TinyModel.make(.qwen3)
        #expect(throws: BasaltError.self) {
            _ = try PHMAdapterApplicator.apply(
                to: model,
                spec: AdapterSpec(method: .phm(hypercomplexDim: 12, factorRank: 1))
            )
        }
        #expect(!model.namedModules().contains { $0.1 is LoRALayer })
    }

    @Test(
        "Equal-parameter micro-benchmark: LoRA(r=2) vs Compacter PHM(n=8, factorRank=2, sharedBasis=true)",
        .enabled(if: ProcessInfo.processInfo.environment["BASALT_BENCH"] == "1")
    )
    func equalParameterMicroBenchmark() throws {
        let loraMetrics = try PHMTestSupport.measureTrainingLatency(
            spec: AdapterSpec(method: .lora(rank: 2, scale: 16.0))
        )
        let phmMetrics = try PHMTestSupport.measureTrainingLatency(
            spec: AdapterSpec(method: .phm(hypercomplexDim: 8, factorRank: 2, sharedBasis: true, scale: 16.0))
        )
        let phmCompact = try PHMTestSupport.measureTrainingLatency(
            spec: AdapterSpec(method: .phm(hypercomplexDim: 8, factorRank: 1, sharedBasis: true, scale: 16.0))
        )
        print(String(format: "[PHM_BENCH] lora(r=2): params=%d fwd=%.3fms step=%.3fms | phm(n=8,r=2,shared): params=%d fwd=%.3fms step=%.3fms | phm(n=8,r=1,shared): params=%d fwd=%.3fms step=%.3fms", loraMetrics.params, loraMetrics.forwardMs, loraMetrics.stepMs, phmMetrics.params, phmMetrics.forwardMs, phmMetrics.stepMs, phmCompact.params, phmCompact.forwardMs, phmCompact.stepMs))
        #expect(loraMetrics.params == 8192)
        #expect(phmMetrics.params == 8704)
        #expect(phmCompact.params == 4608)
        #expect(loraMetrics.forwardMs > 0 && phmMetrics.forwardMs > 0)
        #expect(loraMetrics.stepMs > 0 && phmMetrics.stepMs > 0)
    }
}
