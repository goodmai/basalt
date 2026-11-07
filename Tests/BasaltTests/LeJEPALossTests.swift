import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
@testable import BasaltCore
@testable import BasaltInference

@Suite("JEPAHead & LeJEPALoss: 4-layer BatchNorm MLP, invariance MSE, and Epps-Pulley SIGReg (#385)", .serialized)
struct LeJEPALossTests {
    @Test("JEPAHead.trainableParameters() includes lora_a/lora_b and full-rank projector weights/biases, excluding frozen base weights and BatchNorm running stats")
    func jepaHeadTrainableParameters() throws {
        let backbone = try LeJEPABackboneTests.tinyQwen3Encoder()
        let baseKeys = Set(backbone.parameters().flattened().map { "backbone.\($0.0)" })
        _ = try MLXInferenceEngine.attachLoRA(to: backbone, layers: [0, 1], rank: 4)
        let head = JEPAHead(backbone: backbone, inputDim: 64, hiddenDim: 128, outputDim: 32, isCausal: false)
        eval(head)

        let allKeys = Set(head.parameters().flattened().map(\.0))
        let trainableKeys = Set(head.trainableParameters().flattened().map(\.0))

        #expect(allKeys.contains("projector.bn1.running_mean"))
        #expect(allKeys.contains("projector.bn1.running_var"))
        #expect(!trainableKeys.contains(where: { $0.contains("running_mean") || $0.contains("running_var") }))
        #expect(trainableKeys.isDisjoint(with: baseKeys))

        let loraTrainable = trainableKeys.filter { $0.hasPrefix("backbone.") }
        let projTrainable = trainableKeys.filter { $0.hasPrefix("projector.") }
        #expect(!loraTrainable.isEmpty)
        #expect(loraTrainable.allSatisfy { $0.hasSuffix(".lora_a") || $0.hasSuffix(".lora_b") })
        #expect(projTrainable == [
            "projector.fc1.weight", "projector.bn1.weight", "projector.bn1.bias",
            "projector.fc2.weight", "projector.bn2.weight", "projector.bn2.bias",
            "projector.fc3.weight", "projector.bn3.weight", "projector.bn3.bias",
            "projector.fc4.weight"
        ])
    }

    @Test("LeJEPALoss yields invariance = 0 on identical views, penalizes collapsed Z = 0 via SIGReg, and scales effectiveLambda inversely with N",
          arguments: [(16, Float(0.02)), (32, Float(0.01)), (64, Float(0.02))])
    func invarianceAndSIGRegBehavior(sampleCount: Int, baseLambda: Float) {
        let dim = 32
        let lossFn = LeJEPALoss(baseLambda: baseLambda, referenceSamples: 2048, sliceCount: 128, knotCount: 17)
        let directions = LeJEPALoss.unitDirections(projDim: dim, sliceCount: 128, seed: 7)

        let singleScene = MLXRandom.normal([1, 1, dim], dtype: .float32, key: MLXRandom.key(11))
        let identicalViews = MLX.broadcast(singleScene, to: [sampleCount / 8, 8, dim]).reshaped(sampleCount, dim)
        let identicalOut = lossFn(identicalViews, directions: directions)

        let collapsedZero = MLXArray.zeros([sampleCount, dim], type: Float.self)
        let collapsedOut = lossFn(collapsedZero, directions: directions)

        let gaussian = MLXRandom.normal([sampleCount, dim], dtype: .float32, key: MLXRandom.key(29))
        let gaussianOut = lossFn(gaussian, directions: directions)
        eval(
            identicalOut.invariance, identicalOut.sigReg,
            collapsedOut.invariance, collapsedOut.sigReg,
            gaussianOut.invariance, gaussianOut.sigReg
        )

        #expect(identicalOut.invariance.item(Float.self) < 1e-6)
        #expect(collapsedOut.sigReg.item(Float.self) > gaussianOut.sigReg.item(Float.self))
        #expect(collapsedOut.sigReg.item(Float.self) > 1.0)
        #expect(gaussianOut.invariance.item(Float.self) > 0.1)

        let expectedLambda = baseLambda * (2048.0 / Float(sampleCount))
        #expect(abs(collapsedOut.effectiveLambda - expectedLambda) < 1e-6)
    }

    @Test("LeJEPALoss produces finite, non-zero gradients through MLXNN.valueAndGrad on JEPAHead")
    func valueAndGradFlow() throws {
        let backbone = try LeJEPABackboneTests.tinyQwen3Encoder()
        _ = try MLXInferenceEngine.attachLoRA(to: backbone, layers: [0, 1], rank: 4)
        let loraB = backbone.trainableParameters().flattened().filter { $0.0.hasSuffix(".lora_b") }
        backbone.update(parameters: ModuleParameters.unflattened(loraB.map { ($0.0, MLXRandom.normal($0.1.shape, scale: 0.02, key: MLXRandom.key(5))) }))
        let head = JEPAHead(backbone: backbone, inputDim: 64, hiddenDim: 128, outputDim: 32, isCausal: false)
        head.train(true)
        eval(head)

        let lossFn = LeJEPALoss(baseLambda: 0.02, referenceSamples: 16, sliceCount: 64, knotCount: 9)
        let directions = LeJEPALoss.unitDirections(projDim: 32, sliceCount: 64, seed: 13)
        eval(directions)

        let ids = MLXRandom.randInt(low: 1, high: 200, [16, 12], key: MLXRandom.key(19)).asType(.int32)
        let mask = MLXArray.ones([16, 12], type: Int32.self)
        eval(ids, mask)

        let vg = valueAndGrad(model: head) { model, arrays -> [MLXArray] in
            let (_, projected) = model(arrays[0], attentionMask: arrays[1])
            return [lossFn(projected, directions: arrays[2]).total]
        }
        let (values, grads) = vg(head, [ids, mask, directions])
        eval(values, grads)

        let lossValue = values[0].item(Float.self)
        #expect(lossValue.isFinite && lossValue > 0)

        let flatGrads = grads.flattened()
        #expect(!flatGrads.isEmpty)
        for (key, grad) in flatGrads {
            let maxAbs = MLX.max(MLX.abs(grad)).item(Float.self)
            #expect(maxAbs.isFinite, "gradient \(key) is not finite")
            #expect(maxAbs > 0, "gradient \(key) is zero")
        }
    }
}
