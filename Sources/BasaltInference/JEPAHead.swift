import Foundation
import MLX
import MLXNN
import MLXEmbedders
import BasaltCore

final class JEPAProjector: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "bn2") var bn2: BatchNorm
    @ModuleInfo(key: "fc3") var fc3: Linear
    @ModuleInfo(key: "bn3") var bn3: BatchNorm
    @ModuleInfo(key: "fc4") var fc4: Linear

    let inputDim: Int
    let hiddenDim: Int
    let outputDim: Int

    init(inputDim: Int, hiddenDim: Int = 2048, outputDim: Int = 512) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.outputDim = outputDim
        _fc1.wrappedValue = Linear(inputDim, outputDim, bias: false)
        _bn1.wrappedValue = BatchNorm(featureCount: outputDim)
        _fc2.wrappedValue = Linear(outputDim, hiddenDim, bias: false)
        _bn2.wrappedValue = BatchNorm(featureCount: hiddenDim)
        _fc3.wrappedValue = Linear(hiddenDim, hiddenDim, bias: false)
        _bn3.wrappedValue = BatchNorm(featureCount: hiddenDim)
        _fc4.wrappedValue = Linear(hiddenDim, outputDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = gelu(bn1(fc1(x.asType(.float32))))
        h = gelu(bn2(fc2(h)))
        h = gelu(bn3(fc3(h)))
        return fc4(h)
    }
}

final class JEPAHead: Module {
    @ModuleInfo(key: "backbone") var backbone: Module
    @ModuleInfo(key: "projector") var projector: JEPAProjector
    let isCausal: Bool
    let poolingStrategy: Pooling.Strategy

    init(
        backbone: Module,
        inputDim: Int,
        hiddenDim: Int = 2048,
        outputDim: Int = 512,
        isCausal: Bool = false,
        poolingStrategy: Pooling.Strategy = .mean
    ) {
        _backbone.wrappedValue = backbone
        _projector.wrappedValue = JEPAProjector(inputDim: inputDim, hiddenDim: hiddenDim, outputDim: outputDim)
        self.isCausal = isCausal
        self.poolingStrategy = poolingStrategy
        super.init()
    }

    func encode(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let output = LeJEPABackbone.encode(
            model: backbone, inputIds: inputIds, attentionMask: attentionMask, isCausal: isCausal)
        return LeJEPABackbone.pool(output, mask: attentionMask, strategy: poolingStrategy)
    }

    func callAsFunction(
        _ inputIds: MLXArray, attentionMask: MLXArray? = nil
    ) -> (embeddings: MLXArray, projected: MLXArray) {
        let embeddings = encode(inputIds, attentionMask: attentionMask)
        let projected = projector(embeddings)
        return (embeddings, projected)
    }
}
