import BasaltCore
import Foundation
import MLX
import MLXNN

package struct EAGLE3LayerTapConfig: Sendable, Equatable {
    package var lowLayer: Int
    package var midLayer: Int
    package var highLayer: Int

    package init(lowLayer: Int, midLayer: Int, highLayer: Int) {
        self.lowLayer = max(0, lowLayer)
        self.midLayer = max(0, midLayer)
        self.highLayer = max(0, highLayer)
    }

    package static func standard(numLayers: Int) -> EAGLE3LayerTapConfig {
        let n = max(3, numLayers)
        return EAGLE3LayerTapConfig(
            lowLayer: min(2, n - 3),
            midLayer: n / 2,
            highLayer: n - 1
        )
    }
}

package final class EAGLE3FusionHead: Module {
    @ModuleInfo(key: "fc_fuse") package var fcFuse: Linear
    @ModuleInfo(key: "norm") package var norm: RMSNorm
    @ModuleInfo(key: "gate_proj") package var gateProj: Linear
    @ModuleInfo(key: "up_proj") package var upProj: Linear
    @ModuleInfo(key: "down_proj") package var downProj: Linear
    @ModuleInfo(key: "lm_head") package var lmHead: Linear
    package let hiddenSize: Int
    package let vocabSize: Int
    package let tapConfig: EAGLE3LayerTapConfig

    package init(
        hiddenSize: Int,
        intermediateSize: Int? = nil,
        vocabSize: Int,
        tapConfig: EAGLE3LayerTapConfig = .standard(numLayers: 24)
    ) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.tapConfig = tapConfig
        let inter = intermediateSize ?? (hiddenSize * 2)
        self._fcFuse.wrappedValue = Linear(hiddenSize * 3, hiddenSize, bias: false)
        self._norm.wrappedValue = RMSNorm(dimensions: hiddenSize)
        self._gateProj.wrappedValue = Linear(hiddenSize, inter, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, inter, bias: false)
        self._downProj.wrappedValue = Linear(inter, hiddenSize, bias: false)
        self._lmHead.wrappedValue = Linear(hiddenSize, vocabSize, bias: false)
        super.init()
    }

    package func fuseFeatures(low: MLXArray, mid: MLXArray, high: MLXArray) -> MLXArray {
        let concat = concatenated([low, mid, high], axis: -1)
        let fused = fcFuse(concat)
        let normed = norm(fused)
        let mlp = downProj(silu(gateProj(normed)) * upProj(normed))
        return fused + mlp
    }

    package func draftLogits(low: MLXArray, mid: MLXArray, high: MLXArray) -> MLXArray {
        lmHead(fuseFeatures(low: low, mid: mid, high: high))
    }

    package func topKCandidates(
        low: MLXArray,
        mid: MLXArray,
        high: MLXArray,
        k: Int
    ) -> (tokens: MLXArray, logits: MLXArray) {
        let logits = draftLogits(low: low, mid: mid, high: high)
        let clampedK = min(max(1, k), vocabSize)
        let topIdx = argSort(-logits, axis: -1)[.ellipsis, 0 ..< clampedK].asType(.int32)
        return (topIdx, logits)
    }

    package static func trimRollbackCount(draftedSteps: Int, acceptedSteps: Int) -> Int {
        max(0, draftedSteps - max(0, acceptedSteps))
    }
}

package struct DiffuCDScheduler: Sendable, Equatable {
    package var alphaMax: Float
    package var gamma: Float
    package var minTimestep: Float

    package init(alphaMax: Float = 0.6, gamma: Float = 1.5, minTimestep: Float = 0.15) {
        self.alphaMax = max(0, alphaMax)
        self.gamma = max(0.1, gamma)
        self.minTimestep = min(max(minTimestep, 0), 1)
    }

    package func scale(at normalizedTimestep: Float) -> Float {
        let t = min(max(normalizedTimestep, 0), 1)
        guard t >= minTimestep else { return 0 }
        return alphaMax * powf(t, gamma)
    }

    package func extrapolate(
        strongPrediction: MLXArray,
        weakPrediction: MLXArray,
        normalizedTimestep: Float
    ) -> MLXArray {
        let alpha = scale(at: normalizedTimestep)
        guard alpha > 0 else { return strongPrediction }
        let delta = strongPrediction.asType(.float32) - weakPrediction.asType(.float32)
        return (strongPrediction.asType(.float32) + alpha * delta).asType(strongPrediction.dtype)
    }
}
