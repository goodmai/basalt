import Foundation
import MLX
import MLXLMCommon
import MLXNN

package final class DSALightningIndexer: Module {
    package struct Config: Codable, Sendable, Equatable {
        package var indexHeads: Int
        package var indexHeadDim: Int
        package var topK: Int
        package var ropeHeadDim: Int
        package var ropeTheta: Float
        package var ropeInterleave: Bool
        package var rmsNormEps: Float
        package var useMetal: Bool

        package init(
            indexHeads: Int,
            indexHeadDim: Int,
            topK: Int,
            ropeHeadDim: Int = 0,
            useMetal: Bool = true
        ) {
            self.indexHeads = max(1, indexHeads)
            self.indexHeadDim = max(1, indexHeadDim)
            self.topK = max(1, topK)
            self.ropeHeadDim = max(0, ropeHeadDim)
            self.ropeTheta = 10_000.0
            self.ropeInterleave = false
            self.rmsNormEps = 1e-6
            self.useMetal = useMetal
        }
    }

    package let config: Config
    package let weightScale: Float
    let rope: (any RoPELayer)?

    @ModuleInfo(key: "wq_b") package var wqB: Linear
    @ModuleInfo(key: "wk") package var wk: Linear
    @ModuleInfo(key: "k_norm") package var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") package var weightsProj: Linear

    package init(hiddenSize: Int, qInputDim: Int? = nil, config: Config) {
        self.config = config
        self.weightScale = 1.0 / Float(config.indexHeadDim).squareRoot()
        let qDim = qInputDim ?? hiddenSize
        self._wqB.wrappedValue = Linear(qDim, config.indexHeads * config.indexHeadDim, bias: false)
        self._wk.wrappedValue = Linear(hiddenSize, config.indexHeadDim, bias: false)
        self._kNorm.wrappedValue = LayerNorm(dimensions: config.indexHeadDim, eps: config.rmsNormEps)
        self._weightsProj.wrappedValue = Linear(hiddenSize, config.indexHeads, bias: false)
        if config.ropeHeadDim > 0 {
            self.rope = initializeRope(
                dims: config.ropeHeadDim,
                base: config.ropeTheta,
                traditional: config.ropeInterleave,
                scalingConfig: nil,
                maxPositionEmbeddings: 131_072)
        } else {
            self.rope = nil
        }
    }

    package func selectTopK(
        x: MLXArray,
        qInput: MLXArray? = nil,
        cache: DSALatentKVCache? = nil,
        causal: Bool = true,
        qOffset: Int? = nil
    ) -> (indices: MLXArray, scores: MLXArray) {
        let batch = x.dim(0)
        let length = x.dim(1)
        let priorLen = qOffset ?? (cache?.cachedKIndex?.dim(1) ?? 0)
        let ropeOffset: RoPEOffset? = priorLen > 0 ? .scalar(priorLen) : nil
        let (qIndex, kIndex, weights) = projectInputs(
            x: x, qInput: qInput, batch: batch, length: length, offset: ropeOffset)
        let allK = cache?.updateIndexerKeys(kIndex) ?? kIndex
        let scores = DSALightningIndexerMetal.score(
            q: qIndex,
            k: allK,
            weights: weights,
            options: DSALightningIndexerMetal.ScoreOptions(
                causal: causal, qOffset: priorLen, useMetal: config.useMetal && !training))
        let indices = DSALightningIndexerMetal.selectTopK(
            scores: scores, topK: config.topK, causal: causal, qOffset: priorLen)
        return (indices, scores)
    }

    package func projectInputs(
        x: MLXArray,
        qInput: MLXArray? = nil,
        batch: Int,
        length: Int,
        offset: RoPEOffset?
    ) -> (q: MLXArray, k: MLXArray, weights: MLXArray) {
        let rawQ = wqB(qInput ?? x).reshaped(batch, length, config.indexHeads, config.indexHeadDim)
        let rawK = kNorm(wk(x))
        let (qRot, kRot) = applyIndexerRoPE(q: rawQ, k: rawK, offset: offset)
        let weights = weightsProj(x).asType(.float32) * weightScale
        return (qRot, kRot, weights)
    }

    private func applyIndexerRoPE(
        q: MLXArray,
        k: MLXArray,
        offset: RoPEOffset?
    ) -> (MLXArray, MLXArray) {
        guard let rope, config.ropeHeadDim > 0 else { return (q, k) }
        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.expandedDimensions(axis: 1)
        if config.ropeHeadDim >= config.indexHeadDim {
            let qR = applyRotaryPosition(rope, to: qT, offset: offset).transposed(0, 2, 1, 3)
            let kR = applyRotaryPosition(rope, to: kT, offset: offset).squeezed(axis: 1)
            return (qR, kR)
        }
        let qParts = split(qT, indices: [config.ropeHeadDim], axis: -1)
        let kParts = split(kT, indices: [config.ropeHeadDim], axis: -1)
        let qR = concatenated([applyRotaryPosition(rope, to: qParts[0], offset: offset), qParts[1]], axis: -1)
            .transposed(0, 2, 1, 3)
        let kR = concatenated([applyRotaryPosition(rope, to: kParts[0], offset: offset), kParts[1]], axis: -1)
            .squeezed(axis: 1)
        return (qR, kR)
    }
}
