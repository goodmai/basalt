import MLX
import MLXLMCommon

struct ExcludeTopChoices: LogitProcessor {
    static let defaultThreshold: Float = 0.1

    let threshold: Float
    let probability: Float
    let randomState: MLXRandom.RandomState

    init(threshold: Float = defaultThreshold, probability: Float, seed: UInt64?) {
        self.threshold = threshold
        self.probability = probability
        randomState = seed.map { MLXRandom.RandomState(seed: $0 &+ 1) } ?? MLXRandom.RandomState()
    }

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        let probabilities = softmax(logits.asType(.float32), axis: -1)
        let top = probabilities .> threshold
        let leastOfTop = MLX.where(top, probabilities, MLXArray(Float.infinity)).min(axis: -1, keepDims: true)
        let draw = MLXRandom.uniform(0 ..< 1, [1], key: randomState) .< probability
        let excluded = top .&& (probabilities .> leastOfTop) .&& draw
        return MLX.where(excluded, MLXArray(-Float.infinity).asType(logits.dtype), logits)
    }

    mutating func didSample(token: MLXArray) {}
}
