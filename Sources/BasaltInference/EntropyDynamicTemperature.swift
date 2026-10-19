import MLX
import MLXLMCommon

// MARK: — Entropy-based dynamic temperature (EDT)
//
// A static temperature is wrong in both directions at once: too hot on a syntactic
// token the model is sure of (a bracket, a recalled constant), too cold once it has
// slid into a loop. EDT (Zhang et al., arXiv:2403.14541) picks the temperature per
// step from the entropy of the next-token distribution:
//
//     T_t = T0 · N^(θ / H_t)  +  γ_rep · R_t,   clamped to `range`
//
// with N ∈ (0, 1). H_t → 0 sends N^(θ/H) → 0 (near-greedy); a flat distribution sends
// it towards 1 (T → T0). This is the paper's form. Issue #40 wrote N^((θ/ln N)·H − 1),
// which is not what the paper defines: at H = 0 it gives T0 / N = 1.25·T0 instead of
// ~0, so confident tokens are never near-greedy, and it grows as e^(θH)·T0/N without
// bound (5.6·T0 at H = 1 nat). γ_rep · R_t is the issue's addition: R_t is
// RepetitionGuard's cycle score, so a loop as it forms warms the sampler up.
//
// It plugs into mlx-swift-lm as a LogitProcessor (GenerationComponents), so the
// stock TokenIterator, penalty processors and sampler stay as they are. The stock
// sampler divides by T0, so the processor scales the logits by T0 / T_t and the
// effective temperature becomes T_t. Min-P then runs in the sampler on the rescaled
// distribution. Everything per step is MLX ops on the logits — no CPU pass over the vocab.

public struct EntropyDynamicTemperature: LogitProcessor {
    /// EDT's base N and exponent θ, the values issue #40 asks for; γ_rep is ours.
    public static let defaultN: Float = 0.8
    public static let defaultTheta: Float = 1.5
    public static let defaultGammaRep: Float = 0.5
    /// Floor keeps T_t > 0 (H = 0 would otherwise divide by zero); ceiling is the API's
    /// own temperature limit.
    public static let range: ClosedRange<Float> = 0.05...2.0

    /// R_t over generated tokens: probe and threshold in tokens, mirroring
    /// RepetitionGuard's 64 characters / 4 occurrences.
    static let probeTokens = 16
    static let threshold = 4
    static let minTokens = 64
    static let retainTokens = 512

    let baseTemperature: Float
    let n: Float
    let theta: Float
    let gammaRep: Float

    /// Generated tokens already read back to the CPU, for R_t.
    private(set) var history: [Int] = []
    /// Tokens not yet read back. Reading the newest token would force the step that is still
    /// running on the GPU and break TokenIterator's asyncEval pipelining; two steps back the
    /// iterator has already read it, so `item()` is free. R_t therefore lags by two tokens.
    private var pending: [MLXArray] = []
    private(set) var repetition: Float = 0

    public init(
        baseTemperature: Float,
        n: Float = defaultN, theta: Float = defaultTheta, gammaRep: Float = defaultGammaRep
    ) {
        self.baseTemperature = baseTemperature
        self.n = n
        self.theta = theta
        self.gammaRep = gammaRep
    }

    /// Shannon entropy (nats) of softmax(logits) along the last axis, kept as a [.., 1] array.
    static func entropy(_ logits: MLXArray) -> MLXArray {
        let z = logits.asType(.float32)
        let logp = z - logSumExp(z, axis: -1, keepDims: true)
        let p = exp(logp)
        // -inf logits (masked tokens) give 0 · -inf = NaN; they contribute 0.
        return -sum(MLX.where(p .> 0, p * logp, MLXArray(Float(0))), axis: -1, keepDims: true)
    }

    /// T_t for a given entropy and repetition index.
    func temperature(entropy: MLXArray, repetition: Float) -> MLXArray {
        let t = baseTemperature * pow(MLXArray(n), theta / entropy) + gammaRep * repetition
        return clip(t, min: Self.range.lowerBound, max: Self.range.upperBound)
    }

    public mutating func prompt(_ prompt: MLXArray) {
        // The prompt is not part of the generation's own loop; R_t starts from zero.
    }

    public func process(logits: MLXArray) -> MLXArray {
        let t = temperature(entropy: Self.entropy(logits), repetition: repetition)
        return logits * (baseTemperature / t)
    }

    public mutating func didSample(token: MLXArray) {
        pending.append(token)
        while pending.count > 2 {
            history.append(pending.removeFirst().item(Int.self))
        }
        if history.count > Self.retainTokens {
            history.removeFirst(history.count - Self.retainTokens)
        }
        repetition = history.count < Self.minTokens ? 0 : Float(
            RepetitionGuard.score(tokens: history, probeSize: Self.probeTokens, threshold: Self.threshold))
    }
}
