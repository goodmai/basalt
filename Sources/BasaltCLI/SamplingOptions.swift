import ArgumentParser
import BasaltInference

struct SamplingOptions: ParsableArguments {
    @Option(
        name: .customLong("reasoning-effort"),
        help: "Chat-template reasoning budget (Qwen3.8: xhigh | medium | low), or `none` to disable thinking entirely"
    )
    var reasoningEffort: String?

    @Option(name: .customLong("top-k"), help: "Keep only the k most likely tokens (unset disables)")
    var topK: Int?

    @Option(name: .customLong("min-p"), help: "Drop tokens below this fraction of the top token's probability")
    var minP: Float?

    @Option(name: .customLong("seed"), help: "Seed the sampler so a non-greedy run is reproducible")
    var seed: UInt64?

    @Option(name: .customLong("repetition-penalty"), help: "Penalty on recently emitted tokens, e.g. 1.1 (default: off)")
    var repetitionPenalty: Float?

    @Flag(
        name: .customLong("experimental-samplers"),
        help: "Accept the experimental dry_* (DRY repetition penalty) and xtc_* (exclude top choices) request fields. Measured on a 2-bit Qwen3.8, DRY turned a verbatim loop that RepetitionGuard stops into a loop with corrupted numbers that runs to max_tokens; on models that do not loop it changed nothing. Without this flag those fields are refused with 400"
    )
    var experimentalSamplers = false

    func configure(_ configuration: inout EngineConfiguration) {
        configuration.reasoningEffort = reasoningEffort
        configuration.repetitionPenalty = repetitionPenalty
        (configuration.topK, configuration.minP, configuration.seed) = (topK, minP, seed)
        configuration.experimentalSamplers = experimentalSamplers
    }
}
