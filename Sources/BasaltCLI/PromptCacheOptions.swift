import ArgumentParser
import BasaltInference

struct PromptCacheOptions: ParsableArguments {
    @Option(
        name: .customLong("prompt-cache-mb"),
        help: "Memory for the prompt-cache prefix tree shared by all requests, in MB (default: half of what Metal's recommended working set leaves after the weights; 0 disables)"
    )
    var promptCacheMB: Int?

    @Option(
        name: .customLong("hybrid-anchors"),
        help: "Recurrent-state snapshots kept per conversation at message boundaries (assistant turn starts, tool result and reasoning ends, at least 512 tokens apart) on hybrid models such as Qwen3.5, Nemotron-H or LFM2, so a prompt edited mid-history resumes from the last one before the edit; each costs one recurrent state from the prompt-cache budget (default: 8; 0 disables)"
    )
    var hybridAnchors = 8

    @Flag(
        name: .customLong("cache-blend"),
        help: "Reuse the KV of prompt chunks (split after blank lines) in any order, recomputing only the most deviating tokens (CacheBlend, approximate; Llama and Qwen3 only; takes half the prompt-cache budget)"
    )
    var cacheBlend = false

    @Option(
        name: .customLong("cache-blend-ratio"),
        help: "Share of reused chunk tokens recomputed by --cache-blend, 0...1 (default: 0.3; 0 is fastest, 1 ≈ a full prefill up to numerical noise)"
    )
    var cacheBlendRatio: Float = 0.3

    func configure(_ configuration: inout EngineConfiguration) {
        configuration.promptCacheBytes = promptCacheMB.map { min(max(0, $0), 1 << 30) << 20 }
        configuration.cacheBlendRatio = cacheBlend ? min(max(0, cacheBlendRatio), 1) : nil
        configuration.hybridAnchors = max(0, hybridAnchors)
    }
}
