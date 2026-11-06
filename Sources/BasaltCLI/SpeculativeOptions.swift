import ArgumentParser
import BasaltInference

struct SpeculativeOptions: ParsableArguments {
    @Flag(
        name: .customLong("speculative-lookup"),
        help: "Draft up to 16 tokens per step by matching the last 3-5 tokens against the prompt and output, verified in one forward pass (faster when the answer copies its context, slower on free prose; greedy output can differ at floating-point near-ties; attention-only models and dense recurrent hybrids such as Qwen3.5 or Nemotron-H, without --kv-bits)"
    )
    var speculativeLookup = false

    @Flag(
        name: .customLong("speculative-recycling"),
        help: "With speculative lookup on, draft up to 6 tokens from the model's own argmax successors (Token Recycling) when no n-gram matches; measured neutral to 1.2x over lookup alone on reasoning-style text, no gain on prose"
    )
    var speculativeRecycling = false

    @Flag(
        name: .customLong("mtp"),
        help: "Draft with the model's multi-token-prediction head (mtp.safetensors, e.g. MTPLX packs of Qwen3.5/3.8 and Bonsai 2), verified in one forward pass and exact at any temperature (greedy drafts, the target samples every token); requests without JSON constraints, speculative lookup, --kv-bits or --kv-zones and prompts up to 8192 tokens (prefilled in one forward; a follow-up turn reuses the stored prefix from the prompt cache and prefills only the new tail)"
    )
    var mtp = false

    @Option(
        name: .customLong("mtp-block"),
        help: "With --mtp, the largest verify round: one token the target already chose plus up to N-1 drafted by the MTP head. Rounds start at one drafted token, grow by one after four fully accepted rounds and drop back to one on a rejection (default 4; Qwen3.5-family targets rewind up to 7 drafted tokens, others clamp to their limit; 2 drafts one token every round)"
    )
    var mtpBlock = 4

    @Flag(
        name: .customLong("grammar-jump-forward"),
        help: "Feed XGrammar-forced spans in one forward pass during constrained decoding (opt-in; JSON schema and forced tool calls)"
    )
    var grammarJumpForward = false

    func configure(_ configuration: inout EngineConfiguration) {
        (configuration.speculativeLookup, configuration.speculativeRecycling) = (speculativeLookup, speculativeRecycling)
        (configuration.mtp, configuration.mtpBlockSize) = (mtp, mtpBlock)
        configuration.grammarJumpForward = grammarJumpForward
    }
}
