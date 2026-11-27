import ArgumentParser
import BasaltCore

extension AttentionBackend: ExpressibleByArgument {}

struct KVCacheOptions: ParsableArguments {
    @Option(
        name: .customLong("kv-bits"),
        help: "Quantize the KV cache to 4 or 8 bits, k8v4 (8-bit keys, 4-bit values; needs --kv-zones; experimental k8v4h also Walsh–Hadamard-rotates the values; experimental turbo3, turbo4 and turbo8v4 run the TurboQuant cache on full-attention layers, without --kv-zones, measured in #471 at 8–13% slower decode than 4-bit and no smaller; experimental polar4 stores the zoned middle's keys as one byte per RoPE pair, 5-bit phase and 3-bit modulus, on Qwen3, Qwen3.5 and LFM2 only, measured no better than affine in #351; experimental kvarn and kvarn-qjl run 2-bit variance-balanced tile compression with optional 1-bit QJL residual flash-decode, #550/#634; experimental gaugerope stores each full-attention token as a per-RoPE-frequency latent carrier plus a joint key-residual and value PCA code, calibrated on the first 512 tokens of each request or once by models convert --to gaugerope, at most 0.6x the k8v4 bytes, on Qwen 3.5, 3.6 and 3.8 only, measured worse than k8v4 in #647), or `auto`: fp16 until a request's max_tokens does not fit, then 8-bit, k8v4 with --kv-zones, then 4-bit (default: full precision)"
    )
    var bits: KVBitsSetting?

    @Flag(name: .customLong("kv-zones"), help: KVBitsSetting.zonesHelp)
    var zoned: Bool = false

    @Option(name: .customLong("kv-evict"), help: KVBitsSetting.evictHelp)
    var evict: Double?

    @Option(name: .customLong("kv-head-split"), help: ArgumentHelp("Experimental: keep the whole context only in this share (0.05...1) of the KV heads of each full-attention layer, the heads whose loss hurts a long-range recall most (scored once per model by ablation: how much keeping only sinks and a window in that one head lowers the log-probability of a passkey answer; cached in ~/.basalt/head-scores). Measured on hybrids at 16k: 0.5 kept passkey 5/5 on Qwen3.8-27B and 4/5 on LFM2.5-24B-A2B but lost multikey recall (2/3 and 1/3 against 3/3 dense); 0.25 lost more. Below 0.5 it loses facts; the other heads keep 128 sink and 256 recent tokens (DuoAttention). Not with uniform --kv-bits; with --kv-zones --kv-bits P (experimental) the kept heads hold their exact sinks and recent window and quantize the tokens between them at P, e.g. k8v4. Only for requests without a prompt-cache namespace, like --kv-zones"))
    var headSplit: Double?

    @Option(name: .customLong("quest-top-k"), help: ArgumentHelp("Experimental: on long prompts, each decode step of a full-attention layer attends to 128 sink tokens, the last 256 or more, and only these many 64-token blocks between them, picked per KV head by their key bounds (Quest). Models whose attention reads the whole cache, like Gemma 4, stay dense. Not with --kv-bits, --kv-zones or --kv-head-split, and only for requests without a prompt-cache namespace"))
    var questTopK: Int?

    @Option(name: .customLong("attention-backend"), help: ArgumentHelp("Experimental: full-attention Metal backend for non-cached requests: standard (default MLX SDPA), sagemx (SageAttention/MXAttention UOS+PNQ mean-centered kernel, #551/#634), moba (MoBA centroid-routed block-sparse Metal kernel, #551/#634), or nsa (experimental mean-pooled, fixed-gate 3-branch NSA approximation, #845)"))
    var attentionBackend: AttentionBackend = .standard

    @Option(name: .customLong("moba-top-k"), help: ArgumentHelp("With --attention-backend moba or nsa, number of 64-token blocks selected alongside fixed or window branches (1...4096, default 8)"))
    var mobaTopK: Int = 8

    @Option(name: .customLong("nsa-window"), help: ArgumentHelp("With --attention-backend nsa, sliding-window token count for the local branch (1...65536, default 256)"))
    var nsaWindow: Int = 256

    @Option(name: .customLong("rope-yarn"), help: ArgumentHelp("Extend the context of Qwen 3.5, 3.6 and 3.8 (qwen3_5) with YaRN by this factor (e.g. 2 for about 512K tokens, 4 for about 1M); the model's config is staged with rope_type yarn and original_max_position_embeddings set to its native context. YaRN is static: short prompts pay its small cost too, so use it only for contexts past the native one. Other models, Gemma 4 included, load with their stock RoPE and a warning"))
    var ropeYarn: Double?

    var zones: KVZones? { KVBitsSetting.zones(zoned, evicting: evict) }

    func validateCombination() throws {
        if let ropeYarn, !(ropeYarn > 1 && ropeYarn <= 64) { throw ValidationError("--rope-yarn takes a factor in (1, 64]") }
        try KVBitsSetting.validate(zones: zoned, bits: bits, evict: evict)
        if let headSplit {
            guard (0.05 ... 1).contains(headSplit) else { throw ValidationError("--kv-head-split takes a share of heads in 0.05...1") }
            guard bits == nil || zoned else {
                throw ValidationError("--kv-head-split does not combine with uniform --kv-bits; add --kv-zones to quantize only the middle of the heads that keep the whole context")
            }
        }
        if let questTopK {
            guard (1 ... 4096).contains(questTopK) else { throw ValidationError("--quest-top-k takes a number of blocks in 1...4096") }
            guard bits == nil, !zoned, headSplit == nil else {
                throw ValidationError("--quest-top-k does not combine with --kv-bits, --kv-zones or --kv-head-split")
            }
        }
        guard (1 ... 4096).contains(mobaTopK) else { throw ValidationError("--moba-top-k takes a number of blocks in 1...4096") }
        guard (1 ... 65_536).contains(nsaWindow) else { throw ValidationError("--nsa-window takes a window size in 1...65536") }
        if attentionBackend != .standard {
            guard bits == nil, !zoned, headSplit == nil, questTopK == nil else {
                throw ValidationError("--attention-backend \(attentionBackend.rawValue) does not combine with --kv-bits, --kv-zones, --kv-head-split or --quest-top-k")
            }
        }
    }
}

extension ServeCommand {
    var kvBits: KVBitsSetting? { kvCache.bits }
    var kvZones: Bool { kvCache.zoned }
    var kvEvict: Double? { kvCache.evict }
}
