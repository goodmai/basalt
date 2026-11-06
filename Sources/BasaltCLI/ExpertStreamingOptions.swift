import ArgumentParser
import BasaltCore

struct ExpertStreamingOptions: ParsableArguments {
    @Flag(
        name: .customLong("experimental-expert-streaming"),
        help: "Experimental (#225): load a mixture-of-experts model with only its non-expert weights resident and read routed experts from the safetensors shards (pread, F_NOCACHE) into a bounded LRU expert cache on demand, so a model in the Streaming fit tier loads. Greedy output matches the resident model; decode is slower by the cache misses. Needs an mlx-lm converted checkpoint with stacked expert tensors"
    )
    var enabled = false

    @Option(
        name: .customLong("expert-cache-gb"),
        help: "With --experimental-expert-streaming: size of the expert cache in GB (default: 25% of the expert weights)"
    )
    var cacheGB: Double?

    var setting: ExpertStreaming? {
        enabled ? ExpertStreaming(cacheBytes: cacheGB.map { ByteCount(Int64($0 * 1_073_741_824)) }) : nil
    }

    func validate() throws {
        guard let cacheGB else { return }
        guard enabled else { throw ValidationError("--expert-cache-gb needs --experimental-expert-streaming") }
        guard cacheGB > 0, cacheGB.isFinite else { throw ValidationError("--expert-cache-gb must be more than 0, got \(cacheGB)") }
    }
}
