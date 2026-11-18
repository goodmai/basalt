import ArgumentParser
import BasaltCore

struct ExpertStreamingOptions: ParsableArguments {
    @Flag(
        name: [.customLong("expert-streaming"), .customLong("experimental-expert-streaming")],
        help: "Load a mixture-of-experts model with only its non-expert weights resident and read routed experts from the safetensors shards (batched asynchronous reads, F_NOCACHE) straight into fixed expert slots; auto-enabled on local stacked SwitchLinear checkpoints in the Streaming fit tier (#225, #226); --experimental-expert-streaming is kept as an alias"
    )
    var enabled = false

    @Flag(
        name: .customLong("no-expert-streaming"),
        help: "Disable automatic SSD expert streaming on local stacked SwitchLinear checkpoints in the Streaming fit tier (same as BASALT_NO_EXPERT_STREAMING=1)"
    )
    var disabled = false

    @Option(
        name: .customLong("expert-cache-gb"),
        help: "With --expert-streaming: size of the expert cache in GB (default: 25% of the expert weights)"
    )
    var cacheGB: Double?

    var setting: ExpertStreaming? {
        enabled && !disabled ? ExpertStreaming(cacheBytes: cacheGB.map { ByteCount(Int64($0 * 1_073_741_824)) }) : nil
    }

    func validate() throws {
        if enabled, disabled {
            throw ValidationError("--expert-streaming and --no-expert-streaming cannot be used together")
        }
        guard let cacheGB else { return }
        guard enabled, !disabled else { throw ValidationError("--expert-cache-gb needs --expert-streaming") }
        guard cacheGB > 0, cacheGB <= 65_536, cacheGB.isFinite else {
            throw ValidationError("--expert-cache-gb must be in (0, 65536], got \(cacheGB)")
        }
    }
}
