import ArgumentParser

struct ExperimentalServeOptions: ParsableArguments {
    @Flag(
        name: .customLong("experimental-zero-copy-body"),
        help: "Wrap each collected REST request body in Data without copying it, in the JSON nesting check and the chat pipeline. Measured on Gemma 4 26B-A4B with 1-8 images at the 32 MB limit: process footprint peak 70-95 MB lower, MLX peak unchanged, TTFT no better (#345)"
    )
    var zeroCopyBody = false

    @Flag(
        name: .customLong("experimental-smallm-kernel"),
        help: "Run 2-8 row forwards (speculative-lookup and MTP verify windows, batched decode) of 4-bit affine linear layers through basalt's small-M kernel, which decodes each weight word once for all rows; each row equals the stock one-token result bit for bit (#208). Same as BASALT_SMALLM_KERNEL=1. Off by default until measured end to end"
    )
    var smallMKernel = false
}
