import ArgumentParser

struct ExperimentalServeOptions: ParsableArguments {
    @Flag(
        name: .customLong("experimental-zero-copy-body"),
        help: "Wrap each collected REST request body in Data without copying it, in the JSON nesting check and the chat pipeline. Measured on Gemma 4 26B-A4B with 1-8 images at the 32 MB limit: process footprint peak 70-95 MB lower, MLX peak unchanged, TTFT no better (#345)"
    )
    var zeroCopyBody = false

    @Flag(
        name: .customLong("experimental-smallm-kernel"),
        help: "Run 4-8 row forwards (speculative-lookup and MTP verify windows, batched decode) of 4-bit affine linear layers through basalt's simdgroup-matrix kernel, which decodes each weight once for all rows. On M4 Pro, Qwen3.8-27B shapes: 1.5-1.7x the one-token time at any of 4-8 rows, where stock takes 1.7-1.9x at 4 and 3.3-3.7x at 8; results differ from stock by bf16 rounding (#208). Same as BASALT_SMALLM_KERNEL=1. Off by default until greedy identity is checked on real models"
    )
    var smallMKernel = false

    @Flag(
        name: .customLong("experimental-speech-analyzer"),
        help: "Serve /v1/audio/transcriptions and the basalt_transcribe MCP tool through macOS SpeechAnalyzer on the Neural Engine when the loaded model has no audio encoder. The first request for a language may install its system speech assets (#222, #230). Off by default until its word error rate is measured"
    )
    var speechAnalyzer = false
}
