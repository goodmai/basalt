import ArgumentParser
import BasaltCore
import BasaltSpeech

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
        help: "Serve /v1/audio/transcriptions, the basalt_transcribe MCP tool and audio in /v1/pipeline through macOS SpeechAnalyzer on the Neural Engine when the loaded model has no audio encoder. The first request for a language may install its system speech assets (#222, #230). Off by default until its word error rate is measured"
    )
    var speechAnalyzer = false

    @Option(
        name: .customLong("image-fallback"),
        help: "ocr: when the loaded model takes no images, read each image of a chat request with Apple's on-device Vision OCR and send its Markdown as a text block ([image N, OCR]) instead of refusing the request; responses say so in the X-Basalt-Image-Fallback header and, when not streamed, a basalt field (#403). Off by default until its quality is measured"
    )
    var imageFallback: ImageFallback?

    enum ImageFallback: String, ExpressibleByArgument {
        case ocr
    }

    func attachMediaFallbacks(to orchestrator: ModelOrchestratorActor) async {
        if speechAnalyzer { await orchestrator.setSpeechFallback(SpeechAnalyzerTranscriber()) }
        await orchestrator.setDocumentTranscriber(VisionDocumentTranscriber(), imageFallback: imageFallback == .ocr)
    }
}
