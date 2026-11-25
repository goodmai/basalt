import ArgumentParser
import BasaltCore
import BasaltInference
import Foundation
import BasaltSpeech

extension MACaQActivationMode: ExpressibleByArgument {}

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
        name: .customLong("experimental-q3-kernel"),
        help: "Run one-row (decode) forwards of MACaQ 3-bit affine layers, dense and MoE experts, through basalt's 3-bit kernel, which reads packed weights as 16-bit words and drops MLX's K % 512 gate for its fast path. On M4 Pro: 17408x5120 dense 173-177 us vs 181-184 for MLX, gpt-oss expert projections 77-82 vs 81-92 us (within run-to-run noise); results match MLX within bf16 rounding (#653). Same as BASALT_Q3_KERNEL=1"
    )
    var q3Kernel = false

    @Flag(
        name: .customLong("experimental-kv-shared-row"),
        help: "On Gemma 4 12B, 26B-A4B and 31B (attention_k_eq_v), store each global-layer KV head as one 640-value row [rotated keys | unrotated values | rotated values] instead of 512 keys + 512 values: 0.625x global-layer KV bytes, with the k_norm weight folded into the queries and stock SDPA over two views of the row (#645). Requests without a prompt-cache namespace only. Not with --kv-bits, --kv-zones, --kv-head-split, --quest-top-k or --attention-backend"
    )
    var kvSharedRow = false

    @Flag(
        name: [.customLong("speech-analyzer"), .customLong("experimental-speech-analyzer")],
        help: "Serve /v1/audio/transcriptions, the basalt_transcribe MCP tool and audio in /v1/pipeline through macOS SpeechAnalyzer on the Neural Engine when the loaded model has no audio encoder (and, for /v1/pipeline, no --pipeline-audio-model is set). The first request for a language may install its system speech assets (#222, #226, #230); --experimental-speech-analyzer is kept as an alias"
    )
    var speechAnalyzer = false

    @Option(
        name: .customLong("image-fallback"),
        help: "ocr: when the loaded model takes no images, read each image of a chat request with Apple's on-device Vision OCR and send its Markdown as a text block ([image N, OCR]) instead of refusing the request; responses say so in the X-Basalt-Image-Fallback header and, when not streamed, a basalt field (#403). Off by default until its quality is measured"
    )
    var imageFallback: ImageFallback?

    @Option(
        name: [.customLong("ane-vision-tower"), .customLong("experimental-ane-vision-tower")],
        help: "Run PaddleOCR-VL's vision encoder on the Neural Engine from a Core AI tower bundle (metadata.json + tower.aimodel, fixed patch-count buckets), specialized when the model loads and tracked in ~/.basalt/cache/coreai-vision/ (#224, #226, #400); images whose patch grid fits no bucket stay on the GPU tower. Build the bundle with swift scripts/paddle_tower_export.swift <snapshot> <out>; --experimental-ane-vision-tower is kept as an alias"
    )
    var aneVisionTower: String?

    @Option(
        name: .customLong("telemetry-confidence-every"),
        help: "Measure per-token confidence (top-token probability, entropy, top-two margin) on every Nth generation and add it to /api/v1/stats and /api/v1/metrics; 0 measures only requests that ask for it. Costs one GPU sync per measured token (#450)"
    )
    var telemetryConfidenceEvery = 0

    @Option(
        name: .customLong("experimental-macaq-activations"),
        help: "Experimental (#547): fused MACaQ activation microscaling mode inside the prologue-MMA Metal kernel: w4a16 (default), w4a8 (dynamic FP8 E4M3) or w4a4 (dynamic OCP MXFP4)"
    )
    var macaqActivations: MACaQActivationMode = .fp16

    @Flag(
        name: .customLong("experimental-dllm"),
        help: "Decode discrete diffusion language models (Dream, DiffuCoder, LLaDA) with Fast-dLLM DualBlockKVCache and confidence-threshold parallel unmasking (#574). Off by default until real-model throughput and quality are measured"
    )
    var dllm = false

    @Flag(
        name: .customLong("experimental-dflash"),
        help: "Draft 8-16 tokens in one parallel block-diffusion forward pass conditioned on target hidden states (DFlash) and verify losslessly against the target model (#574). Off by default until real-model speedup is measured"
    )
    var dflash = false

    func applyKernelEnvironment(_ set: (String, String) -> Void = { setenv($0, $1, 1) }) {
        if macaqActivations != .fp16 { set("BASALT_MACAQ_ACTIVATIONS", macaqActivations.rawValue) }
    }

    func validate() throws {
        if telemetryConfidenceEvery < 0 { throw ValidationError("--telemetry-confidence-every takes 0 or a positive count, got \(telemetryConfidenceEvery)") }
        if dllm, dflash {
            throw ValidationError("--experimental-dllm and --experimental-dflash cannot be combined: Fast-dLLM decodes a diffusion language model directly while DFlash drafts for an autoregressive target")
        }
        if dllm || dflash {
            throw ValidationError("\(dllm ? "--experimental-dllm" : "--experimental-dflash") is not wired to the engine yet (#574): generation would run on the standard decode path")
        }
        if let aneVisionTower, !CoreAIVisionTower.isBundle(URL(fileURLWithPath: aneVisionTower)) {
            throw ValidationError("--ane-vision-tower \(aneVisionTower) is not a Core AI vision tower bundle (metadata.json and tower.aimodel)")
        }
    }

    enum ImageFallback: String, ExpressibleByArgument {
        case ocr
    }

    func attachMediaFallbacks(to orchestrator: some ModelOrchestrating) async {
        if speechAnalyzer { await orchestrator.setSpeechFallback(SpeechAnalyzerTranscriber()) }
        await orchestrator.setDocumentTranscriber(VisionDocumentTranscriber(), imageFallback: imageFallback == .ocr)
        await orchestrator.telemetry.sampleConfidence(every: telemetryConfidenceEvery)
    }
}
