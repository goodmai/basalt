import ArgumentParser
import BasaltBench
import BasaltCore
import BasaltInference
import Foundation

struct VideoCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video",
        abstract: "Generate synchronized audio-visual .mp4/.mov clips on Apple Silicon via phased UMA MLX execution",
        subcommands: [VideoGenerateSubcommand.self],
        defaultSubcommand: VideoGenerateSubcommand.self
    )
}

struct VideoGenerateSubcommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Synthesize a video clip with phased text conditioning, TeaCache DiT denoising, tiled 3D-VAE and native AVAssetWriter muxing"
    )

    @Option(name: .long, help: "Video DiT model repo ID from the local cache or local checkpoint directory")
    var model: String

    @Option(name: .long, help: "Visual and motion prompt")
    var prompt: String

    @Option(name: [.customShort("o"), .long], help: "Output .mp4 or .mov file path")
    var output: String

    @Option(name: .long, help: "Optional negative prompt")
    var negativePrompt: String?

    @Option(name: .long, help: "Number of video frames to synthesize (default: 25)")
    var frames: Int = 25

    @Option(name: .long, help: "Frames per second (default: 24)")
    var fps: Int = 24

    @Option(name: .long, help: "Flow-matching denoising steps (default: 20)")
    var steps: Int = 20

    @Option(name: .long, help: "Dimensions as <width>x<height>, multiples of 16 (default: 512x512)")
    var size: String = "512x512"

    @Option(name: .long, help: "Video codec: h264 or hevc (default: h264)")
    var codec: String = "h264"

    @Option(name: .long, help: "TeaCache skip threshold (default: 0.12)")
    var teacacheThreshold: Double = 0.12

    @Option(name: .long, help: "Optional Sliding Tile Attention window")
    var staWindow: Int?

    @Option(name: .long, help: "Deterministic RNG seed")
    var seed: UInt64?

    @Flag(name: .long, inversion: .prefixedNo, help: "Include synchronized audio track (default: true)")
    var audio: Bool = true

    func buildRequest() throws -> VideoGenerationRequest {
        let dims = try VideoGenerationRequest.parseSize(size)
        guard let vCodec = VideoCodecKind(rawValue: codec.lowercased()) else {
            throw ValidationError("--codec must be h264 or hevc, got '\(codec)'")
        }
        let expandedOut = (output as NSString).expandingTildeInPath
        let ext = URL(fileURLWithPath: expandedOut).pathExtension.lowercased()
        let container: VideoContainerFormat = switch ext {
        case "mov": .mov
        case "mp4", "": .mp4
        default: throw ValidationError("--output extension must be .mp4 or .mov, got '.\(ext)'")
        }
        var config = VideoGenerationConfig(
            guidanceScale: 5.0,
            seed: seed,
            codec: vCodec,
            includeAudio: audio,
            teacacheThreshold: teacacheThreshold
        )
        config.staWindow = staWindow
        var request = try VideoGenerationRequest(
            model: model,
            prompt: prompt,
            dimensions: dims,
            temporal: VideoTemporalSpec(frames: frames, fps: fps, steps: steps),
            containerFormat: container,
            config: config
        )
        request.negativePrompt = negativePrompt
        return request
    }

    mutating func run() async throws {
        let request = try buildRequest()
        let resolvedDir = try Self.resolveModelDirectory(model)
        let outURL = URL(fileURLWithPath: (output as NSString).expandingTildeInPath)
        try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let runner = try PhasedVideoRunnerActor(
            identifier: model,
            options: .init(directory: resolvedDir)
        )
        let result = try await GPULock.withLock {
            let stream = try await runner.generate(request)
            var completed: VideoGenerationResult?
            for try await event in stream {
                switch event {
                case .progress(let prog):
                    let pct = Int((prog.fractionCompleted * 100).rounded())
                    fputs("\r[\(prog.phase.rawValue)] \(pct)% — \(prog.message)", stderr)
                case .completed(let finalResult):
                    completed = finalResult
                }
            }
            fputs("\n", stderr)
            guard let completed else {
                throw BasaltError.modelInferenceError(details: "video generation finished without output")
            }
            return completed
        }
        try result.videoData.write(to: outURL, options: .atomic)
        print("Wrote \(outURL.path) (\(result.width)x\(result.height), \(result.frames) frames @ \(result.fps) fps, mode=\(result.executionMode.rawValue))")
    }

    static func resolveModelDirectory(_ spec: String) throws -> URL? {
        let expanded = (spec as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue {
            return URL(fileURLWithPath: expanded)
        }
        if let cached = ModelCache.resolve(repoId: spec) {
            return cached
        }
        throw ValidationError("Model '\(spec)' is not a local directory or cached Hugging Face snapshot")
    }
}
