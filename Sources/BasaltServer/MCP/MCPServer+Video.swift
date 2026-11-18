import BasaltCore
import Foundation

package enum MCPVideoToolName: String, Sendable, Equatable, CaseIterable {
    case videoGenerate = "video_generate"

    static func isAllowed(_ name: String, by filter: MCPToolFilter) -> Bool {
        !filter.isExplicitlyDisabled(name) && !filter.disabled.isSuperset(of: MCPToolFilter.Group.media.tools)
    }
}

package struct MCPVideoToolOutput: Codable, Sendable, Equatable {
    package let assetID: String
    package let path: String
    package let url: String
    package let width: Int
    package let height: Int
    package let frames: Int
    package let fps: Int
    package let durationSeconds: Double
    package let format: String
    package let codec: String
    package let hasAudio: Bool
    package let seed: UInt64
    package let steps: Int
    package let teacacheSkipped: Int
    package let executionMode: String
    package let model: String

    init(result: VideoGenerationResult, persisted: VideoArtifactStore.PersistedVideo) {
        self.assetID = persisted.assetID
        self.path = persisted.path
        self.url = persisted.url
        self.width = result.width
        self.height = result.height
        self.frames = result.frames
        self.fps = result.fps
        self.durationSeconds = result.durationSeconds
        self.format = result.containerFormat.rawValue
        self.codec = result.codec.rawValue
        self.hasAudio = result.hasAudioTrack
        self.seed = result.seed
        self.steps = result.stepsExecuted
        self.teacacheSkipped = result.teacacheSkippedSteps
        self.executionMode = result.executionMode.rawValue
        self.model = result.model
    }

    enum CodingKeys: String, CodingKey {
        case path, url, width, height, frames, fps, format, codec, seed, steps, model
        case assetID = "asset_id"
        case durationSeconds = "duration_seconds"
        case hasAudio = "has_audio"
        case teacacheSkipped = "teacache_skipped"
        case executionMode = "execution_mode"
    }
}

extension MCPServer {
    func generateVideoTool(_ args: [String: Any]?) async throws(MCPFailure) -> ToolResult {
        guard let synthesizer = videoSynthesizer else {
            throw .methodNotFound("video_generate needs a video model: configure a video synthesizer on this server")
        }
        let request = try MCPArguments.videoGeneration(args, descriptor: synthesizer.descriptor)
        do {
            let stream = try await synthesizer.generate(request)
            var finalResult: VideoGenerationResult?
            for try await event in stream {
                if case .completed(let done) = event {
                    finalResult = done
                }
            }
            guard let result = finalResult else {
                throw BasaltError.modelInferenceError(details: "video: synthesizer finished without a completed artifact")
            }
            let persisted = try VideoArtifactStore.persist(result, in: videoStoreDirectory)
            return .json(MCPVideoToolOutput(result: result, persisted: persisted))
        } catch let error as BasaltError {
            throw MCPFailure(error)
        } catch {
            throw .internalError(error.localizedDescription)
        }
    }
}

extension MCPArguments {
    static func videoGeneration(_ args: [String: Any]?, descriptor: VideoModelDescriptor) throws(MCPFailure) -> VideoGenerationRequest {
        let prompt = try string("prompt", in: args) ?? ""
        let negative = try string("negative_prompt", in: args)
        let model = try string("model", in: args)
        let size = try string("size", in: args)
        let frames = try integer("frames", in: args, 1 ... VideoGenerationRequest.maxFrames) ?? descriptor.defaultFrames
        let fps = try integer("fps", in: args, 1 ... 120) ?? descriptor.defaultFPS
        let steps = try integer("steps", in: args, 1 ... 100) ?? descriptor.defaultSteps
        let rawFormat = try string("format", in: args)
        let rawCodec = try string("codec", in: args)
        let seedVal = try integer("seed", in: args, 0 ... Int.max).map { UInt64($0) }
        let teacache = try number("teacache_threshold", in: args).map(Double.init) ?? 0.12
        let includeAudio = try optionalBoolean("include_audio", in: args) ?? descriptor.supportsAudio
        do throws(BasaltError) {
            let dims = if let size, !size.isEmpty { try VideoGenerationRequest.parseSize(size) } else { VideoDimensions() }
            let container = try WireVideoRequest.parseContainer(rawFormat)
            let codec = try WireVideoRequest.parseVideoCodec(rawCodec)
            let config = VideoGenerationConfig(
                guidanceScale: 5.0,
                seed: seedVal,
                codec: codec,
                includeAudio: includeAudio,
                teacacheThreshold: teacache
            )
            var request = try VideoGenerationRequest(
                model: (model?.isEmpty == false ? model : nil) ?? descriptor.identifier,
                prompt: prompt,
                dimensions: dims,
                temporal: VideoTemporalSpec(frames: frames, fps: fps, steps: steps),
                containerFormat: container,
                config: config
            )
            request.negativePrompt = negative
            return request
        } catch {
            throw MCPFailure(error)
        }
    }
}

extension MCPToolCatalog {
    static let generateVideo = MCPToolDescriptor(
        name: MCPVideoToolName.videoGenerate.rawValue,
        description: "Synthesize a synchronized audio-visual .mp4 or .mov clip on Apple Silicon via phased UMA MLX execution (text conditioning → TeaCache DiT denoising → tiled 3D-VAE + native AVAssetWriter muxing), persist it under ~/.basalt/videos/<sha256>.<ext>, and return its path, URL and metadata.",
        inputSchema: ResearchTool.object([
            "prompt": ["type": "string", "description": "Visual and motion description of the video to synthesize"],
            "negative_prompt": ["type": "string", "description": "Optional elements to avoid"],
            "model": ["type": "string", "description": "Optional video DiT model identifier"],
            "size": ["type": "string", "description": "Dimensions as <width>x<height>, multiples of 16 (default 512x512)"],
            "frames": ["type": "integer", "minimum": 1, "maximum": VideoGenerationRequest.maxFrames, "description": "Frame count (default 25)"],
            "fps": ["type": "integer", "minimum": 1, "maximum": 120, "description": "Frames per second (default 24)"],
            "steps": ["type": "integer", "minimum": 1, "maximum": 100, "description": "Denoising steps"],
            "format": ["type": "string", "enum": VideoContainerFormat.allCases.map(\.rawValue), "description": "Container format: mp4 or mov"],
            "codec": ["type": "string", "enum": VideoCodecKind.allCases.map(\.rawValue), "description": "Video codec: h264 or hevc"],
            "include_audio": ["type": "boolean", "description": "Include synchronized audio track (default true)"],
            "teacache_threshold": ["type": "number", "description": "TeaCache residual skip threshold (default 0.12)"],
            "seed": ["type": "integer", "minimum": 0, "description": "Deterministic RNG seed"],
        ], required: ["prompt"])
    )
}
