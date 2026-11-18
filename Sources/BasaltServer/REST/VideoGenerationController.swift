import BasaltCore
import CryptoKit
import Foundation
import HTTPTypes
import Hummingbird

struct VideoArtifactStore: Sendable {
    struct PersistedVideo: Sendable, Equatable {
        let assetID: String
        let path: String
        let url: String
    }

    static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".basalt/videos", directoryHint: .isDirectory)
    }

    static func persist(_ result: VideoGenerationResult, in directory: URL?) throws(BasaltError) -> PersistedVideo {
        let root = (directory ?? defaultDirectory).standardizedFileURL
        do {
            try FileManager.default.createDirectory(
                at: root,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
        } catch {
            throw .invalidRequestStructure(details: "video: failed to prepare store directory: \(error.localizedDescription)")
        }
        let digest = SHA256.hash(data: result.videoData).map { String(format: "%02x", $0) }.joined()
        let assetID = "\(digest).\(result.containerFormat.fileExtension)"
        let fileURL = root.appending(component: assetID)
        do {
            try result.videoData.write(to: fileURL, options: [.atomic])
            try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: fileURL.path)
        } catch {
            throw .modelInferenceError(details: "video: failed to persist generated artifact: \(error.localizedDescription)")
        }
        return PersistedVideo(assetID: assetID, path: fileURL.path, url: "/v1/videos/content/\(assetID)")
    }

    static func parseAssetID(_ id: String) -> VideoContainerFormat? {
        let parts = id.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 2, parts[0].utf8.count == 64 else { return nil }
        let isHex = parts[0].unicodeScalars.allSatisfy {
            ("0" ... "9").contains($0) || ("a" ... "f").contains($0)
        }
        guard isHex else { return nil }
        return VideoContainerFormat(rawValue: String(parts[1]))
    }

    static func readValidatedAsset(id: String, in directory: URL?) -> (data: Data, format: VideoContainerFormat)? {
        guard let format = parseAssetID(id) else { return nil }
        let root = (directory ?? defaultDirectory).standardizedFileURL
        let fileURL = root.appending(component: id).standardizedFileURL
        guard fileURL.deletingLastPathComponent() == root else { return nil }
        guard let data = try? Data(contentsOf: fileURL, options: .mappedIfSafe) else { return nil }
        return (data, format)
    }
}

struct VideoGenerationController: Sendable {
    static let generationsPath = "/v1/videos/generations"
    static let bodyLimit = 32 * 1024 * 1024

    static let modelHeader = HTTPField.Name("X-Basalt-Video-Model")!
    static let modeHeader = HTTPField.Name("X-Basalt-Video-Execution-Mode")!
    static let codecHeader = HTTPField.Name("X-Basalt-Video-Codec")!

    let synthesizer: (any VideoSynthesizer)?
    var pipelines: PipelineRegistry? = nil
    var storeDirectory: URL? = nil

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.post("/videos/generations") { try await self.generations(request: $0, context: $1) }
        group.get("/videos/content/{id}") { try await self.content(request: $0, context: $1) }
        group.get("/videos/models") { try await self.models(request: $0, context: $1) }
    }

    private func resolveSynthesizer() async -> (any VideoSynthesizer)? {
        if let live = await pipelines?.system(ComposeSpec.defaultPipeline)?.video { return live }
        return synthesizer
    }

    func models(request: Request, context: BasicRequestContext) async throws -> Response {
        let active = await resolveSynthesizer()
        let items = active.map { [$0.descriptor] } ?? []
        var buffer = ByteBuffer()
        buffer.writeBytes(try ImagesController.makeJSONEncoder().encode(VideoModelsEnvelope(models: items)))
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }

    func content(request: Request, context: BasicRequestContext) async throws -> Response {
        let raw = context.parameters.get("id")?.removingPercentEncoding ?? ""
        guard !request.uri.path.contains(".."), VideoArtifactStore.parseAssetID(raw) != nil else {
            return ProviderErrorResponse.response(status: .badRequest, message: "invalid video asset id", shape: .openAI)
        }
        guard let loaded = VideoArtifactStore.readValidatedAsset(id: raw, in: storeDirectory) else {
            return ProviderErrorResponse.response(status: .notFound, message: "video asset not found", shape: .openAI)
        }
        var buffer = ByteBuffer()
        buffer.writeBytes(loaded.data)
        let headers: HTTPFields = [
            .contentType: loaded.format.contentType,
            .cacheControl: "private, max-age=86400",
        ]
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: buffer))
    }

    func generations(request: Request, context: BasicRequestContext) async throws -> Response {
        let bodyData: Data
        do {
            let collected = try await request.body.collect(upTo: Self.bodyLimit)
            bodyData = Data(collected.readableBytesView)
        } catch let error as any HTTPResponseError where error.status == .contentTooLarge {
            return ProviderErrorResponse.response(
                status: .contentTooLarge,
                message: "a video generation request is limited to \(Self.bodyLimit >> 20) MB",
                shape: .openAI
            )
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.generationsPath)
        }
        do {
            guard let active = await resolveSynthesizer() else {
                throw BasaltError.invalidRequestStructure(details: "video: no video synthesizer is configured on this server")
            }
            let parsed = try WireVideoRequest.parse(bodyData, descriptor: active.descriptor)
            return try await execute(parsed, active: active, started: .now)
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.generationsPath)
        }
    }
}

extension VideoGenerationController {
    func execute(
        _ genRequest: VideoGenerationRequest,
        active: any VideoSynthesizer,
        started: ContinuousClock.Instant
    ) async throws -> Response {
        let stream = try await active.generate(genRequest)
        if genRequest.stream {
            return Self.sseResponse(
                stream: stream,
                active: active,
                codec: genRequest.config.codec,
                storeDirectory: storeDirectory,
                started: started
            )
        }
        var completedResult: VideoGenerationResult?
        for try await event in stream {
            if case .completed(let done) = event {
                completedResult = done
            }
        }
        guard let result = completedResult else {
            throw BasaltError.modelInferenceError(details: "video: synthesizer finished without a completed artifact")
        }
        let persisted = try VideoArtifactStore.persist(result, in: storeDirectory)
        let elapsed = max(result.elapsedMs, ImagesController.elapsedMilliseconds(since: started))
        let item = VideoWireItem(result: result, persisted: persisted, elapsedMs: elapsed)
        let envelope = VideosResponseEnvelope(created: Int(Date().timeIntervalSince1970), data: [item])
        var buffer = ByteBuffer()
        buffer.writeBytes(try ImagesController.makeJSONEncoder().encode(envelope))
        let headers: HTTPFields = [
            .contentType: "application/json",
            Self.modelHeader: result.model,
            Self.modeHeader: result.executionMode.rawValue,
            Self.codecHeader: result.codec.rawValue,
        ]
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: buffer))
    }

    static func sseResponse(
        stream: AsyncThrowingStream<VideoGenerationEvent, any Error>,
        active: any VideoSynthesizer,
        codec: VideoCodecKind,
        storeDirectory: URL?,
        started: ContinuousClock.Instant
    ) -> Response {
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                await streamSSE(stream: stream, storeDirectory: storeDirectory, started: started, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        let headers: HTTPFields = [
            .contentType: "text/event-stream",
            .cacheControl: "no-cache",
            .connection: "keep-alive",
            modelHeader: active.modelIdentifier,
            codecHeader: codec.rawValue,
        ]
        return Response(status: .ok, headers: headers, body: ResponseBody(asyncSequence: body))
    }

    private static func streamSSE(
        stream: AsyncThrowingStream<VideoGenerationEvent, any Error>,
        storeDirectory: URL?,
        started: ContinuousClock.Instant,
        continuation: AsyncStream<ByteBuffer>.Continuation
    ) async {
        let encoder = ImagesController.makeJSONEncoder()
        do {
            for try await event in stream {
                switch event {
                case .progress(let prog):
                    let wire = VideoProgressSSEEvent(progress: prog)
                    let json = try encoder.encode(wire)
                    continuation.yield(ByteBuffer(string: "data: \(String(decoding: json, as: UTF8.self))\n\n"))
                case .completed(let result):
                    let persisted = try VideoArtifactStore.persist(result, in: storeDirectory)
                    let elapsed = max(result.elapsedMs, ImagesController.elapsedMilliseconds(since: started))
                    let wire = VideoCompletedSSEEvent(result: result, persisted: persisted, elapsedMs: elapsed)
                    let json = try encoder.encode(wire)
                    continuation.yield(ByteBuffer(string: "data: \(String(decoding: json, as: UTF8.self))\n\n"))
                }
            }
            continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
        } catch {
            let message = ModelsController.publicMessage(for: error)
            if let json = try? JSONSerialization.data(withJSONObject: ["type": "error", "message": message, "error": ["message": message]]) {
                continuation.yield(ByteBuffer(string: "event: error\ndata: \(String(decoding: json, as: UTF8.self))\n\n"))
            }
        }
        continuation.finish()
    }
}

struct VideoModelsEnvelope: Codable, Sendable, Equatable {
    let models: [VideoModelDescriptor]
}

struct VideosResponseEnvelope: Codable, Sendable, Equatable {
    let created: Int
    let data: [VideoWireItem]
}

struct VideoWireItem: Codable, Sendable, Equatable {
    let assetID: String
    let url: String
    let b64MP4: String
    let width: Int
    let height: Int
    let frames: Int
    let fps: Int
    let durationSeconds: Double
    let format: String
    let codec: String
    let audioCodec: String?
    let hasAudio: Bool
    let seed: UInt64
    let steps: Int
    let teacacheSkipped: Int
    let executionMode: String
    let phasesCompleted: [String]
    let model: String
    let elapsedMs: Int

    init(result: VideoGenerationResult, persisted: VideoArtifactStore.PersistedVideo, elapsedMs: Int) {
        self.assetID = persisted.assetID
        self.url = persisted.url
        self.b64MP4 = result.videoData.base64EncodedString()
        self.width = result.width
        self.height = result.height
        self.frames = result.frames
        self.fps = result.fps
        self.durationSeconds = result.durationSeconds
        self.format = result.containerFormat.rawValue
        self.codec = result.codec.rawValue
        self.audioCodec = result.audioCodec?.rawValue
        self.hasAudio = result.hasAudioTrack
        self.seed = result.seed
        self.steps = result.stepsExecuted
        self.teacacheSkipped = result.teacacheSkippedSteps
        self.executionMode = result.executionMode.rawValue
        self.phasesCompleted = result.phasesCompleted.map(\.rawValue)
        self.model = result.model
        self.elapsedMs = elapsedMs
    }

    enum CodingKeys: String, CodingKey {
        case url, width, height, frames, fps, format, codec, seed, steps, model
        case assetID = "asset_id"
        case b64MP4 = "b64_mp4"
        case durationSeconds = "duration_seconds"
        case audioCodec = "audio_codec"
        case hasAudio = "has_audio"
        case teacacheSkipped = "teacache_skipped"
        case executionMode = "execution_mode"
        case phasesCompleted = "phases_completed"
        case elapsedMs = "elapsed_ms"
    }
}

struct VideoProgressSSEEvent: Codable, Sendable, Equatable {
    let type: String
    let phase: String
    let step: Int
    let totalSteps: Int
    let fractionCompleted: Double
    let teacacheSkipped: Bool
    let b64Preview: String?
    let message: String

    init(progress: VideoGenerationProgress) {
        self.type = "video_generation.progress"
        self.phase = progress.phase.rawValue
        self.step = progress.step
        self.totalSteps = progress.totalSteps
        self.fractionCompleted = progress.fractionCompleted
        self.teacacheSkipped = progress.teacacheSkipped
        self.b64Preview = progress.previewPNG?.base64EncodedString()
        self.message = progress.message
    }

    enum CodingKeys: String, CodingKey {
        case type, phase, step, message
        case totalSteps = "total_steps"
        case fractionCompleted = "fraction_completed"
        case teacacheSkipped = "teacache_skipped"
        case b64Preview = "b64_preview"
    }
}

struct VideoCompletedSSEEvent: Codable, Sendable, Equatable {
    let type: String
    let item: VideoWireItem

    init(result: VideoGenerationResult, persisted: VideoArtifactStore.PersistedVideo, elapsedMs: Int) {
        self.type = "video_generation.completed"
        self.item = VideoWireItem(result: result, persisted: persisted, elapsedMs: elapsedMs)
    }

    func encode(to encoder: any Encoder) throws {
        var keyed = encoder.container(keyedBy: DynamicTypeKey.self)
        try keyed.encode(type, forKey: DynamicTypeKey("type"))
        try item.encode(to: encoder)
    }

    init(from decoder: any Decoder) throws {
        let keyed = try decoder.container(keyedBy: DynamicTypeKey.self)
        self.type = try keyed.decode(String.self, forKey: DynamicTypeKey("type"))
        self.item = try VideoWireItem(from: decoder)
    }

    private struct DynamicTypeKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(_ string: String) { self.stringValue = string }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
}

struct WireVideoRequest: Decodable {
    var model: String?
    var prompt: String?
    var negativePrompt: String?
    var size: String?
    var width: Int?
    var height: Int?
    var frames: Int?
    var fps: Int?
    var steps: Int?
    var format: String?
    var containerFormat: String?
    var codec: String?
    var audioCodec: String?
    var includeAudio: Bool?
    var audioSampleRate: Int?
    var audioChannels: Int?
    var seed: UInt64?
    var guidanceScale: Double?
    var teacacheThreshold: Double?
    var staWindow: Int?
    var previewEverySteps: Int?
    var stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model, prompt, size, width, height, frames, fps, steps, format, codec, seed, stream
        case negativePrompt = "negative_prompt"
        case containerFormat = "container_format"
        case audioCodec = "audio_codec"
        case includeAudio = "include_audio"
        case audioSampleRate = "audio_sample_rate"
        case audioChannels = "audio_channels"
        case guidanceScale = "guidance_scale"
        case teacacheThreshold = "teacache_threshold"
        case staWindow = "sta_window"
        case previewEverySteps = "preview_every_steps"
    }

    static func parse(_ data: Data, descriptor: VideoModelDescriptor) throws(BasaltError) -> VideoGenerationRequest {
        let wire: WireVideoRequest
        do {
            wire = try JSONDecoder().decode(WireVideoRequest.self, from: data)
        } catch {
            throw .invalidRequestStructure(details: "video: body must be valid JSON matching /v1/videos/generations")
        }
        return try wire.buildRequest(descriptor: descriptor)
    }

    func buildRequest(descriptor: VideoModelDescriptor) throws(BasaltError) -> VideoGenerationRequest {
        let dims = try resolveDimensions()
        let container = try Self.parseContainer(containerFormat ?? format)
        let vCodec = try Self.parseVideoCodec(codec)
        let aCodec = try Self.parseAudioCodec(audioCodec)
        var config = VideoGenerationConfig(
            guidanceScale: guidanceScale ?? 5.0,
            seed: seed,
            codec: vCodec,
            includeAudio: includeAudio ?? descriptor.supportsAudio,
            teacacheThreshold: teacacheThreshold ?? 0.12
        )
        config.audioCodec = aCodec
        if let audioSampleRate { config.audioSampleRate = audioSampleRate }
        if let audioChannels { config.audioChannels = audioChannels }
        config.staWindow = staWindow
        config.previewEverySteps = previewEverySteps ?? 0
        let temporal = VideoTemporalSpec(
            frames: frames ?? descriptor.defaultFrames,
            fps: fps ?? descriptor.defaultFPS,
            steps: steps ?? descriptor.defaultSteps
        )
        var request = try VideoGenerationRequest(
            model: (model?.isEmpty == false ? model : nil) ?? descriptor.identifier,
            prompt: prompt ?? "",
            dimensions: dims,
            temporal: temporal,
            containerFormat: container,
            config: config
        )
        request.negativePrompt = negativePrompt
        request.stream = stream ?? false
        return request
    }

    private func resolveDimensions() throws(BasaltError) -> VideoDimensions {
        if let size, !size.isEmpty {
            return try VideoGenerationRequest.parseSize(size)
        }
        if let width, let height {
            return VideoDimensions(width: width, height: height)
        }
        return VideoDimensions()
    }

    static func parseContainer(_ raw: String?) throws(BasaltError) -> VideoContainerFormat {
        guard let raw, !raw.isEmpty else { return .mp4 }
        guard let fmt = VideoContainerFormat(rawValue: raw.lowercased()) else {
            throw .invalidRequestStructure(details: "video: format must be mp4 or mov")
        }
        return fmt
    }

    static func parseVideoCodec(_ raw: String?) throws(BasaltError) -> VideoCodecKind {
        guard let raw, !raw.isEmpty else { return .h264 }
        guard let parsed = VideoCodecKind(rawValue: raw.lowercased()) else {
            throw .invalidRequestStructure(details: "video: codec must be h264 or hevc")
        }
        return parsed
    }

    static func parseAudioCodec(_ raw: String?) throws(BasaltError) -> VideoAudioCodecKind {
        guard let raw, !raw.isEmpty else { return .aac }
        guard let parsed = VideoAudioCodecKind(rawValue: raw.lowercased()) else {
            throw .invalidRequestStructure(details: "video: audio_codec must be aac or pcm")
        }
        return parsed
    }
}
