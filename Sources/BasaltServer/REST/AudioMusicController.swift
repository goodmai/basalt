import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct AudioMusicController: Sendable {
    static let musicPath = "/v1/audio/music"
    static let modelsPath = "/v1/audio/music/models"
    static let bodyLimit = 256 * 1024

    private static let modelHeader = HTTPField.Name("X-Basalt-Music-Model")!
    private static let sampleRateHeader = HTTPField.Name("X-Basalt-Music-Sample-Rate")!

    let synthesizer: (any MusicSynthesizer)?
    var pipelines: PipelineRegistry? = nil

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.post("/audio/music", use: music)
        group.get("/audio/music/models", use: models)
    }

    private func resolveSynthesizer() async -> (any MusicSynthesizer)? {
        if let live = await pipelines?.system(ComposeSpec.defaultPipeline)?.music { return live }
        return synthesizer
    }

    @Sendable
    func models(request: Request, context: BasicRequestContext) async throws -> Response {
        let active = await resolveSynthesizer()
        let list = active.map { [$0.descriptor] } ?? [MusicModelDescriptor.aceStep15()]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var buffer = ByteBuffer()
        buffer.writeBytes(try encoder.encode(ModelsEnvelope(models: list)))
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }

    @Sendable
    func music(request: Request, context: BasicRequestContext) async throws -> Response {
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: Self.bodyLimit)
        } catch let error as any HTTPResponseError where error.status == .contentTooLarge {
            return ProviderErrorResponse.response(
                status: .contentTooLarge,
                message: "a music synthesis request is limited to \(Self.bodyLimit >> 10) KB",
                shape: .openAI
            )
        }
        do {
            let parsed = try Self.parseRequest(Data(body.readableBytesView))
            guard let active = await resolveSynthesizer() else {
                throw BasaltError.invalidRequestStructure(details: "music: no music synthesizer is configured on this server")
            }
            let started = ContinuousClock.now
            let stream = try await active.synthesize(parsed)
            return try await Self.makeResponse(stream: stream, request: parsed, active: active, started: started)
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.musicPath)
        }
    }

    static func parseRequest(_ data: Data) throws(BasaltError) -> MusicGenerationRequest {
        let wire: WireMusicRequest
        do {
            wire = try JSONDecoder().decode(WireMusicRequest.self, from: data)
        } catch {
            throw .invalidRequestStructure(details: "music: body must be valid JSON matching /v1/audio/music")
        }
        return try wire.toMusicRequest()
    }
}

extension AudioMusicController {
    struct ModelsEnvelope: Codable, Sendable, Equatable {
        let models: [MusicModelDescriptor]
    }

    struct WireMusicRequest: Decodable {
        var model: String?
        var prompt: String?
        var lyrics: String?
        var tags: [String]?
        var durationSeconds: Double?
        var duration: Double?
        var bpm: Double?
        var steps: Int?
        var guidanceScale: Double?
        var seed: UInt64?
        var responseFormat: String?
        var streamFormat: String?

        enum CodingKeys: String, CodingKey {
            case model, prompt, lyrics, tags, duration, bpm, steps, seed
            case durationSeconds = "duration_seconds"
            case guidanceScale = "guidance_scale"
            case responseFormat = "response_format"
            case streamFormat = "stream_format"
        }

        func toMusicRequest() throws(BasaltError) -> MusicGenerationRequest {
            let fmt = try Self.parseFormat(responseFormat)
            let str = try Self.parseStream(streamFormat)
            let opts = MusicGenerationOptions(
                lyrics: lyrics,
                tags: tags ?? [],
                bpm: bpm ?? 120.0,
                steps: steps ?? 8,
                guidanceScale: guidanceScale ?? 4.0,
                seed: seed
            )
            return try MusicGenerationRequest(
                model: (model?.isEmpty == false ? model : nil) ?? "ACE-Step/ACE-Step-1.5-3.5B",
                prompt: prompt ?? "",
                durationSeconds: durationSeconds ?? duration ?? 4.0,
                responseFormat: fmt,
                streamFormat: str,
                options: opts
            )
        }

        private static func parseFormat(_ raw: String?) throws(BasaltError) -> SpeechAudioFormat {
            guard let raw, !raw.isEmpty else { return .wav }
            guard let fmt = SpeechAudioFormat(rawValue: raw.lowercased()), fmt.isUncompressed else {
                throw .invalidRequestStructure(details: "music: response_format must be wav, pcm or pcm_24k")
            }
            return fmt
        }

        private static func parseStream(_ raw: String?) throws(BasaltError) -> SpeechStreamFormat {
            guard let raw, !raw.isEmpty else { return .audio }
            guard let fmt = SpeechStreamFormat(rawValue: raw.lowercased()) else {
                throw .invalidRequestStructure(details: "music: stream_format must be audio or sse")
            }
            return fmt
        }
    }

    struct MusicDeltaEvent: Encodable, Sendable {
        let type = "music.delta"
        let audio: String
        let sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case type, audio
            case sampleRate = "sample_rate"
        }
    }

    struct MusicDoneEvent: Encodable, Sendable {
        let type = "music.done"
        let receipt: MusicReceipt
    }
}

extension AudioMusicController {
    static func makeResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        request: MusicGenerationRequest,
        active: any MusicSynthesizer,
        started: ContinuousClock.Instant
    ) async throws -> Response {
        if request.streamFormat == .sse {
            return sseResponse(stream: stream, request: request, active: active, started: started)
        }
        if request.responseFormat == .wav {
            return try await wavResponse(stream: stream, active: active)
        }
        return pcmResponse(stream: stream, request: request, active: active)
    }

    private static func wavResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        active: any MusicSynthesizer
    ) async throws -> Response {
        var allSamples: [Float] = []
        var rate = active.descriptor.defaultSampleRate
        for try await chunk in stream {
            if chunk.sampleRate > 0 { rate = chunk.sampleRate }
            allSamples.append(contentsOf: chunk.samples)
        }
        let wav = WAVEncoder.encode(samples: allSamples, sampleRate: rate)
        var buffer = ByteBuffer()
        buffer.writeBytes(wav)
        let headers: HTTPFields = [
            .contentType: SpeechAudioFormat.wav.contentType,
            modelHeader: active.modelIdentifier,
            sampleRateHeader: String(rate),
        ]
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: buffer))
    }

    private static func pcmResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        request: MusicGenerationRequest,
        active: any MusicSynthesizer
    ) -> Response {
        let forces24k = request.responseFormat == .pcm24k
        let declaredRate = forces24k ? 24_000 : active.descriptor.defaultSampleRate
        let body = AsyncThrowingStream<ByteBuffer, any Error> { continuation in
            let task = Task {
                do {
                    for try await chunk in stream where !chunk.samples.isEmpty {
                        let outChunk = forces24k && chunk.sampleRate != 24_000 ? chunk.resampled(to: 24_000) : chunk
                        var buffer = ByteBuffer()
                        buffer.writeBytes(outChunk.encodePCM16LE())
                        continuation.yield(buffer)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        let headers: HTTPFields = [
            .contentType: SpeechAudioFormat.pcm.contentType,
            modelHeader: active.modelIdentifier,
            sampleRateHeader: String(declaredRate),
        ]
        return Response(status: .ok, headers: headers, body: ResponseBody(asyncSequence: body))
    }

    private static func sseResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        request: MusicGenerationRequest,
        active: any MusicSynthesizer,
        started: ContinuousClock.Instant
    ) -> Response {
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                await streamMusicSSE(stream: stream, request: request, active: active, started: started, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        let headers: HTTPFields = [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive"]
        return Response(status: .ok, headers: headers, body: ResponseBody(asyncSequence: body))
    }

    private static func streamMusicSSE(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        request: MusicGenerationRequest,
        active: any MusicSynthesizer,
        started: ContinuousClock.Instant,
        continuation: AsyncStream<ByteBuffer>.Continuation
    ) async {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let forces24k = request.responseFormat == .pcm24k
        var totalSamples = 0
        var rate = forces24k ? 24_000 : active.descriptor.defaultSampleRate
        do {
            for try await chunk in stream where !chunk.samples.isEmpty {
                let outChunk = forces24k && chunk.sampleRate != 24_000 ? chunk.resampled(to: 24_000) : chunk
                if outChunk.sampleRate > 0 { rate = outChunk.sampleRate }
                totalSamples += outChunk.samples.count
                let delta = MusicDeltaEvent(audio: outChunk.encodePCM16LE().base64EncodedString(), sampleRate: rate)
                if let json = try? encoder.encode(delta) {
                    continuation.yield(ByteBuffer(string: "event: music.delta\ndata: \(String(decoding: json, as: UTF8.self))\n\n"))
                }
            }
            let parts = started.duration(to: .now).components
            let elapsedMs = Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
            let receipt = MusicReceipt(
                model: active.modelIdentifier,
                sampleRate: rate,
                durationSeconds: Double(totalSamples) / Double(max(1, rate)),
                stepsExecuted: request.options.steps,
                elapsedMs: elapsedMs,
                seed: request.options.seed ?? 0
            )
            if let json = try? encoder.encode(MusicDoneEvent(receipt: receipt)) {
                continuation.yield(ByteBuffer(string: "event: music.done\ndata: \(String(decoding: json, as: UTF8.self))\n\n"))
            }
        } catch {
            let message = ModelsController.publicMessage(for: error)
            if let json = try? JSONSerialization.data(withJSONObject: ["type": "error", "error": ["message": message]]) {
                continuation.yield(ByteBuffer(string: "event: error\ndata: \(String(decoding: json, as: UTF8.self))\n\n"))
            }
        }
        continuation.finish()
    }
}
