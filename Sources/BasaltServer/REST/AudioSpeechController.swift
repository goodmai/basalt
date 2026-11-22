import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct AudioSpeechController: Sendable {
    static let speechPath = "/v1/audio/speech"
    static let voicesPath = "/v1/audio/voices"
    static let bodyLimit = 256 * 1024

    private static let modelHeader = HTTPField.Name("X-Basalt-Speech-Model")!
    private static let computeHeader = HTTPField.Name("X-Basalt-Speech-Compute")!
    private static let fallbackHeader = HTTPField.Name("X-Basalt-Speech-Fallback-From")!
    private static let sampleRateHeader = HTTPField.Name("X-Basalt-Speech-Sample-Rate")!

    let synthesizer: (any SpeechSynthesizer)?
    var pipelines: PipelineRegistry? = nil
    var allowedReferenceDirectories: [URL] = []

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.post("/audio/speech", use: speech)
        group.get("/audio/voices", use: voices)
    }

    private func resolveSynthesizer() async -> (any SpeechSynthesizer)? {
        if let live = await pipelines?.system(ComposeSpec.defaultPipeline)?.speech { return live }
        return synthesizer
    }

    @Sendable
    func voices(request: Request, context: BasicRequestContext) async throws -> Response {
        let active = await resolveSynthesizer()
        let items = if let active { await active.voices } else { [VoiceSpec]() }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var buffer = ByteBuffer()
        buffer.writeBytes(try encoder.encode(VoicesEnvelope(voices: items)))
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }

    @Sendable
    func speech(request: Request, context: BasicRequestContext) async throws -> Response {
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: Self.bodyLimit)
        } catch let error as any HTTPResponseError where error.status == .contentTooLarge {
            return ProviderErrorResponse.response(
                status: .contentTooLarge,
                message: "a speech synthesis request is limited to \(Self.bodyLimit >> 10) KB",
                shape: .openAI
            )
        }
        do {
            let speechRequest = try Self.parseRequest(Data(body.readableBytesView), allowedReferenceDirectories: allowedReferenceDirectories)
            guard let active = await resolveSynthesizer() else {
                throw BasaltError.invalidRequestStructure(details: "speech: no speech synthesizer is configured on this server")
            }
            let started = ContinuousClock.now
            let decision = await active.resolveRoute(speechRequest)
            let stream = try await active.synthesize(speechRequest)
            return try await Self.makeResponse(stream: stream, request: speechRequest, decision: decision, started: started)
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.speechPath)
        }
    }

    static func parseRequest(_ data: Data, allowedReferenceDirectories: [URL] = []) throws(BasaltError) -> SpeechRequest {
        let wire: WireSpeechRequest
        do {
            wire = try JSONDecoder().decode(WireSpeechRequest.self, from: data)
        } catch {
            throw .invalidRequestStructure(details: "speech: body must be valid JSON matching /v1/audio/speech")
        }
        return try wire.toSpeechRequest(allowedReferenceDirectories: allowedReferenceDirectories)
    }
}

extension AudioSpeechController {
    struct VoicesEnvelope: Codable, Sendable, Equatable {
        let voices: [VoiceSpec]
    }

    struct WireSpeechRequest: Decodable {
        var model: String?
        var input: String?
        var voice: String?
        var responseFormat: String?
        var speed: Double?
        var streamFormat: String?
        var language: String?
        var instruct: String?
        var exaggeration: Double?
        var preferANE: Bool?
        var referenceAudio: String?

        enum CodingKeys: String, CodingKey {
            case model, input, voice, speed, language, instruct, exaggeration
            case responseFormat = "response_format"
            case streamFormat = "stream_format"
            case preferANE = "prefer_ane"
            case referenceAudio = "reference_audio"
        }

        func toSpeechRequest(allowedReferenceDirectories: [URL] = []) throws(BasaltError) -> SpeechRequest {
            guard let input else {
                throw .invalidRequestStructure(details: "speech: input text is required")
            }
            let resolvedRef: String?
            if let referenceAudio {
                let clean = referenceAudio.trimmingCharacters(in: .whitespacesAndNewlines)
                if clean.isEmpty {
                    resolvedRef = nil
                } else {
                    let target = MCPImages.resolved(URL(fileURLWithPath: clean))
                    guard allowedReferenceDirectories.contains(where: { MCPImages.contains(MCPImages.resolved($0), target) }) else {
                        throw .invalidRequestStructure(
                            details: "speech: client reference_audio paths are disabled on /v1/audio/speech unless inside an allowed reference directory; configure pipeline.speak.reference_audio on the server")
                    }
                    resolvedRef = target.path
                }
            } else {
                resolvedRef = nil
            }
            let format = try Self.parseFormat(responseFormat)
            let stream = try Self.parseStream(streamFormat)
            var request = try SpeechRequest(
                model: model ?? ComposeSpec.appleSpeech,
                input: input,
                voice: voice ?? "alloy",
                responseFormat: format,
                speed: speed ?? 1.0,
                streamFormat: stream
            )
            try request.configure(
                language: language,
                instruct: instruct,
                exaggeration: exaggeration,
                preferANE: preferANE ?? false,
                referenceAudio: resolvedRef
            )
            return request
        }

        private static func parseFormat(_ raw: String?) throws(BasaltError) -> SpeechAudioFormat {
            guard let raw, !raw.isEmpty else { return .wav }
            guard let format = SpeechAudioFormat(rawValue: raw) else {
                throw .invalidRequestStructure(details: "speech: response_format must be wav, pcm, pcm_24k, mp3, opus, aac or flac")
            }
            guard format.isUncompressed else {
                throw .invalidRequestStructure(details: "speech: compressed container '\(format.rawValue)' is not configured; use wav, pcm or pcm_24k")
            }
            return format
        }

        private static func parseStream(_ raw: String?) throws(BasaltError) -> SpeechStreamFormat {
            guard let raw, !raw.isEmpty else { return .audio }
            guard let stream = SpeechStreamFormat(rawValue: raw) else {
                throw .invalidRequestStructure(details: "speech: stream_format must be audio or sse")
            }
            return stream
        }
    }

    struct AudioDeltaEvent: Encodable, Sendable {
        let type = "audio.delta"
        let audio: String
        let delta: String
        let sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case type, audio, delta
            case sampleRate = "sample_rate"
        }
    }

    struct AudioDoneEvent: Encodable, Sendable {
        let type = "audio.done"
        let sampleRate: Int
        let durationSeconds: Double
        let receipt: SpeechReceipt

        enum CodingKeys: String, CodingKey {
            case type, receipt
            case sampleRate = "sample_rate"
            case durationSeconds = "duration_seconds"
        }
    }
}

extension AudioSpeechController {
    static func makeResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        request: SpeechRequest,
        decision: SpeechRouteDecision,
        started: ContinuousClock.Instant
    ) async throws -> Response {
        if request.streamFormat == .sse {
            return sseResponse(stream: stream, request: request, decision: decision, started: started)
        }
        if request.responseFormat == .wav {
            return try await wavResponse(stream: stream, decision: decision, started: started)
        }
        return pcmResponse(stream: stream, request: request, decision: decision)
    }

    private static func wavResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        decision: SpeechRouteDecision,
        started: ContinuousClock.Instant
    ) async throws -> Response {
        var allSamples: [Float] = []
        var rate = SpeechRequest.defaultSampleRate
        var ttfaMs: Double?
        for try await chunk in stream {
            if chunk.sampleRate > 0 { rate = chunk.sampleRate }
            if !chunk.samples.isEmpty {
                if ttfaMs == nil { ttfaMs = milliseconds(from: started, to: .now) }
                allSamples.append(contentsOf: chunk.samples)
            }
        }
        let elapsedMs = milliseconds(from: started, to: .now)
        let audioMs = Double(allSamples.count) / Double(max(1, rate)) * 1_000
        let rtf = audioMs > 0 ? elapsedMs / audioMs : 0
        let receipt = decision.receipt(ttfaMs: ttfaMs ?? elapsedMs, audioMs: audioMs, rtf: rtf)
        let wav = WAVEncoder.encode(samples: allSamples, sampleRate: rate)
        var buffer = ByteBuffer()
        buffer.writeBytes(wav)
        return Response(status: .ok, headers: audioHeaders(contentType: SpeechAudioFormat.wav.contentType, rate: rate, receipt: receipt), body: ResponseBody(byteBuffer: buffer))
    }

    private static func pcmResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        request: SpeechRequest,
        decision: SpeechRouteDecision
    ) -> Response {
        let forces24k = request.responseFormat == .pcm24k
        let declaredRate = forces24k
            ? SpeechRequest.defaultSampleRate
            : (TTSModelFamily.detect(identifier: decision.selected.model)?.defaultSampleRate ?? SpeechRequest.defaultSampleRate)
        let body = AsyncThrowingStream<ByteBuffer, any Error> { continuation in
            let task = Task {
                do {
                    for try await chunk in stream where !chunk.samples.isEmpty {
                        let outChunk = forces24k && chunk.sampleRate != SpeechRequest.defaultSampleRate
                            ? chunk.resampled(to: SpeechRequest.defaultSampleRate)
                            : chunk
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
        var headers: HTTPFields = [
            .contentType: SpeechAudioFormat.pcm.contentType,
            modelHeader: decision.selected.model,
            computeHeader: decision.selected.compute.rawValue,
            sampleRateHeader: String(declaredRate),
        ]
        if let fallback = decision.fallbackFrom { headers[fallbackHeader] = fallback }
        return Response(status: .ok, headers: headers, body: ResponseBody(asyncSequence: body))
    }

    private static func sseResponse(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        request: SpeechRequest,
        decision: SpeechRouteDecision,
        started: ContinuousClock.Instant
    ) -> Response {
        let forces24k = request.responseFormat == .pcm24k
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                await streamSSE(stream: stream, forces24k: forces24k, decision: decision, started: started, continuation: continuation)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        let headers: HTTPFields = [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive"]
        return Response(status: .ok, headers: headers, body: ResponseBody(asyncSequence: body))
    }

    private static func streamSSE(
        stream: AsyncThrowingStream<AudioChunk, any Error>,
        forces24k: Bool,
        decision: SpeechRouteDecision,
        started: ContinuousClock.Instant,
        continuation: AsyncStream<ByteBuffer>.Continuation
    ) async {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var totalSamples = 0
        var rate = SpeechRequest.defaultSampleRate
        var ttfaMs: Double?
        do {
            for try await rawChunk in stream {
                let chunk = forces24k && rawChunk.sampleRate != SpeechRequest.defaultSampleRate && !rawChunk.samples.isEmpty
                    ? rawChunk.resampled(to: SpeechRequest.defaultSampleRate)
                    : rawChunk
                if chunk.sampleRate > 0 { rate = chunk.sampleRate }
                guard !chunk.samples.isEmpty else { continue }
                if ttfaMs == nil { ttfaMs = milliseconds(from: started, to: .now) }
                totalSamples += chunk.samples.count
                let encoded = chunk.encodePCM16LE().base64EncodedString()
                let delta = AudioDeltaEvent(audio: encoded, delta: encoded, sampleRate: rate)
                if let json = try? encoder.encode(delta) {
                    continuation.yield(ByteBuffer(string: "event: audio.delta\ndata: \(String(decoding: json, as: UTF8.self))\n\n"))
                }
            }
            let elapsedMs = milliseconds(from: started, to: .now)
            let audioMs = Double(totalSamples) / Double(max(1, rate)) * 1_000
            let receipt = decision.receipt(ttfaMs: ttfaMs ?? elapsedMs, audioMs: audioMs, rtf: audioMs > 0 ? elapsedMs / audioMs : 0)
            let done = AudioDoneEvent(sampleRate: rate, durationSeconds: audioMs / 1_000, receipt: receipt)
            if let json = try? encoder.encode(done) {
                continuation.yield(ByteBuffer(string: "event: audio.done\ndata: \(String(decoding: json, as: UTF8.self))\n\n"))
            }
        } catch {
            let message = ModelsController.publicMessage(for: error)
            if let json = try? JSONSerialization.data(withJSONObject: ["type": "error", "error": ["message": message]]) {
                continuation.yield(ByteBuffer(string: "event: error\ndata: \(String(decoding: json, as: UTF8.self))\n\n"))
            }
        }
        continuation.finish()
    }

    private static func audioHeaders(contentType: String, rate: Int, receipt: SpeechReceipt) -> HTTPFields {
        var headers: HTTPFields = [
            .contentType: contentType,
            modelHeader: receipt.model,
            computeHeader: receipt.compute.rawValue,
            sampleRateHeader: String(rate),
        ]
        if let fallback = receipt.fallbackFrom { headers[fallbackHeader] = fallback }
        return headers
    }

    private static func milliseconds(from start: ContinuousClock.Instant, to end: ContinuousClock.Instant) -> Double {
        let parts = start.duration(to: end).components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }
}
