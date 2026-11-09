import Testing
import Foundation
import AVFoundation
import Synchronization
import Hummingbird
import HummingbirdTesting
import NIOCore
import HTTPTypes
@testable import BasaltCore
@testable import BasaltServer

final class ScriptedTranscriber: AudioTranscriber {
    let requests = Mutex<[TranscriptionRequest]>([])
    let result = Transcription(
        text: "hello world", language: "en",
        segments: [.init(start: 0, end: 0.5, text: "hello"), .init(start: 0.5, end: 1, text: " world")])

    func transcribe(_ request: TranscriptionRequest) async throws(BasaltError) -> Transcription {
        requests.withLock { $0.append(request) }
        return result
    }
}

enum AudioFixture {
    static func wav(rate: Int, channels: Int, bits: Int, payload: Data) -> Data {
        func le<T: FixedWidthInteger>(_ value: T) -> Data { withUnsafeBytes(of: value.littleEndian) { Data($0) } }
        let blockAlign = channels * bits / 8
        var data = Data("RIFF".utf8) + le(UInt32(36 + payload.count)) + Data("WAVEfmt ".utf8)
        data += le(UInt32(16)) + le(UInt16(1)) + le(UInt16(channels)) + le(UInt32(rate))
        data += le(UInt32(rate * blockAlign)) + le(UInt16(blockAlign)) + le(UInt16(bits))
        return data + Data("data".utf8) + le(UInt32(payload.count)) + payload
    }

    static func sine(seconds: Double = 1, rate: Int = 44_100, channels: Int = 2, amplitude: Double = 0.5) -> Data {
        var payload = Data()
        for frame in 0 ..< Int(seconds * Double(rate)) {
            let value = Int16(amplitude * Double(Int16.max) * sin(2 * .pi * 440 * Double(frame) / Double(rate)))
            for _ in 0 ..< channels { withUnsafeBytes(of: value.littleEndian) { payload.append(contentsOf: $0) } }
        }
        return wav(rate: rate, channels: channels, bits: 16, payload: payload)
    }

    struct File {
        let name: String
        let type: String?
        let data: Data
    }

    static func form(_ fields: [(String, String)], file: File? = File(name: "clip.wav", type: "audio/wav", data: sine()),
                     boundary: String = "XyZ-boundary") -> Data {
        var body = Data()
        for (name, value) in fields {
            body += Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".utf8)
        }
        if let file {
            let type = file.type.map { "Content-Type: \($0)\r\n" } ?? ""
            body += Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(file.name)\"\r\n\(type)\r\n".utf8)
            body += file.data + Data("\r\n".utf8)
        }
        return body + Data("--\(boundary)--\r\n".utf8)
    }
}

@Suite("Speech to text: multipart parsing and audio decoding (#230)")
struct TranscriptionCoreTests {
    @Test("A multipart form yields each field by name, with a quoted file name that holds a semicolon and binary content kept intact")
    func parsesForm() throws {
        let binary = Data([0, 13, 10, 45, 45, 255])
        let body = AudioFixture.form([("model", "nemotron"), ("language", " en ")], file: .init(name: "a;b.wav", type: "audio/wav", data: binary))
        let form = try MultipartForm(body, boundary: "XyZ-boundary")
        #expect(try form.text("model") == "nemotron")
        #expect(try form.text("language") == "en")
        #expect(form.parts["file"]?.filename == "a;b.wav")
        #expect(form.parts["file"]?.contentType == "audio/wav")
        #expect(form.parts["file"]?.body == binary)
    }

    @Test("Malformed forms are refused: a field twice, no closing boundary, too many fields, an empty or multi-line boundary")
    func refusesMalformedForms() {
        let twice = AudioFixture.form([("model", "a"), ("model", "b")])
        #expect(throws: BasaltError.self) { try MultipartForm(twice, boundary: "XyZ-boundary") }
        let unclosed = Data("--XyZ-boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nvalue".utf8)
        #expect(throws: BasaltError.self) { try MultipartForm(unclosed, boundary: "XyZ-boundary") }
        let many = AudioFixture.form((0 ... MultipartForm.maxParts).map { ("f\($0)", "v") }, file: nil)
        #expect(throws: BasaltError.self) { try MultipartForm(many, boundary: "XyZ-boundary") }
        #expect(throws: BasaltError.self) { try MultipartForm(AudioFixture.form([]), boundary: "") }
        #expect(throws: BasaltError.self) { try MultipartForm(AudioFixture.form([]), boundary: "a\r\nb") }
    }

    @Test("Upload file names with path separators, dot names or control characters are refused; plain names pass")
    func fileNames() {
        for bad in ["../etc/passwd.wav", "a/b.wav", "a\\b.wav", "..", "", "clip\u{0}.wav", "clip\n.wav", String(repeating: "a", count: 256) + ".wav"] {
            #expect(!TranscriptionsController.isPlainFileName(bad), "\(bad.debugDescription)")
        }
        for good in ["meeting 1.wav", "запись.mp3", "a..b.flac"] {
            #expect(TranscriptionsController.isPlainFileName(good), "\(good)")
        }
    }

    @Test("A 44.1 kHz stereo WAV decodes to 16 kHz mono floats with its amplitude kept")
    func decodesAndResamples() throws {
        let clip = try AudioClip(encoded: AudioFixture.sine(), format: "WAV")
        #expect(abs(clip.samples.count - 16_000) <= 32)
        let peak = clip.samples.map(abs).max() ?? 0
        #expect(peak > 0.4 && peak < 0.6)
        #expect(abs(clip.seconds - 1) < 0.01)
    }

    @Test("Audio over 30 minutes is refused from its header, before it is decoded")
    func refusesLongAudio() {
        let silence = AudioFixture.wav(rate: 8_000, channels: 1, bits: 8, payload: Data(repeating: 0x80, count: 8_000 * 1_801))
        #expect(silence.count < AudioClip.maxBytes)
        #expect(throws: BasaltError.self) { try AudioClip(encoded: silence, format: "wav") }
    }

    @Test("Bytes that are not audio, an empty file, a file over 25 MB and an unlisted format are refused")
    func refusesBadAudio() {
        #expect(throws: BasaltError.self) { try AudioClip(encoded: Data("<html>not audio</html>".utf8), format: "wav") }
        #expect(throws: BasaltError.self) { try AudioClip(encoded: Data(), format: "wav") }
        #expect(throws: BasaltError.self) { try AudioClip(encoded: Data(count: AudioClip.maxBytes + 1), format: "wav") }
        #expect(throws: BasaltError.self) { try AudioClip(encoded: AudioFixture.sine(), format: "ogg") }
    }

    @Test("A request takes ISO-639-1 languages and temperatures in 0...1, and nothing else")
    func requestValidation() throws {
        let clip = try AudioClip(samples: [0.1, 0.2])
        #expect(try TranscriptionRequest(audio: clip, language: "en-US", temperature: 0.4).language == "en-us")
        #expect(try TranscriptionRequest(audio: clip).temperature == 0)
        #expect(throws: BasaltError.self) { try TranscriptionRequest(audio: clip, language: "../en") }
        #expect(throws: BasaltError.self) { try TranscriptionRequest(audio: clip, temperature: 1.5) }
        #expect(throws: BasaltError.self) { try AudioClip(samples: []) }
        #expect(throws: BasaltError.self) { try AudioClip(samples: [0.1, .nan]) }
        #expect(throws: BasaltError.self) { try AudioClip(samples: [.infinity]) }
    }

    @Test("A clip converts back into an AVAudioPCMBuffer in another format, as the SpeechAnalyzer fallback feeds it")
    func convertsToTargetFormat() throws {
        let clip = try AudioClip(encoded: AudioFixture.sine(), format: "wav")
        let target = try #require(AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 8_000, channels: 1, interleaved: true))
        let buffer = try clip.buffer(in: target)
        #expect(abs(Int(buffer.frameLength) - 8_000) <= 32)
    }
}

@Suite("Speech to text: /v1/audio/transcriptions and basalt_transcribe (#230)")
struct TranscriptionAPITests {
    private func app(engine: MockInferenceEngine = MockInferenceEngine(), fallback: (any AudioTranscriber)? = nil) async throws
        -> Application<RouterResponder<BasicRequestContext>> {
        let orchestrator = ModelOrchestratorActor(engine: engine, telemetry: TelemetryTracker())
        await orchestrator.setSpeechFallback(fallback)
        return Application(router: RESTServer(orchestrator: orchestrator, config: try ServerConfig(modelPath: "org/mock-model")).makeRouter())
    }

    private func post(_ client: some TestClientProtocol, _ body: Data, contentType: String = "multipart/form-data; boundary=XyZ-boundary")
        async throws -> (status: HTTPResponse.Status, type: String?, body: String) {
        try await client.execute(uri: "/v1/audio/transcriptions", method: .post, headers: [.contentType: contentType], body: ByteBuffer(bytes: body)) {
            ($0.status, $0.headers[.contentType], String(buffer: $0.body))
        }
    }

    private func json(_ text: String) -> [String: Any] {
        (try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]) ?? [:]
    }

    @Test("A model without an audio encoder refuses with 400 in the OpenAI error shape")
    func refusesWithoutEncoder() async throws {
        try await app().test(.router) { (client) async throws in
            let response = try await post(client, AudioFixture.form([("model", "whisper-1")]))
            #expect(response.status == .badRequest)
            let error = json(response.body)["error"] as? [String: Any]
            #expect((error?["message"] as? String)?.contains("no audio encoder") == true)
        }
    }

    @Test("json, text and verbose_json come back in the OpenAI shapes, and the engine sees 16 kHz audio, the language and the temperature")
    func responseFormats() async throws {
        let transcriber = ScriptedTranscriber()
        try await app(fallback: transcriber).test(.router) { (client) async throws in
            let plain = try await post(client, AudioFixture.form([("model", "nemotron")]))
            #expect(plain.status == .ok)
            #expect(json(plain.body)["text"] as? String == "hello world")
            let text = try await post(client, AudioFixture.form([("response_format", "text")]))
            #expect(text.body == "hello world")
            #expect(text.type?.hasPrefix("text/plain") == true)
            let verbose = json(try await post(client, AudioFixture.form([("response_format", "verbose_json"), ("language", "en"), ("temperature", "0.2")])).body)
            #expect(verbose["task"] as? String == "transcribe")
            #expect(verbose["language"] as? String == "en")
            #expect(abs((verbose["duration"] as? Double ?? 0) - 1) < 0.01)
            let segments = verbose["segments"] as? [[String: Any]]
            #expect(segments?.count == 2)
            #expect(segments?.last?["id"] as? Int == 1)
            #expect(segments?.last?["text"] as? String == " world")
        }
        let last = try #require(transcriber.requests.withLock { $0.last })
        #expect(last.language == "en" && last.temperature == 0.2)
        #expect(abs(last.audio.samples.count - 16_000) <= 32)
    }

    @Test("A model with an audio encoder transcribes itself, ahead of the fallback")
    func engineAheadOfFallback() async throws {
        let engine = MockInferenceEngine()
        await engine.setMockTranscription(Transcription(text: "from the model"))
        let fallback = ScriptedTranscriber()
        try await app(engine: engine, fallback: fallback).test(.router) { (client) async throws in
            #expect(json(try await post(client, AudioFixture.form([])).body)["text"] as? String == "from the model")
        }
        #expect(fallback.requests.withLock { $0.isEmpty })
    }

    @Test("Bad uploads are refused before the engine: wrong content type 415, no file, srt, temperature 2, a text file, a traversal file name")
    func refusesBadUploads() async throws {
        let transcriber = ScriptedTranscriber()
        try await app(fallback: transcriber).test(.router) { (client) async throws in
            #expect(try await post(client, Data("{}".utf8), contentType: "application/json").status == .unsupportedMediaType)
            #expect(try await post(client, AudioFixture.form([("model", "x")], file: nil)).status == .badRequest)
            #expect(try await post(client, AudioFixture.form([("response_format", "srt")])).status == .badRequest)
            #expect(try await post(client, AudioFixture.form([("temperature", "2")])).status == .badRequest)
            #expect(try await post(client, AudioFixture.form([], file: .init(name: "a.wav", type: "text/html", data: AudioFixture.sine()))).status == .badRequest)
            #expect(try await post(client, AudioFixture.form([], file: .init(name: "../../a.wav", type: nil, data: AudioFixture.sine()))).status == .badRequest)
            #expect(try await post(client, AudioFixture.form([], file: .init(name: "noextension", type: nil, data: AudioFixture.sine()))).status == .badRequest)
        }
        #expect(transcriber.requests.withLock { $0.isEmpty })
    }

    @Test("A body over the 25 MB upload limit is refused with 413")
    func refusesLargeBody() async throws {
        try await app(fallback: ScriptedTranscriber()).test(.router) { (client) async throws in
            let big = AudioFixture.form([], file: .init(name: "a.wav", type: "audio/wav", data: Data(count: TranscriptionsController.bodyLimit)))
            #expect(try await post(client, big).status == .contentTooLarge)
        }
    }

    @Test("Other POST routes still require JSON; multipart is only taken on the transcription route")
    func multipartOnlyOnTranscriptions() async throws {
        try await app().test(.router) { (client) async throws in
            let status = try await client.execute(
                uri: "/v1/chat/completions", method: .post,
                headers: [.contentType: "multipart/form-data; boundary=XyZ-boundary"], body: ByteBuffer(bytes: AudioFixture.form([]))) { $0.status }
            #expect(status == .unsupportedMediaType)
        }
    }

    @Test("The basalt_transcribe MCP tool takes base64 audio and returns the text; tools/list names it")
    func mcpTool() async throws {
        let orchestrator = ModelOrchestratorActor(engine: MockInferenceEngine(), telemetry: TelemetryTracker())
        await orchestrator.setSpeechFallback(ScriptedTranscriber())
        let capturer = OutputCapturer()
        let server = MCPServer(orchestrator: orchestrator) { capturer.capture($0) }
        #expect(MCPTool(name: "basalt_transcribe") == .media(.transcribe))
        await server.dispatch(#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#)
        #expect(capturer.lines.last?.contains("basalt_transcribe") == true)
        let audio = AudioFixture.sine().base64EncodedString()
        await server.dispatch(#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"basalt_transcribe","arguments":{"audio":"\#(audio)","format":"wav"}}}"#)
        #expect(capturer.lines.last?.contains("hello world") == true)
        await server.dispatch(#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"basalt_transcribe","arguments":{"audio":"AAAA","format":"exe"}}}"#)
        #expect(capturer.lines.last?.contains("\"error\"") == true)
    }
}
