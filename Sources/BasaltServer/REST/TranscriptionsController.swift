import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct TranscriptionsController: Sendable {
    static let path = "/v1/audio/transcriptions"
    static let bodyLimit = AudioClip.maxBytes + 64 * 1024
    static let fileTypes: Set<String> = ["application/octet-stream", "video/mp4", "video/mpeg"]

    let orchestrator: ModelOrchestratorActor

    enum ResponseFormat: String {
        case json, text
        case verboseJSON = "verbose_json"
    }

    struct Upload {
        let request: TranscriptionRequest
        let format: ResponseFormat
    }

    struct Verbose: Encodable {
        struct Segment: Encodable {
            let id: Int
            let start: Double
            let end: Double
            let text: String
        }

        let task = "transcribe"
        let language: String?
        let duration: Double
        let text: String
        let segments: [Segment]
    }

    @Sendable
    func transcriptions(request: Request, context: BasicRequestContext) async throws -> Response {
        let body: ByteBuffer
        do {
            body = try await request.body.collect(upTo: Self.bodyLimit)
        } catch let error as any HTTPResponseError where error.status == .contentTooLarge {
            return ProviderErrorResponse.response(
                status: .contentTooLarge, message: "an audio upload is limited to \(AudioClip.maxBytes >> 20) MB", shape: .openAI)
        }
        do {
            let upload = try Self.upload(Data(body.readableBytesView), contentType: request.headers[.contentType])
            return try Self.response(try await orchestrator.transcribe(upload.request), for: upload)
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.path)
        }
    }

    static func upload(_ body: Data, contentType: String?) throws(BasaltError) -> Upload {
        guard let type = contentType.flatMap(MediaType.init(from:)), type.isType(.multipartForm),
              let parameter = type.parameter, parameter.name.lowercased() == "boundary" else {
            throw .invalidRequestStructure(details: "send the audio as multipart/form-data with a boundary")
        }
        let form = try MultipartForm(body, boundary: parameter.value)
        guard let file = form.parts["file"] else { throw .invalidRequestStructure(details: "the form needs a file field with the audio") }
        if let model = try form.text("model"), model.count > 256 {
            throw .invalidRequestStructure(details: "model must be at most 256 characters")
        }
        let format = try responseFormat(form.text("response_format"))
        let temperature = try form.text("temperature").map { text throws(BasaltError) in
            guard let value = Double(text) else { throw .invalidRequestStructure(details: "temperature must be a number") }
            return value
        }
        let language = try form.text("language").flatMap { $0.isEmpty ? nil : $0 }
        let request = try TranscriptionRequest(audio: try clip(file), language: language, temperature: temperature)
        return Upload(request: request, format: format)
    }

    static func clip(_ file: MultipartForm.Part) throws(BasaltError) -> AudioClip {
        if let type = file.contentType?.split(separator: ";").first.map(String.init), !type.hasPrefix("audio/"), !fileTypes.contains(type) {
            throw .invalidRequestStructure(details: "the file has content type \(type.prefix(64)); send audio/*, video/mp4 or application/octet-stream")
        }
        guard let name = file.filename, isPlainFileName(name) else {
            throw .invalidRequestStructure(details: "the file needs a plain file name (no path separators or control characters) of at most 255 bytes")
        }
        guard let dot = name.lastIndex(of: "."), dot != name.startIndex else {
            throw .invalidRequestStructure(details: "the file name needs an extension that names the audio format, such as .wav")
        }
        return try AudioClip(encoded: file.body, format: String(name[name.index(after: dot)...]))
    }

    static func isPlainFileName(_ name: String) -> Bool {
        !name.isEmpty && name.utf8.count <= 255 && name != "." && name != ".."
            && !name.unicodeScalars.contains { $0 == "/" || $0 == "\\" || $0.properties.generalCategory == .control }
    }

    static func responseFormat(_ text: String?) throws(BasaltError) -> ResponseFormat {
        guard let text, !text.isEmpty else { return .json }
        if let format = ResponseFormat(rawValue: text) { return format }
        guard text == "srt" || text == "vtt" else {
            throw .invalidRequestStructure(details: "response_format must be json, text or verbose_json")
        }
        throw .invalidRequestStructure(details: "response_format \(text) needs word timestamps, which basalt does not produce yet; use json, text or verbose_json")
    }

    static func response(_ result: Transcription, for upload: Upload) throws -> Response {
        var buffer = ByteBuffer()
        switch upload.format {
        case .text:
            buffer.writeString(result.text)
            return Response(status: .ok, headers: [.contentType: "text/plain; charset=utf-8"], body: ResponseBody(byteBuffer: buffer))
        case .json:
            buffer.writeBytes(try JSONEncoder().encode(["text": result.text]))
        case .verboseJSON:
            let segments = result.segments.enumerated().map { Verbose.Segment(id: $0, start: $1.start, end: $1.end, text: $1.text) }
            buffer.writeBytes(try JSONEncoder().encode(Verbose(
                language: result.language ?? upload.request.language, duration: upload.request.audio.seconds,
                text: result.text, segments: segments)))
        }
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }
}
