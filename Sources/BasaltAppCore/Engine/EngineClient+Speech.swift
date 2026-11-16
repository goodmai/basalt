import BasaltCore
import Foundation

public struct AudioTranscriptionCall: Sendable, Equatable {
    public var wav: Data
    public var filename: String
    public var language: String?

    public init(wav: Data, filename: String = "voice.wav", language: String? = nil) {
        (self.wav, self.filename, self.language) = (wav, filename, language)
    }

    public func multipartBody(boundary: String) -> Data {
        var body = Data()
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\njson\r\n".utf8))
        if let language, !language.isEmpty {
            body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n\(language)\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\nContent-Type: audio/wav\r\n\r\n".utf8))
        body.append(wav)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        return body
    }
}

public struct SpeechSynthesisCall: Sendable, Equatable, Encodable {
    public var input: String
    public var voice: String
    public var speed: Double
    public var responseFormat: String
    public var model: String?
    public var language: String?

    public init(
        input: String,
        voice: String = "alloy",
        speed: Double = 1.0,
        responseFormat: String = "wav",
        model: String? = nil,
        language: String? = nil
    ) {
        (self.input, self.voice, self.speed, self.responseFormat, self.model, self.language) =
            (input, voice, speed, responseFormat, model, language)
    }

    enum CodingKeys: String, CodingKey {
        case input, voice, speed, model, language
        case responseFormat = "response_format"
    }
}

extension EngineClient {
    public func transcribeAudio(_ wav: Data, filename: String = "voice.wav", language: String? = nil) async throws -> String {
        try await transcribeAudio(AudioTranscriptionCall(wav: wav, filename: filename, language: language))
    }

    public func synthesizeSpeech(_ text: String, voice: String = "alloy", speed: Double = 1.0) async throws -> Data {
        try await synthesizeSpeech(SpeechSynthesisCall(input: text, voice: voice, speed: speed))
    }

    static func configureSpeech(_ client: inout EngineClient, _ endpoint: EngineEndpoint, session: URLSession) {
        client.transcribeAudio = { call in
            guard !call.wav.isEmpty else { throw EngineError.refused(status: 400, message: "audio recording is empty") }
            let boundary = "BasaltVoice-\(UUID().uuidString)"
            let body = call.multipartBody(boundary: boundary)
            var request = endpoint.request(
                "v1/audio/transcriptions", method: "POST", body: body,
                contentType: "multipart/form-data; boundary=\(boundary)")
            request.timeoutInterval = 180
            let (data, response) = try await session.data(for: request)
            try check(response, data)
            guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                  let text = json["text"] as? String else {
                throw EngineError.malformed("transcription response missing text")
            }
            return text
        }
        client.synthesizeSpeech = { call in
            let trimmed = call.input.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { throw EngineError.refused(status: 400, message: "speech input is empty") }
            let body = try JSONEncoder().encode(call)
            var request = endpoint.request("v1/audio/speech", method: "POST", body: body)
            request.timeoutInterval = 180
            let (data, response) = try await session.data(for: request)
            try check(response, data)
            return data
        }
    }
}
