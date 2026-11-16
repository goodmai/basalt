import Foundation

public enum ImageSSEOutcome: Sendable, Equatable {
    case done
    case partial(step: Int, totalSteps: Int, png: Data, width: Int, height: Int)
    case completed(GeneratedImagePart, inlineBytes: Data?, contentURL: String?, elapsedMs: Int)
    case ignored
}

public enum ImageSSEDecoder {
    private struct Envelope: Decodable {
        var type: String
        var step: Int?
        var totalSteps: Int?
        var assetID: String?
        var url: String?
        var b64JSON: String?
        var revisedPrompt: String?
        var width: Int?
        var height: Int?
        var hasAlpha: Bool?
        var seed: UInt64?
        var steps: Int?
        var teacacheSkipped: Int?
        var model: String?
        var licenseTier: String?
        var elapsedMs: Int?
        var boxes: [ChatImageBox]?
        var message: String?

        enum CodingKeys: String, CodingKey {
            case type, step, url, width, height, seed, steps, model, boxes, message
            case totalSteps = "total_steps"
            case assetID = "asset_id"
            case b64JSON = "b64_json"
            case revisedPrompt = "revised_prompt"
            case hasAlpha = "has_alpha"
            case teacacheSkipped = "teacache_skipped"
            case licenseTier = "license_tier"
            case elapsedMs = "elapsed_ms"
        }
    }

    public static func decode(_ event: SSEEvent, call: ImageGenerationCall) throws(EngineError) -> ImageSSEOutcome {
        let trimmed = event.data.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == "[DONE]" { return .done }
        guard !trimmed.isEmpty else { return .ignored }
        let envelope: Envelope
        do {
            envelope = try JSONDecoder().decode(Envelope.self, from: Data(trimmed.utf8))
        } catch {
            throw .malformed("image SSE JSON: \(error)")
        }
        switch envelope.type {
        case "image_generation.partial_image":
            return try decodePartial(envelope, fallbackSteps: call.steps)
        case "image_generation.completed":
            return try decodeCompleted(envelope, call: call)
        case "error":
            throw .refused(status: 500, message: envelope.message ?? "image generation failed")
        default:
            return .ignored
        }
    }

    private static func decodePartial(_ envelope: Envelope, fallbackSteps: Int) throws(EngineError) -> ImageSSEOutcome {
        guard let b64 = envelope.b64JSON, let png = Data(base64Encoded: b64) else {
            throw .malformed("partial image missing b64_json")
        }
        return .partial(
            step: envelope.step ?? 0,
            totalSteps: envelope.totalSteps ?? fallbackSteps,
            png: png,
            width: envelope.width ?? 256,
            height: envelope.height ?? 256
        )
    }

    private static func decodeCompleted(_ envelope: Envelope, call: ImageGenerationCall) throws(EngineError) -> ImageSSEOutcome {
        let totalSteps = envelope.steps ?? envelope.totalSteps ?? call.steps
        var part = GeneratedImagePart(
            assetID: envelope.assetID ?? "",
            prompt: call.prompt,
            model: envelope.model ?? call.model ?? "z-image-turbo",
            width: envelope.width ?? 1024,
            height: envelope.height ?? 1024,
            totalSteps: totalSteps
        )
        part.beforeFilePath = call.beforeFilePath
        part.hasAlpha = envelope.hasAlpha ?? false
        part.revisedPrompt = envelope.revisedPrompt
        part.seed = envelope.seed
        part.currentStep = totalSteps
        part.teacacheSkipped = envelope.teacacheSkipped ?? 0
        part.licenseTier = envelope.licenseTier
        part.boxes = envelope.boxes ?? []
        let inlineBytes: Data?
        if let b64 = envelope.b64JSON, !b64.isEmpty {
            guard let decoded = Data(base64Encoded: b64) else {
                throw .malformed("completed image invalid b64_json")
            }
            inlineBytes = decoded
        } else {
            inlineBytes = nil
        }
        return .completed(part, inlineBytes: inlineBytes, contentURL: envelope.url, elapsedMs: envelope.elapsedMs ?? 0)
    }
}

extension EngineClient {
    public func streamImageGeneration(_ call: ImageGenerationCall) -> AsyncThrowingStream<ImageStreamEvent, any Error> {
        streamImage(call)
    }

    public func streamImageGeneration(
        prompt: String,
        model: String? = nil,
        size: String = "1024x1024",
        steps: Int = 8,
        referenceImages: [Data] = [],
        beforeFilePath: String? = nil
    ) -> AsyncThrowingStream<ImageStreamEvent, any Error> {
        streamImage(ImageGenerationCall(
            prompt: prompt,
            model: model,
            size: size,
            steps: steps,
            referenceImages: referenceImages,
            beforeFilePath: beforeFilePath
        ))
    }

    static func configureImages(_ client: inout EngineClient, _ endpoint: EngineEndpoint, session: URLSession) {
        client.fetchImageContent = { assetID in
            let safe = (assetID as NSString).lastPathComponent
            let (data, response) = try await session.data(for: endpoint.request("v1/images/content/\(safe)"))
            try check(response, data)
            return data
        }
        client.streamImage = { call in
            imageStream(endpoint, call, session: session)
        }
    }

    private static func imageStream(
        _ endpoint: EngineEndpoint,
        _ call: ImageGenerationCall,
        session: URLSession
    ) -> AsyncThrowingStream<ImageStreamEvent, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let body = try call.wireBody()
                    var request = endpoint.request(call.endpointPath, method: "POST", body: body)
                    request.timeoutInterval = 900
                    let (bytes, response) = try await session.bytes(for: request)
                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        var errorBody = Data()
                        for try await byte in bytes.prefix(4096) { errorBody.append(byte) }
                        try check(response, errorBody)
                    }
                    try await relayImageEvents(bytes, endpoint: endpoint, call: call, session: session, to: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish(throwing: EngineError.cancelled)
                } catch let error as EngineError {
                    continuation.finish(throwing: error)
                } catch {
                    continuation.finish(throwing: Task.isCancelled ? EngineError.cancelled : EngineError.transport(error.localizedDescription))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func relayImageEvents(
        _ bytes: URLSession.AsyncBytes,
        endpoint: EngineEndpoint,
        call: ImageGenerationCall,
        session: URLSession,
        to continuation: AsyncThrowingStream<ImageStreamEvent, any Error>.Continuation
    ) async throws {
        var parser = SSEParser()
        var pending: [UInt8] = []
        for try await byte in bytes {
            pending.append(byte)
            guard byte == UInt8(ascii: "\n") else { continue }
            for event in parser.feed(pending) {
                let done = try await emitImageOutcome(event, endpoint: endpoint, call: call, session: session, to: continuation)
                if done { return }
            }
            pending.removeAll(keepingCapacity: true)
        }
        if !pending.isEmpty {
            pending.append(contentsOf: [UInt8(ascii: "\n"), UInt8(ascii: "\n")])
            for event in parser.feed(pending) {
                let done = try await emitImageOutcome(event, endpoint: endpoint, call: call, session: session, to: continuation)
                if done { return }
            }
        }
    }

    private static func emitImageOutcome(
        _ event: SSEEvent,
        endpoint: EngineEndpoint,
        call: ImageGenerationCall,
        session: URLSession,
        to continuation: AsyncThrowingStream<ImageStreamEvent, any Error>.Continuation
    ) async throws -> Bool {
        switch try ImageSSEDecoder.decode(event, call: call) {
        case .done:
            return true
        case .ignored:
            return false
        case .partial(let step, let totalSteps, let png, let width, let height):
            continuation.yield(.partial(step: step, totalSteps: totalSteps, png: png, width: width, height: height))
            return false
        case .completed(let part, let inlineBytes, let contentURL, let elapsedMs):
            let resolvedBytes = try await resolveImageBytes(
                inlineBytes: inlineBytes, contentURL: contentURL, assetID: part.assetID, endpoint: endpoint, session: session
            )
            continuation.yield(.completed(part, imageBytes: resolvedBytes, elapsedMs: elapsedMs))
            return false
        }
    }

    private static func resolveImageBytes(
        inlineBytes: Data?,
        contentURL: String?,
        assetID: String,
        endpoint: EngineEndpoint,
        session: URLSession
    ) async throws -> Data {
        if let inlineBytes, !inlineBytes.isEmpty { return inlineBytes }
        let path: String
        if let contentURL, !contentURL.isEmpty {
            path = contentURL.hasPrefix("/") ? String(contentURL.dropFirst()) : contentURL
        } else if !assetID.isEmpty {
            path = "v1/images/content/\((assetID as NSString).lastPathComponent)"
        } else {
            throw EngineError.malformed("completed image missing both b64_json and content url")
        }
        let (data, response) = try await session.data(for: endpoint.request(path))
        try check(response, data)
        return data
    }
}
