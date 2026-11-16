import Foundation
import Hummingbird
import HTTPTypes
import BasaltCore

struct ImagesController: Sendable {
    static let generationsPath = "/v1/images/generations"
    static let editsPath = "/v1/images/edits"
    static let bodyLimit = 64 * 1024 * 1024

    static let modelHeader = HTTPField.Name("X-Basalt-Image-Model")!
    static let computeHeader = HTTPField.Name("X-Basalt-Image-Compute")!
    static let licenseHeader = HTTPField.Name("X-Basalt-Image-License-Tier")!
    static let fallbackHeader = HTTPField.Name("X-Basalt-Image-Fallback-From")!

    let synthesizer: (any ImageSynthesizer)?
    var pipelines: PipelineRegistry? = nil
    var storeDirectory: URL? = nil

    func add(to group: RouterGroup<BasicRequestContext>) {
        group.post("/images/generations") { try await self.generations(request: $0, context: $1) }
        group.post("/images/edits") { try await self.edits(request: $0, context: $1) }
        group.get("/images/content/{id}") { try await self.content(request: $0, context: $1) }
        group.get("/images/content/**") { _, _ in Self.invalidAssetResponse() }
        group.get("/images/models") { try await self.models(request: $0, context: $1) }
    }

    private func resolveSynthesizer() async -> (any ImageSynthesizer)? {
        if let synthesizer { return synthesizer }
        return await pipelines?.system(ComposeSpec.defaultPipeline)?.image
    }

    func models(request: Request, context: BasicRequestContext) async throws -> Response {
        let active = await resolveSynthesizer()
        let items: [ImageModelDescriptor]
        if let routed = active as? RoutedImageSynthesizer {
            items = routed.allDescriptors
        } else if let active {
            items = [active.descriptor]
        } else {
            items = []
        }
        let encoder = Self.makeJSONEncoder()
        var buffer = ByteBuffer()
        buffer.writeBytes(try encoder.encode(ModelsEnvelope(models: items)))
        return Response(status: .ok, headers: [.contentType: "application/json"], body: ResponseBody(byteBuffer: buffer))
    }

    func content(request: Request, context: BasicRequestContext) async throws -> Response {
        let raw = context.parameters.get("id")?.removingPercentEncoding ?? ""
        guard !request.uri.path.contains(".."), ImageArtifactStore.parseAssetID(raw) != nil else {
            return Self.invalidAssetResponse()
        }
        guard let loaded = ImageArtifactStore.readValidatedAsset(id: raw, in: storeDirectory) else {
            return ProviderErrorResponse.response(status: .notFound, message: "image asset not found", shape: .openAI)
        }
        var buffer = ByteBuffer()
        buffer.writeBytes(loaded.data)
        let headers: HTTPFields = [
            .contentType: loaded.format.contentType,
            .cacheControl: "private, max-age=86400",
        ]
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: buffer))
    }

    static func invalidAssetResponse() -> Response {
        ProviderErrorResponse.response(status: .badRequest, message: "invalid image asset id", shape: .openAI)
    }

    func generations(request: Request, context: BasicRequestContext) async throws -> Response {
        let data: Data
        switch await collectBody(request) {
        case .success(let payload): data = payload
        case .failure(let errorResponse): return errorResponse
        }
        do {
            guard let active = await resolveSynthesizer() else {
                throw BasaltError.invalidRequestStructure(details: "image: no image synthesizer is configured on this server")
            }
            let parsed = try WireImageRequest.parseGeneration(
                data,
                defaultModel: active.modelIdentifier,
                defaultSteps: active.descriptor.defaultSteps
            )
            return try await execute(parsed, active: active, started: .now)
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.generationsPath)
        }
    }

    func edits(request: Request, context: BasicRequestContext) async throws -> Response {
        let data: Data
        switch await collectBody(request) {
        case .success(let payload): data = payload
        case .failure(let errorResponse): return errorResponse
        }
        do {
            guard let active = await resolveSynthesizer() else {
                throw BasaltError.invalidRequestStructure(details: "image: no image synthesizer is configured on this server")
            }
            let parsed = try Self.parseEditPayload(
                data,
                contentType: request.headers[.contentType],
                defaultModel: active.modelIdentifier,
                defaultSteps: active.descriptor.defaultSteps
            )
            return try await execute(parsed, active: active, started: .now)
        } catch {
            return ProviderErrorResponse.response(for: error, path: Self.editsPath)
        }
    }

    private func collectBody(_ request: Request) async -> Result<Data, Response> {
        do {
            let buffer = try await request.body.collect(upTo: Self.bodyLimit)
            return .success(Data(buffer.readableBytesView))
        } catch let error as any HTTPResponseError where error.status == .contentTooLarge {
            return .failure(ProviderErrorResponse.response(
                status: .contentTooLarge,
                message: "an image request is limited to \(Self.bodyLimit >> 20) MB",
                shape: .openAI
            ))
        } catch {
            return .failure(ProviderErrorResponse.response(for: error, path: Self.generationsPath))
        }
    }
}

extension ImagesController {
    static func parseEditPayload(
        _ data: Data,
        contentType: String?,
        defaultModel: String,
        defaultSteps: Int
    ) throws(BasaltError) -> ImageGenerationRequest {
        let parsed: ImageGenerationRequest
        if let media = contentType.flatMap(MediaType.init(from:)), media.isType(.multipartForm) {
            guard let param = media.parameter, param.name.lowercased() == "boundary" else {
                throw .invalidRequestStructure(details: "image: multipart/form-data requires a boundary parameter")
            }
            parsed = try WireImageRequest.parseMultipartEdit(
                data,
                boundary: param.value,
                defaultModel: defaultModel,
                defaultSteps: defaultSteps
            )
        } else {
            parsed = try WireImageRequest.parseJSONEdit(
                data,
                defaultModel: defaultModel,
                defaultSteps: defaultSteps
            )
        }
        guard parsed.isEditRequest || parsed.isAudioConditioned else {
            throw .invalidRequestStructure(details: "image: edits require at least one reference image, mask or audio input")
        }
        return parsed
    }

    func execute(
        _ genRequest: ImageGenerationRequest,
        active: any ImageSynthesizer,
        started: ContinuousClock.Instant
    ) async throws -> Response {
        let decision = await active.resolveRoute(genRequest)
        let stream = try await active.generate(genRequest)
        if genRequest.stream {
            return Self.sseResponse(
                stream: stream,
                active: active,
                decision: decision,
                storeDirectory: storeDirectory,
                started: started
            )
        }
        var finalArtifact: ImageCompletedArtifact?
        for try await event in stream {
            if case .completed(let completed) = event {
                finalArtifact = completed
            }
        }
        guard let artifact = finalArtifact else {
            throw BasaltError.modelInferenceError(details: "image: synthesizer finished without a completed artifact")
        }
        let persisted = try ImageArtifactStore.persist(artifact, in: storeDirectory)
        let elapsed = max(artifact.elapsedMs, Self.elapsedMilliseconds(since: started))
        let item = ImageWireItem(artifact: artifact, persisted: persisted, elapsedMs: elapsed)
        let envelope = ImagesResponseEnvelope(created: Int(Date().timeIntervalSince1970), data: [item])
        var buffer = ByteBuffer()
        buffer.writeBytes(try Self.makeJSONEncoder().encode(envelope))
        var headers: HTTPFields = [
            .contentType: "application/json",
            Self.modelHeader: artifact.model,
            Self.computeHeader: artifact.compute.rawValue,
            Self.licenseHeader: artifact.licenseTier.rawValue,
        ]
        if let fallback = decision.fallbackFrom {
            headers[Self.fallbackHeader] = fallback
        }
        return Response(status: .ok, headers: headers, body: ResponseBody(byteBuffer: buffer))
    }
}

extension ImagesController {
    static func sseResponse(
        stream: AsyncThrowingStream<ImageGenerationEvent, any Error>,
        active: any ImageSynthesizer,
        decision: ImageRouteDecision,
        storeDirectory: URL?,
        started: ContinuousClock.Instant
    ) -> Response {
        let body = AsyncStream<ByteBuffer> { continuation in
            let task = Task {
                await streamSSE(
                    stream: stream,
                    storeDirectory: storeDirectory,
                    started: started,
                    continuation: continuation
                )
            }
            continuation.onTermination = { _ in task.cancel() }
        }
        var headers: HTTPFields = [
            .contentType: "text/event-stream",
            .cacheControl: "no-cache",
            .connection: "keep-alive",
            modelHeader: decision.selectedModel,
            computeHeader: decision.compute.rawValue,
            licenseHeader: active.descriptor.licenseTier.rawValue,
        ]
        if let fallback = decision.fallbackFrom {
            headers[fallbackHeader] = fallback
        }
        return Response(status: .ok, headers: headers, body: ResponseBody(asyncSequence: body))
    }

    private static func streamSSE(
        stream: AsyncThrowingStream<ImageGenerationEvent, any Error>,
        storeDirectory: URL?,
        started: ContinuousClock.Instant,
        continuation: AsyncStream<ByteBuffer>.Continuation
    ) async {
        let encoder = makeJSONEncoder()
        do {
            for try await event in stream {
                switch event {
                case .partial(let frame):
                    let wire = ImagePartialSSEEvent(frame: frame)
                    let json = try encoder.encode(wire)
                    continuation.yield(ByteBuffer(string: "data: \(String(decoding: json, as: UTF8.self))\n\n"))
                case .completed(let artifact):
                    let persisted = try ImageArtifactStore.persist(artifact, in: storeDirectory)
                    let elapsed = max(artifact.elapsedMs, elapsedMilliseconds(since: started))
                    let wire = ImageCompletedSSEEvent(artifact: artifact, persisted: persisted, elapsedMs: elapsed)
                    let json = try encoder.encode(wire)
                    continuation.yield(ByteBuffer(string: "data: \(String(decoding: json, as: UTF8.self))\n\n"))
                }
            }
            continuation.yield(ByteBuffer(string: "data: [DONE]\n\n"))
        } catch {
            let message = ModelsController.publicMessage(for: error)
            if let json = try? JSONSerialization.data(withJSONObject: ["type": "error", "error": ["message": message]]) {
                continuation.yield(ByteBuffer(string: "data: \(String(decoding: json, as: UTF8.self))\n\n"))
            }
        }
        continuation.finish()
    }

    static func makeJSONEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    static func elapsedMilliseconds(since started: ContinuousClock.Instant) -> Int {
        let parts = started.duration(to: .now).components
        let ms = Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
        return max(1, Int(ms.rounded()))
    }
}
