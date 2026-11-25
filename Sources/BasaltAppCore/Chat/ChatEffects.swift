import Foundation

public struct ChatEnvironment: Sendable {
    public var engine: @Sendable () async -> EngineClient?
    public var persist: @Sendable (ConversationSnapshot) async -> Void
    public var answered: @Sendable (ConversationSnapshot) async -> Void
    public var mediaStore: AppMediaStore

    public init(
        engine: @escaping @Sendable () async -> EngineClient?,
        persist: @escaping @Sendable (ConversationSnapshot) async -> Void,
        answered: @escaping @Sendable (ConversationSnapshot) async -> Void = { _ in },
        mediaStore: AppMediaStore = AppMediaStore()
    ) {
        (self.engine, self.persist, self.answered, self.mediaStore) = (engine, persist, answered, mediaStore)
    }
}

public enum ChatEffects {
    public static let runID = "run"

    public static func handler(_ environment: ChatEnvironment) -> @Sendable (ChatEffect) -> EffectRun<ChatAction> {
        { effect in
            switch effect {
            case .cancel: return .cancel(id: runID)
            case .persist(let conversation):
                return .run(id: "persist-\(conversation.id.raw)") { _ in await environment.persist(conversation) }
            case .answered(let conversation):
                return .run(id: "answered-\(conversation.id.raw)") { _ in await environment.answered(conversation) }
            case .start(let run, let request):
                return .run(id: runID) { send in await stream(run, request, environment, send) }
            case .perceive(let run, let call):
                return .run(id: runID) { send in await perceive(run, call, environment, send) }
            case .generateImage(let run, let call):
                return .run(id: runID) { send in await generateImage(run, call, environment, send) }
            }
        }
    }

    static func stream(_ run: RunID, _ request: ChatRequest, _ environment: ChatEnvironment, _ send: @escaping @Sendable (ChatAction) async -> Void) async {
        guard let engine = await environment.engine() else { return await send(.failed(run, .notRunning)) }
        let clock = ContinuousClock()
        let started = clock.now
        var first: ContinuousClock.Instant?
        do {
            for try await delta in engine.chat(request) {
                if first == nil, delta.isContent { first = clock.now }
                await send(.delta(run, delta))
            }
            guard !Task.isCancelled else { return }
            let end = clock.now
            await send(.finished(run, RunTiming(timeToFirstToken: first.map { $0 - started }, streaming: end - (first ?? started))))
        } catch {
            guard !Task.isCancelled else { return }
            await send(.failed(run, error as? EngineError ?? .transport(error.localizedDescription)))
        }
    }
}

extension ChatEffects {
    static func perceive(_ run: RunID, _ call: PipelineCall, _ environment: ChatEnvironment, _ send: @escaping @Sendable (ChatAction) async -> Void) async {
        guard let engine = await environment.engine() else { return await send(.failed(run, .notRunning)) }
        do {
            let resolved = try await resolvePipelineCall(call)
            let response = try await engine.pipeline(resolved)
            guard !Task.isCancelled else { return }
            await send(.perceived(run, PipelineResult(response, count: resolved.attachments.count)))
        } catch {
            guard !Task.isCancelled else { return }
            await send(.failed(run, error as? EngineError ?? .transport(error.localizedDescription)))
        }
    }

    static func generateImage(
        _ run: RunID,
        _ call: ImageGenerationCall,
        _ environment: ChatEnvironment,
        _ send: @escaping @Sendable (ChatAction) async -> Void
    ) async {
        guard let engine = await environment.engine() else { return await send(.failed(run, .notRunning)) }
        var completed = false
        do {
            let resolved = try await resolveReferenceImages(call)
            for try await event in engine.streamImageGeneration(resolved) {
                switch event {
                case .partial(let step, let totalSteps, let png, let width, let height):
                    await send(.imagePartial(run, step: step, totalSteps: totalSteps, png: png, width: width, height: height))
                case .completed(var part, let imageBytes, let elapsedMs):
                    let savedPath = try environment.mediaStore.store(imageBytes, format: resolved.outputFormat)
                    part.filePath = savedPath
                    if part.assetID.isEmpty {
                        part.assetID = URL(fileURLWithPath: savedPath).lastPathComponent
                    }
                    if part.beforeFilePath == nil {
                        part.beforeFilePath = resolved.beforeFilePath
                    }
                    part.previewPNG = nil
                    completed = true
                    let stage = StageReceipt(stage: "imagine", model: part.model, ms: elapsedMs)
                    await send(.imageCompleted(run, part, stage))
                }
            }
            if !completed, !Task.isCancelled {
                await send(.failed(run, .malformed("image stream ended before completion")))
            }
        } catch {
            guard !Task.isCancelled else { return }
            await send(.failed(run, error as? EngineError ?? .transport(error.localizedDescription)))
        }
    }

    static func resolveReferenceImages(_ call: ImageGenerationCall) async throws(EngineError) -> ImageGenerationCall {
        guard !call.referenceImagePaths.isEmpty else { return call }
        var resolved = call
        var loaded = call.referenceImages
        for path in call.referenceImagePaths {
            guard let bytes = await readBoundedFile(at: path) else {
                throw .refused(status: 400, message: "Reference image could not be read: \(URL(fileURLWithPath: path).lastPathComponent)")
            }
            loaded.append(bytes)
        }
        resolved.referenceImages = loaded
        resolved.referenceImagePaths = []
        return resolved
    }

    private static func resolvePipelineCall(_ call: PipelineCall) async throws(EngineError) -> PipelineCall {
        guard call.attachments.contains(where: { $0.data.isEmpty }) else { return call }
        var resolved: [ChatAttachment] = []
        for item in call.attachments {
            if !item.data.isEmpty {
                resolved.append(item)
            } else if let path = item.sourcePath, let bytes = await readBoundedFile(at: path),
                      let loaded = ChatAttachment(name: item.name, data: bytes, id: item.id, sourcePath: path) {
                resolved.append(loaded)
            } else {
                throw .refused(status: 400, message: "Attachment could not be read: \(item.name)")
            }
        }
        return PipelineCall(attachments: resolved, task: call.task)
    }

    @concurrent
    private static func readBoundedFile(at path: String) async -> Data? {
        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        guard let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              values.isRegularFile == true,
              let size = values.fileSize,
              size > 0, size <= ChatAttachment.byteLimit else { return nil }
        return try? Data(contentsOf: url)
    }
}

extension ChatDelta {
    var isContent: Bool {
        switch self {
        case .text, .reasoning, .toolCall: true
        case .usage, .finish: false
        }
    }
}
