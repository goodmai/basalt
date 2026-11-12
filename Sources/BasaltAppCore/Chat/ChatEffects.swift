import Foundation

public struct ChatEnvironment: Sendable {
    public var engine: @Sendable () async -> EngineClient?
    public var persist: @Sendable (ConversationSnapshot) async -> Void

    public init(engine: @escaping @Sendable () async -> EngineClient?, persist: @escaping @Sendable (ConversationSnapshot) async -> Void) {
        (self.engine, self.persist) = (engine, persist)
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
            case .start(let run, let request):
                return .run(id: runID) { send in await stream(run, request, environment, send) }
            case .perceive(let run, let call):
                return .run(id: runID) { send in await perceive(run, call, environment, send) }
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
            let response = try await engine.pipeline(call)
            guard !Task.isCancelled else { return }
            await send(.perceived(run, PipelineResult(response, count: call.attachments.count)))
        } catch {
            guard !Task.isCancelled else { return }
            await send(.failed(run, error as? EngineError ?? .transport(error.localizedDescription)))
        }
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
