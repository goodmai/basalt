import BasaltAppCore
import BasaltCore
import Foundation
import Observation

public typealias ChatStore = Store<ChatState, ChatAction, ChatEffect>
public typealias MonitorStore = Store<MonitorState, MonitorAction, MonitorEffect>

@Observable @MainActor
public final class AppModel {
    public private(set) var engine = EngineState()
    public private(set) var engineLog: [String] = []
    public private(set) var summaries: [ConversationSummary] = []
    public var selection: ConversationID?
    public private(set) var chats: [ConversationID: ChatStore] = [:]
    public let monitor: MonitorStore
    @ObservationIgnored let supervisor: EngineSupervisor
    @ObservationIgnored let repository: ConversationRepository
    @ObservationIgnored public let model: String

    public init(launch: HelperLaunch, model: String, storeURL: URL?, environment: EngineEnvironment = .live) throws {
        let supervisor = EngineSupervisor(launch: launch, environment: environment)
        (self.supervisor, self.model) = (supervisor, model)
        repository = ConversationRepository(modelContainer: try ConversationRepository.container(at: storeURL))
        monitor = Store(MonitorState(), reduce: MonitorReducer.reduce, handle: MonitorReducer.handler(
            stats: {
                guard let endpoint = await supervisor.endpoint else { throw EngineError.notRunning }
                return try await EngineClient.http(endpoint).stats()
            }, every: .seconds(2), sleep: { try await Task.sleep(for: $0) }))
    }

    public func start() async {
        _ = try? await repository.recoverInterrupted()
        await reload()
        let states = supervisor.states
        Task { [weak self] in
            for await state in states { await self?.engineChanged(state) }
        }
        await supervisor.start()
    }

    public func stop() async { await supervisor.stop() }

    public func restart() async {
        await supervisor.stop()
        await supervisor.start()
    }

    func engineChanged(_ state: EngineState) async {
        engine = state
        engineLog = await supervisor.log
    }

    public func reload() async {
        summaries = (try? await repository.summaries()) ?? []
    }

    public func newConversation() {
        let conversation = ConversationSnapshot(id: ConversationID(), model: model)
        chats[conversation.id] = makeStore(conversation)
        selection = conversation.id
    }

    public func store(for id: ConversationID) async -> ChatStore? {
        if let store = chats[id] { return store }
        guard let conversation = try? await repository.load(id) else { return nil }
        let store = chats[id] ?? makeStore(conversation)
        chats[id] = store
        return store
    }

    public func delete(_ id: ConversationID) async {
        chats[id]?.send(.stop)
        chats[id] = nil
        if selection == id { selection = nil }
        try? await repository.delete(id)
        await reload()
    }

    func makeStore(_ conversation: ConversationSnapshot) -> ChatStore {
        let (supervisor, repository) = (supervisor, repository)
        let environment = ChatEnvironment(
            engine: { await supervisor.endpoint.map { EngineClient.http($0) } },
            persist: { [weak self] snapshot in
                try? await repository.save(snapshot)
                await self?.reload()
            })
        return Store(ChatState(conversation: conversation), reduce: ChatReducer.reduce, handle: ChatEffects.handler(environment))
    }
}
