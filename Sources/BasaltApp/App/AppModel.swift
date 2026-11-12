import BasaltAppCore
import BasaltCore
import Foundation
import Observation

public typealias ChatStore = Store<ChatState, ChatAction, ChatEffect>
public typealias MonitorStore = Store<MonitorState, MonitorAction, MonitorEffect>
public typealias LibraryStore = Store<LibraryState, LibraryAction, LibraryEffect>

@Observable @MainActor
public final class AppModel {
    public static let defaultModel = "mlx-community/gemma-4-12B-it-qat-4bit"

    public private(set) var engine = EngineState()
    public private(set) var engineLog: [String] = []
    public private(set) var summaries: [ConversationSummary] = []
    public var selection: ConversationID?
    public private(set) var chats: [ConversationID: ChatStore] = [:]
    public private(set) var quickAsk: ChatStore?
    public let monitor: MonitorStore
    public let library: LibraryStore
    public let drafts: DraftStore
    @ObservationIgnored let supervisor: EngineSupervisor
    @ObservationIgnored let repository: ConversationRepository
    @ObservationIgnored let configuredModel: String
    public var model: String { library.state.models.first(where: \.isLoaded)?.id ?? configuredModel }
    @ObservationIgnored private var listening = false

    @ObservationIgnored public var answered: @MainActor (ConversationSnapshot) -> Void = { _ in }

    public init(launch: HelperLaunch, model: String, storeURL: URL?, environment: EngineEnvironment = .live, drafts: DraftStore = .defaults()) throws {
        let supervisor = EngineSupervisor(launch: launch, environment: environment)
        (self.supervisor, self.configuredModel, self.drafts) = (supervisor, model, drafts)
        repository = ConversationRepository(modelContainer: try ConversationRepository.container(at: storeURL))
        monitor = Store(MonitorState(), reduce: MonitorReducer.reduce, handle: MonitorReducer.handler(
            stats: {
                guard let endpoint = await supervisor.endpoint else { throw EngineError.notRunning }
                return try await EngineClient.http(endpoint).stats()
            }, every: .seconds(2), sleep: { try await Task.sleep(for: $0) },
            compose: {
                guard let endpoint = await supervisor.endpoint else { return nil }
                return try await EngineClient.http(endpoint).compose()
            }))
        library = Store(LibraryState(pending: UserDefaults.standard.stringArray(forKey: Self.pendingKey) ?? []), reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(
            LibraryEnvironment(
                client: { await supervisor.endpoint.map { LibraryClient.live($0) } },
                savePending: { UserDefaults.standard.set($0, forKey: AppModel.pendingKey) },
                chose: { id in UserDefaults.standard.set(id, forKey: "model") })))
    }

    public func start() async {
        if !listening {
            listening = true
            _ = try? await repository.recoverInterrupted()
            await reload()
            let states = supervisor.states
            Task { [weak self] in
                for await state in states { await self?.engineChanged(state) }
            }
        }
        await supervisor.start()
    }

    public func stop() async { await supervisor.shutdown() }

    public func readyEndpoint(within limit: Duration = .seconds(180)) async -> EngineEndpoint? {
        let deadline = ContinuousClock.now + limit
        while ContinuousClock.now < deadline {
            if case .ready = engine.phase, let endpoint = await supervisor.endpoint { return endpoint }
            try? await Task.sleep(for: .milliseconds(250))
        }
        return nil
    }

    public func startConversation(text: String, attachments: [ChatAttachment] = []) {
        newConversation()
        guard let id = selection, let store = chats[id] else { return }
        store.send(.draftChanged(text))
        for attachment in attachments { store.send(.attach(attachment)) }
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
    }

    nonisolated static let pendingKey = "pendingDownloads"

    public func switchEngine(_ arguments: [String]) async {
        let current = await supervisor.launch
        guard current.arguments != arguments else { return }
        await supervisor.relaunch(HelperLaunch(executable: current.executable, arguments: arguments, environment: current.environment))
    }

    public func restart() async {
        await supervisor.shutdown()
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

    public func openQuickAsk() -> ChatStore {
        if let quickAsk { return quickAsk }
        let store = makeStore(ConversationSnapshot(id: ConversationID(), model: model))
        quickAsk = store
        return store
    }

    public func dismissQuickAsk() {
        quickAsk?.send(.stop)
        quickAsk = nil
    }

    public func promoteQuickAsk() async {
        guard let store = quickAsk, !store.state.conversation.messages.isEmpty else { return }
        let id = store.state.conversation.id
        (quickAsk, chats[id], selection) = (nil, store, id)
        try? await repository.save(store.state.conversation)
        await reload()
    }

    func persists(_ id: ConversationID) -> Bool { quickAsk?.state.conversation.id != id }

    func makeStore(_ conversation: ConversationSnapshot) -> ChatStore {
        let (supervisor, repository) = (supervisor, repository)
        let environment = ChatEnvironment(
            engine: { await supervisor.endpoint.map { EngineClient.http($0) } },
            persist: { [weak self] snapshot in
                guard await self?.persists(snapshot.id) == true else { return }
                try? await repository.save(snapshot)
                await self?.reload()
            },
            answered: { [weak self] snapshot in
                guard await self?.persists(snapshot.id) == true else { return }
                await self?.answered(snapshot)
            })
        let store = Store(ChatState(conversation: conversation), reduce: ChatReducer.reduce, handle: ChatEffects.handler(environment))
        let draft = drafts.load(conversation.id)
        if !draft.isEmpty { store.send(.draftChanged(draft)) }
        return store
    }
}
