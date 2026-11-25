import BasaltAppCore
import BasaltCore
import Foundation
import Observation

public typealias ChatStore = Store<ChatState, ChatAction, ChatEffect>
public typealias MonitorStore = Store<MonitorState, MonitorAction, MonitorEffect>
public typealias LibraryStore = Store<LibraryState, LibraryAction, LibraryEffect>

@MainActor
private final class MainActorCallback<Value: Sendable> {
    var invoke: (@MainActor (Value) -> Void)?
}

@Observable @MainActor
public final class AppModel {
    public static let defaultModel = "mlx-community/gemma-4-12B-it-qat-4bit"

    public private(set) var engine = EngineState()
    public private(set) var engineLog: [String] = []
    public private(set) var summaries: [ConversationSummary] = []
    public var searchQuery = ""
    public var isSearching = false
    public internal(set) var matchedConversationIDs: Set<ConversationID>?
    public var selection: ConversationID?
    public internal(set) var chats: [ConversationID: ChatStore] = [:]
    public internal(set) var quickAsk: ChatStore?
    public private(set) var update: AvailableUpdate?
    public internal(set) var latestComposeEvent: ComposeStreamEvent?
    public let monitor: MonitorStore
    public let library: LibraryStore
    public let composeProfiles: ComposeProfilesStore
    public let voice: AudioSessionController
    public let drafts: DraftStore
    @ObservationIgnored let supervisor: EngineSupervisor
    @ObservationIgnored let repository: ConversationRepository
    @ObservationIgnored let cloudEnvironment: @Sendable () -> [String: String]
    public internal(set) var configuredModel: String
    public var model: String { library.state.models.first(where: \.isLoaded)?.id ?? configuredModel }
    @ObservationIgnored private var listening = false
    @ObservationIgnored var composeEventsTask: Task<Void, Never>?

    @ObservationIgnored public var answered: @MainActor (ConversationSnapshot) -> Void = { _ in }

    public init(
        launch: HelperLaunch,
        model: String,
        storeURL: URL?,
        environment: EngineEnvironment = .live,
        drafts: DraftStore = .defaults(),
        cloudEnvironment: @escaping @Sendable () -> [String: String] = { CloudKeys.environment(defaults: .standard, keychain: .live()) }
    ) throws {
        let supervisor = EngineSupervisor(launch: launch, environment: environment)
        (self.supervisor, self.configuredModel, self.drafts, self.cloudEnvironment) = (supervisor, model, drafts, cloudEnvironment)
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
        let onChosen = MainActorCallback<String>()
        library = Store(LibraryState(pending: UserDefaults.standard.stringArray(forKey: Self.pendingKey) ?? []), reduce: LibraryReducer.reduce, handle: LibraryEffects.handler(
            LibraryEnvironment(
                client: { await supervisor.endpoint.map { LibraryClient.live($0) } },
                savePending: { UserDefaults.standard.set($0, forKey: AppModel.pendingKey) },
                chose: { id in
                    UserDefaults.standard.set(id, forKey: EngineChoice.modelKey)
                    await supervisor.updateLaunchArguments(["serve", "--model", id])
                    await MainActor.run { onChosen.invoke?(id) }
                })))
        let initialCompose = UserDefaults.standard.string(forKey: EngineChoice.composeKey) ?? ""
        composeProfiles = Self.makeComposeProfilesStore(
            initialCompose: initialCompose,
            supervisor: supervisor,
            cloudEnvironment: cloudEnvironment
        )
        voice = AudioSessionController(engine: { await supervisor.endpoint.map { EngineClient.http($0) } })
        onChosen.invoke = { [weak self] id in self?.configuredModel = id }
        voice.onSubmitTranscript = { [weak self] text in self?.submitVoiceTranscript(text) }
    }

    public func start(arguments: [String]? = nil) async {
        if !listening {
            listening = true
            _ = try? await repository.recoverInterrupted()
            await reload()
            let states = supervisor.states
            Task { [weak self] in
                for await state in states { await self?.engineChanged(state) }
            }
        }
        if let arguments {
            await switchEngine(arguments)
        } else {
            await supervisor.start()
        }
    }

    public func stop() async {
        composeEventsTask?.cancel()
        composeEventsTask = nil
        await supervisor.shutdown()
    }

    public func checkForUpdate(defaults: UserDefaults = .standard, now: Date = Date(), fetch: @Sendable () async -> AvailableUpdate? = { await UpdateCheck.latest() }) async {
        guard defaults.object(forKey: UpdateCheck.enabledKey) as? Bool ?? true,
              UpdateCheck.due(last: defaults.object(forKey: UpdateCheck.lastKey) as? Date, now: now) else { return }
        defaults.set(now, forKey: UpdateCheck.lastKey)
        update = await fetch()
    }

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

    public func restart() async {
        let current = await supervisor.launch
        let freshEnv = Self.refreshedEnvironment(preserving: current.environment, cloud: cloudEnvironment())
        await supervisor.relaunch(
            HelperLaunch(executable: current.executable, arguments: current.arguments, environment: freshEnv)
        )
    }

    func engineChanged(_ state: EngineState) async {
        engine = state
        engineLog = await supervisor.log
        watchComposeEvents(for: state)
    }

    public func reload() async {
        summaries = (try? await repository.summaries()) ?? []
        let trimmed = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            await updateSearch(searchQuery)
        }
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
        let store = chats.removeValue(forKey: id)
        store?.send(.stop)
        if selection == id { selection = nil }
        try? await repository.delete(id)
        await reload()
    }

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
                await self?.handleAnswered(snapshot)
            })
        let store = Store(ChatState(conversation: conversation), reduce: ChatReducer.reduce, handle: ChatEffects.handler(environment))
        let draft = drafts.load(conversation.id)
        if !draft.isEmpty { store.send(.draftChanged(draft)) }
        return store
    }
}
