import BasaltAppCore
import BasaltCore
import Foundation

extension AppModel {
    static func makeComposeProfilesStore(
        initialCompose: String,
        supervisor: EngineSupervisor
    ) -> ComposeProfilesStore {
        Store(
            ComposeProfilesState(activeCompose: initialCompose),
            reduce: ComposeProfilesReducer.reduce,
            handle: { effect in
                switch effect {
                case .applyProfile(let id):
                    return .run(id: "apply-compose-profile") { send in
                        guard let yaml = ComposeProfileCatalog.bundledYAML[id] else {
                            await send(.failed("Unknown compose profile: \(id)"))
                            return
                        }
                        if let endpoint = await supervisor.endpoint {
                            do {
                                let diff = try await EngineClient.http(endpoint).applyCompose(ComposeApplyCall(yaml: yaml))
                                if let refusal = diff.refusal {
                                    await send(.failed(refusal))
                                    return
                                }
                                if diff.restart == nil {
                                    await supervisor.updateLaunchArguments(["serve", "--compose", id])
                                    UserDefaults.standard.set(id, forKey: EngineChoice.composeKey)
                                    await send(.activated(id))
                                    return
                                }
                            } catch let EngineError.refused(status, message) where status != 404 {
                                await send(.failed(message))
                                return
                            } catch {}
                        }
                        guard !Task.isCancelled else { return }
                        let current = await supervisor.launch
                        UserDefaults.standard.set(id, forKey: EngineChoice.composeKey)
                        await supervisor.relaunch(
                            HelperLaunch(
                                executable: current.executable,
                                arguments: ["serve", "--compose", id],
                                environment: current.environment
                            )
                        )
                        await send(.activated(id))
                    }
                }
            }
        )
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

    func persists(_ id: ConversationID) -> Bool {
        quickAsk?.state.conversation.id != id
    }
}
