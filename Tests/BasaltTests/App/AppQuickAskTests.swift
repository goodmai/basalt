import BasaltCore
import Foundation
import Synchronization
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app phase 2: cloud keys from the Keychain into the engine's environment, drafts that survive a relaunch, quick ask kept out of history until it is opened as a conversation (#434 phase 2)")
struct AppQuickAskTests {
    @Test("Only well-formed variable names with a stored, non-empty key reach the engine's environment")
    func cloudKeys() {
        let keychain = KeychainClient.memory()
        #expect(keychain.write("OPENROUTER_API_KEY", "sk-test") && keychain.write("EMPTY_KEY", "") && keychain.write("bad-name", "x"))
        #expect(CloudKeys.environment(["OPENROUTER_API_KEY", "EMPTY_KEY", "bad-name", "MISSING_KEY"], keychain: keychain) == ["OPENROUTER_API_KEY": "sk-test"])
        keychain.delete("OPENROUTER_API_KEY")
        #expect(keychain.read("OPENROUTER_API_KEY") == nil && CloudKeys.isName("_A1") && !CloudKeys.isName("1A"))
    }

    @Test("A draft is saved per conversation and an emptied draft is removed")
    func drafts() throws {
        let suite = "basalt-tests-\(UUID().uuidString)"
        defer { UserDefaults.standard.removePersistentDomain(forName: suite) }
        let (drafts, id) = (DraftStore.defaults(suite), ConversationID())
        drafts.save(id, "half a question")
        #expect(drafts.load(id) == "half a question" && drafts.load(ConversationID()).isEmpty)
        drafts.save(id, "")
        #expect(UserDefaults(suiteName: suite)?.object(forKey: "draft.\(id.raw.uuidString)") == nil)
    }

    @MainActor
    @Test("Quick ask stays out of history until ⌘↩ opens it as a conversation; a restored draft is in the new store")
    func quickAsk() async throws {
        let saved = Mutex<[ConversationID: String]>([:])
        let drafts = DraftStore(load: { id in saved.withLock { $0[id] ?? "" } }, save: { id, text in saved.withLock { $0[id] = text } })
        let model = try AppModel(launch: HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/false"), arguments: []), model: "org/model", storeURL: nil, drafts: drafts)
        let store = model.openQuickAsk()
        #expect(model.openQuickAsk() === store)
        await model.promoteQuickAsk()
        #expect(model.quickAsk === store && model.selection == nil)
        store.send(.draftChanged("What is basalt?"))
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        await model.reload()
        #expect(model.summaries.isEmpty && !store.state.conversation.messages.isEmpty)
        await model.promoteQuickAsk()
        let id = store.state.conversation.id
        #expect(model.quickAsk == nil && model.selection == id && model.chats[id] === store && model.summaries.map(\.id) == [id])
        let fresh = ConversationID()
        drafts.save(fresh, "unsent")
        #expect(model.makeStore(ConversationSnapshot(id: fresh, model: "org/model")).state.draft == "unsent")
        model.openQuickAsk().send(.draftChanged("discard me"))
        model.dismissQuickAsk()
        #expect(model.quickAsk == nil)
    }
}
