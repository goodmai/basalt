import AppKit
import BasaltCore
import Foundation
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app phase 2 views render in a hosting view: menu bar, quick ask with an answer and a failure, settings, the root window (#434 phase 2)")
@MainActor
struct AppViewRenderTests {
    static func model() throws -> AppModel {
        try AppModel(launch: HelperLaunch(executable: URL(fileURLWithPath: "/usr/bin/false"), arguments: []), model: "org/model", storeURL: nil,
                     drafts: DraftStore(load: { _ in "" }, save: { _, _ in }))
    }

    static func render(_ view: some View) -> NSSize {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
        host.layoutSubtreeIfNeeded()
        return host.fittingSize
    }

    @Test("Every phase 2 view lays out with a non-zero size")
    func renders() async throws {
        let model = try Self.model()
        let store = model.openQuickAsk()
        #expect(Self.render(QuickAskView(store: store) {}).width > 0)
        store.send(.draftChanged("What is basalt?"))
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
        await store.settle()
        #expect(Self.render(QuickAskView(store: store) {}).height > 0)
        #expect(Self.render(MenuBarContent(model: model) {}).width > 0)
        #expect(Self.render(SettingsView(keychain: .memory())).width > 0)
        #expect(Self.render(RootView(model: model)).width > 0)
        var library = LibraryState(pending: ["org/coming"])
        _ = LibraryReducer.reduce(&library, .listed([LibraryModel(id: "org/on", sizeBytes: 1 << 30, isLoaded: true), LibraryModel(id: "org/off", sizeBytes: 2 << 30, isLoaded: false)]))
        _ = LibraryReducer.reduce(&library, .fitted("org/off", ModelFit(fits: false, level: "insufficient", requiredBytes: 4, availableBytes: 2, warning: "too big")))
        _ = LibraryReducer.reduce(&library, .downloadFailed("org/coming", "offline"))
        library.problem = "a problem"
        let libraryStore = Store(library, reduce: LibraryReducer.reduce, handle: { _ in .none })
        #expect(Self.render(LibraryView(store: libraryStore)).width > 0)
        var conversation = ConversationSnapshot(id: ConversationID(), model: "org/model")
        var answer = MessageSnapshot(id: MessageID(), role: .assistant, parts: [.text("Pay 42 today.")])
        answer.receipt = Receipt(model: "org/model")
        answer.receipt?.stages = [StageReceipt(stage: "perceive", model: "apple-vision", ms: 80)]
        conversation.messages = [MessageSnapshot(id: MessageID(), role: .user, parts: [.text("Total?"), .attachment(AttachmentPart(name: "bill.pdf", kind: "pdf", perceived: "Total 42"))]), answer]
        var chat = ChatState(conversation: conversation)
        chat.attachments = [try #require(ChatAttachment(name: "memo.wav", data: Data([1])))]
        #expect(Self.render(ChatView(store: Store(chat, reduce: ChatReducer.reduce, handle: { _ in .none }))).width > 0)
        #expect(AttachmentChip.symbol("pdf") == "doc.richtext" && AttachmentChip.symbol("audio") == "waveform" && AttachmentChip.symbol("image") == "photo")
        #expect(MenuBarContent.status(.ready(port: 1)) == "Ready" && MenuBarContent.ready(.ready(port: 1)) && !MenuBarContent.ready(.stopped))
        #expect(MenuBarContent.status(.backingOff(attempt: 2)).contains("2") && MenuBarContent.status(.failed(log: [])).contains("failing"))
    }

    @Test("The live Keychain stores, reads and deletes a key under its own service")
    func keychain() {
        let keychain = KeychainClient.live(service: "app.basalt.tests.\(UUID().uuidString)")
        guard keychain.write("BASALT_TEST_KEY", "sk-test") else { return }
        #expect(keychain.read("BASALT_TEST_KEY") == "sk-test")
        #expect(keychain.write("BASALT_TEST_KEY", "sk-new") && keychain.read("BASALT_TEST_KEY") == "sk-new")
        keychain.delete("BASALT_TEST_KEY")
        #expect(keychain.read("BASALT_TEST_KEY") == nil)
    }
}
