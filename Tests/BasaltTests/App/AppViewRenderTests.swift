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
