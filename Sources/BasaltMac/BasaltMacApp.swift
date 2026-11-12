import BasaltApp
import BasaltAppCore
import Foundation
import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let model = BasaltMacApp.makeModel()
    var quickAsk: QuickAskController?
    var openMain: (@MainActor () -> Void)?
    private var hotKey: HotKey?
    private var services: ServiceProvider?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let model else { return }
        AppServices.model = model
        let services = ServiceProvider(model: model) { [weak self] in
            self?.openMain?()
            NSApplication.shared.activate()
        }
        (self.services, NSApplication.shared.servicesProvider) = (services, services)
        model.answered = { AnswerNotifications.post($0) }
        AnswerNotifications.request()
        let quickAsk = QuickAskController(model: model) { [weak self] in
            self?.openMain?()
            NSApplication.shared.activate()
        }
        self.quickAsk = quickAsk
        if UserDefaults.standard.object(forKey: "quickAsk") as? Bool ?? true { hotKey = HotKey { quickAsk.toggle() } }
        Task { await model.start() }
        Task { await model.checkForUpdate() }
    }

    func applicationWillTerminate(_ notification: Notification) { hotKey?.unregister() }

    func application(_ application: NSApplication, open urls: [URL]) { services?.open(urls) }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model, !model.engine.isIdle else { return .terminateNow }
        Task {
            await model.stop()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

struct MenuBarLabel: View {
    let delegate: AppDelegate
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Image(systemName: "cube.transparent")
            .accessibilityLabel("Basalt")
            .onAppear { delegate.openMain = { openWindow(id: "main") } }
    }
}

@main
struct BasaltMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    private var model: AppModel? { delegate.model }

    var body: some Scene {
        WindowGroup("Basalt", id: "main") {
            if let model {
                RootView(model: model).frame(minWidth: 720, minHeight: 480)
            } else {
                ContentUnavailableView("The engine is missing", systemImage: "exclamationmark.triangle",
                                       description: Text("Basalt.app carries its engine in Contents/Helpers/basalt. Set BASALT_HELPER to run from a build folder."))
            }
        }
        Window("Monitor", id: "monitor") {
            if let model { MonitorView(store: model.monitor) }
        }
        .keyboardShortcut("m", modifiers: [.command, .option])
        Window("Models", id: "library") {
            if let model { LibraryView(store: model.library) }
        }
        .keyboardShortcut("l", modifiers: [.command, .option])
        MenuBarExtra {
            if let model { MenuBarContent(model: model) { delegate.quickAsk?.toggle() } }
        } label: {
            MenuBarLabel(delegate: delegate)
        }
        .menuBarExtraStyle(.window)
        Settings { SettingsView(keychain: .live()) { arguments in await delegate.model?.switchEngine(arguments) } }
    }

    static func makeModel() -> AppModel? {
        guard let helper = helperURL() else { return nil }
        let model = UserDefaults.standard.string(forKey: "model") ?? AppModel.defaultModel
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appending(path: "Basalt")
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let metallib = Bundle.main.bundleURL.appending(path: "Contents/Resources/mlx.metallib")
        let names = (UserDefaults.standard.string(forKey: CloudKeys.defaultsKey) ?? "").split(separator: "\n").map(String.init)
        var environment = CloudKeys.environment(names, keychain: .live())
        if FileManager.default.fileExists(atPath: metallib.path) { environment["BASALT_METALLIB"] = metallib.path }
        return try? AppModel(
            launch: HelperLaunch(executable: helper, arguments: EngineChoice.arguments(model: model, compose: UserDefaults.standard.string(forKey: EngineChoice.composeKey)), environment: environment), model: model,
            storeURL: support.appending(path: "Conversations.store"))
    }

    static func helperURL() -> URL? {
        let inBundle = Bundle.main.bundleURL.pathExtension == "app"
        let candidates = [
            ProcessInfo.processInfo.environment["BASALT_HELPER"].map { URL(fileURLWithPath: $0) },
            inBundle ? Bundle.main.bundleURL.appending(path: "Contents/Helpers/basalt") : nil,
            inBundle ? nil : Bundle.main.executableURL?.deletingLastPathComponent().appending(path: "Basalt"),
        ]
        return candidates.compactMap { $0 }.first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }
}
