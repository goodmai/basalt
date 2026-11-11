import BasaltApp
import BasaltAppCore
import Foundation
import SwiftUI

@main
struct BasaltMacApp: App {
    @State private var model: AppModel? = BasaltMacApp.makeModel()
    @AppStorage("model") private var modelID = BasaltMacApp.defaultModel

    static let defaultModel = "mlx-community/gemma-4-12B-it-qat-4bit"

    var body: some Scene {
        WindowGroup("Basalt") {
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
        Settings {
            Form {
                TextField("Model", text: $modelID)
                Text("The engine loads this model at its next start.").font(.caption).foregroundStyle(.secondary)
            }
            .padding(20)
            .frame(width: 420)
        }
    }

    static func makeModel() -> AppModel? {
        guard let helper = helperURL() else { return nil }
        let model = UserDefaults.standard.string(forKey: "model") ?? defaultModel
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appending(path: "Basalt")
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        return try? AppModel(
            launch: HelperLaunch(executable: helper, arguments: ["serve", "--model", model]), model: model,
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
