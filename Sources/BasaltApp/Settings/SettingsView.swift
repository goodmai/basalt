import BasaltAppCore
import ServiceManagement
import SwiftUI

public struct SettingsView: View {
    let keychain: KeychainClient
    let restart: @MainActor ([String]) async -> Void

    public init(keychain: KeychainClient, restart: @escaping @MainActor ([String]) async -> Void = { _ in }) { (self.keychain, self.restart) = (keychain, restart) }

    public var body: some View {
        TabView {
            Tab("General", systemImage: "gearshape") { GeneralSettings() }
            Tab("Engine", systemImage: "cpu") { EngineSettings(restart: restart) }
            Tab("Cloud Keys", systemImage: "key") { CloudKeySettings(keychain: keychain) }
        }
        .frame(width: 480)
    }
}

struct GeneralSettings: View {
    @AppStorage("model") private var model = AppModel.defaultModel
    @AppStorage("quickAsk") private var quickAsk = true
    @AppStorage(UpdateCheck.enabledKey) private var checkUpdates = true
    @State private var openAtLogin = SMAppService.mainApp.status == .enabled
    @State private var loginError: String?

    var body: some View {
        Form {
            TextField("Model", text: $model)
            Text("The engine loads this model at its next start.").font(.caption).foregroundStyle(.secondary)
            Toggle("Quick ask on ⌥Space", isOn: $quickAsk)
            Toggle("Check GitHub for new versions once a day", isOn: $checkUpdates)
            Toggle("Open at login", isOn: $openAtLogin)
                .onChange(of: openAtLogin) { _, on in
                    do {
                        try on ? SMAppService.mainApp.register() : SMAppService.mainApp.unregister()
                        loginError = nil
                    } catch {
                        (loginError, openAtLogin) = (error.localizedDescription, SMAppService.mainApp.status == .enabled)
                    }
                }
            if let loginError { Text(loginError).font(.caption).foregroundStyle(.red) }
        }
        .padding(20)
    }
}

struct CloudKeySettings: View {
    let keychain: KeychainClient
    @AppStorage(CloudKeys.defaultsKey) private var stored = ""
    @State private var name = ""
    @State private var value = ""

    var names: [String] { stored.split(separator: "\n").map(String.init) }

    var body: some View {
        Form {
            Section {
                ForEach(names, id: \.self) { key in
                    HStack {
                        Text(key).monospaced()
                        Spacer()
                        Button("Remove", role: .destructive) {
                            keychain.delete(key)
                            stored = names.filter { $0 != key }.joined(separator: "\n")
                        }
                    }
                }
            } footer: {
                Text("Kept in the Keychain and given to the engine as environment variables at its next start, for compose cloud models.").font(.caption)
            }
            HStack {
                TextField("Variable", text: $name, prompt: Text("OPENROUTER_API_KEY")).monospaced()
                SecureField("Key", text: $value)
                Button("Add") {
                    guard keychain.write(name, value) else { return }
                    stored = (names.filter { $0 != name } + [name]).joined(separator: "\n")
                    (name, value) = ("", "")
                }
                .disabled(!CloudKeys.isName(name) || value.isEmpty)
            }
        }
        .padding(20)
    }
}

struct EngineSettings: View {
    let restart: @MainActor ([String]) async -> Void
    @AppStorage(EngineChoice.modelKey) private var model = AppModel.defaultModel
    @AppStorage(EngineChoice.composeKey) private var compose = ""
    @State private var choosing = false

    var arguments: [String] { EngineChoice.arguments(model: model, compose: compose) }

    var body: some View {
        Form {
            Picker("Run", selection: Binding(get: { compose.isEmpty ? 0 : 1 }, set: { if $0 == 0 { compose = "" } else { choosing = true } })) {
                Text("One model").tag(0)
                Text("A compose system").tag(1)
            }
            if compose.isEmpty {
                LabeledContent("Model", value: model)
            } else {
                LabeledContent("Compose file", value: compose)
                Button("Choose Another…") { choosing = true }
            }
            Text("basalt \(arguments.joined(separator: " "))").font(.caption.monospaced()).foregroundStyle(.secondary)
            Button("Restart the Engine with This") { Task { await restart(arguments) } }
        }
        .padding(20)
        .fileImporter(isPresented: $choosing, allowedContentTypes: [.yaml, .data]) { result in
            if let url = try? result.get() { compose = url.path }
        }
    }
}
