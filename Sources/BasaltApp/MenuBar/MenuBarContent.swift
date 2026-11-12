import BasaltAppCore
import SwiftUI

public struct MenuBarContent: View {
    let model: AppModel
    let quickAsk: () -> Void
    @Environment(\.openWindow) private var openWindow

    public init(model: AppModel, quickAsk: @escaping () -> Void) { (self.model, self.quickAsk) = (model, quickAsk) }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(Self.status(model.engine.phase), systemImage: Self.ready(model.engine.phase) ? "circle.fill" : "circle.dotted")
                .foregroundStyle(Self.ready(model.engine.phase) ? .green : .secondary)
            Text(model.model).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            Divider()
            Button("New Conversation") { show { model.newConversation() } }
            Button("Quick Ask  ⌥Space", action: quickAsk)
            if !model.summaries.isEmpty {
                Divider()
                ForEach(model.summaries.prefix(5)) { summary in
                    Button(summary.title.isEmpty ? "New conversation" : summary.title) { show { model.selection = summary.id } }.lineLimit(1)
                }
            }
            Divider()
            engineButtons
            Button("Models") { openWindow(id: "library") }
            Button("Monitor") { openWindow(id: "monitor") }
            SettingsLink { Text("Settings…") }
            Button("Quit Basalt") { NSApplication.shared.terminate(nil) }
        }
        .buttonStyle(.plain)
        .padding(14)
        .frame(width: 280, alignment: .leading)
    }

    @ViewBuilder var engineButtons: some View {
        if model.engine.isIdle {
            Button("Start Engine") { Task { await model.start() } }
        } else {
            Button("Restart Engine") { Task { await model.restart() } }
            Button("Stop Engine") { Task { await model.stop() } }
        }
    }

    func show(_ change: () -> Void) {
        change()
        openWindow(id: "main")
        NSApplication.shared.activate()
    }

    static func ready(_ phase: EngineState.Phase) -> Bool {
        if case .ready = phase { true } else { false }
    }

    static func status(_ phase: EngineState.Phase) -> String {
        switch phase {
        case .stopped: "Engine stopped"
        case .spawning: "Starting the engine"
        case .loading: "Loading the model"
        case .ready: "Ready"
        case .backingOff(let attempt): "Restarting (attempt \(attempt))"
        case .failed: "The engine keeps failing"
        }
    }
}
