import BasaltAppCore
import SwiftUI

public struct MenuBarContent: View {
    let model: AppModel
    let quickAsk: () -> Void
    @Environment(\.openWindow) private var openWindow

    public init(model: AppModel, quickAsk: @escaping () -> Void) {
        (self.model, self.quickAsk) = (model, quickAsk)
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            statusHeader
            if let update = model.update {
                Link("Update available: \(update.version)", destination: update.url)
                    .font(.callout)
            }
            Divider()
            menuRow("New Conversation", systemImage: "square.and.pencil") { show { model.newConversation() } }
            menuRow("Quick Ask  ⌥Space", systemImage: "sparkles", action: quickAsk)
            if !model.summaries.isEmpty {
                Divider()
                ForEach(model.summaries.prefix(5)) { summary in
                    Button(summary.title.isEmpty ? "New conversation" : summary.title) {
                        show { model.selection = summary.id }
                    }
                    .buttonStyle(.plain)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                }
            }
            Divider()
            menuRow("Models", systemImage: "square.stack.3d.up") { openWindow(id: "library") }
            menuRow("Monitor", systemImage: "gauge.with.dots.needle.33percent") { openWindow(id: "monitor") }
            SettingsLink { Label("Settings…", systemImage: "gearshape") }
                .buttonStyle(.plain)
            menuRow("Quit Basalt", systemImage: "power") { NSApplication.shared.terminate(nil) }
        }
        .padding(14)
        .frame(width: 280, alignment: .leading)
    }

    private var statusHeader: some View {
        GlassEffectContainer(spacing: 8) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    Image(systemName: Self.ready(model.engine.phase) ? "circle.fill" : "circle.dotted")
                        .font(.caption)
                        .foregroundStyle(Self.ready(model.engine.phase) ? .green : .secondary)
                    Text(Self.status(model.engine.phase))
                        .font(.callout.weight(.semibold))
                    Spacer()
                }
                Text(model.model)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                HStack(spacing: 6) {
                    engineButtons
                }
            }
            .padding(10)
            .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 14))
        }
    }

    @ViewBuilder
    private var engineButtons: some View {
        if model.engine.isIdle {
            Button("Start Engine") { Task { await model.start() } }
                .controlSize(.small)
                .adaptiveGlassButton(prominent: true)
        } else {
            Button("Restart Engine") { Task { await model.restart() } }
                .controlSize(.small)
                .adaptiveGlassButton()
            Button("Stop Engine") { Task { await model.stop() } }
                .controlSize(.small)
                .adaptiveGlassButton()
        }
    }

    private func menuRow(_ title: LocalizedStringKey, systemImage: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
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
        case .stopped: String(localized: "Engine stopped")
        case .spawning: String(localized: "Starting the engine")
        case .loading: String(localized: "Loading the model")
        case .ready: String(localized: "Ready")
        case .backingOff(let attempt): String(localized: "Restarting (attempt \(attempt))")
        case .failed: String(localized: "The engine keeps failing")
        }
    }
}
