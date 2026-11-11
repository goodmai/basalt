import BasaltAppCore
import SwiftUI

public struct RootView: View {
    @Bindable var model: AppModel

    public init(model: AppModel) { self.model = model }

    public var body: some View {
        NavigationSplitView {
            List(selection: $model.selection) {
                ForEach(model.summaries) { summary in
                    Text(summary.title.isEmpty ? "New conversation" : summary.title)
                        .lineLimit(1)
                        .tag(summary.id)
                        .contextMenu {
                            Button("Delete", role: .destructive) { Task { await model.delete(summary.id) } }
                        }
                }
            }
            .navigationSplitViewColumnWidth(min: 200, ideal: 240)
            .toolbar {
                Button("New Conversation", systemImage: "square.and.pencil") { model.newConversation() }
                    .keyboardShortcut("n")
            }
        } detail: {
            VStack(spacing: 0) {
                EngineBanner(model: model)
                if let id = model.selection {
                    ConversationDetail(model: model, id: id).id(id)
                } else {
                    ContentUnavailableView("No Conversation", systemImage: "bubble.left.and.text.bubble.right",
                                           description: Text("Start one with ⌘N."))
                }
            }
        }
        .task { await model.start() }
    }
}

struct ConversationDetail: View {
    let model: AppModel
    let id: ConversationID
    @State private var store: ChatStore?

    var body: some View {
        Group {
            if let store { ChatView(store: store) } else { ProgressView() }
        }
        .task { store = await model.store(for: id) }
    }
}

struct EngineBanner: View {
    let model: AppModel

    var body: some View {
        switch model.engine.phase {
        case .ready: EmptyView()
        case .stopped: bar("Engine stopped", systemImage: "stop.circle", action: ("Start", { Task { await model.start() } }))
        case .spawning: bar("Starting the engine and loading \(model.model)…", systemImage: "hourglass", action: nil)
        case .loading: bar("Loading \(model.model)…", systemImage: "hourglass", action: nil)
        case .backingOff(let attempt): bar("The engine stopped unexpectedly; restarting (attempt \(attempt))", systemImage: "arrow.clockwise", action: nil)
        case .failed(let log):
            bar("The engine keeps failing: \(log.last ?? "no output")", systemImage: "exclamationmark.triangle",
                action: ("Restart", { Task { await model.restart() } }))
        }
    }

    func bar(_ text: String, systemImage: String, action: (String, () -> Void)?) -> some View {
        HStack {
            Label(text, systemImage: systemImage).lineLimit(2)
            Spacer()
            if let action { Button(action.0, action: action.1) }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
