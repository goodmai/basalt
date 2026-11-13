import BasaltAppCore
import SwiftUI

public struct RootView: View {
    @Bindable var model: AppModel
    @SceneStorage("selection") private var restored = ""

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
            detailContent
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .safeAreaBar(edge: .top) {
                    EngineStatusBar(model: model)
                }
        }
        .onAppear {
            if model.selection == nil, let uuid = UUID(uuidString: restored) {
                model.selection = ConversationID(uuid)
            }
        }
        .onChange(of: model.selection) { _, id in restored = id?.raw.uuidString ?? "" }
    }

    @ViewBuilder
    private var detailContent: some View {
        if let id = model.selection {
            ConversationDetail(model: model, id: id).id(id)
        } else {
            ContentUnavailableView(
                "No Conversation",
                systemImage: "bubble.left.and.text.bubble.right",
                description: Text("Start one with ⌘N.")
            )
        }
    }
}

struct ConversationDetail: View {
    let model: AppModel
    let id: ConversationID
    @State private var store: ChatStore?

    var body: some View {
        Group {
            if let store {
                ChatView(store: store)
            } else {
                ProgressView().accessibilityLabel("Loading the conversation")
            }
        }
        .task { store = await model.store(for: id) }
        .onChange(of: store?.state.draft ?? "") { _, draft in model.drafts.save(id, draft) }
    }
}

struct EngineStatusBar: View {
    let model: AppModel
    @Namespace private var glassNamespace
    @Environment(\.glassSurfacePolicy) private var policy

    var body: some View {
        switch model.engine.phase {
        case .ready:
            EmptyView()
        case .stopped:
            bar("Engine stopped", systemImage: "stop.circle", tint: .secondary,
                action: ("Start", { Task { await model.start() } }))
        case .spawning:
            bar("Starting the engine and loading \(model.model)…", systemImage: "hourglass", tint: .secondary, action: nil)
        case .loading:
            bar("Loading \(model.model)…", systemImage: "hourglass", tint: .secondary, action: nil)
        case .backingOff(let attempt):
            bar("The engine stopped unexpectedly; restarting (attempt \(attempt))",
                systemImage: "arrow.clockwise", tint: .orange, action: nil)
        case .failed(let log):
            bar("The engine keeps failing: \(log.last ?? "no output")",
                systemImage: "exclamationmark.triangle.fill", tint: .red,
                action: ("Restart", { Task { await model.restart() } }))
        }
    }

    private func bar(
        _ text: LocalizedStringKey,
        systemImage: String,
        tint: Color,
        action: (LocalizedStringKey, () -> Void)?
    ) -> some View {
        GlassEffectContainer(spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: systemImage)
                    .foregroundStyle(tint)
                Text(text)
                    .font(.callout)
                    .lineLimit(2)
                Spacer(minLength: 8)
                if let action {
                    Button(action.0, action: action.1)
                        .controlSize(.small)
                        .adaptiveGlassButton(prominent: true)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .adaptiveGlass(.chromeRegular, in: .capsule)
            .glassEffectID("engine-status-pill", in: glassNamespace)
            .glassEffectTransition(policy.usesMorphAnimation ? .matchedGeometry : .identity)
        }
        .frame(maxWidth: 760)
        .padding(.horizontal, 16)
        .padding(.top, 8)
        .padding(.bottom, 4)
    }
}
