import BasaltAppCore
import SwiftUI

public struct QuickAskView: View {
    let store: ChatStore
    let promote: () -> Void
    @FocusState private var focused: Bool

    public init(store: ChatStore, promote: @escaping () -> Void) { (self.store, self.promote) = (store, promote) }

    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Ask Basalt", text: Binding(get: { store.state.draft }, set: { store.send(.draftChanged($0)) }))
                .textFieldStyle(.plain)
                .font(.title3)
                .focused($focused)
                .onSubmit { if store.state.canSend { store.send(.send(message: MessageID(), reply: MessageID(), run: RunID())) } }
            if let reply = store.state.conversation.messages.last, reply.role == .assistant {
                ScrollView { MessageRow(message: reply).frame(maxWidth: .infinity, alignment: .leading) }
                    .frame(maxHeight: 360)
                Text("⌘↩ open as a conversation · Esc close").font(.caption).foregroundStyle(.secondary)
            }
            if case .failed(let error) = store.state.run {
                Label(ChatView.describe(error), systemImage: "exclamationmark.triangle").font(.callout).foregroundStyle(.secondary)
            }
        }
        .padding(18)
        .frame(width: 640)
        .glassEffect(in: .rect(cornerRadius: 22))
        .background { Button("Open as Conversation", action: promote).keyboardShortcut(.return, modifiers: .command).hidden() }
        .onAppear { focused = true }
    }
}
