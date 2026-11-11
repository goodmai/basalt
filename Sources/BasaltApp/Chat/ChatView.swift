import BasaltAppCore
import SwiftUI

struct ChatView: View {
    let store: ChatStore

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 18) {
                        ForEach(store.state.conversation.messages) { MessageRow(message: $0).id($0.id) }
                    }
                    .padding(20)
                    .frame(maxWidth: 820)
                    .frame(maxWidth: .infinity)
                }
                .onChange(of: store.state.conversation.messages.last?.text) {
                    if let last = store.state.conversation.messages.last { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
            if case .failed(let error) = store.state.run {
                Label(ChatView.describe(error), systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 20)
            }
            Composer(store: store)
        }
        .navigationTitle(store.state.conversation.title.isEmpty ? "New Conversation" : store.state.conversation.title)
    }

    static func describe(_ error: EngineError) -> String {
        switch error {
        case .notRunning, .starting: "The engine is not ready yet."
        case .refused(let status, let message): "The engine refused the request (\(status)): \(message)"
        case .malformed: "The engine sent an answer that could not be read."
        case .transport(let message): "Lost the connection to the engine: \(message)"
        case .cancelled: "Stopped."
        }
    }
}

struct MessageRow: View {
    let message: MessageSnapshot

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 80)
                Text(message.text)
                    .textSelection(.enabled)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 9)
                    .background(.tint.opacity(0.15), in: .rect(cornerRadius: 16))
            }
        } else {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(message.parts.enumerated()), id: \.offset) { _, part in PartView(part: part) }
                if message.state == .streaming, message.parts.isEmpty { ProgressView().controlSize(.small) }
                ReceiptLine(message: message)
            }
        }
    }
}

struct PartView: View {
    let part: MessagePart

    var body: some View {
        switch part {
        case .text(let text):
            Text((try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text))
                .textSelection(.enabled)
        case .reasoning(let text):
            DisclosureGroup("Thinking") {
                Text(text).foregroundStyle(.secondary).textSelection(.enabled)
            }
        case .toolCall(let call):
            Label("\(call.name)(\(call.arguments))", systemImage: "wrench.and.screwdriver").font(.callout.monospaced())
        }
    }
}

struct ReceiptLine: View {
    let message: MessageSnapshot

    var body: some View {
        let parts = [
            message.receipt?.model,
            message.receipt?.tokensPerSecond.map { String(format: "%.1f tok/s", $0) },
            message.state == .stopped ? "stopped" : nil,
            message.state == .failed ? "failed" : nil,
        ].compactMap { $0 }
        if !parts.isEmpty, message.state != .streaming {
            Text(parts.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
        }
    }
}

struct Composer: View {
    let store: ChatStore
    @FocusState private var focused: Bool

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: Binding(get: { store.state.draft }, set: { store.send(.draftChanged($0)) }), axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1 ... 10)
                .focused($focused)
                .onSubmit(send)
            if case .streaming = store.state.run {
                Button("Stop", systemImage: "stop.circle.fill") { store.send(.stop) }
                    .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Send", systemImage: "arrow.up.circle.fill", action: send)
                    .disabled(!store.state.canSend)
            }
        }
        .labelStyle(.iconOnly)
        .font(.title3)
        .padding(12)
        .background(.background.secondary, in: .rect(cornerRadius: 18))
        .padding(16)
        .onAppear { focused = true }
    }

    func send() {
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
    }
}
