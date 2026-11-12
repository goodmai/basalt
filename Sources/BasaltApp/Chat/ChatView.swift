import BasaltAppCore
import SwiftUI

struct ChatView: View {
    let store: ChatStore
    @State private var selected: MessageID?
    @State private var inspecting = false

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 18) {
                        ForEach(store.state.conversation.messages) { message in
                            MessageRow(message: message).id(message.id)
                                .onTapGesture { if message.role == .assistant { (selected, inspecting) = (message.id, true) } }
                        }
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
        .dropDestination(for: URL.self) { urls, _ in
            let accepted = urls.filter { ChatAttachment.accepts($0.lastPathComponent) }
            for url in accepted { Composer.attach(url, to: store) }
            return !accepted.isEmpty
        }
        .navigationTitle(store.state.conversation.title.isEmpty ? "New Conversation" : store.state.conversation.title)
        .inspector(isPresented: $inspecting) { InspectorView(store: store, selected: selected).inspectorColumnWidth(min: 260, ideal: 320) }
        .toolbar {
            Button("Inspector", systemImage: "sidebar.trailing") { inspecting.toggle() }.keyboardShortcut("i", modifiers: [.command, .option])
        }
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
                VStack(alignment: .trailing, spacing: 6) {
                    ForEach(Array(message.attachments.enumerated()), id: \.offset) { _, attachment in AttachmentChip(attachment: attachment) }
                    if !message.text.isEmpty {
                        Text(message.text)
                            .textSelection(.enabled)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 9)
                            .background(.tint.opacity(0.15), in: .rect(cornerRadius: 16))
                    }
                }
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
        case .attachment(let attachment):
            AttachmentChip(attachment: attachment)
        }
    }
}

struct AttachmentChip: View {
    let attachment: AttachmentPart

    var body: some View {
        DisclosureGroup {
            Text(attachment.perceived.isEmpty ? "Reading…" : attachment.perceived)
                .font(.callout).foregroundStyle(.secondary).textSelection(.enabled).frame(maxWidth: 520, alignment: .leading)
        } label: {
            Label(attachment.name, systemImage: Self.symbol(attachment.kind))
        }
        .padding(.horizontal, 12).padding(.vertical, 6)
        .background(.background.secondary, in: .rect(cornerRadius: 12))
    }

    static func symbol(_ kind: String) -> String {
        switch kind {
        case "pdf": "doc.richtext"
        case "audio": "waveform"
        default: "photo"
        }
    }
}

struct ReceiptLine: View {
    let message: MessageSnapshot

    var body: some View {
        let parts = [
            message.receipt?.model,
            message.receipt?.tokensPerSecond.map { String(format: "%.1f tok/s", $0) },
            message.receipt?.stages.map { $0.map { "\($0.stage) \($0.model) \($0.ms) ms" }.joined(separator: ", ") },
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

    @State private var importing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !store.state.attachments.isEmpty {
                HStack {
                    ForEach(store.state.attachments) { attachment in
                        Button(attachment.name, systemImage: "xmark.circle.fill") { store.send(.detach(attachment.id)) }
                            .labelStyle(.titleAndIcon).font(.callout).buttonStyle(.bordered)
                    }
                }
            }
            HStack(alignment: .bottom, spacing: 8) {
                Button("Attach", systemImage: "paperclip") { importing = true }
                field
            }
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.pdf, .image, .audio], allowsMultipleSelection: true) { result in
            for url in (try? result.get()) ?? [] { Composer.attach(url, to: store) }
        }
    }

    static func attach(_ url: URL, to store: ChatStore) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        if let data = try? Data(contentsOf: url), let attachment = ChatAttachment(name: url.lastPathComponent, data: data) { store.send(.attach(attachment)) }
    }

    var field: some View {
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
