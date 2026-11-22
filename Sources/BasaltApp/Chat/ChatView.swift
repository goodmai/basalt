import BasaltAppCore
import SwiftUI

struct ChatView: View {
    let store: ChatStore
    var model: AppModel? = nil
    var voice: AudioSessionController? = nil
    @State private var selected: MessageID?
    @State private var inspecting = false
    @Environment(\.glassSurfacePolicy) private var policy

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    ForEach(store.state.conversation.messages) { message in
                        messageView(message)
                            .id(message.id)
                            .onTapGesture {
                                if message.role == .assistant {
                                    (selected, inspecting) = (message.id, true)
                                }
                            }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 20)
                .padding(.bottom, 12)
                .frame(maxWidth: 820)
                .frame(maxWidth: .infinity)
            }
            .scrollEdgeEffectStyle(policy.usesLiquidGlass ? .automatic : .hard, for: [.top, .bottom])
            .onScrollGeometryChange(for: Bool.self) { geo in
                geo.contentSize.height <= geo.containerSize.height
                    || (geo.contentOffset.y + geo.containerSize.height >= geo.contentSize.height - 36)
            } action: { _, atBottom in
                if atBottom != store.state.scrollPin.isPinned {
                    store.send(.scrolled(atBottom: atBottom))
                }
            }
            .onChange(of: store.state.conversation.messages.last?.text) {
                if store.state.scrollPin.isPinned, let last = store.state.conversation.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
            .onChange(of: store.state.scrollPin) { previous, current in
                if !previous.isPinned, current.isPinned, let last = store.state.conversation.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
            .safeAreaBar(edge: .bottom) {
                bottomChrome
            }
        }
        .dropDestination(for: URL.self) { urls, _ in
            let accepted = urls.filter { ChatAttachment.accepts($0.lastPathComponent) }
            for url in accepted { ComposerGlassBar.attach(url, to: store) }
            return !accepted.isEmpty
        }
        .navigationTitle(store.state.conversation.title.isEmpty ? "New Conversation" : store.state.conversation.title)
        .inspector(isPresented: $inspecting) {
            InspectorView(store: store, selected: selected).inspectorColumnWidth(min: 260, ideal: 320)
        }
        .toolbar {
            if let model {
                ComposePickerMenu(model: model)
            }
            if let voice {
                VoiceToolbarItems(voice: voice)
            }
            Button("Inspector", systemImage: "sidebar.trailing") { inspecting.toggle() }
                .keyboardShortcut("i", modifiers: [.command, .option])
        }
    }

    private func messageView(_ message: MessageSnapshot) -> some View {
        let branch = store.state.branches.position(for: message.id, in: store.state.conversation)
        let speaking = voice?.store.state.speakingMessage == message.id || voice?.store.state.synthesizingMessage == message.id
        return MessageRow(
            message: message,
            branch: branch,
            onSelectBranch: { index in store.send(.selectBranch(message: message.id, index: index)) },
            onRegenerate: { store.send(.regenerateFrom(message: message.id, reply: MessageID(), run: RunID())) },
            onEditImage: { part in store.send(.editGeneratedImage(part)) },
            onReadAloud: voice.map { controller in { controller.store.send(.readAloud(message.id, message.text)) } },
            isSpeaking: speaking
        )
    }

    private var bottomChrome: some View {
        VStack(spacing: 8) {
            if let voice, voice.store.state.isActive {
                VoiceStatusPill(voice: voice)
            }
            if store.state.scrollPin.showsJumpControl {
                JumpToLatestPill(unreadDeltas: store.state.scrollPin.unreadDeltas) {
                    store.send(.jumpToBottom)
                }
            }
            if case .failed(let error) = store.state.run {
                Label(ChatView.describe(error), systemImage: "exclamationmark.triangle")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 6)
            }
            ComposerGlassBar(store: store, showsImageToggle: model != nil || store.state.usesImageComposer)
        }
    }

    static func describe(_ error: EngineError) -> String {
        switch error {
        case .notRunning, .starting: String(localized: "The engine is not ready yet.")
        case .refused(let status, let message): String(localized: "The engine refused the request (\(status)): \(message)")
        case .malformed: String(localized: "The engine sent an answer that could not be read.")
        case .transport(let message): String(localized: "Lost the connection to the engine: \(message)")
        case .cancelled: String(localized: "Stopped.")
        }
    }
}

struct VoiceToolbarItems: View {
    let voice: AudioSessionController

    var body: some View {
        Button(voice.store.state.isRecording ? "Finish Recording" : "Record Voice",
               systemImage: voice.store.state.isRecording ? "record.circle.fill" : "mic") {
            voice.store.send(voice.store.state.isRecording ? .pushToTalkReleased : .pushToTalkPressed)
        }
        Button("Hands-Free Voice",
               systemImage: voice.store.state.handsFree ? "waveform.circle.fill" : "waveform.circle") {
            voice.store.send(.setHandsFree(!voice.store.state.handsFree))
        }
    }
}

struct VoiceStatusPill: View {
    let voice: AudioSessionController

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: voice.store.state.isRecording ? "mic.fill" : "waveform")
                .foregroundStyle(voice.store.state.isRecording ? .red : .accentColor)
            Text(statusText)
                .font(.caption.weight(.medium))
            if voice.store.state.isSpeaking {
                Button("Stop Speaking") { voice.store.send(.stopSpeaking) }
                    .buttonStyle(.plain)
                    .font(.caption)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .adaptiveGlass(.chromeRegular, in: .capsule)
    }

    private var statusText: LocalizedStringKey {
        switch voice.store.state.capture {
        case .recording: "Listening…"
        case .transcribing: "Transcribing…"
        case .idle: voice.store.state.isSpeaking ? "Speaking…" : "Hands-free active"
        }
    }
}

struct MessageRow: View {
    let message: MessageSnapshot
    var branch: BranchPosition? = nil
    var onSelectBranch: (Int) -> Void = { _ in }
    var onRegenerate: () -> Void = {}
    var onEditImage: (GeneratedImagePart) -> Void = { _ in }
    var onReadAloud: (() -> Void)? = nil
    var isSpeaking = false

    var body: some View {
        content
            .accessibilityElement(children: .combine)
            .accessibilityLabel(message.role == .user ? Text("You: \(summary)") : Text("Basalt: \(summary)"))
            .contextMenu {
                if message.role == .assistant {
                    if let onReadAloud, !message.text.isEmpty {
                        Button(isSpeaking ? "Stop Speaking" : "Read Aloud",
                               systemImage: isSpeaking ? "speaker.slash" : "speaker.wave.2",
                               action: onReadAloud)
                    }
                    if message.state != .streaming {
                        Button("Regenerate", systemImage: "arrow.clockwise", action: onRegenerate)
                    }
                }
            }
    }

    var summary: String {
        if !message.text.isEmpty { return message.text }
        if let image = message.images.first { return image.prompt }
        return message.attachments.map(\.name).joined(separator: ", ")
    }

    @ViewBuilder var content: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 80)
                VStack(alignment: .trailing, spacing: 6) {
                    ForEach(Array(message.attachments.enumerated()), id: \.offset) { _, attachment in
                        AttachmentChip(attachment: attachment)
                    }
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
                ForEach(Array(message.parts.enumerated()), id: \.offset) { _, part in
                    PartView(part: part, isStreaming: message.state == .streaming, onEditImage: onEditImage)
                }
                if message.state == .streaming, message.parts.isEmpty {
                    ProgressView().controlSize(.small).accessibilityLabel("Answering")
                }
                if let branch {
                    HStack(spacing: 10) {
                        BranchSwitcherView(branch: branch, onSelect: onSelectBranch)
                        ReceiptLine(message: message)
                    }
                } else {
                    ReceiptLine(message: message)
                }
            }
        }
    }
}

struct BranchSwitcherView: View {
    let branch: BranchPosition
    let onSelect: (Int) -> Void

    var body: some View {
        HStack(spacing: 4) {
            Button("Previous Answer", systemImage: "chevron.left") { onSelect(branch.index - 1) }
                .labelStyle(.iconOnly)
                .buttonStyle(.plain)
                .disabled(!branch.hasPrevious)
            Text(verbatim: branch.counter)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Button("Next Answer", systemImage: "chevron.right") { onSelect(branch.index + 1) }
                .labelStyle(.iconOnly)
                .buttonStyle(.plain)
                .disabled(!branch.hasNext)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

struct PartView: View {
    let part: MessagePart
    var isStreaming: Bool = false
    var onEditImage: (GeneratedImagePart) -> Void = { _ in }

    var body: some View {
        switch part {
        case .text(let text):
            if MarkdownBlockParser.hasStructuredBlocks(text) {
                MarkdownBlocksView(markdown: text)
            } else {
                Text((try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text))
                    .textSelection(.enabled)
            }
        case .reasoning(let text):
            DisclosureGroup("Thinking") {
                Text(text).foregroundStyle(.secondary).textSelection(.enabled)
            }
        case .toolCall(let call):
            Label("\(call.name)(\(call.arguments))", systemImage: "wrench.and.screwdriver").font(.callout.monospaced())
        case .attachment(let attachment):
            AttachmentChip(attachment: attachment)
        case .image(let image):
            GeneratedImageCard(part: image, isStreaming: isStreaming, onEdit: onEditImage)
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
            message.receipt?.stages.flatMap { stages in stages.contains(where: \.isCloud) ? String(format: "☁︎ cloud $%.4f", stages.compactMap(\.cost).reduce(0, +)) : nil },
            message.receipt?.stages.map { $0.map { "\($0.stage) \($0.model) \($0.ms) ms" }.joined(separator: ", ") },
            message.state == .stopped ? "stopped" : nil,
            message.state == .failed ? "failed" : nil,
        ].compactMap { $0 }
        if !parts.isEmpty, message.state != .streaming {
            Text(parts.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
        }
    }
}
