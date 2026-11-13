import BasaltAppCore
import SwiftUI
import UniformTypeIdentifiers

struct ComposerGlassBar: View {
    let store: ChatStore
    @Namespace private var glassNamespace
    @FocusState private var focused: Bool
    @State private var importing = false
    @Environment(\.glassSurfacePolicy) private var policy

    private var morphTransition: GlassEffectTransition {
        policy.usesMorphAnimation ? .matchedGeometry : .identity
    }

    var body: some View {
        GlassEffectContainer(spacing: 10) {
            VStack(alignment: .leading, spacing: 8) {
                if !store.state.attachments.isEmpty {
                    attachmentStrip
                }
                inputRow
            }
        }
        .frame(maxWidth: 820)
        .padding(.horizontal, 16)
        .padding(.top, 6)
        .padding(.bottom, 12)
        .frame(maxWidth: .infinity)
        .fileImporter(
            isPresented: $importing,
            allowedContentTypes: [.pdf, .image, .audio],
            allowsMultipleSelection: true
        ) { result in
            for url in (try? result.get()) ?? [] {
                Self.attach(url, to: store)
            }
        }
        .onAppear { focused = true }
    }

    private var attachmentStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(store.state.attachments) { attachment in
                    Button(attachment.name, systemImage: "xmark.circle.fill") {
                        store.send(.detach(attachment.id))
                    }
                    .labelStyle(.titleAndIcon)
                    .font(.callout)
                    .adaptiveGlassButton()
                    .glassEffectID(ComposerGlassID.attachmentChip(attachment.id), in: glassNamespace)
                    .glassEffectTransition(morphTransition)
                    .accessibilityLabel("Remove \(attachment.name)")
                }
            }
        }
    }

    private var inputRow: some View {
        HStack(alignment: .bottom, spacing: 10) {
            Button("Attach", systemImage: "paperclip") { importing = true }
                .labelStyle(.iconOnly)
                .font(.title3)
                .controlSize(.large)
                .adaptiveGlassButton()
                .glassEffectID(ComposerGlassID.attach, in: glassNamespace)

            TextField(
                "Message",
                text: Binding(
                    get: { store.state.draft },
                    set: { store.send(.draftChanged($0)) }
                ),
                axis: .vertical
            )
            .textFieldStyle(.plain)
            .font(.body)
            .lineLimit(1 ... 10)
            .focused($focused)
            .onSubmit(send)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .frame(minHeight: 38)
            .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 20))
            .glassEffectID(ComposerGlassID.field, in: glassNamespace)

            submitOrStopButton
        }
    }

    @ViewBuilder
    private var submitOrStopButton: some View {
        if case .streaming = store.state.run {
            Button("Stop", systemImage: "stop.fill") { store.send(.stop) }
                .labelStyle(.iconOnly)
                .font(.title3)
                .controlSize(.large)
                .tint(.red)
                .adaptiveGlassButton(prominent: true)
                .keyboardShortcut(".", modifiers: .command)
                .glassEffectID(ComposerGlassID.submitOrStop, in: glassNamespace)
                .glassEffectTransition(morphTransition)
        } else {
            Button("Send", systemImage: "arrow.up", action: send)
                .labelStyle(.iconOnly)
                .font(.title3)
                .controlSize(.large)
                .adaptiveGlassButton(prominent: true)
                .disabled(!store.state.canSend)
                .glassEffectID(ComposerGlassID.submitOrStop, in: glassNamespace)
                .glassEffectTransition(morphTransition)
        }
    }

    private func send() {
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
    }

    static func attach(_ url: URL, to store: ChatStore) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        if let data = try? Data(contentsOf: url),
           let attachment = ChatAttachment(name: url.lastPathComponent, data: data) {
            store.send(.attach(attachment))
        }
    }
}

struct JumpToLatestPill: View {
    let unreadDeltas: Int
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Group {
                if unreadDeltas > 0 {
                    Label("Jump to Latest (\(unreadDeltas))", systemImage: "arrow.down.circle.fill")
                } else {
                    Label("Jump to Latest", systemImage: "arrow.down.circle.fill")
                }
            }
            .font(.callout.weight(.medium))
            .padding(.horizontal, 14)
            .padding(.vertical, 7)
        }
        .buttonStyle(.plain)
        .adaptiveGlass(.chromeInteractive, in: .capsule)
    }
}
