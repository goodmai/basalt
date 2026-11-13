import AppKit
import BasaltAppCore
import SwiftUI

public struct QuickAskView: View {
    let store: ChatStore
    let promote: () -> Void
    @Namespace private var glassNamespace
    @FocusState private var focused: Bool
    @Environment(\.glassSurfacePolicy) private var policy

    public init(store: ChatStore, promote: @escaping () -> Void) {
        (self.store, self.promote) = (store, promote)
    }

    private var morphTransition: GlassEffectTransition {
        policy.usesMorphAnimation ? .matchedGeometry : .identity
    }

    private var assistantReply: MessageSnapshot? {
        guard let last = store.state.conversation.messages.last, last.role == .assistant else { return nil }
        return last
    }

    public var body: some View {
        GlassEffectContainer(spacing: 14) {
            VStack(alignment: .leading, spacing: 10) {
                promptSurface
                if let reply = assistantReply {
                    answerSurface(reply)
                } else if case .failed(let error) = store.state.run {
                    errorSurface(error)
                }
            }
        }
        .foregroundStyle(.primary)
        .frame(width: 640)
        .padding(12)
        .background {
            Button("Open as Conversation", action: promote)
                .keyboardShortcut(.return, modifiers: .command)
                .hidden()
        }
        .onAppear { focused = true }
    }

    private var promptSurface: some View {
        HStack(spacing: 10) {
            Image(systemName: "sparkles")
                .font(.title3)
                .foregroundStyle(.tint)
            TextField(
                "Ask Basalt",
                text: Binding(
                    get: { store.state.draft },
                    set: { store.send(.draftChanged($0)) }
                )
            )
            .textFieldStyle(.plain)
            .font(.title3)
            .focused($focused)
            .onSubmit(submitIfReady)

            if case .streaming = store.state.run {
                Button("Stop", systemImage: "stop.fill") { store.send(.stop) }
                    .labelStyle(.iconOnly)
                    .tint(.red)
                    .adaptiveGlassButton(prominent: true)
                    .keyboardShortcut(".", modifiers: .command)
            } else {
                Button("Send", systemImage: "arrow.up", action: submitIfReady)
                    .labelStyle(.iconOnly)
                    .adaptiveGlassButton(prominent: true)
                    .disabled(!store.state.canSend)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 22))
        .glassEffectID(QuickAskGlassID.promptBar, in: glassNamespace)
        .glassEffectTransition(morphTransition)
    }

    private func answerSurface(_ reply: MessageSnapshot) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            ViewThatFits(in: .vertical) {
                MessageRow(message: reply)
                    .frame(maxWidth: .infinity, alignment: .leading)
                ScrollView {
                    MessageRow(message: reply)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 260)
            }

            HStack(spacing: 8) {
                Text("⌘↩ open as a conversation · Esc close")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if !reply.text.isEmpty {
                    Button("Copy", systemImage: "doc.on.doc") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(reply.text, forType: .string)
                    }
                    .controlSize(.small)
                    .adaptiveGlassButton()
                }
                Button("Open as Conversation", systemImage: "arrow.up.forward.app", action: promote)
                    .controlSize(.small)
                    .adaptiveGlassButton(prominent: true)
            }
        }
        .padding(16)
        .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 22))
        .glassEffectID(QuickAskGlassID.answerBody, in: glassNamespace)
        .glassEffectTransition(morphTransition)
    }

    private func errorSurface(_ error: EngineError) -> some View {
        Label(ChatView.describe(error), systemImage: "exclamationmark.triangle.fill")
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .adaptiveGlass(.chromeRegular, in: .rect(cornerRadius: 18))
            .glassEffectTransition(morphTransition)
    }

    private func submitIfReady() {
        guard store.state.canSend else { return }
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
    }
}
