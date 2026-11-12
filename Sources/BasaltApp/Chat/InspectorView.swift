import BasaltAppCore
import SwiftUI

struct InspectorView: View {
    let store: ChatStore
    let selected: MessageID?
    @State private var prompt = ""

    var answer: MessageSnapshot? {
        let messages = store.state.conversation.messages
        return messages.first { $0.id == selected && $0.role == .assistant } ?? messages.last { $0.role == .assistant }
    }

    var question: MessageSnapshot? {
        let messages = store.state.conversation.messages
        guard let answer, let index = messages.firstIndex(where: { $0.id == answer.id }), index > 0 else { return nil }
        return messages[index - 1]
    }

    var body: some View {
        Form {
            Section("Conversation") {
                LabeledContent("Model", value: store.state.conversation.model ?? "the engine's model")
                TextField("System prompt", text: $prompt, prompt: Text("None"), axis: .vertical)
                    .lineLimit(3 ... 12)
                    .onSubmit { store.send(.systemPromptChanged(prompt)) }
                Button("Save System Prompt") { store.send(.systemPromptChanged(prompt)) }
                    .disabled(prompt == (store.state.conversation.systemPrompt ?? ""))
            }
            if let receipt = answer?.receipt {
                Section("Answer") { ReceiptDetails(receipt: receipt) }
            }
            if let attachments = question?.attachments, !attachments.isEmpty {
                Section("Attachments") {
                    ForEach(Array(attachments.enumerated()), id: \.offset) { _, attachment in
                        DisclosureGroup(attachment.name) {
                            Text(attachment.perceived.isEmpty ? "Not read yet" : attachment.perceived).font(.callout).textSelection(.enabled)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .onAppear { prompt = store.state.conversation.systemPrompt ?? "" }
    }
}

struct ReceiptDetails: View {
    let receipt: Receipt

    var body: some View {
        LabeledContent("Model", value: receipt.model ?? "unknown")
        LabeledContent("Prompt tokens", value: "\(receipt.promptTokens)")
        LabeledContent("Completion tokens", value: "\(receipt.completionTokens)")
        if let ttft = receipt.timeToFirstToken {
            LabeledContent("First token", value: ttft.formatted(.units(allowed: [.seconds, .milliseconds], width: .abbreviated)))
        }
        if let speed = receipt.tokensPerSecond { LabeledContent("Speed", value: String(format: "%.1f tok/s", speed)) }
        ForEach(Array((receipt.stages ?? []).enumerated()), id: \.offset) { _, stage in
            LabeledContent(stage.stage.capitalized, value: "\(stage.model) · \(stage.ms) ms" + (stage.cost.map { String(format: " · cloud $%.4f", $0) } ?? ""))
        }
    }
}
