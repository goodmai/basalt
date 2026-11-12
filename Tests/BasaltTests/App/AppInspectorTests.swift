import AppKit
import BasaltCore
import Foundation
import SwiftUI
import Testing
@testable import BasaltApp
@testable import BasaltAppCore

@Suite("Mac app inspector: the conversation's system prompt, and a receipt with tokens, first-token time, speed, stages and perceived attachments (#434 phase 3c)")
@MainActor
struct AppInspectorTests {
    @Test("A system prompt is trimmed and saved, an empty one clears it, and an unchanged one saves nothing")
    func systemPrompt() {
        var state = ChatState(conversation: ConversationSnapshot(id: ConversationID()))
        #expect(ChatReducer.reduce(&state, .systemPromptChanged("  Be brief.  ")) == [.persist(state.conversation)] && state.conversation.systemPrompt == "Be brief.")
        #expect(ChatReducer.reduce(&state, .systemPromptChanged("Be brief.")).isEmpty)
        #expect(ChatReducer.reduce(&state, .systemPromptChanged(" ")).count == 1 && state.conversation.systemPrompt == nil)
        #expect(ChatReducer.reduce(&state, .systemPromptChanged("")).isEmpty)
    }

    @Test("The inspector lays out the selected answer's receipt and the attachments of its question")
    func renders() {
        var conversation = ConversationSnapshot(id: ConversationID(), model: "org/model", systemPrompt: "Be brief.")
        var answer = MessageSnapshot(id: MessageID(), role: .assistant, parts: [.text("Pay 42.")])
        var receipt = Receipt(model: "org/model")
        (receipt.promptTokens, receipt.completionTokens, receipt.timeToFirstToken, receipt.tokensPerSecond) = (30, 4, .milliseconds(240), 18.5)
        receipt.stages = [StageReceipt(stage: "perceive", model: "apple-vision", ms: 80), StageReceipt(stage: "answer", model: "org/model", ms: 900)]
        answer.receipt = receipt
        conversation.messages = [MessageSnapshot(id: MessageID(), role: .user, parts: [.attachment(AttachmentPart(name: "bill.pdf", kind: "pdf", perceived: "Total 42"))]), answer]
        let store = Store(ChatState(conversation: conversation), reduce: ChatReducer.reduce, handle: { _ in .none })
        for selected in [answer.id, nil] {
            let host = NSHostingView(rootView: InspectorView(store: store, selected: selected).frame(width: 320, height: 600))
            host.layoutSubtreeIfNeeded()
            #expect(host.fittingSize.width > 0)
        }
        #expect(InspectorView(store: store, selected: nil).question?.attachments.first?.name == "bill.pdf")
    }
}
