import BasaltAppCore
import Foundation

extension AppModel {
    public func startImageConversation(prompt: String, attachments: [ChatAttachment] = []) {
        newConversation()
        guard let id = selection, let store = chats[id] else { return }
        if !store.state.imageMode {
            store.send(.toggleImageMode)
        }
        store.send(.draftChanged(prompt))
        for attachment in attachments {
            store.send(.attach(attachment))
        }
        store.send(.send(message: MessageID(), reply: MessageID(), run: RunID()))
    }

    public func regenerateSelectedAnswer() {
        guard let id = selection, let store = chats[id] else { return }
        store.send(.regenerate(reply: MessageID(), run: RunID()))
    }
}
