import BasaltAppCore
import Foundation
import SwiftData

@Model
final class ConversationRecord {
    @Attribute(.unique) var id: UUID
    var title: String
    var created: Date
    var updated: Date
    var model: String?
    var systemPrompt: String?
    @Relationship(deleteRule: .cascade, inverse: \MessageRecord.conversation) var messages: [MessageRecord] = []

    init(id: UUID, created: Date) {
        (self.id, self.title, self.created, self.updated) = (id, "", created, created)
    }
}

@Model
final class MessageRecord {
    @Attribute(.unique) var id: UUID
    var order: Int
    var role: String
    var state: String
    var parts: Data
    var receipt: Data?
    var text: String
    var conversation: ConversationRecord?

    init(id: UUID, order: Int) {
        (self.id, self.order, self.role, self.state, self.parts, self.text) = (id, order, "", "", Data(), "")
    }
}

enum RecordMapping {
    static func fill(_ record: MessageRecord, from message: MessageSnapshot, order: Int) {
        record.order = order
        record.role = message.role.rawValue
        record.state = message.state.rawValue
        record.parts = (try? JSONEncoder().encode(message.parts)) ?? Data()
        record.receipt = message.receipt.flatMap { try? JSONEncoder().encode($0) }
        record.text = message.text
    }

    static func snapshot(_ record: MessageRecord) -> MessageSnapshot {
        var message = MessageSnapshot(
            id: MessageID(record.id), role: MessageSnapshot.Role(rawValue: record.role) ?? .user,
            parts: (try? JSONDecoder().decode([MessagePart].self, from: record.parts)) ?? [.text(record.text)],
            state: MessageSnapshot.State(rawValue: record.state) ?? .complete)
        message.receipt = record.receipt.flatMap { try? JSONDecoder().decode(Receipt.self, from: $0) }
        return message
    }

    static func snapshot(_ record: ConversationRecord) -> ConversationSnapshot {
        var conversation = ConversationSnapshot(id: ConversationID(record.id), model: record.model, systemPrompt: record.systemPrompt)
        conversation.title = record.title
        conversation.messages = record.messages.sorted { $0.order < $1.order }.map(snapshot)
        return conversation
    }
}
