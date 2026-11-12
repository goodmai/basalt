import BasaltCore
import Foundation

public struct Identifier<Tag>: Hashable, Sendable, Codable {
    public let raw: UUID

    public init(_ raw: UUID = UUID()) { self.raw = raw }

    public init(from decoder: any Decoder) throws { raw = try decoder.singleValueContainer().decode(UUID.self) }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(raw)
    }
}

public struct ConversationSummary: Sendable, Equatable, Identifiable {
    public let id: ConversationID
    public var title: String
    public var updated: Date

    public init(id: ConversationID, title: String, updated: Date) { (self.id, self.title, self.updated) = (id, title, updated) }
}

public enum ConversationTag {}
public enum MessageTag {}
public enum RunTag {}

public typealias ConversationID = Identifier<ConversationTag>
public typealias MessageID = Identifier<MessageTag>
public typealias RunID = Identifier<RunTag>

public enum MessagePart: Sendable, Equatable, Codable {
    case text(String)
    case reasoning(String)
    case toolCall(ToolCallOutput)
    case attachment(AttachmentPart)
}

public struct AttachmentPart: Sendable, Equatable, Codable {
    public var name: String
    public var kind: String
    public var perceived: String

    public init(name: String, kind: String, perceived: String = "") { (self.name, self.kind, self.perceived) = (name, kind, perceived) }
}

public struct StageReceipt: Sendable, Equatable, Codable {
    public var stage: String
    public var model: String
    public var ms: Int
    public var cost: Double?

    public init(stage: String, model: String, ms: Int, cost: Double? = nil) { (self.stage, self.model, self.ms, self.cost) = (stage, model, ms, cost) }

    public var isCloud: Bool { cost != nil }
}

public struct Receipt: Sendable, Equatable, Codable {
    public var model: String?
    public var promptTokens = 0
    public var completionTokens = 0
    public var timeToFirstToken: Duration?
    public var tokensPerSecond: Double?
    public var stages: [StageReceipt]?

    public init(model: String? = nil) { self.model = model }
}

public struct MessageSnapshot: Sendable, Equatable, Identifiable {
    public enum Role: String, Sendable, Equatable, Codable { case system, user, assistant }
    public enum State: String, Sendable, Equatable, Codable { case complete, streaming, stopped, failed }

    public let id: MessageID
    public var role: Role
    public var parts: [MessagePart]
    public var state: State
    public var receipt: Receipt?

    public init(id: MessageID, role: Role, parts: [MessagePart] = [], state: State = .complete) {
        (self.id, self.role, self.parts, self.state) = (id, role, parts, state)
    }

    public var text: String {
        parts.reduce(into: "") { text, part in if case .text(let piece) = part { text += piece } }
    }

    public var attachments: [AttachmentPart] {
        parts.compactMap { part in if case .attachment(let attachment) = part { attachment } else { nil } }
    }

    public var prompt: String {
        let attached = attachments.filter { !$0.perceived.isEmpty }.map { "Attached \($0.kind) \($0.name):\n\($0.perceived)\n\n" }
        return attached.joined() + text
    }
}

public struct ConversationSnapshot: Sendable, Equatable, Identifiable {
    public let id: ConversationID
    public var title = ""
    public var model: String?
    public var systemPrompt: String?
    public var messages: [MessageSnapshot] = []

    public init(id: ConversationID, model: String? = nil, systemPrompt: String? = nil) {
        (self.id, self.model, self.systemPrompt) = (id, model, systemPrompt)
    }
}

public struct ChatRequest: Sendable, Equatable, Encodable {
    public struct Message: Sendable, Equatable, Encodable {
        public var role: MessageSnapshot.Role
        public var content: String
    }

    public struct StreamOptions: Sendable, Equatable, Encodable {
        public var includeUsage = true

        enum CodingKeys: String, CodingKey { case includeUsage = "include_usage" }
    }

    public var model: String?
    public var messages: [Message]
    public var stream = true
    public var streamOptions = StreamOptions()

    enum CodingKeys: String, CodingKey {
        case model, messages, stream
        case streamOptions = "stream_options"
    }

    public init(_ conversation: ConversationSnapshot) {
        model = conversation.model
        let system = conversation.systemPrompt.map { [Message(role: .system, content: $0)] } ?? []
        messages = system + conversation.messages.filter { !$0.prompt.isEmpty }.map { Message(role: $0.role, content: $0.prompt) }
    }
}
