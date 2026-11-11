import BasaltCore
import Foundation

public struct Identifier<Tag>: Hashable, Sendable {
    public let raw: UUID

    public init(_ raw: UUID = UUID()) { self.raw = raw }
}

public enum ConversationTag {}
public enum MessageTag {}
public enum RunTag {}

public typealias ConversationID = Identifier<ConversationTag>
public typealias MessageID = Identifier<MessageTag>
public typealias RunID = Identifier<RunTag>

public enum MessagePart: Sendable, Equatable {
    case text(String)
    case reasoning(String)
    case toolCall(ToolCallOutput)
}

public struct Receipt: Sendable, Equatable {
    public var model: String?
    public var promptTokens = 0
    public var completionTokens = 0
    public var timeToFirstToken: Duration?
    public var tokensPerSecond: Double?

    public init(model: String? = nil) { self.model = model }
}

public struct MessageSnapshot: Sendable, Equatable, Identifiable {
    public enum Role: String, Sendable, Equatable, Codable { case system, user, assistant }
    public enum State: String, Sendable, Equatable { case complete, streaming, stopped, failed }

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
        messages = system + conversation.messages.filter { !$0.text.isEmpty }.map { Message(role: $0.role, content: $0.text) }
    }
}
