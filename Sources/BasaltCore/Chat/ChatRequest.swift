import Foundation

public struct ChatMessageInput: Codable, Sendable, Equatable {
    public enum Role: String, Codable, Sendable {
        case system, user, assistant, tool
    }

    public let role: Role
    public let text: String
    public let toolCalls: [ToolCallOutput]?
    public let toolCallId: String?
    public let toolName: String?
    private let attachedImages: [ImageInput]?

    public var images: [ImageInput] { attachedImages ?? [] }

    enum CodingKeys: String, CodingKey {
        case role, text, toolCalls, toolCallId, toolName
        case attachedImages = "images"
    }

    public init(
        role: Role, text: String, toolCalls: [ToolCallOutput]? = nil,
        toolCallId: String? = nil, toolName: String? = nil, images: [ImageInput] = []
    ) {
        self.role = role
        self.text = text
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.toolName = toolName
        attachedImages = images.isEmpty ? nil : images
    }

    public static func plainText(_ messages: [ChatMessageInput]) -> String {
        messages.flatMap { message -> [String] in [message.text] + (message.toolCalls ?? []).map(\.arguments) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }

    public static func splittingPrefill(_ messages: [ChatMessageInput]) -> (turns: [ChatMessageInput], prefill: String)? {
        let trailing = messages.reversed().prefix { $0.role == .assistant && $0.toolCalls?.isEmpty != false }
        guard !trailing.isEmpty, trailing.count < messages.count else { return nil }
        return (Array(messages.dropLast(trailing.count)), trailing.reversed().map(\.text).joined())
    }

    public static func leadingSystem(_ messages: [ChatMessageInput]) -> [ChatMessageInput] {
        let firstUser = messages.firstIndex { $0.role == .user } ?? messages.endIndex
        let leading = joined(messages[..<firstUser].filter { $0.role == .system }.map(\.text))
        var rest: [ChatMessageInput] = []
        for (index, message) in messages.enumerated() where !(index < firstUser && message.role == .system) {
            let message = message.role == .system ? ChatMessageInput(role: .user, text: "System: \(message.text)", images: message.images) : message
            if message.role == .user, let previous = rest.last, previous.role == .user {
                rest[rest.count - 1] = ChatMessageInput(
                    role: .user, text: joined([previous.text, message.text]) ?? "", images: previous.images + message.images)
            } else {
                rest.append(message)
            }
        }
        return (leading.map { [ChatMessageInput(role: .system, text: $0)] } ?? []) + rest
    }

    public static func alternating(_ messages: [ChatMessageInput]) -> [ChatMessageInput] {
        let placed = leadingSystem(messages)
        var turns: [(role: Role, texts: [String], images: [ImageInput])] = []
        for message in placed where message.role != .system {
            let role: Role = message.role == .tool ? .user : message.role
            let text = message.role == .tool
                ? "Tool result (\(message.toolName ?? message.toolCallId ?? "tool")): \(message.text)"
                : ([message.text] + (message.toolCalls ?? []).map { "\($0.name)(\($0.arguments))" })
                    .filter { !$0.isEmpty }.joined(separator: "\n")
            if turns.last?.role == role {
                turns[turns.count - 1].texts.append(text)
                turns[turns.count - 1].images += message.images
            } else {
                turns.append((role, [text], message.images))
            }
        }
        guard turns.contains(where: { $0.role == .user }) else {
            return [ChatMessageInput(role: .user, text: plainText(messages), images: messages.flatMap(\.images))]
        }
        if turns[0].role == .assistant {
            let opening = joined(turns.removeFirst().texts).map { "Assistant: \($0)" }
            turns[0].texts.insert(contentsOf: opening.map { [$0] } ?? [], at: 0)
        }
        let leading = placed.first.flatMap { $0.role == .system ? [$0] : nil } ?? []
        return leading + turns.map { ChatMessageInput(role: $0.role, text: joined($0.texts) ?? "", images: $0.images) }
    }

    private static func joined(_ texts: [String]) -> String? {
        let parts = texts.filter { !$0.isEmpty }
        return parts.isEmpty ? nil : parts.joined(separator: "\n\n")
    }
}

/// How much the model should think before answering.
///
/// Expressed as intent, not as a vendor knob: each ModelProfile maps these cases
/// onto whatever its chat template actually understands. Qwen reads
/// `reasoning_effort` plus `enable_thinking`; Gemma has no knob at all and
/// ignores the policy entirely.
public enum ReasoningPolicy: Sendable, Equatable {
    /// Let the template pick. For Qwen3.x that means `xhigh`, which can spend the
    /// entire token budget reasoning — fine for hard problems, ruinous for
    /// "what is 127 + 458".
    case modelDefault
    /// Emit an already-closed thinking block so the model answers directly.
    case disabled
    /// A named budget the template understands (`xhigh` | `medium` | `low`).
    case effort(String)

    public init?(rawValue: String?) {
        switch rawValue?.lowercased() {
        case nil:                     return nil
        case "none", "off", "false":  self = .disabled
        case "default", "auto":       self = .modelDefault
        // Levels pass through as named: each profile maps them onto its own template's
        // vocabulary (Qwen spells the top level `xhigh`, gpt-oss spells it `high`).
        case .some(let value):        self = .effort(value)  // already lowercased
        }
    }
}
