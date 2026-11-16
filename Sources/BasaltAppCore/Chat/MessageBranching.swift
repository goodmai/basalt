import Foundation

public struct BranchPosition: Sendable, Equatable {
    public var index: Int
    public var count: Int

    public init(index: Int, count: Int) {
        (self.index, self.count) = (index, count)
    }

    public var hasPrevious: Bool { index > 0 }

    public var hasNext: Bool { index + 1 < count }

    public var label: String { "‹ \(index + 1)/\(count) ›" }

    public var counter: String { "\(index + 1)/\(count)" }
}

public struct MessageBranchGroup: Sendable, Equatable, Codable {
    public var variants: [[MessageSnapshot]]
    public var selected: Int

    public init(variants: [[MessageSnapshot]], selected: Int) {
        (self.variants, self.selected) = (variants, selected)
    }
}

extension MessageSnapshot: Codable {
    enum CodingKeys: String, CodingKey {
        case id, role, parts, state, receipt
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(MessageID.self, forKey: .id)
        role = try container.decode(Role.self, forKey: .role)
        parts = try container.decode([MessagePart].self, forKey: .parts)
        state = try container.decode(State.self, forKey: .state)
        receipt = try container.decodeIfPresent(Receipt.self, forKey: .receipt)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(role, forKey: .role)
        try container.encode(parts, forKey: .parts)
        try container.encode(state, forKey: .state)
        try container.encodeIfPresent(receipt, forKey: .receipt)
    }
}

public struct MessageBranches: Sendable, Equatable, Codable {
    public static let rootAnchor = "root"
    private var groups: [String: MessageBranchGroup] = [:]

    public init() {}

    public var isEmpty: Bool { groups.isEmpty }

    public func position(for message: MessageID, in conversation: ConversationSnapshot) -> BranchPosition? {
        guard !groups.isEmpty, let index = conversation.messages.firstIndex(where: { $0.id == message }) else { return nil }
        let anchor = Self.anchorKey(at: index, in: conversation.messages)
        guard let group = groups[anchor], group.variants.count > 1 else { return nil }
        return BranchPosition(index: group.selected, count: group.variants.count)
    }

    public mutating func fork(in conversation: inout ConversationSnapshot, target: MessageID) -> Bool {
        guard let rawIndex = conversation.messages.firstIndex(where: { $0.id == target }) else { return false }
        let forkIndex: Int
        if conversation.messages[rawIndex].role == .assistant {
            forkIndex = rawIndex
        } else if rawIndex + 1 < conversation.messages.count, conversation.messages[rawIndex + 1].role == .assistant {
            forkIndex = rawIndex + 1
        } else {
            return false
        }
        let anchor = Self.anchorKey(at: forkIndex, in: conversation.messages)
        let currentTail = Array(conversation.messages[forkIndex...])
        if var existing = groups[anchor] {
            if existing.variants.indices.contains(existing.selected) {
                existing.variants[existing.selected] = currentTail
            }
            existing.variants.append([])
            existing.selected = existing.variants.count - 1
            groups[anchor] = existing
        } else {
            groups[anchor] = MessageBranchGroup(variants: [currentTail, []], selected: 1)
        }
        conversation.messages.removeSubrange(forkIndex...)
        return true
    }

    public mutating func recordHead(_ message: MessageSnapshot, in conversation: ConversationSnapshot) {
        guard !groups.isEmpty, let index = conversation.messages.firstIndex(where: { $0.id == message.id }) else { return }
        let anchor = Self.anchorKey(at: index, in: conversation.messages)
        guard var group = groups[anchor], group.variants.indices.contains(group.selected) else { return }
        group.variants[group.selected] = Array(conversation.messages[index...])
        groups[anchor] = group
    }

    public mutating func select(in conversation: inout ConversationSnapshot, message: MessageID, index targetIndex: Int) -> Bool {
        guard let msgIndex = conversation.messages.firstIndex(where: { $0.id == message }) else { return false }
        let anchor = Self.anchorKey(at: msgIndex, in: conversation.messages)
        guard var group = groups[anchor], group.variants.indices.contains(targetIndex), targetIndex != group.selected else { return false }
        if group.variants.indices.contains(group.selected) {
            group.variants[group.selected] = Array(conversation.messages[msgIndex...])
        }
        let replacement = group.variants[targetIndex]
        guard !replacement.isEmpty else { return false }
        conversation.messages.replaceSubrange(msgIndex..., with: replacement)
        group.selected = targetIndex
        groups[anchor] = group
        return true
    }

    private static func anchorKey(at index: Int, in messages: [MessageSnapshot]) -> String {
        index > 0 ? messages[index - 1].id.raw.uuidString : rootAnchor
    }
}
