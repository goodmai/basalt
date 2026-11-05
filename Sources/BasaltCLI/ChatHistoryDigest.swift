import BasaltCore

struct ChatHistoryDigest: Sendable, Equatable {
    static let header = "Earlier turns of this conversation, folded to fit the context (the start of each):"
    static let lineLength = 160
    static let maxLines = 64
    static let replyFloor = 1024

    private(set) var lines: [String] = []
    private(set) var turns: [ChatMessageInput]

    init(turns: [ChatMessageInput] = []) {
        self.turns = turns
    }

    mutating func record(_ user: String, _ reply: String) {
        turns += [ChatMessageInput(role: .user, text: user), ChatMessageInput(role: .assistant, text: reply)]
    }

    func request(_ prompt: String, maxTokens: Int, sessionId: String) -> GenerationRequest {
        guard !lines.isEmpty else { return ChatCommand.request(prompt, history: turns, maxTokens: maxTokens, sessionId: sessionId) }
        let note = Self.header + "\n" + lines.joined(separator: "\n") + "\n\n"
        guard let first = turns.first else {
            return ChatCommand.request(note + prompt, history: [], maxTokens: maxTokens, sessionId: sessionId)
        }
        let history = [ChatMessageInput(role: first.role, text: note + first.text)] + turns.dropFirst()
        return ChatCommand.request(prompt, history: history, maxTokens: maxTokens, sessionId: sessionId)
    }

    mutating func foldOlderHalf() -> Bool {
        guard !turns.isEmpty else { return false }
        var count = min(turns.count, max(2, turns.count / 4 * 2))
        while count < turns.count, turns[count].role != .user { count += 1 }
        lines = Array((lines + turns.prefix(count).map(Self.line)).suffix(Self.maxLines))
        turns.removeFirst(count)
        return true
    }

    mutating func fitted(
        _ prompt: String, maxTokens: Int, sessionId: String, room: @Sendable (GenerationRequest) async -> Int
    ) async -> GenerationRequest {
        var request = request(prompt, maxTokens: maxTokens, sessionId: sessionId)
        while await room(request) < min(maxTokens, Self.replyFloor), foldOlderHalf() {
            request = self.request(prompt, maxTokens: maxTokens, sessionId: sessionId)
        }
        return request
    }

    static func line(_ message: ChatMessageInput) -> String {
        let flat = message.text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        return "\(message.role.rawValue): " + (flat.count > lineLength ? String(flat.prefix(lineLength)) + "…" : flat)
    }
}
