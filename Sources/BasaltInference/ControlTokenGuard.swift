import Foundation
import BasaltCore

struct ControlTokenGuard: Sendable, Equatable {
    static let none = ControlTokenGuard(tokens: [])
    static let joiner = "\u{2060}"

    let tokens: [String]

    init(tokens: [String]) {
        self.tokens = Set(tokens.filter(Self.isMarkup)).sorted { $0.count > $1.count }
    }

    init(modelDirectory: URL) {
        self.init(tokens: Self.addedTokens(in: modelDirectory))
    }

    static func isMarkup(_ token: String) -> Bool {
        token.count >= 3 && token.contains(where: { "<[|".contains($0) }) && !token.contains(where: \.isWhitespace)
    }

    static func addedTokens(in directory: URL) -> [String] {
        func json(_ name: String) -> [String: Any]? {
            (try? Data(contentsOf: directory.appendingPathComponent(name)))
                .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        }
        let fromTokenizer = (json("tokenizer.json")?["added_tokens"] as? [[String: Any]] ?? []).compactMap { $0["content"] as? String }
        let fromConfig = (json("tokenizer_config.json")?["added_tokens_decoder"] as? [String: [String: Any]] ?? [:])
            .values.compactMap { $0["content"] as? String }
        return fromTokenizer + fromConfig
    }

    func neutralize(_ text: String) -> String {
        tokens.reduce(text) { text, token in
            guard text.contains(token) else { return text }
            return text.replacingOccurrences(of: token, with: String(token.prefix(1)) + Self.joiner + token.dropFirst())
        }
    }

    func neutralize(_ messages: [ChatMessageInput]) -> [ChatMessageInput] {
        guard !tokens.isEmpty else { return messages }
        return messages.map { message in
            ChatMessageInput(
                role: message.role, text: neutralize(message.text),
                toolCalls: message.toolCalls?.map { ToolCallOutput(id: $0.id, name: $0.name, arguments: neutralize($0.arguments)) },
                toolCallId: message.toolCallId, toolName: message.toolName)
        }
    }
}
