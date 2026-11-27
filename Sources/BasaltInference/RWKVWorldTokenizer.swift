import BasaltCore
import Foundation
import MLXLMCommon

struct RWKVWorldTokenizer: MLXLMCommon.Tokenizer {
    static let endOfText = "<|rwkv_tokenizer_end_of_text|>"
    let vocabulary: RWKVWorldVocabulary

    static func load(from directory: URL) throws -> RWKVWorldTokenizer? {
        let url = directory.appending(component: RWKVWorldVocabulary.fileName)
        guard FileManager.default.fileExists(atPath: url.path),
              !FileManager.default.fileExists(atPath: directory.appending(component: "tokenizer.json").path) else { return nil }
        return RWKVWorldTokenizer(vocabulary: try RWKVWorldVocabulary(contents: String(contentsOf: url, encoding: .utf8)))
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { vocabulary.encode(text) }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { vocabulary.decode(tokenIds) }

    func convertTokenToId(_ token: String) -> Int? { token == Self.endOfText ? 0 : vocabulary.id(of: Array(token.utf8)) }

    func convertIdToToken(_ id: Int) -> String? {
        id == 0 ? Self.endOfText : vocabulary.bytes(of: id).map { String(decoding: $0, as: UTF8.self) }
    }

    var bosToken: String? { Self.endOfText }
    var eosToken: String? { "\n\n" }
    var unknownToken: String? { nil }

    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        let turns = messages.map { message -> String in
            let role = (message["role"] as? String ?? "user").capitalized
            return "\(role == "Assistant" ? "Assistant" : role == "System" ? "System" : "User"): \(message["content"] as? String ?? "")"
        }
        return vocabulary.encode(turns.joined(separator: "\n\n") + "\n\nAssistant:")
    }
}
