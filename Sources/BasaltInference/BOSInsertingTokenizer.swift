import BasaltCore
import Foundation
import MLXLMCommon

struct BOSInsertingTokenizer: MLXLMCommon.Tokenizer {
    let base: any MLXLMCommon.Tokenizer
    let bos: Int

    static func wrapping(_ base: any MLXLMCommon.Tokenizer, directory: URL) -> any MLXLMCommon.Tokenizer {
        guard ModelCache.config(at: directory)?["force_bos_token_insert"] as? Bool == true,
              let bos = base.bosToken.flatMap(base.convertTokenToId) else { return base }
        return BOSInsertingTokenizer(base: base, bos: bos)
    }

    func prefixed(_ ids: [Int]) -> [Int] { ids.first == bos ? ids : [bos] + ids }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let ids = base.encode(text: text, addSpecialTokens: addSpecialTokens)
        return addSpecialTokens ? prefixed(ids) : ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { base.decode(tokenIds: tokenIds, skipSpecialTokens: skipSpecialTokens) }
    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }
    var bosToken: String? { base.bosToken }
    var eosToken: String? { base.eosToken }
    var unknownToken: String? { base.unknownToken }

    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] {
        prefixed(try base.applyChatTemplate(messages: messages, tools: tools, additionalContext: additionalContext))
    }
}
