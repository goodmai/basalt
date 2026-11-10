import Foundation
import MLX
import MLXLMCommon
import BasaltCore

struct NoRepeatNGram: LogitProcessor {
    static let deepSeekOCR = (size: 20, window: 50, whitelist: ["<td>", "</td>"])

    private final class History {
        var tokens: [Int] = []
        var unread: MLXArray?
    }

    let size: Int
    let window: Int
    let whitelist: Set<Int>
    private var history = History()

    init(size: Int, window: Int, whitelist: Set<Int>) {
        (self.size, self.window, self.whitelist) = (size, window, whitelist)
    }

    static func banned(_ tokens: [Int], size: Int, window: Int) -> Set<Int> {
        guard size > 1, tokens.count >= size else { return [] }
        let prefix = tokens.suffix(size - 1)
        let start = max(0, tokens.count - window)
        guard start <= tokens.count - size else { return [] }
        var banned: Set<Int> = []
        for i in start ... tokens.count - size where tokens[i ..< i + size - 1].elementsEqual(prefix) {
            banned.insert(tokens[i + size - 1])
        }
        return banned
    }

    mutating func prompt(_ prompt: MLXArray) {
        history.tokens.removeAll(keepingCapacity: true)
        history.unread = nil
    }

    func process(logits: MLXArray) -> MLXArray {
        if let token = history.unread {
            history.unread = nil
            history.tokens.append(token.item(Int.self))
        }
        let banned = Self.banned(history.tokens, size: size, window: window).subtracting(whitelist).sorted()
        guard !banned.isEmpty else { return logits }
        let indices = MLXArray(banned.map(Int32.init))[.newAxis, 0...]
        let blocked = MLXArray(Array(repeating: -Float.infinity, count: banned.count))[.newAxis, 0...].asType(logits.dtype)
        return putAlong(logits, indices, values: blocked, axis: -1)
    }

    mutating func didSample(token: MLXArray) {
        history.unread = token
    }
}

extension MLXInferenceEngine {
    func ocrNoRepeat(for request: GenerationRequest) async -> (@Sendable () -> NoRepeatNGram)? {
        guard request.ocrMode != nil, let container else { return nil }
        let setting = NoRepeatNGram.deepSeekOCR
        let whitelist = await container.perform { context in Set(setting.whitelist.compactMap { context.tokenizer.convertTokenToId($0) }) }
        return { NoRepeatNGram(size: setting.size, window: setting.window, whitelist: whitelist) }
    }
}
