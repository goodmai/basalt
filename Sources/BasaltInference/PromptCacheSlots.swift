import MLX
import MLXLMCommon

final class PromptCache: @unchecked Sendable {
    let layers: [KVCache]

    init(_ layers: [KVCache]) {
        self.layers = layers
    }

    var tokenCount: Int {
        layers.map(\.offset).min() ?? 0
    }

    var byteCount: Int {
        layers.reduce(0) { total, layer in
            total + layer.innerState().reduce(0) { $0 + $1.nbytes }
        }
    }

    var isRewindable: Bool {
        !layers.isEmpty && canTrimPromptCache(layers)
    }

    func rewind(to count: Int) -> Bool {
        guard isRewindable, count <= tokenCount else { return false }
        trimPromptCache(layers, numTokens: tokenCount - count)
        return layers.allSatisfy { $0.offset == count }
    }

    func shrink(to count: Int) -> Bool {
        guard count > 0, rewind(to: count) else { return false }
        for case let layer as KVCacheSimple in layers where !layer.state.isEmpty {
            layer.state = layer.state.map { contiguous($0) }
            eval(layer.innerState())
        }
        return true
    }
}

enum PromptCacheBudget {
    static func automatic(physicalMemory: Int, weights: Int) -> Int {
        max(0, physicalMemory / 4 * 3 - weights) / 2
    }
}

struct PromptCacheLease: Sendable {
    let cache: PromptCache
    let reusedTokens: Int
}

struct PromptCacheSlots: Sendable {
    struct Slot: Sendable {
        var tokens: [Int]
        let cache: PromptCache
        let lastUse: Int
    }

    let byteBudget: Int
    private(set) var slots: [String: Slot] = [:]
    private var clock = 0

    init(byteBudget: Int) {
        self.byteBudget = byteBudget
    }

    var parkedByteLimit: Int { byteBudget / 8 }

    var byteCount: Int {
        slots.values.reduce(0) { $0 + $1.cache.byteCount }
    }

    static func reusablePrefixLength(cached: [Int], prompt: [Int]) -> Int {
        let common = zip(cached, prompt).prefix { $0 == $1 }.count
        return max(0, min(common, prompt.count - 1))
    }

    mutating func lease(_ key: String, for prompt: [Int]) -> PromptCacheLease? {
        guard let slot = slots.removeValue(forKey: key) else { return nil }
        let common = Self.reusablePrefixLength(cached: slot.tokens, prompt: prompt)
        guard common > 0, slot.cache.rewind(to: common) else { return nil }
        return PromptCacheLease(cache: slot.cache, reusedTokens: common)
    }

    mutating func store(_ key: String, tokens: [Int], cache: PromptCache) {
        guard byteBudget > 0, cache.tokenCount >= tokens.count, cache.shrink(to: tokens.count),
              cache.byteCount <= byteBudget
        else { return }
        clock += 1
        slots[key] = Slot(tokens: tokens, cache: cache, lastUse: clock)
        let inactive = slots.filter { $0.key != key }.sorted { $0.value.lastUse < $1.value.lastUse }.map(\.key)
        for other in inactive where byteCount > byteBudget {
            park(other)
        }
        for other in inactive where byteCount > byteBudget {
            slots[other] = nil
        }
    }

    mutating func removeAll() {
        slots.removeAll()
    }

    private mutating func park(_ key: String) {
        guard let slot = slots[key], slot.cache.byteCount > parkedByteLimit else { return }
        let bytesPerToken = max(1, slot.cache.byteCount / max(1, slot.cache.tokenCount))
        let keep = min(slot.tokens.count, parkedByteLimit / bytesPerToken)
        if slot.cache.shrink(to: keep) {
            slots[key]?.tokens = Array(slot.tokens.prefix(keep))
        } else {
            slots[key] = nil
        }
    }
}
