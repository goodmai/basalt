import Foundation
import MLX
import MLXLMCommon

extension PromptCache {
    var isHybrid: Bool {
        layers.contains { $0 is ArraysCache }
            && layers.allSatisfy { $0 is ArraysCache || type(of: $0) == KVCacheSimple.self }
    }

    func isSnapshot(of count: Int) -> Bool {
        isHybrid && layers.allSatisfy { $0.offset == count }
    }

    func snapshot(at count: Int) -> PromptCache {
        let copies = PromptCache(layers.map { $0.copy() })
        for case let recurrent as ArraysCache in copies.layers {
            recurrent.offset = count
        }
        eval(copies.layers.flatMap { $0.innerState() })
        return copies
    }
}

enum HybridPrefixSnapshot {
    static func prefillHead(
        _ prompt: MLXArray, into cache: PromptCache, reused: Int,
        model: any LanguageModel, prefill: PrefillParameters
    ) throws -> (kept: PromptCache?, start: Int) {
        guard cache.isHybrid else { return (cache, reused) }
        let boundary = prompt.dim(-1) - 1
        guard boundary > reused else { return (nil, reused) }
        let head = LMInput(tokens: prompt[.ellipsis, reused ..< boundary])
        if case .tokens(let rest) = try model.prepare(head, cache: cache.layers, state: nil, prefill: prefill),
           rest.tokens.size > 0 {
            _ = model(rest[text: .newAxis], cache: cache.layers, state: nil)
            eval(cache.layers.flatMap { $0.innerState() })
        }
        return (cache.snapshot(at: boundary), boundary)
    }

    static func restorable(_ path: [PromptCacheTree.Step]) -> [PromptCacheTree.Step] {
        guard path.first?.node.layers.contains(where: \.isEmpty) == true else { return path }
        guard let end = path.lastIndex(where: { $0.used == $0.node.tokens.count && !$0.node.recurrent.isEmpty })
        else { return [] }
        return Array(path[...end])
    }

    static func recurrentLayers(of cache: PromptCache) -> [Int: KVCache] {
        Dictionary(uniqueKeysWithValues: cache.layers.enumerated().compactMap { index, layer in
            layer is ArraysCache ? (index, layer) : nil
        })
    }
}
