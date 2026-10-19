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

    func snapshot(at count: Int, state: LMOutput.State?) -> PromptCache {
        let copies = PromptCache(layers.map { $0.copy() })
        for case let recurrent as ArraysCache in copies.layers {
            recurrent.offset = count
        }
        copies.state = state
        eval(copies.layers.flatMap { $0.innerState() })
        return copies
    }
}

struct RecurrentSnapshot {
    let layers: [Int: KVCache]
    let state: LMOutput.State?

    init?(of cache: PromptCache) {
        guard cache.isHybrid else { return nil }
        layers = Dictionary(uniqueKeysWithValues: cache.layers.enumerated().compactMap { index, layer in
            layer is ArraysCache ? (index, layer) : nil
        })
        state = cache.state
    }

    var bytes: Int {
        layers.values.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    }
}

enum HybridPrefixSnapshot {
    static func prefillHead(
        _ prompt: MLXArray, into cache: PromptCache, reused: Int, boundary: Int,
        model: any LanguageModel, prefill: PrefillParameters
    ) throws -> (kept: PromptCache?, start: Int, state: LMOutput.State?) {
        guard cache.isHybrid else { return (cache, reused, cache.state) }
        guard boundary > reused, boundary < prompt.dim(-1) else { return (nil, reused, cache.state) }
        let head = LMInput(tokens: prompt[.ellipsis, reused ..< boundary])
        var state = cache.state
        switch try model.prepare(head, cache: cache.layers, state: state, prefill: prefill) {
        case .tokens(let rest):
            if rest.tokens.size > 0 {
                state = model(rest[text: .newAxis], cache: cache.layers, state: state).state ?? state
            }
        case .logits(let output):
            state = output.state ?? state
        }
        eval(cache.layers.flatMap { $0.innerState() })
        return (cache.snapshot(at: boundary, state: state), boundary, state)
    }

    static func templateSuffixLength(_ container: ModelContainer, _ makeInput: (String) -> UserInput) async -> Int {
        guard let a = try? await container.prepare(input: makeInput("a")).text.tokens.asArray(Int.self),
              let b = try? await container.prepare(input: makeInput("b")).text.tokens.asArray(Int.self)
        else { return 1 }
        return max(1, zip(a.reversed(), b.reversed()).prefix { $0 == $1 }.count)
    }

    static func restorable(_ path: [PromptCacheTree.Step]) -> [PromptCacheTree.Step] {
        guard path.first?.node.layers.contains(where: \.isEmpty) == true else { return path }
        guard let end = path.lastIndex(where: { $0.used == $0.node.tokens.count && $0.node.snapshot != nil })
        else { return [] }
        return Array(path[...end])
    }
}
