import Foundation
import MLX
import MLXLMCommon
import BasaltCore

extension KVCache {
    var isSnapshotOnly: Bool {
        self is ArraysCache || type(of: self) == RotatingKVCache.self
    }
}

extension PromptCache {
    var isHybrid: Bool {
        layers.contains { $0.isSnapshotOnly }
            && layers.allSatisfy { $0.isSnapshotOnly || type(of: $0) == KVCacheSimple.self }
    }

    func isSnapshot(of count: Int) -> Bool {
        isHybrid && layers.allSatisfy { $0.isSnapshotOnly ? $0.offset == count : $0.offset >= count }
    }

    func snapshot(at count: Int, state: LMOutput.State?) -> PromptCache {
        let kept = PromptCache(layers.map { layer in
            guard layer.isSnapshotOnly else { return layer }
            let copy = layer.copy()
            (copy as? ArraysCache)?.offset = count
            return copy
        })
        kept.state = state
        eval(kept.layers.filter { $0.isSnapshotOnly }.flatMap { $0.innerState() })
        return kept
    }
}

struct RecurrentSnapshot {
    let layers: [Int: KVCache]
    let state: LMOutput.State?

    init?(of cache: PromptCache) {
        guard cache.isHybrid else { return nil }
        layers = Dictionary(uniqueKeysWithValues: cache.layers.enumerated().compactMap { index, layer in
            layer.isSnapshotOnly ? (index, layer) : nil
        })
        state = cache.state
    }

    var bytes: Int {
        layers.values.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    }
}

enum HybridPrefixSnapshot {
    static func prefillHead(
        _ prompt: MLXArray, into cache: PromptCache, reused: Int, cuts: [Int] = [], boundary: Int,
        model: any LanguageModel, prefill: PrefillParameters
    ) throws -> (anchors: [PromptCache], kept: PromptCache?, start: Int, state: LMOutput.State?) {
        guard cache.isHybrid else { return ([], cache, reused, cache.state) }
        guard boundary > reused, boundary < prompt.dim(-1) else { return ([], nil, reused, cache.state) }
        var state = cache.state
        func advance(_ range: Range<Int>) throws -> PromptCache {
            let slice = LMInput(tokens: prompt[.ellipsis, range])
            switch try model.prepare(slice, cache: cache.layers, state: state, prefill: prefill) {
            case .tokens(let rest):
                if rest.tokens.size > 0 {
                    state = model(rest[text: .newAxis], cache: cache.layers, state: state).state ?? state
                }
            case .logits(let output):
                state = output.state ?? state
            }
            eval(cache.layers.flatMap { $0.innerState() })
            return cache.snapshot(at: range.upperBound, state: state)
        }
        var start = reused
        var anchors: [PromptCache] = []
        for cut in cuts.sorted() where cut > start && cut < boundary {
            anchors.append(try advance(start ..< cut))
            start = cut
        }
        return (anchors, try advance(start ..< boundary), boundary, state)
    }

    static let minAnchorSpacing = 512

    static func anchorPrefixes(_ turns: [ChatMessageInput]) -> [[ChatMessageInput]] {
        turns.indices.dropFirst().reversed().flatMap { index -> [[ChatMessageInput]] in
            let head = Array(turns[..<index])
            let opensAssistant = turns[index].role == .assistant || turns[index - 1].role == .tool
            let reasoning = turns[index].role == .assistant
                ? MLXInferenceEngine.firstMarker(MLXInferenceEngine.strayThinkingClosers, in: turns[index].text) : nil
            return (opensAssistant ? [head] : [])
                + (reasoning.map { [head + [ChatMessageInput(role: .assistant, text: String(turns[index].text[..<$0.upperBound]))]] } ?? [])
        }
    }

    static func templateSuffixLength(_ container: ModelContainer, _ makeInput: @Sendable (String) -> UserInput) async -> Int {
        guard let a = try? await container.prepare(input: makeInput("a")).text.tokens.asArray(Int.self),
              let b = try? await container.prepare(input: makeInput("b")).text.tokens.asArray(Int.self)
        else { return 1 }
        return max(1, zip(a.reversed(), b.reversed()).prefix { $0 == $1 }.count)
    }

    static func templateKey(_ parts: [Any]) -> String? {
        guard JSONSerialization.isValidJSONObject(parts),
              let data = try? JSONSerialization.data(withJSONObject: parts, options: [.sortedKeys])
        else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    static func restorable(_ path: [PromptCacheTree.Step]) -> [PromptCacheTree.Step] {
        guard path.first?.node.layers.contains(where: \.isEmpty) == true else { return path }
        guard let end = path.lastIndex(where: { $0.used == $0.node.tokens.count && $0.node.snapshot != nil })
        else { return [] }
        return Array(path[...end])
    }
}
