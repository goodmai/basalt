import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import BasaltCore

struct CacheBlendLayer {
    let inputNorm: UnaryLayer
    let postNorm: UnaryLayer
    let q: UnaryLayer
    let k: UnaryLayer
    let v: UnaryLayer
    let o: UnaryLayer
    let qNorm: UnaryLayer?
    let kNorm: UnaryLayer?
    let mlp: UnaryLayer
    let rope: any RoPELayer
}

struct CacheBlendSegment {
    let range: Range<Int>
    let cached: [[MLXArray]]?
}

struct CacheBlendPass {
    let caches: [KVCacheSimple]
    let layers: [[MLXArray]]
    let recomputed: Int
    let exactPrefix: Int
}

struct CacheBlendModel {
    let embed: Embedding
    let layers: [CacheBlendLayer]
    let kvHeads: Int
    let headDim: Int

    init?(_ model: any LanguageModel) {
        guard model is LlamaModel || model is Qwen3Model,
              let heads = (model as? KVCacheDimensionProvider)?.kvHeads, let kvHeads = heads.first, kvHeads > 0
        else { return nil }
        let modules = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        guard let embed = modules["model.embed_tokens"] as? Embedding,
              let keys = modules["model.layers.0.self_attn.k_proj"] as? Linear
        else { return nil }
        var layers: [CacheBlendLayer] = []
        while let layer = Self.layer(modules, prefix: "model.layers.\(layers.count)") {
            layers.append(layer)
        }
        guard layers.count == heads.count else { return nil }
        self.embed = embed
        self.layers = layers
        self.kvHeads = kvHeads
        self.headDim = keys.weight.dim(0) / kvHeads
    }

    private static func layer(_ modules: [String: Module], prefix: String) -> CacheBlendLayer? {
        func unary(_ name: String) -> UnaryLayer? { modules["\(prefix).\(name)"] as? UnaryLayer }
        guard let inputNorm = unary("input_layernorm"), let postNorm = unary("post_attention_layernorm"),
              let q = unary("self_attn.q_proj"), let k = unary("self_attn.k_proj"),
              let v = unary("self_attn.v_proj"), let o = unary("self_attn.o_proj"),
              let mlp = unary("mlp"), let rope = modules["\(prefix).self_attn.rope"] as? any RoPELayer
        else { return nil }
        return CacheBlendLayer(
            inputNorm: inputNorm, postNorm: postNorm, q: q, k: k, v: v, o: o,
            qNorm: unary("self_attn.q_norm"), kNorm: unary("self_attn.k_norm"), mlp: mlp, rope: rope)
    }

    func prefill(
        tokens: [Int], offset: Int, past: [[MLXArray]], segments: [CacheBlendSegment], ratio: Float, step: Int = 512
    ) -> CacheBlendPass {
        let count = tokens.count
        let hits = segments.filter { $0.cached != nil }.reduce(0) { $0 + $1.range.count }
        let keep = hits - Int((Float(hits) * min(max(ratio, 0), 1)).rounded(.down))
        let scale = pow(Float(headDim), -0.5)
        var hidden = embed(MLXArray(tokens.map(Int32.init)))[.newAxis]
        var active = Array(0 ..< count)
        var gather: MLXArray?
        var caches: [KVCacheSimple] = []
        var kept: [[MLXArray]] = []
        for (index, layer) in layers.enumerated() {
            var keys: MLXArray
            var values: MLXArray
            var queries: MLXArray?
            if gather != nil, active.isEmpty {
                let cached = assemble(segments, layer: index, like: kept[0][0])
                (keys, values) = (cached[0], cached[1])
            } else {
                let x = layer.inputNorm(hidden)
                var q = layer.q(x).reshaped(1, active.count, -1, headDim)
                var k = layer.k(x).reshaped(1, active.count, kvHeads, headDim)
                values = layer.v(x).reshaped(1, active.count, kvHeads, headDim).transposed(0, 2, 1, 3)
                if let qNorm = layer.qNorm { q = qNorm(q) }
                if let kNorm = layer.kNorm { k = kNorm(k) }
                queries = q.transposed(0, 2, 1, 3)
                keys = k.transposed(0, 2, 1, 3)
                if let gather {
                    let cached = assemble(segments, layer: index, like: keys)
                    keys = take(concatenated([cached[0], keys], axis: 2), gather, axis: 2)
                    values = take(concatenated([cached[1], values], axis: 2), gather, axis: 2)
                } else if index == 1, keep > 0 {
                    let cached = assemble(segments, layer: index, like: keys)
                    let deviation = (keys - cached[0]).asType(.float32).square().sum(axes: [0, 1, 3])
                        + (values - cached[1]).asType(.float32).square().sum(axes: [0, 1, 3])
                    active = Self.recomputedPositions(segments, deviation: deviation.asArray(Float.self), dropping: keep)
                    let chosen = MLXArray(active.map(Int32.init))
                    queries = active.isEmpty ? nil : take(q.transposed(0, 2, 1, 3), chosen, axis: 2)
                    hidden = active.isEmpty ? hidden : take(hidden, chosen, axis: 1)
                    var map = Array(0 ..< count)
                    for (slot, position) in active.enumerated() { map[position] = count + slot }
                    gather = MLXArray(map.map(Int32.init))
                }
            }
            kept.append([keys, values])
            let rotated = layer.rope(keys, offset: offset)
            let allKeys = past.isEmpty ? rotated : concatenated([past[index][0], rotated], axis: 2)
            let allValues = past.isEmpty ? values : concatenated([past[index][1], values], axis: 2)
            let cache = KVCacheSimple()
            cache.state = [allKeys, allValues]
            caches.append(cache)
            guard index < layers.count - 1, let queries else {
                eval(allKeys, allValues)
                continue
            }
            let everyToken = active.count == count
            let positions = MLXArray(active.map { Int32(offset + $0) })
            let rotatedQueries = everyToken
                ? layer.rope(queries, offset: offset)
                : layer.rope(queries.transposed(2, 1, 0, 3), offset: positions).transposed(2, 1, 0, 3)
            var blocks: [MLXArray] = []
            for lower in stride(from: 0, to: active.count, by: max(1, step)) {
                let upper = min(active.count, lower + max(1, step))
                let visible = offset + active[upper - 1] + 1
                let mask: MLXFast.ScaledDotProductAttentionMaskMode = everyToken
                    ? .causal
                    : .array(positions[lower ..< upper].expandedDimensions(axis: 1)
                        .>= MLXArray((0 ..< visible).map(Int32.init)).expandedDimensions(axis: 0))
                let attended = MLXFast.scaledDotProductAttention(
                    queries: rotatedQueries[0..., 0..., lower ..< upper, 0...],
                    keys: allKeys[.ellipsis, ..<visible, 0...], values: allValues[.ellipsis, ..<visible, 0...],
                    scale: scale, mask: mask)
                var block = hidden[0..., lower ..< upper, 0...]
                block = block + layer.o(attended.transposed(0, 2, 1, 3).reshaped(1, upper - lower, -1))
                block = block + layer.mlp(layer.postNorm(block))
                eval(block)
                blocks.append(block)
            }
            hidden = concatenated(blocks, axis: 1)
            eval(hidden, allKeys, allValues)
        }
        let skipped = Set(0 ..< count).subtracting(active)
        return CacheBlendPass(
            caches: caches, layers: kept, recomputed: active.count - (count - hits),
            exactPrefix: skipped.min() ?? count)
    }

    static let chunkHead = 32

    static func recomputedPositions(_ segments: [CacheBlendSegment], deviation: [Float], dropping count: Int) -> [Int] {
        let priority = segments.flatMap { segment in
            segment.range.map { segment.cached == nil ? 2 : $0 - segment.range.lowerBound < chunkHead ? 1 : 0 }
        }
        return deviation.indices
            .sorted { (priority[$0], deviation[$0]) > (priority[$1], deviation[$1]) }
            .dropLast(count).sorted()
    }

    var rotatesPurely: Bool {
        layers.allSatisfy { $0.rope is RoPE || $0.rope is Llama3RoPE }
    }

    func unrotated(_ caches: [KVCache], _ range: Range<Int>) -> [[MLXArray]] {
        let positions = MLXArray(range.map(Int32.init))
        let probe = layers[0].rope(MLXArray((0 ..< headDim).map { $0 == 0 ? Float(1) : 0 }).reshaped(1, 1, 1, -1), offset: 1)
        let traditional = abs(probe[0, 0, 0, 1]).item(Float.self) > 1e-6
        let sign = MLXArray((0 ..< headDim).map { (traditional ? $0 % 2 == 1 : $0 >= headDim / 2) ? Float(-1) : 1 })
        return zip(layers, caches).enumerated().map { index, pair in
            guard index > 0 else { return [] }
            let state = pair.1.state
            let keys = state[0][.ellipsis, range, 0...]
            let flipped = (keys * sign.asType(keys.dtype)).transposed(2, 1, 0, 3)
            let back = pair.0.rope(flipped, offset: positions).transposed(2, 1, 0, 3) * sign.asType(keys.dtype)
            return [back, state[1][.ellipsis, range, 0...]]
        }
    }

    private func assemble(_ segments: [CacheBlendSegment], layer: Int, like keys: MLXArray) -> [MLXArray] {
        [0, 1].map { part in
            concatenated(segments.map { segment in
                segment.cached?[layer][part].asType(keys.dtype)
                    ?? MLXArray.zeros([1, kvHeads, segment.range.count, headDim], dtype: keys.dtype)
            }, axis: 2)
        }
    }
}

struct CacheBlendOutcome {
    let cache: PromptCache
    let tokens: Int
    let reused: Int
    let recomputed: Int
    let exactPrefix: Int
}

final class CacheBlendStore: @unchecked Sendable {
    struct Key: Hashable {
        let namespace: String
        let tokens: [Int]
    }

    struct Entry {
        let layers: [[MLXArray]]
        let bytes: Int
        var lastUse: Int
    }

    struct State {
        var entries: [Key: Entry] = [:]
        var bytes = 0
        var clock = 0
        var texts: [Int: String] = [:]
    }

    let ratio: Float
    let byteBudget: Int
    private var state = State()
    private let lock = NSLock()

    private func withState<T>(_ body: (inout State) -> T) -> T {
        lock.withLock { body(&state) }
    }

    init(ratio: Float, byteBudget: Int) {
        self.ratio = ratio
        self.byteBudget = byteBudget
    }

    var byteCount: Int { withState { $0.bytes } }
    var chunkCount: Int { withState { $0.entries.count } }

    static let minimumChunk = 16
    static let maximumTokens = 16384
    static let minimumSaving: Float = 0.2

    static func blends(reused: Int, count: Int, offset: Int, ratio: Float) -> Bool {
        count > 0 && offset + count <= maximumTokens
            && (1 - min(max(ratio, 0), 1)) * Float(reused) / Float(count) >= minimumSaving
    }

    static func chunks(_ texts: [String]) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var start = 0
        for index in texts.indices {
            let blank = texts[index].contains("\n\n")
                || (index > 0 && texts[index - 1].hasSuffix("\n") && texts[index].hasPrefix("\n"))
            if blank {
                ranges.append(start ..< index + 1)
                start = index + 1
            }
        }
        if start < texts.count { ranges.append(start ..< texts.count) }
        return ranges
    }

    func chunks(_ tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer) -> [Range<Int>] {
        let known = withState { $0.texts }
        let decoded = Dictionary(uniqueKeysWithValues: Set(tokens).filter { known[$0] == nil }.map {
            ($0, tokenizer.decode(tokenIds: [$0], skipSpecialTokens: false))
        })
        if !decoded.isEmpty {
            withState { state in state.texts.merge(decoded) { cached, _ in cached } }
        }
        return Self.chunks(tokens.map { known[$0] ?? decoded[$0] ?? "" })
    }

    func extend(
        _ cache: PromptCache, prompt: [Int], from start: Int, namespace: String,
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer, prefillStep: Int = 512,
        ratio: Float? = nil, storing: Bool = true
    ) -> CacheBlendOutcome? {
        let end = prompt.count - 1
        guard start < end, let blender = CacheBlendModel(model),
              cache.layers.count == blender.layers.count, cache.layers.allSatisfy({ $0 is KVCacheSimple })
        else { return nil }
        let ranges = chunks(Array(prompt[..<end]), tokenizer: tokenizer).filter { $0.upperBound > start }
        let segments = withState { state in
            ranges.map { range -> CacheBlendSegment in
                let local = max(range.lowerBound, start) - start ..< range.upperBound - start
                let key = Key(namespace: namespace, tokens: Array(prompt[range]))
                guard range.lowerBound >= start, range.count >= Self.minimumChunk, var entry = state.entries[key] else {
                    return CacheBlendSegment(range: local, cached: nil)
                }
                state.clock += 1
                entry.lastUse = state.clock
                state.entries[key] = entry
                return CacheBlendSegment(range: local, cached: entry.layers)
            }
        }
        let hits = segments.filter { $0.cached != nil }.reduce(0) { $0 + $1.range.count }
        let count = end - start
        let reused: Int
        let caches: [KVCache]
        let recomputed: Int
        let exactPrefix: Int
        let layers: (Range<Int>) -> [[MLXArray]]?
        if !Self.blends(reused: hits, count: count, offset: start, ratio: ratio ?? self.ratio) {
            var position = start
            while position < end {
                if Task.isCancelled { return nil }
                let step = min(end, position + max(1, prefillStep))
                _ = model(MLXArray(prompt[position ..< step].map(Int32.init)).reshaped(1, -1), cache: cache.layers)
                eval(cache.layers.flatMap { $0.innerState() })
                position = step
            }
            (caches, reused, recomputed, exactPrefix) = (cache.layers, 0, 0, end)
            let region = blender.rotatesPurely ? blender.unrotated(cache.layers, start ..< end) : nil
            layers = { range in
                region?.map { layer in layer.map { $0[.ellipsis, (range.lowerBound - start) ..< (range.upperBound - start), 0...] } }
            }
        } else {
            let pass = blender.prefill(
                tokens: Array(prompt[start ..< end]), offset: start,
                past: start == 0 ? [] : cache.layers.map(\.state), segments: segments, ratio: ratio ?? self.ratio,
                step: prefillStep)
            (caches, reused, recomputed, exactPrefix) = (pass.caches, hits, pass.recomputed, start + pass.exactPrefix)
            layers = { range in
                pass.layers.enumerated().map { index, layer in
                    index == 0 ? [] : layer.map { $0[.ellipsis, (range.lowerBound - start) ..< (range.upperBound - start), 0...] }
                }
            }
        }
        let stored = zip(ranges, segments).compactMap { range, segment -> (Key, [[MLXArray]])? in
            guard storing, range.lowerBound >= start, range.count >= Self.minimumChunk, segment.cached == nil,
                  let sliced = layers(range)
            else { return nil }
            return (Key(namespace: namespace, tokens: Array(prompt[range])), sliced.map { $0.map { contiguous($0) } })
        }
        eval(stored.flatMap { $0.1.joined() })
        withState { state in
            for (key, layers) in stored where state.entries[key] == nil {
                let bytes = layers.joined().reduce(0) { $0 + $1.nbytes }
                guard bytes <= byteBudget else { continue }
                state.clock += 1
                state.entries[key] = Entry(layers: layers, bytes: bytes, lastUse: state.clock)
                state.bytes += bytes
            }
            while state.bytes > byteBudget, let victim = state.entries.min(by: { $0.value.lastUse < $1.value.lastUse }) {
                state.bytes -= victim.value.bytes
                state.entries[victim.key] = nil
            }
        }
        return CacheBlendOutcome(
            cache: PromptCache(caches), tokens: count, reused: reused,
            recomputed: recomputed, exactPrefix: exactPrefix)
    }
}
