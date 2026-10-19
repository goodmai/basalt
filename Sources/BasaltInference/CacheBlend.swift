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
        tokens: [Int], offset: Int, past: [[MLXArray]], segments: [CacheBlendSegment], ratio: Float
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
            if let gather, active.isEmpty {
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
                    let reused = MLXArray(segments.flatMap { Array(repeating: $0.cached != nil, count: $0.range.count) })
                    let ranked = argSort(which(reused, deviation, MLXArray(Float.infinity))).asArray(Int32.self)
                    active = ranked.dropFirst(keep).map(Int.init).sorted()
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
            let positions = MLXArray(active.map { Int32(offset + $0) })
            let rotatedQueries = layer.rope(queries.transposed(2, 1, 0, 3), offset: positions).transposed(2, 1, 0, 3)
            let mask = positions.expandedDimensions(axis: 1)
                .>= MLXArray((0 ..< offset + count).map(Int32.init)).expandedDimensions(axis: 0)
            let attended = MLXFast.scaledDotProductAttention(
                queries: rotatedQueries, keys: allKeys, values: allValues, scale: scale, mask: .array(mask))
            hidden = hidden + layer.o(attended.transposed(0, 2, 1, 3).reshaped(1, active.count, -1))
            hidden = hidden + layer.mlp(layer.postNorm(hidden))
            eval(hidden, allKeys, allValues)
        }
        let skipped = Set(0 ..< count).subtracting(active)
        return CacheBlendPass(
            caches: caches, layers: kept, recomputed: active.count - (count - hits),
            exactPrefix: skipped.min() ?? count)
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
        var boundaries: [Int: Bool] = [:]
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

    static func chunks(_ tokens: [Int], isBoundary: (Int) -> Bool) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var start = 0
        for (index, token) in tokens.enumerated() where isBoundary(token) {
            ranges.append(start ..< index + 1)
            start = index + 1
        }
        if start < tokens.count { ranges.append(start ..< tokens.count) }
        return ranges
    }

    func chunks(_ tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer) -> [Range<Int>] {
        withState { state in
            Self.chunks(tokens) { token in
                if let known = state.boundaries[token] { return known }
                let boundary = tokenizer.decode(tokenIds: [token], skipSpecialTokens: false).contains("\n\n")
                state.boundaries[token] = boundary
                return boundary
            }
        }
    }

    func extend(
        _ cache: PromptCache, prompt: [Int], from start: Int, namespace: String,
        model: any LanguageModel, tokenizer: any MLXLMCommon.Tokenizer
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
                guard range.lowerBound >= start, var entry = state.entries[key] else {
                    return CacheBlendSegment(range: local, cached: nil)
                }
                state.clock += 1
                entry.lastUse = state.clock
                state.entries[key] = entry
                return CacheBlendSegment(range: local, cached: entry.layers)
            }
        }
        let pass = blender.prefill(
            tokens: Array(prompt[start ..< end]), offset: start,
            past: start == 0 ? [] : cache.layers.map(\.state), segments: segments, ratio: ratio)
        let stored = zip(ranges, segments).filter { $0.0.lowerBound >= start && $0.1.cached == nil }.map { range, segment in
            (Key(namespace: namespace, tokens: Array(prompt[range])),
             pass.layers.map { layer in layer.map { contiguous($0[.ellipsis, segment.range, 0...]) } })
        }
        eval(stored.flatMap { $0.1.joined() })
        withState { state in
            for (key, layers) in stored where state.entries[key] == nil {
                state.clock += 1
                let bytes = layers.joined().reduce(0) { $0 + $1.nbytes }
                state.entries[key] = Entry(layers: layers, bytes: bytes, lastUse: state.clock)
                state.bytes += bytes
            }
            while state.bytes > byteBudget, let victim = state.entries.min(by: { $0.value.lastUse < $1.value.lastUse }) {
                state.bytes -= victim.value.bytes
                state.entries[victim.key] = nil
            }
        }
        let reused = segments.filter { $0.cached != nil }.reduce(0) { $0 + $1.range.count }
        return CacheBlendOutcome(
            cache: PromptCache(pass.caches), tokens: end - start, reused: reused,
            recomputed: pass.recomputed, exactPrefix: start + pass.exactPrefix)
    }
}
