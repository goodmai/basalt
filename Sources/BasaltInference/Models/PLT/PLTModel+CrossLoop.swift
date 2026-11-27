import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension PLTAttention {
    func crossLoop(_ x: MLXArray, caches: [KVCache], batch: Int) -> MLXArray {
        let p = project(x, caches: caches, batch: batch)
        var outputs = [MLXFast.scaledDotProductAttention(queries: p.queries[0], keys: p.keys[0], values: p.values[0], scale: scale, mask: .none)]
        for loop in 1 ..< caches.count {
            let global = MLXFast.scaledDotProductAttention(queries: p.queries[loop], keys: p.keys[0], values: p.values[0], scale: scale, mask: .none)
            let local = PLTMasks.window(p.queries[loop], p.keys[loop], p.values[loop], left: config.window, scale: scale)
            let g = gate(x[(loop * batch) ..< ((loop + 1) * batch)])
            outputs.append(g * global + (1 - g) * local)
        }
        return output(concatenated(outputs, axis: 0))
    }

    private func project(_ x: MLXArray, caches: [KVCache], batch: Int) -> (queries: [MLXArray], keys: [MLXArray], values: [MLXArray]) {
        let rows = x.dim(0)
        let offset = caches[0].ropeOffset
        let queries = applyRotaryPosition(rope, to: wq(x).reshaped(rows, 1, config.heads, -1).transposed(0, 2, 1, 3), offset: offset)
        let keys = applyRotaryPosition(rope, to: wk(x).reshaped(rows, 1, config.kvHeads, -1).transposed(0, 2, 1, 3), offset: offset)
        let values = wv(x).reshaped(rows, 1, config.kvHeads, -1).transposed(0, 2, 1, 3)
        var result: (queries: [MLXArray], keys: [MLXArray], values: [MLXArray]) = ([], [], [])
        for (loop, cache) in caches.enumerated() {
            let range = (loop * batch) ..< ((loop + 1) * batch)
            let (k, v) = cache.update(keys: keys[range], values: values[range])
            result.queries.append(queries[range])
            result.keys.append(k)
            result.values.append(v)
        }
        return result
    }
}

extension PLTDecoderLayer {
    func crossLoop(_ x: MLXArray, caches: [KVCache], batch: Int) -> MLXArray {
        let h = x + attention.crossLoop(inputNorm(x), caches: caches, batch: batch)
        return h + mlp(postNorm(h))
    }
}

extension PLTInner {
    func crossLoopStep(_ tokens: MLXArray, cache: [KVCache]) -> MLXArray {
        let embedded = embedTokens(tokens)
        let batch = embedded.dim(0)
        let inputs = [embedded] + (1 ..< config.loops).map { loop in
            config.embeddingScale * embedded + config.hiddenScale * carried(cache[shiftIndex(loop)], like: embedded)
        }
        var h = concatenated(inputs, axis: 0)
        for (index, layer) in layers.enumerated() {
            h = layer.crossLoop(h, caches: (0 ..< config.loops).map { cache[$0 * config.layers + index] }, batch: batch)
        }
        for loop in 1 ..< config.loops {
            let previous = h[((loop - 1) * batch) ..< (loop * batch)]
            let stored = config.normalizePerLoop ? norm(previous) : previous
            _ = cache[shiftIndex(loop)].update(keys: expandedDimensions(stored, axis: 1), values: MLXArray.zeros([batch, 1, 1, 1], dtype: stored.dtype))
        }
        return norm(h[((config.loops - 1) * batch)...])
    }

    func carried(_ cache: KVCache, like hidden: MLXArray) -> MLXArray {
        guard cache.offset > 0, let stored = cache.state.first else { return MLXArray.zeros(like: hidden) }
        return stored[0..., 0, (cache.offset - 1) ..< cache.offset, 0...].asType(hidden.dtype)
    }
}
