import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum PaddleOCRVLVision {
    static func bilinear(_ grid: MLXArray, height: Int, width: Int) -> MLXArray {
        func axis(_ size: Int, _ input: Int) -> (low: MLXArray, high: MLXArray, weight: MLXArray) {
            let positions = size == 1
                ? MLXArray([Float(0)])
                : (MLXArray(0 ..< size).asType(.float32) + 0.5) * Float(input) / Float(size) - 0.5
            let low = clip(floor(positions).asType(.int32), min: 0, max: input - 1)
            let high = clip(floor(positions).asType(.int32) + 1, min: 0, max: input - 1)
            return (low, high, positions - low.asType(.float32))
        }
        let (rows, columns) = (axis(height, grid.dim(0)), axis(width, grid.dim(1)))
        let flat = grid.reshaped(-1, grid.dim(-1))
        func gather(_ r: MLXArray, _ c: MLXArray) -> MLXArray {
            flat.take((r.reshaped(-1, 1) * grid.dim(1) + c.reshaped(1, -1)).flattened(), axis: 0).reshaped(height, width, -1)
        }
        let rw = rows.weight.reshaped(height, 1, 1), cw = columns.weight.reshaped(1, width, 1)
        return (1 - rw) * (1 - cw) * gather(rows.low, columns.low) + (1 - rw) * cw * gather(rows.low, columns.high)
            + rw * (1 - cw) * gather(rows.high, columns.low) + rw * cw * gather(rows.high, columns.high)
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray { PaddleOCRVLText.rotateHalf(x) }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let side: Int

        init(_ config: PaddleOCRVLConfiguration.Vision) {
            side = config.imageSize / config.patchSize
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels, outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize), stride: .init(config.patchSize))
            _positionEmbedding.wrappedValue = Embedding(embeddingCount: side * side, dimensions: config.hiddenSize)
        }

        func callAsFunction(_ patches: MLXArray, frame: THW) -> MLXArray {
            let embedded = patchEmbedding(patches).reshaped(patches.dim(0), -1)
            let table = positionEmbedding(MLXArray(0 ..< side * side)).reshaped(side, side, -1)
            let positions = PaddleOCRVLVision.bilinear(table, height: frame.h, width: frame.w).asType(table.dtype)
            return embedded + tiled(positions.reshaped(frame.h * frame.w, -1), repetitions: [frame.t, 1])
        }
    }

    final class Attention: Module {
        let heads: Int
        let scale: Float
        @ModuleInfo var qkv: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(dimensions: Int, heads: Int) {
            self.heads = heads
            scale = pow(Float(dimensions / heads), -0.5)
            _qkv.wrappedValue = Linear(dimensions, dimensions * 3, bias: true)
            _outProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let n = x.dim(0)
            let parts = qkv(x).reshaped(n, 3, heads, -1).transposed(1, 0, 2, 3)
            func rotated(_ t: MLXArray) -> MLXArray {
                (t * cos + PaddleOCRVLVision.rotateHalf(t) * sin).asType(t.dtype)
            }
            let (q, k, v) = (rotated(parts[0]), rotated(parts[1]), parts[2])
            let output = MLXFast.scaledDotProductAttention(
                queries: q.transposed(1, 0, 2)[.newAxis], keys: k.transposed(1, 0, 2)[.newAxis],
                values: v.transposed(1, 0, 2)[.newAxis], scale: scale, mask: .none)
            return outProj(output[0].transposed(1, 0, 2).reshaped(n, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dimensions: Int, hidden: Int) {
            _fc1.wrappedValue = Linear(dimensions, hidden)
            _fc2.wrappedValue = Linear(hidden, dimensions)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: PaddleOCRVLConfiguration.Vision) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dimensions: config.hiddenSize, heads: config.heads)
            _mlp.wrappedValue = MLP(dimensions: config.hiddenSize, hidden: config.intermediateSize)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let h = x + attention(norm1(x), cos: cos, sin: sin)
            return h + mlp(norm2(h))
        }
    }

    final class Projector: Module {
        let merge: Int
        @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        init(_ config: PaddleOCRVLConfiguration.Vision, textHidden: Int) {
            merge = config.mergeSize
            let merged = config.hiddenSize * merge * merge
            _preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _linear1.wrappedValue = Linear(merged, merged)
            _linear2.wrappedValue = Linear(merged, textHidden)
        }

        func callAsFunction(_ x: MLXArray, frame: THW) -> MLXArray {
            let (h, w) = (frame.h / merge, frame.w / merge)
            let blocks = preNorm(x).reshaped(frame.t, h, merge, w, merge, -1).transposed(0, 1, 3, 2, 4, 5)
            return linear2(gelu(linear1(blocks.reshaped(frame.t * h * w, -1))))
        }
    }

    final class Model: Module {
        let headDim: Int
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var layers: [EncoderLayer]
        @ModuleInfo(key: "post_layernorm") var postNorm: LayerNorm
        @ModuleInfo var projector: Projector

        init(_ config: PaddleOCRVLConfiguration.Vision, textHidden: Int) {
            headDim = config.hiddenSize / config.heads
            _embeddings.wrappedValue = Embeddings(config)
            _layers.wrappedValue = (0 ..< config.layers).map { _ in EncoderLayer(config) }
            _postNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _projector.wrappedValue = Projector(config, textHidden: textHidden)
        }

        func rotary(_ frame: THW) -> (cos: MLXArray, sin: MLXArray) {
            let dims = headDim / 2
            let invFreq = 1 / pow(MLXArray(Float(10_000)), MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32) / Float(dims))
            let ids = MLXArray(0 ..< frame.t * frame.h * frame.w) % (frame.h * frame.w)
            let table = outer(MLXArray(0 ..< max(frame.h, frame.w)).asType(.float32), invFreq)
            let freqs = concatenated([table[floorDivide(ids, frame.w)], table[ids % frame.w]], axis: -1)
            let full = tiled(freqs[0..., .newAxis, 0...], repetitions: [1, 1, 2])
            return (MLX.cos(full), MLX.sin(full))
        }

        func callAsFunction(_ patches: MLXArray, frames: [THW]) -> MLXArray {
            var start = 0
            return concatenated(frames.map { frame in
                let count = frame.t * frame.h * frame.w
                defer { start += count }
                var h = embeddings(patches[start ..< start + count], frame: frame)
                let (cos, sin) = rotary(frame)
                for layer in layers { h = layer(h, cos: cos, sin: sin) }
                return projector(postNorm(h), frame: frame)
            }, axis: 0)
        }
    }
}
