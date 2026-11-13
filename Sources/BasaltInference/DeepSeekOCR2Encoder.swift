import BasaltCore
import Foundation
import MLX
import MLXNN

enum DeepSeekOCR2Encoder {
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let scale: Float
        let rope: RoPE
        @ModuleInfo(key: "q_proj") var queryProjection: Linear
        @ModuleInfo(key: "k_proj") var keyProjection: Linear
        @ModuleInfo(key: "v_proj") var valueProjection: Linear
        @ModuleInfo(key: "o_proj") var outputProjection: Linear

        init(_ config: DeepSeekOCR2Configuration.Encoder) {
            let headDim = config.dim / config.attentionHeads
            (heads, kvHeads, scale) = (config.attentionHeads, config.keyValueHeads, pow(Float(headDim), -0.5))
            rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
            _queryProjection.wrappedValue = Linear(config.dim, heads * headDim, bias: true)
            _keyProjection.wrappedValue = Linear(config.dim, kvHeads * headDim, bias: true)
            _valueProjection.wrappedValue = Linear(config.dim, kvHeads * headDim, bias: true)
            _outputProjection.wrappedValue = Linear(heads * headDim, config.dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
            let (b, l) = (x.dim(0), x.dim(1))
            let q = rope(queryProjection(x).reshaped(b, l, heads, -1).transposed(0, 2, 1, 3))
            let k = rope(keyProjection(x).reshaped(b, l, kvHeads, -1).transposed(0, 2, 1, 3))
            let v = valueProjection(x).reshaped(b, l, kvHeads, -1).transposed(0, 2, 1, 3)
            let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .array(mask))
            return outputProjection(attended.transposed(0, 2, 1, 3).reshaped(b, l, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(_ config: DeepSeekOCR2Configuration.Encoder) {
            _gate.wrappedValue = Linear(config.dim, config.hiddenSize, bias: false)
            _up.wrappedValue = Linear(config.dim, config.hiddenSize, bias: false)
            _down.wrappedValue = Linear(config.hiddenSize, config.dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

        init(_ config: DeepSeekOCR2Configuration.Encoder) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _inputNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.eps)
            _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.eps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
            let h = x + attention(inputNorm(x), mask: mask)
            return h + mlp(postAttentionNorm(h))
        }
    }

    final class Queries: Module {
        @ModuleInfo(key: "qwen2_encoder") var encoder: Encoder

        init(_ config: DeepSeekOCR2Configuration.Encoder) {
            _encoder.wrappedValue = Encoder(config)
        }
    }

    final class Encoder: Module {
        @ParameterInfo(key: "query_768") var tileQueries: MLXArray
        @ParameterInfo(key: "query_1024") var globalQueries: MLXArray
        let layers: [Layer]
        let norm: RMSNorm

        init(_ config: DeepSeekOCR2Configuration.Encoder) {
            _tileQueries.wrappedValue = zeros([144, config.dim])
            _globalQueries.wrappedValue = zeros([256, config.dim])
            layers = (0 ..< config.layers).map { _ in Layer(config) }
            norm = RMSNorm(dimensions: config.dim, eps: config.eps)
        }

        static func mask(images: Int) -> MLXArray {
            let (row, column) = (MLXArray(0 ..< 2 * images)[0..., .newAxis], MLXArray(0 ..< 2 * images)[.newAxis])
            return (column .< images) .|| (column .<= row)
        }

        func callAsFunction(_ samFeatures: MLXArray) throws(BasaltError) -> MLXArray {
            let (b, dim) = (samFeatures.dim(0), samFeatures.dim(-1))
            let images = samFeatures.reshaped(b, -1, dim)
            let count = images.dim(1)
            guard let queries = [tileQueries, globalQueries].first(where: { $0.dim(0) == count }) else {
                throw .invalidRequestStructure(details: "DeepSeek-OCR-2 has learned queries for 768 and 1024 pixel views, not for a view of \(count) patches")
            }
            var h = concatenated([images, broadcast(queries.asType(images.dtype)[.newAxis], to: [b, count, dim])], axis: 1)
            let mask = Self.mask(images: count)
            for layer in layers { h = layer(h, mask: mask) }
            return norm(h[0..., count...])
        }
    }
}
