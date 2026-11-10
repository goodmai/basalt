import Foundation
import MLX
import MLXNN

enum DeepSeekOCRResample {
    static func cubic(_ distance: Double, a: Double) -> Double {
        let t = abs(distance)
        if t <= 1 { return ((a + 2) * t - (a + 3)) * t * t + 1 }
        return t < 2 ? (((t - 5) * t + 8) * t - 4) * a : 0
    }

    static func antialiased(from input: Int, to output: Int) -> [[Double]] {
        let stretch = output < input ? Double(input) / Double(output) : 1
        return (0 ..< output).map { o in
            let center = (Double(o) + 0.5) / Double(output) * Double(input) - 0.5
            let weights = (0 ..< input).map { cubic((Double($0) - center) / stretch, a: -0.5) }
            let total = weights.reduce(0, +)
            return total > 1e-8 ? weights.map { $0 / total } : weights
        }
    }

    static func linear(from input: Int, to output: Int) -> [[Double]] {
        let scale = Double(input) / Double(output)
        return (0 ..< output).map { o in
            let position = max((Double(o) + 0.5) * scale - 0.5, 0)
            let low = min(Int(position.rounded(.down)), input - 1)
            var row = [Double](repeating: 0, count: input)
            row[low] += 1 - (position - Double(low))
            row[min(low + 1, input - 1)] += position - Double(low)
            return row
        }
    }

    static func matrix(_ rows: [[Double]]) -> MLXArray {
        MLXArray(rows.flatMap { $0.map(Float.init) }, [rows.count, rows.first?.count ?? 0])
    }

    static func positions(_ grid: MLXArray, side: Int) -> MLXArray {
        guard grid.dim(0) != side else { return grid }
        let weights = matrix(antialiased(from: grid.dim(0), to: side))
        let rows = matmul(weights, grid.asType(.float32).reshaped(grid.dim(0), -1)).reshaped(side, grid.dim(1), -1)
        return matmul(weights[.newAxis], rows).asType(grid.dtype)
    }

    static func relative(_ table: MLXArray, side: Int) -> MLXArray {
        let length = 2 * side - 1
        let resized = table.dim(0) == length
            ? table : matmul(matrix(linear(from: table.dim(0), to: length)), table.asType(.float32)).asType(table.dtype)
        let index = (0 ..< side).flatMap { i in (0 ..< side).map { Int32(i - $0 + side - 1) } }
        return resized[MLXArray(index, [side, side])]
    }
}

enum DeepSeekOCRCLIP {
    final class Embeddings: Module {
        @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: DeepSeekOCRConfiguration.CLIP) {
            let side = config.imageSize / config.patchSize
            _classEmbedding.wrappedValue = zeros([config.width])
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: 3, outputChannels: config.width, kernelSize: .init(config.patchSize), stride: .init(config.patchSize), bias: false)
            _positionEmbedding.wrappedValue = Embedding(embeddingCount: side * side + 1, dimensions: config.width)
        }

        func callAsFunction(_ samFeatures: MLXArray) -> MLXArray {
            let (b, side) = (samFeatures.dim(0), samFeatures.dim(1))
            let dtype = positionEmbedding.weight.dtype
            let patches = samFeatures.reshaped(b, side * side, -1)
            let cls = broadcast(classEmbedding, to: [b, 1, classEmbedding.dim(0)]).asType(dtype)
            let table = positionEmbedding.weight
            let source = Int(Double(table.dim(0) - 1).squareRoot())
            let grid = DeepSeekOCRResample.positions(table[1...].reshaped(source, source, -1), side: side)
            let positions = concatenated([table[..<1], grid.reshaped(side * side, -1)], axis: 0)
            return concatenated([cls, patches], axis: 1) + positions.asType(dtype)
        }
    }

    final class Attention: Module {
        let heads: Int
        let scale: Float
        @ModuleInfo(key: "qkv_proj") var qkv: Linear
        @ModuleInfo(key: "out_proj") var out: Linear

        init(dimensions: Int, heads: Int) {
            self.heads = heads
            scale = pow(Float(dimensions / heads), -0.5)
            _qkv.wrappedValue = Linear(dimensions, dimensions * 3)
            _out.wrappedValue = Linear(dimensions, dimensions)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (b, l) = (x.dim(0), x.dim(1))
            let parts = split(qkv(x), parts: 3, axis: -1).map { $0.reshaped(b, l, heads, -1).transposed(0, 2, 1, 3) }
            let output = MLXFast.scaledDotProductAttention(queries: parts[0], keys: parts[1], values: parts[2], scale: scale, mask: .none)
            return out(output.transposed(0, 2, 1, 3).reshaped(b, l, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(dimensions: Int, hidden: Int) {
            _fc1.wrappedValue = Linear(dimensions, hidden)
            _fc2.wrappedValue = Linear(hidden, dimensions)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = fc1(x)
            return fc2(h * sigmoid(1.702 * h))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: DeepSeekOCRConfiguration.CLIP) {
            _attention.wrappedValue = Attention(dimensions: config.width, heads: config.heads)
            _norm1.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-5)
            _norm2.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-5)
            _mlp.wrappedValue = MLP(dimensions: config.width, hidden: config.width * 4)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attention(norm1(x))
            return h + mlp(norm2(h))
        }
    }

    final class Transformer: Module {
        let layers: [EncoderLayer]

        init(_ config: DeepSeekOCRConfiguration.CLIP) {
            layers = (0 ..< config.layers).map { _ in EncoderLayer(config) }
        }
    }

    final class Model: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo(key: "pre_layrnorm") var preNorm: LayerNorm
        @ModuleInfo var transformer: Transformer

        init(_ config: DeepSeekOCRConfiguration.CLIP) {
            _embeddings.wrappedValue = Embeddings(config)
            _preNorm.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-5)
            _transformer.wrappedValue = Transformer(config)
        }

        func callAsFunction(_ samFeatures: MLXArray) -> MLXArray {
            var x = preNorm(embeddings(samFeatures))
            for layer in transformer.layers { x = layer(x) }
            return x
        }
    }
}
