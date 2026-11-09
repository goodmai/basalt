import Foundation
import MLX
import MLXNN

enum DeepSeekOCRSAM {
    final class Attention: Module {
        let heads: Int
        let scale: Float
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear
        @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray
        @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray

        init(dimensions: Int, heads: Int, side: Int) {
            self.heads = heads
            scale = pow(Float(dimensions / heads), -0.5)
            _qkv.wrappedValue = Linear(dimensions, dimensions * 3)
            _proj.wrappedValue = Linear(dimensions, dimensions)
            _relPosH.wrappedValue = zeros([2 * side - 1, dimensions / heads])
            _relPosW.wrappedValue = zeros([2 * side - 1, dimensions / heads])
        }

        func callAsFunction(_ x: MLXArray, queryRows: Int = 1024) -> MLXArray {
            let (b, h, w) = (x.dim(0), x.dim(1), x.dim(2))
            let parts = qkv(x).reshaped(b, h * w, 3, heads, -1).transposed(2, 0, 3, 1, 4)
            let (q, k, v) = (parts[0], parts[1], parts[2])
            let (tableH, tableW) = (DeepSeekOCRResample.relative(relPosH, side: h), DeepSeekOCRResample.relative(relPosW, side: w))
            let rows = max(1, queryRows / w)
            let blocks = stride(from: 0, to: h, by: rows).map { first in
                let last = min(first + rows, h)
                let block = q[0..., 0..., (first * w) ..< (last * w)]
                let grid = block.reshaped(b * heads, last - first, w, -1)
                let relH = matmul(grid, tableH[first ..< last].transposed(0, 2, 1))
                let relW = matmul(grid.transposed(0, 2, 1, 3), tableW.transposed(0, 2, 1)).transposed(0, 2, 1, 3)
                let bias = (relH.expandedDimensions(axis: -1) + relW.expandedDimensions(axis: -2)).reshaped(b, heads, (last - first) * w, h * w)
                return MLXFast.scaledDotProductAttention(queries: block, keys: k, values: v, scale: scale, mask: .array(bias))
            }
            return proj(concatenated(blocks, axis: 2).transposed(0, 2, 1, 3).reshaped(b, h, w, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var lin1: Linear
        @ModuleInfo var lin2: Linear

        init(dimensions: Int, hidden: Int) {
            _lin1.wrappedValue = Linear(dimensions, hidden)
            _lin2.wrappedValue = Linear(hidden, dimensions)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { lin2(gelu(lin1(x))) }
    }

    final class Block: Module {
        let window: Int
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: DeepSeekOCRConfiguration.SAM, layer: Int, grid: Int) {
            window = config.globalAttentionLayers.contains(layer) ? 0 : config.window
            _norm1.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-6)
            _attn.wrappedValue = Attention(dimensions: config.width, heads: config.heads, side: window > 0 ? window : grid)
            _mlp.wrappedValue = MLP(dimensions: config.width, hidden: config.width * 4)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = norm1(x)
            let attended = window > 0 ? windowed(h) : attn(h)
            let y = x + attended
            return y + mlp(norm2(y))
        }

        func windowed(_ x: MLXArray) -> MLXArray {
            let (b, h, w, c, s) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), window)
            let (ph, pw) = ((s - h % s) % s, (s - w % s) % s)
            let padded = ph + pw > 0 ? MLX.padded(x, widths: [0, .init((0, ph)), .init((0, pw)), 0]) : x
            let (hp, wp) = (h + ph, w + pw)
            let windows = padded.reshaped(b, hp / s, s, wp / s, s, c).transposed(0, 1, 3, 2, 4, 5).reshaped(-1, s, s, c)
            let merged = attn(windows).reshaped(b, hp / s, wp / s, s, s, -1).transposed(0, 1, 3, 2, 4, 5).reshaped(b, hp, wp, -1)
            return merged[0..., ..<h, ..<w]
        }
    }

    final class PatchEmbed: Module {
        @ModuleInfo var proj: Conv2d

        init(_ config: DeepSeekOCRConfiguration.SAM) {
            _proj.wrappedValue = Conv2d(
                inputChannels: 3, outputChannels: config.width, kernelSize: .init(config.patchSize), stride: .init(config.patchSize))
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
        let blocks: [Block]
        let neck: [UnaryLayer]
        @ModuleInfo(key: "net_2") var net2: Conv2d
        @ModuleInfo(key: "net_3") var net3: Conv2d

        init(_ config: DeepSeekOCRConfiguration.SAM, imageSize: Int) {
            let grid = imageSize / config.patchSize
            let (neck, down) = (config.neckChannels, config.downsampleChannels)
            _patchEmbed.wrappedValue = PatchEmbed(config)
            _posEmbed.wrappedValue = zeros([1, grid, grid, config.width])
            blocks = (0 ..< config.layers).map { Block(config, layer: $0, grid: grid) }
            self.neck = [
                Conv2d(inputChannels: config.width, outputChannels: neck, kernelSize: 1, bias: false),
                LayerNorm(dimensions: neck, eps: 1e-6),
                Conv2d(inputChannels: neck, outputChannels: neck, kernelSize: 3, padding: 1, bias: false),
                LayerNorm(dimensions: neck, eps: 1e-6),
            ]
            _net2.wrappedValue = Conv2d(inputChannels: neck, outputChannels: down[0], kernelSize: 3, stride: 2, padding: 1, bias: false)
            _net3.wrappedValue = Conv2d(inputChannels: down[0], outputChannels: down[1], kernelSize: 3, stride: 2, padding: 1, bias: false)
        }

        func callAsFunction(_ pixels: MLXArray) -> MLXArray {
            var x = patchEmbed.proj(pixels)
            x = x + DeepSeekOCRResample.positions(posEmbed[0], side: x.dim(1))[.newAxis]
            for block in blocks { x = block(x) }
            for layer in neck { x = layer(x) }
            return net3(net2(x))
        }
    }
}
