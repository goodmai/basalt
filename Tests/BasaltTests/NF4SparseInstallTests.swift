import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon

@Suite("NF4 in some layers and affine in the rest, as a converted MoE checkpoint with affine experts (#357)")
struct NF4SparseInstallTests {
    final class Block: Module {
        @ModuleInfo var proj: Linear
        @ModuleInfo var expert: Linear

        override init() {
            _proj.wrappedValue = Linear(64, 64, bias: false)
            _expert.wrappedValue = Linear(64, 64, bias: false)
        }
    }

    final class Stack: Module {
        let layers: [Block]

        override init() { layers = (0 ..< 3).map { _ in Block() } }
    }

    @Test("install and quantizeRest keep the layer array and place each module, where one sparse update over layers crashed")
    func sparseInstall() throws {
        let stack = Stack()
        let dense = MLXRandom.normal([64, 64], key: MLXRandom.key(6))
        let (codes, scales) = try NF4.quantized(dense, groupSize: 32)
        let (wq, affineScales, biases) = quantized(dense, groupSize: 64, bits: 4)
        var weights = ["layers.1.proj.weight": codes, "layers.1.proj.scales": scales]
        for index in [0, 2] {
            weights["layers.\(index).expert.weight"] = wq
            weights["layers.\(index).expert.scales"] = affineScales
            weights["layers.\(index).expert.biases"] = try #require(biases)
        }
        try NF4Linear.install(in: stack, weights: weights)
        NF4Linear.quantizeRest(in: stack, weights: weights) { _ in (64, 4, .affine) }
        #expect(stack.layers.count == 3)
        #expect(stack.layers[1].proj is NF4Linear)
        #expect(stack.layers[0].expert is QuantizedLinear && stack.layers[2].expert is QuantizedLinear)
        #expect(!(stack.layers[0].proj is NF4Linear) && !(stack.layers[1].expert is QuantizedLinear))
    }
}
