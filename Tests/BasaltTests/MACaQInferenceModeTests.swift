import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing
@testable import BasaltInference

@Suite("MACaQ expert layers can run float32 activations in their scale dtype: MLX's mixed float32 × bfloat16 gather matmul is about 3x slower")
struct MACaQInferenceModeTests {
    static func layer() -> MACaQSwitchLinear {
        let (experts, outs, cols) = (4, 32, 64)
        let w = MLXRandom.normal([experts, outs, cols], key: MLXRandom.key(7101))
        var (wq, sc, bi) = ([MLXArray](), [MLXArray](), [MLXArray]())
        for e in 0 ..< experts {
            let q = MLX.quantized(w[e], groupSize: 32, bits: 4, mode: .affine)
            wq.append(q.wq)
            sc.append(q.scales.asType(.bfloat16))
            bi.append(q.biases!.asType(.bfloat16))
        }
        var spec = MACaQLayerSpec(bits: 4, groupSize: 32, mode: .affine, hadamardBlock: 32)
        spec.perExpertPreScale = true
        spec.hasBias = true
        let layer = MACaQSwitchLinear(
            geometry: MACaQSwitchGeometry(experts: experts, outputs: outs, inputs: cols),
            packed: MACaQPackedWeights(weight: stacked(wq), scales: stacked(sc), biases: stacked(bi)),
            bias: MLXRandom.normal([experts, outs], key: MLXRandom.key(7102)).asType(.bfloat16),
            preScale: MLXArray.ones([experts, cols], dtype: .bfloat16), spec: spec)
        return layer
    }

    @Test("After runInScaleDType a float32 input returns float32 and matches the same input given in the scale dtype")
    func float32InputMatchesScaleDType() {
        let layer = Self.layer()
        MACaQSwitchLinear.runInScaleDType(layer)
        #expect(layer.activationDType == .bfloat16)
        let x = MLXRandom.normal([1, 1, 1, 1, 64], key: MLXRandom.key(7103)).asType(.bfloat16)
        let indices = MLXArray([Int32(0), 3]).reshaped(1, 1, 2)
        let fromFloat = layer(x.asType(.float32), indices)
        let fromHalf = layer(x, indices)
        eval(fromFloat, fromHalf)
        #expect(fromFloat.dtype == .float32)
        #expect(fromHalf.dtype == .bfloat16)
        #expect(abs(fromFloat - fromHalf.asType(.float32)).max().item(Float.self) == 0)
    }

    @Test("Without it (PV-Tuning, conversion) float32 activations stay float32")
    func float32StaysWithoutOptIn() {
        let layer = Self.layer()
        let x = MLXRandom.normal([1, 1, 1, 1, 64], key: MLXRandom.key(7104))
        let indices = MLXArray([Int32(1), 2]).reshaped(1, 1, 2)
        let full = layer(x, indices)
        MACaQSwitchLinear.runInScaleDType(layer)
        let cast = layer(x, indices)
        eval(full, cast)
        #expect(layer.activationDType == .bfloat16)
        #expect(full.dtype == .float32 && cast.dtype == .float32)
        #expect(abs(full - cast).max().item(Float.self) > 0)
    }
}
