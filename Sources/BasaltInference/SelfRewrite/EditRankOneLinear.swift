import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class EditRankOneLinear: Linear, LoRALayer {
    private struct BaseLinearBox {
        let linear: Linear
    }

    private let wrapped: BaseLinearBox
    let scale: Float
    var loraEnabled: Bool = true
    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    var baseLinear: Linear { wrapped.linear }

    override var shape: (Int, Int) { wrapped.linear.shape }

    init(base: Linear, loraA: MLXArray, loraB: MLXArray, scale: Float) {
        self.wrapped = BaseLinearBox(linear: base)
        self.scale = scale
        self._loraA.wrappedValue = loraA
        self._loraB.wrappedValue = loraB
        super.init(weight: base.weight, bias: base.bias)
        train(base.training)
        freeze()
    }

    override func freeze(recursive: Bool = true, keys: [String]? = nil, strict: Bool = false) throws {
        let local = keys ?? filterMap(filter: Self.filterLocalParameters).flattened().map(\.0)
        let filtered = local.filter { $0 != "lora_a" && $0 != "lora_b" }
        try super.freeze(recursive: recursive, keys: filtered, strict: strict)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = wrapped.linear(x)
        if !loraEnabled { return y }
        let z = matmul(matmul(x.asType(loraA.dtype), loraA), loraB)
        return y + (scale * z).asType(y.dtype)
    }

    func reverted() -> Module {
        wrapped.linear
    }

    func fused() -> Module {
        let denseBase = TensorPacker.dense(wrapped.linear).asType(.float32)
        let delta = scale * matmul(loraB.T.asType(.float32), loraA.T.asType(.float32))
        return EditOpExecutor.repackLinear(wrapped.linear, editedFloat32: denseBase + delta)
    }
}
