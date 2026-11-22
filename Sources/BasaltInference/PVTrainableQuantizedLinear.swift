import Foundation
import MLX
import MLXNN

struct TrainableQuantizedKeys {
    private(set) var keys: Set<String> = []

    mutating func unfrozen(_ requested: [String]?) {
        keys = Set(requested ?? []).subtracting(["weight"])
    }

    mutating func frozen(_ requested: [String]?) {
        if let requested { keys.subtract(requested) } else { keys.removeAll() }
    }
}

package final class PVAffineLinear: QuantizedLinear {
    private var trainable = TrainableQuantizedKeys()

    init(from layer: QuantizedLinear) {
        super.init(
            weight: layer.weight, bias: layer.bias, scales: layer.scales, biases: layer.biases,
            groupSize: layer.groupSize, bits: layer.bits)
        freeze()
    }

    func packedLayer() -> QuantizedLinear {
        QuantizedLinear(weight: weight, bias: bias, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }

    func allowTraining(_ keys: [String]) {
        trainable.unfrozen(keys)
    }

    package override func didSetNoGrad(_ noGrad: Set<String>) {
        super.didSetNoGrad(noGrad)
        trainable.frozen(nil)
    }

    package override func noGrad() -> Set<String> {
        super.noGrad().subtracting(trainable.keys)
    }
}
